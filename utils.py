import whisper
import ffmpeg
from deep_translator import GoogleTranslator
import tempfile
import os
import soundfile as sf
import librosa

def is_valid_video_format(filename):
    """
    Check if the video format is supported
    """
    valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4a']
    return os.path.splitext(filename.lower())[1] in valid_extensions

def extract_audio(video_path):
    """
    Extract audio from video or audio file using ffmpeg
    """
    try:
        # Determine file type
        file_extension = os.path.splitext(video_path)[1].lower()
        
        # Create temporary file for audio
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_audio_path = temp_audio.name
        temp_audio.close()

        # If it's already an audio file, just convert it
        if file_extension in ['.m4a', '.mp3', '.wav', '.flac', '.ogg']:
            # Read the audio file
            audio, sample_rate = librosa.load(video_path, sr=None)
            
            # Write to wav
            sf.write(temp_audio_path, audio, sample_rate)
            
            return temp_audio_path

        # If it's a video file, use ffmpeg
        (
            ffmpeg
            .input(video_path)
            .output(temp_audio_path, acodec='pcm_s16le', ac=1, ar='16000')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )

        return temp_audio_path

    except Exception as e:
        raise Exception(f"Error extracting audio: {str(e)}")

def format_timestamp(seconds):
    """
    Format seconds into HH:MM:SS
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def format_timestamp_srt(seconds):
    """
    Format seconds into SRT timestamp format: HH:MM:SS,mmm
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def format_timestamp_vtt(seconds):
    """
    Format seconds into VTT timestamp format: HH:MM:SS.mmm
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

def parse_transcript_segments(transcript_text):
    """
    Parse transcript text into segments with timestamps.
    Returns list of (start_seconds, text) tuples.
    """
    import re
    segments = []
    lines = transcript_text.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip chunk markers
        if line.startswith('[CHUNK'):
            continue

        # Match timestamp pattern [HH:MM:SS]
        match = re.match(r'\[(\d{2}):(\d{2}):(\d{2})\]\s*(.*)', line)
        if match:
            hours, minutes, seconds, text = match.groups()
            start_time = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
            if text.strip():
                segments.append((start_time, text.strip()))
        elif line and not line.startswith('['):
            # Line without timestamp - append to previous segment or create new one
            if segments:
                prev_time, prev_text = segments[-1]
                segments[-1] = (prev_time, prev_text + ' ' + line)
            else:
                segments.append((0, line))

    return segments

def export_to_srt(transcript_text, default_duration=5):
    """
    Convert transcript text to SRT subtitle format.

    Args:
        transcript_text: Transcript with [HH:MM:SS] timestamps
        default_duration: Default duration for last segment in seconds

    Returns:
        SRT formatted string
    """
    segments = parse_transcript_segments(transcript_text)
    if not segments:
        return ""

    srt_lines = []
    for i, (start_time, text) in enumerate(segments):
        # Calculate end time from next segment or add default duration
        if i < len(segments) - 1:
            end_time = segments[i + 1][0]
        else:
            end_time = start_time + default_duration

        # Ensure end time is after start time
        if end_time <= start_time:
            end_time = start_time + default_duration

        # SRT format: index, timestamps, text, blank line
        srt_lines.append(str(i + 1))
        srt_lines.append(f"{format_timestamp_srt(start_time)} --> {format_timestamp_srt(end_time)}")
        srt_lines.append(text)
        srt_lines.append("")

    return "\n".join(srt_lines)

def export_to_vtt(transcript_text, default_duration=5):
    """
    Convert transcript text to WebVTT subtitle format.

    Args:
        transcript_text: Transcript with [HH:MM:SS] timestamps
        default_duration: Default duration for last segment in seconds

    Returns:
        VTT formatted string
    """
    segments = parse_transcript_segments(transcript_text)
    if not segments:
        return "WEBVTT\n\n"

    vtt_lines = ["WEBVTT", ""]
    for i, (start_time, text) in enumerate(segments):
        # Calculate end time from next segment or add default duration
        if i < len(segments) - 1:
            end_time = segments[i + 1][0]
        else:
            end_time = start_time + default_duration

        # Ensure end time is after start time
        if end_time <= start_time:
            end_time = start_time + default_duration

        # VTT format: optional cue id, timestamps, text, blank line
        vtt_lines.append(str(i + 1))
        vtt_lines.append(f"{format_timestamp_vtt(start_time)} --> {format_timestamp_vtt(end_time)}")
        vtt_lines.append(text)
        vtt_lines.append("")

    return "\n".join(vtt_lines)

def export_to_markdown(transcript_text, filename=None, include_timestamps=True):
    """
    Convert transcript text to clean Markdown format.

    Args:
        transcript_text: Transcript with [HH:MM:SS] timestamps
        filename: Original filename for title
        include_timestamps: Whether to include timestamps

    Returns:
        Markdown formatted string
    """
    from datetime import datetime

    segments = parse_transcript_segments(transcript_text)

    # Build header
    lines = []
    title = filename if filename else "Transcription"
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}")

    # Calculate duration from last segment if available
    if segments:
        last_time = segments[-1][0]
        duration_str = format_timestamp(last_time)
        lines.append(f"**Duration:** {duration_str}")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Transcript")
    lines.append("")

    # Add segments
    for start_time, text in segments:
        if include_timestamps:
            timestamp = format_timestamp(start_time)
            lines.append(f"**[{timestamp}]** {text}")
        else:
            lines.append(text)
        lines.append("")

    return "\n".join(lines)

def export_to_json(transcript_text, filename=None):
    """
    Convert transcript text to structured JSON format.

    Args:
        transcript_text: Transcript with [HH:MM:SS] timestamps
        filename: Original filename for metadata

    Returns:
        JSON formatted string
    """
    import json
    from datetime import datetime

    segments = parse_transcript_segments(transcript_text)

    # Build structured data
    data = {
        "metadata": {
            "title": filename if filename else "Transcription",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "generated_by": "Video-Transcription"
        },
        "segments": []
    }

    # Calculate duration from last segment
    if segments:
        last_time = segments[-1][0]
        data["metadata"]["duration_seconds"] = last_time
        data["metadata"]["duration_formatted"] = format_timestamp(last_time)

    # Add segments with calculated end times
    for i, (start_time, text) in enumerate(segments):
        if i < len(segments) - 1:
            end_time = segments[i + 1][0]
        else:
            end_time = start_time + 5  # Default 5 second duration for last segment

        segment_data = {
            "index": i + 1,
            "start": start_time,
            "end": end_time,
            "start_formatted": format_timestamp(start_time),
            "end_formatted": format_timestamp(end_time),
            "text": text
        }
        data["segments"].append(segment_data)

    return json.dumps(data, indent=2, ensure_ascii=False)

def translate_text(text, target_language):
    """
    Translate text to target language
    """
    try:
        translator = GoogleTranslator(source='auto', target=target_language.lower())
        
        if '[' in text and ']' in text:  # Text contains timestamps
            # Split the text into segments and translate each segment separately
            segments = text.split('\n')
            translated_segments = []
            
            for segment in segments:
                if segment.strip():  # Skip empty lines
                    # Extract timestamp if present
                    timestamp = ''
                    content = segment
                    if segment.startswith('[') and ']' in segment:
                        timestamp = segment[:segment.index(']') + 1]
                        content = segment[segment.index(']') + 1:].strip()
                    
                    # Translate content if it's not empty
                    if content:
                        translated_content = translator.translate(content)
                        translated_segments.append(f"{timestamp} {translated_content}" if timestamp else translated_content)
                    else:
                        translated_segments.append(segment)
                else:
                    translated_segments.append(segment)
            
            return '\n'.join(translated_segments)
        else:
            # Translate the entire text at once if no timestamps
            return translator.translate(text)
    except Exception as e:
        raise Exception(f"Error translating text: {str(e)}")

def get_available_languages():
    """
    Returns a dictionary of available languages for translation
    """
    return {
        'af': 'Afrikaans', 'sq': 'Albanian', 'am': 'Amharic', 'ar': 'Arabic',
        'hy': 'Armenian', 'az': 'Azerbaijani', 'eu': 'Basque', 'be': 'Belarusian',
        'bn': 'Bengali', 'bs': 'Bosnian', 'bg': 'Bulgarian', 'ca': 'Catalan',
        'ceb': 'Cebuano', 'zh-CN': 'Chinese (Simplified)', 'zh-TW': 'Chinese (Traditional)',
        'co': 'Corsican', 'hr': 'Croatian', 'cs': 'Czech', 'da': 'Danish',
        'nl': 'Dutch', 'en': 'English', 'eo': 'Esperanto', 'et': 'Estonian',
        'fi': 'Finnish', 'fr': 'French', 'fy': 'Frisian', 'gl': 'Galician',
        'ka': 'Georgian', 'de': 'German', 'el': 'Greek', 'gu': 'Gujarati',
        'ht': 'Haitian Creole', 'ha': 'Hausa', 'haw': 'Hawaiian', 'he': 'Hebrew',
        'hi': 'Hindi', 'hmn': 'Hmong', 'hu': 'Hungarian', 'is': 'Icelandic',
        'ig': 'Igbo', 'id': 'Indonesian', 'ga': 'Irish', 'it': 'Italian',
        'ja': 'Japanese', 'jv': 'Javanese', 'kn': 'Kannada', 'kk': 'Kazakh',
        'km': 'Khmer', 'rw': 'Kinyarwanda', 'ko': 'Korean', 'ku': 'Kurdish',
        'ky': 'Kyrgyz', 'lo': 'Lao', 'la': 'Latin', 'lv': 'Latvian',
        'lt': 'Lithuanian', 'lb': 'Luxembourgish', 'mk': 'Macedonian',
        'mg': 'Malagasy', 'ms': 'Malay', 'ml': 'Malayalam', 'mt': 'Maltese',
        'mi': 'Maori', 'mr': 'Marathi', 'mn': 'Mongolian', 'my': 'Myanmar',
        'ne': 'Nepali', 'no': 'Norwegian', 'ny': 'Nyanja', 'or': 'Odia',
        'ps': 'Pashto', 'fa': 'Persian', 'pl': 'Polish', 'pt': 'Portuguese',
        'pa': 'Punjabi', 'ro': 'Romanian', 'ru': 'Russian', 'sm': 'Samoan',
        'gd': 'Scots Gaelic', 'sr': 'Serbian', 'st': 'Sesotho', 'sn': 'Shona',
        'sd': 'Sindhi', 'si': 'Sinhala', 'sk': 'Slovak', 'sl': 'Slovenian',
        'so': 'Somali', 'es': 'Spanish', 'su': 'Sundanese', 'sw': 'Swahili',
        'sv': 'Swedish', 'tl': 'Tagalog', 'tg': 'Tajik', 'ta': 'Tamil',
        'tt': 'Tatar', 'te': 'Telugu', 'th': 'Thai', 'tr': 'Turkish',
        'tk': 'Turkmen', 'uk': 'Ukrainian', 'ur': 'Urdu', 'ug': 'Uyghur',
        'uz': 'Uzbek', 'vi': 'Vietnamese', 'cy': 'Welsh', 'xh': 'Xhosa',
        'yi': 'Yiddish', 'yo': 'Yoruba', 'zu': 'Zulu'
    }

def transcribe_audio(audio_path, include_timestamps=True):
    """
    Transcribe audio file using Whisper with optional timestamps
    """
    try:
        # Load model
        model = whisper.load_model("base")
        
        # Transcribe
        result = model.transcribe(audio_path)
        
        if include_timestamps:
            # Format with timestamps
            formatted_segments = []
            for segment in result["segments"]:
                timestamp = format_timestamp(segment["start"])
                text = segment["text"].strip()
                formatted_segments.append(f"[{timestamp}] {text}")
            return "\n".join(formatted_segments)
        else:
            # Return plain text
            return result["text"]
    except Exception as e:
        raise Exception(f"Error transcribing audio: {str(e)}")
