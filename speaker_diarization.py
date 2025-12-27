"""
Speaker Diarization for Video Transcription
Detects and labels different speakers in audio/video content.
"""

import os
import re
import tempfile
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class SpeakerSegment:
    """Represents a segment of speech by a speaker."""
    speaker: str
    start: float
    end: float
    text: str = ""


@dataclass
class SpeakerStats:
    """Statistics for a single speaker."""
    speaker_id: str
    display_name: str
    total_time: float
    word_count: int
    segment_count: int
    percentage: float


def parse_timestamp_transcript(transcript: str, default_duration: int = 5) -> List[Tuple[str, str, str]]:
    """
    Parse timestamped transcript into segments with start/end times.

    Args:
        transcript: Transcript text with [HH:MM:SS] timestamps
        default_duration: Default duration for last segment in seconds

    Returns:
        List of (start_time, end_time, text) tuples where times are "HH:MM:SS" format
    """
    segments = []
    lines = transcript.strip().split('\n')

    parsed_lines = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('[CHUNK'):
            continue

        # Match timestamp pattern [HH:MM:SS]
        match = re.match(r'\[(\d{2}:\d{2}:\d{2})\]\s*(.*)', line)
        if match:
            timestamp, text = match.groups()
            if text.strip():
                parsed_lines.append((timestamp, text.strip()))
        elif line and not line.startswith('[') and parsed_lines:
            # Continuation line - append to previous
            prev_time, prev_text = parsed_lines[-1]
            parsed_lines[-1] = (prev_time, prev_text + ' ' + line)

    # Convert to (start, end, text) tuples
    for i, (start_time, text) in enumerate(parsed_lines):
        if i + 1 < len(parsed_lines):
            end_time = parsed_lines[i + 1][0]
        else:
            # Last segment: add default duration
            parts = start_time.split(':')
            total_seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2]) + default_duration
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            end_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        segments.append((start_time, end_time, text))

    return segments


class SpeakerDiarizer:
    """
    Handles speaker diarization using pyannote-audio or fallback methods.
    """

    def __init__(self, hf_token: Optional[str] = None):
        """
        Initialize the diarizer.

        Args:
            hf_token: HuggingFace token for pyannote models (optional)
        """
        self.hf_token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        self._pipeline = None
        self._available = None

    def is_available(self) -> bool:
        """Check if pyannote diarization is available."""
        if self._available is not None:
            return self._available

        try:
            from pyannote.audio import Pipeline
            self._available = self.hf_token is not None
        except ImportError:
            self._available = False

        return self._available

    def _get_pipeline(self):
        """Get or create the diarization pipeline."""
        if self._pipeline is None:
            if not self.is_available():
                raise RuntimeError("Pyannote diarization not available. Install pyannote-audio and set HF_TOKEN.")

            from pyannote.audio import Pipeline
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token
            )
        return self._pipeline

    def diarize(self, audio_path: str, num_speakers: Optional[int] = None) -> List[SpeakerSegment]:
        """
        Perform speaker diarization on an audio file.

        Args:
            audio_path: Path to the audio file
            num_speakers: Optional hint for number of speakers

        Returns:
            List of SpeakerSegment objects with speaker labels and timestamps
        """
        if not self.is_available():
            # Return empty list if diarization not available
            return []

        try:
            pipeline = self._get_pipeline()

            # Run diarization
            if num_speakers:
                diarization = pipeline(audio_path, num_speakers=num_speakers)
            else:
                diarization = pipeline(audio_path)

            # Convert to segments
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append(SpeakerSegment(
                    speaker=speaker,
                    start=turn.start,
                    end=turn.end
                ))

            return segments

        except Exception as e:
            print(f"Diarization error: {e}")
            return []

    def merge_with_transcript(
        self,
        transcript: str,
        segments: List[SpeakerSegment]
    ) -> str:
        """
        Merge diarization results with a timestamped transcript.

        Args:
            transcript: Timestamped transcript text
            segments: Speaker segments from diarization

        Returns:
            Transcript with speaker labels added
        """
        if not segments:
            return transcript

        # Parse timestamps from transcript
        # Expected format: [HH:MM:SS] or [MM:SS] text
        lines = transcript.strip().split('\n')
        result_lines = []

        timestamp_pattern = re.compile(r'\[(\d{1,2}):(\d{2}):?(\d{2})?\]')

        for line in lines:
            match = timestamp_pattern.match(line)
            if match:
                groups = match.groups()
                if groups[2] is not None:
                    # HH:MM:SS format
                    time_seconds = int(groups[0]) * 3600 + int(groups[1]) * 60 + int(groups[2])
                else:
                    # MM:SS format
                    time_seconds = int(groups[0]) * 60 + int(groups[1])

                # Find the speaker for this timestamp
                speaker = self._find_speaker_at_time(segments, time_seconds)
                if speaker:
                    # Insert speaker label after timestamp
                    timestamp_end = match.end()
                    line = f"{line[:timestamp_end]} [{speaker}]{line[timestamp_end:]}"

            result_lines.append(line)

        return '\n'.join(result_lines)

    def _find_speaker_at_time(
        self,
        segments: List[SpeakerSegment],
        time_seconds: float
    ) -> Optional[str]:
        """Find which speaker is speaking at a given time."""
        for segment in segments:
            if segment.start <= time_seconds <= segment.end:
                return segment.speaker
        return None

    def calculate_stats(
        self,
        transcript: str,
        segments: List[SpeakerSegment],
        speaker_names: Optional[Dict[str, str]] = None
    ) -> List[SpeakerStats]:
        """
        Calculate statistics per speaker.

        Args:
            transcript: The transcript text
            segments: Speaker segments
            speaker_names: Optional mapping of speaker IDs to display names

        Returns:
            List of SpeakerStats objects
        """
        if not segments:
            return []

        speaker_names = speaker_names or {}

        # Calculate time per speaker
        speaker_times = defaultdict(float)
        speaker_segments = defaultdict(int)

        for segment in segments:
            duration = segment.end - segment.start
            speaker_times[segment.speaker] += duration
            speaker_segments[segment.speaker] += 1

        total_time = sum(speaker_times.values())

        # Estimate word counts (rough estimate based on time)
        # Average speaking rate is about 150 words per minute
        words_per_second = 150 / 60

        stats = []
        for speaker_id, time in speaker_times.items():
            stats.append(SpeakerStats(
                speaker_id=speaker_id,
                display_name=speaker_names.get(speaker_id, speaker_id),
                total_time=time,
                word_count=int(time * words_per_second),
                segment_count=speaker_segments[speaker_id],
                percentage=(time / total_time * 100) if total_time > 0 else 0
            ))

        # Sort by total time descending
        stats.sort(key=lambda x: x.total_time, reverse=True)

        return stats


def format_speaker_transcript(
    transcript: str,
    speaker_segments: List[SpeakerSegment],
    speaker_names: Optional[Dict[str, str]] = None
) -> str:
    """
    Format a transcript with speaker labels for display.

    Args:
        transcript: The original transcript
        speaker_segments: Speaker diarization segments
        speaker_names: Optional custom speaker names

    Returns:
        Formatted transcript with speaker labels
    """
    if not speaker_segments:
        return transcript

    speaker_names = speaker_names or {}
    diarizer = SpeakerDiarizer()
    labeled_transcript = diarizer.merge_with_transcript(transcript, speaker_segments)

    # Replace speaker IDs with display names
    for speaker_id, display_name in speaker_names.items():
        labeled_transcript = labeled_transcript.replace(
            f"[{speaker_id}]",
            f"[{display_name}]"
        )

    return labeled_transcript


def export_with_speakers_srt(
    transcript: str,
    speaker_segments: List[SpeakerSegment],
    speaker_names: Optional[Dict[str, str]] = None
) -> str:
    """
    Export transcript to SRT format with speaker labels.

    Args:
        transcript: Timestamped transcript
        speaker_segments: Speaker diarization segments
        speaker_names: Optional custom speaker names

    Returns:
        SRT formatted string with speaker labels
    """
    # parse_timestamp_transcript is defined at module level

    speaker_names = speaker_names or {}
    diarizer = SpeakerDiarizer()

    # Parse transcript into segments
    segments = parse_timestamp_transcript(transcript)

    srt_lines = []
    for i, (start_time, end_time, text) in enumerate(segments, 1):
        # Find speaker for this segment
        start_seconds = timestamp_to_seconds(start_time)
        speaker = diarizer._find_speaker_at_time(speaker_segments, start_seconds)

        if speaker:
            display_name = speaker_names.get(speaker, speaker)
            text = f"[{display_name}] {text}"

        # Format times
        start_srt = format_srt_time(start_time)
        end_srt = format_srt_time(end_time)

        srt_lines.append(f"{i}")
        srt_lines.append(f"{start_srt} --> {end_srt}")
        srt_lines.append(text)
        srt_lines.append("")

    return '\n'.join(srt_lines)


def export_with_speakers_vtt(
    transcript: str,
    speaker_segments: List[SpeakerSegment],
    speaker_names: Optional[Dict[str, str]] = None
) -> str:
    """
    Export transcript to VTT format with speaker labels.

    Args:
        transcript: Timestamped transcript
        speaker_segments: Speaker diarization segments
        speaker_names: Optional custom speaker names

    Returns:
        VTT formatted string with speaker labels
    """
    # parse_timestamp_transcript is defined at module level

    speaker_names = speaker_names or {}
    diarizer = SpeakerDiarizer()

    # Parse transcript into segments
    segments = parse_timestamp_transcript(transcript)

    vtt_lines = ["WEBVTT", ""]
    for start_time, end_time, text in segments:
        # Find speaker for this segment
        start_seconds = timestamp_to_seconds(start_time)
        speaker = diarizer._find_speaker_at_time(speaker_segments, start_seconds)

        if speaker:
            display_name = speaker_names.get(speaker, speaker)
            text = f"<v {display_name}>{text}"

        # Format times
        start_vtt = format_vtt_time(start_time)
        end_vtt = format_vtt_time(end_time)

        vtt_lines.append(f"{start_vtt} --> {end_vtt}")
        vtt_lines.append(text)
        vtt_lines.append("")

    return '\n'.join(vtt_lines)


def timestamp_to_seconds(timestamp: str) -> float:
    """Convert timestamp string to seconds."""
    parts = timestamp.split(':')
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return 0


def format_srt_time(timestamp: str) -> str:
    """Format timestamp for SRT format (HH:MM:SS,mmm)."""
    parts = timestamp.split(':')
    if len(parts) == 2:
        # MM:SS format, add hours
        return f"00:{parts[0]}:{parts[1].replace('.', ',')}"
    elif len(parts) == 3:
        # HH:MM:SS format
        return f"{parts[0]}:{parts[1]}:{parts[2].replace('.', ',')}"
    return "00:00:00,000"


def format_vtt_time(timestamp: str) -> str:
    """Format timestamp for VTT format (HH:MM:SS.mmm)."""
    parts = timestamp.split(':')
    if len(parts) == 2:
        # MM:SS format, add hours
        return f"00:{parts[0]}:{parts[1]}"
    elif len(parts) == 3:
        # Already HH:MM:SS format
        return timestamp
    return "00:00:00.000"


# Simple fallback diarization using voice activity detection
class SimpleDiarizer:
    """
    Simple fallback diarizer that uses basic audio analysis.
    Not as accurate as pyannote but works without external dependencies.
    """

    def __init__(self):
        self._available = None

    def is_available(self) -> bool:
        """Check if simple diarization is available."""
        if self._available is not None:
            return self._available

        try:
            import librosa
            import numpy as np
            from sklearn.cluster import AgglomerativeClustering
            self._available = True
        except ImportError:
            self._available = False

        return self._available

    def diarize(
        self,
        audio_path: str,
        num_speakers: int = 2,
        segment_duration: float = 1.0
    ) -> List[SpeakerSegment]:
        """
        Perform simple speaker diarization using audio clustering.

        Args:
            audio_path: Path to audio file
            num_speakers: Expected number of speakers
            segment_duration: Duration of each segment in seconds

        Returns:
            List of SpeakerSegment objects
        """
        if not self.is_available():
            return []

        try:
            import librosa
            import numpy as np
            from sklearn.cluster import AgglomerativeClustering

            # Load audio
            y, sr = librosa.load(audio_path, sr=16000)

            # Extract MFCC features for each segment
            segment_samples = int(segment_duration * sr)
            segments = []
            features = []

            for i in range(0, len(y), segment_samples):
                segment = y[i:i + segment_samples]
                if len(segment) < segment_samples // 2:
                    continue

                # Pad if needed
                if len(segment) < segment_samples:
                    segment = np.pad(segment, (0, segment_samples - len(segment)))

                # Extract MFCC features
                mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=20)
                mfcc_mean = np.mean(mfcc, axis=1)

                features.append(mfcc_mean)
                segments.append({
                    'start': i / sr,
                    'end': min((i + segment_samples) / sr, len(y) / sr)
                })

            if not features:
                return []

            # Cluster features
            features = np.array(features)
            clustering = AgglomerativeClustering(n_clusters=num_speakers)
            labels = clustering.fit_predict(features)

            # Create speaker segments
            result = []
            for i, (segment, label) in enumerate(zip(segments, labels)):
                result.append(SpeakerSegment(
                    speaker=f"Speaker {label + 1}",
                    start=segment['start'],
                    end=segment['end']
                ))

            # Merge consecutive segments from same speaker
            merged = []
            current = None

            for segment in result:
                if current is None:
                    current = segment
                elif segment.speaker == current.speaker:
                    current = SpeakerSegment(
                        speaker=current.speaker,
                        start=current.start,
                        end=segment.end
                    )
                else:
                    merged.append(current)
                    current = segment

            if current:
                merged.append(current)

            return merged

        except Exception as e:
            print(f"Simple diarization error: {e}")
            return []


def get_diarizer(hf_token: Optional[str] = None) -> SpeakerDiarizer:
    """Get the best available diarizer."""
    diarizer = SpeakerDiarizer(hf_token)
    if diarizer.is_available():
        return diarizer

    # Fallback to simple diarizer
    simple = SimpleDiarizer()
    if simple.is_available():
        # Return a wrapper that matches the interface
        class SimpleDiarizerWrapper(SpeakerDiarizer):
            def __init__(self):
                super().__init__()
                self._simple = simple
                self._available = True

            def diarize(self, audio_path, num_speakers=None):
                return self._simple.diarize(
                    audio_path,
                    num_speakers=num_speakers or 2
                )

        return SimpleDiarizerWrapper()

    # Return non-functional diarizer
    return diarizer


# Manual speaker labeling helpers
def create_manual_segments(
    transcript: str,
    speaker_changes: List[Tuple[str, str]]
) -> str:
    """
    Manually add speaker labels to a transcript.

    Args:
        transcript: The transcript text
        speaker_changes: List of (timestamp, speaker_name) tuples
            where timestamp is like "00:30" or "01:15:30"

    Returns:
        Transcript with speaker labels added
    """
    if not speaker_changes:
        return transcript

    # Convert speaker changes to seconds
    changes = []
    for timestamp, speaker in speaker_changes:
        seconds = timestamp_to_seconds(timestamp)
        changes.append((seconds, speaker))

    changes.sort(key=lambda x: x[0])

    # Parse and label transcript
    lines = transcript.strip().split('\n')
    result_lines = []

    timestamp_pattern = re.compile(r'\[(\d{1,2}):(\d{2}):?(\d{2})?\]')
    current_speaker = changes[0][1] if changes else None
    change_index = 0

    for line in lines:
        match = timestamp_pattern.match(line)
        if match:
            groups = match.groups()
            if groups[2] is not None:
                time_seconds = int(groups[0]) * 3600 + int(groups[1]) * 60 + int(groups[2])
            else:
                time_seconds = int(groups[0]) * 60 + int(groups[1])

            # Check if speaker changed
            while change_index < len(changes) - 1 and time_seconds >= changes[change_index + 1][0]:
                change_index += 1
                current_speaker = changes[change_index][1]

            if current_speaker:
                timestamp_end = match.end()
                line = f"{line[:timestamp_end]} [{current_speaker}]{line[timestamp_end:]}"

        result_lines.append(line)

    return '\n'.join(result_lines)
