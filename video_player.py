"""
Interactive video player component with clickable timestamps.
Provides click-to-seek functionality for transcripts.
"""
import streamlit as st
import streamlit.components.v1 as components
import re
import base64
import os
import html


def parse_timestamp_to_seconds(timestamp_str: str) -> float:
    """Convert [HH:MM:SS] timestamp to seconds."""
    match = re.match(r'\[?(\d{2}):(\d{2}):(\d{2})\]?', timestamp_str)
    if match:
        hours, minutes, seconds = map(int, match.groups())
        return hours * 3600 + minutes * 60 + seconds
    return 0


def get_video_base64(video_path: str) -> str:
    """Convert video file to base64 for embedding."""
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    return base64.b64encode(video_bytes).decode()


def escape_for_html(text: str) -> str:
    """Safely escape text for HTML display to prevent XSS."""
    return html.escape(text, quote=True)


def escape_for_js_string(text: str) -> str:
    """Safely escape text for use in JavaScript strings."""
    # Escape backslashes first, then quotes and special chars
    text = text.replace('\\', '\\\\')
    text = text.replace("'", "\\'")
    text = text.replace('"', '\\"')
    text = text.replace('\n', '\\n')
    text = text.replace('\r', '\\r')
    text = text.replace('<', '\\x3c')
    text = text.replace('>', '\\x3e')
    return text


def render_interactive_player(video_path: str, transcript_text: str, player_id: str = "video_player"):
    """
    Render an interactive video player with clickable transcript timestamps.

    Args:
        video_path: Path to the video file
        transcript_text: Transcript text with [HH:MM:SS] timestamps
        player_id: Unique ID for the player (for multiple players on same page)
    """
    # Sanitize player_id to prevent injection
    player_id = re.sub(r'[^a-zA-Z0-9_]', '', player_id)

    # Parse transcript into segments with timestamps
    segments = parse_transcript_with_timestamps(transcript_text)

    # Get video file extension for MIME type
    ext = os.path.splitext(video_path)[1].lower()
    mime_types = {
        '.mp4': 'video/mp4',
        '.mov': 'video/quicktime',
        '.avi': 'video/x-msvideo',
        '.mkv': 'video/x-matroska',
        '.m4a': 'audio/mp4',
        '.wav': 'audio/wav',
        '.mp3': 'audio/mpeg'
    }
    mime_type = mime_types.get(ext, 'video/mp4')

    # Encode video to base64
    video_base64 = get_video_base64(video_path)

    # Build segment data as JSON for safe JavaScript consumption
    segments_json = build_segments_json(segments)

    # Create the combined HTML component with safe DOM manipulation
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            .video-transcript-container {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 100%;
            }}

            .video-container {{
                margin-bottom: 20px;
                background: #000;
                border-radius: 8px;
                overflow: hidden;
            }}

            .video-container video {{
                width: 100%;
                max-height: 400px;
            }}

            .transcript-container {{
                max-height: 400px;
                overflow-y: auto;
                padding: 15px;
                background: #f8f9fa;
                border-radius: 8px;
                border: 1px solid #e9ecef;
            }}

            .transcript-segment {{
                margin-bottom: 12px;
                padding: 8px 12px;
                border-radius: 6px;
                transition: background-color 0.2s;
                line-height: 1.6;
            }}

            .transcript-segment:hover {{
                background: #e9ecef;
            }}

            .transcript-segment.active {{
                background: #d4edda;
                border-left: 3px solid #28a745;
            }}

            .transcript-segment.hidden {{
                display: none;
            }}

            .timestamp {{
                display: inline-block;
                background: #007bff;
                color: white;
                padding: 2px 8px;
                border-radius: 4px;
                font-size: 0.85em;
                font-family: monospace;
                cursor: pointer;
                margin-right: 10px;
                transition: background-color 0.2s;
            }}

            .timestamp:hover {{
                background: #0056b3;
            }}

            .search-container {{
                margin-bottom: 15px;
            }}

            .search-input {{
                width: 100%;
                padding: 10px 15px;
                border: 2px solid #e9ecef;
                border-radius: 6px;
                font-size: 14px;
                outline: none;
                transition: border-color 0.2s;
                box-sizing: border-box;
            }}

            .search-input:focus {{
                border-color: #007bff;
            }}

            .search-results {{
                font-size: 0.9em;
                color: #6c757d;
                margin-top: 5px;
            }}

            .highlight {{
                background: #fff3cd;
                padding: 1px 3px;
                border-radius: 3px;
            }}

            .controls {{
                display: flex;
                gap: 10px;
                margin-bottom: 10px;
            }}

            .control-btn {{
                padding: 6px 12px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 0.9em;
                transition: background-color 0.2s;
            }}

            .control-btn.secondary {{
                background: #6c757d;
                color: white;
            }}

            .control-btn.secondary:hover {{
                background: #545b62;
            }}

            .segment-text {{
                display: inline;
            }}
        </style>
    </head>
    <body>
        <div class="video-transcript-container">
            <div class="video-container">
                <video id="{player_id}" controls>
                    <source src="data:{mime_type};base64,{video_base64}" type="{mime_type}">
                    Your browser does not support video playback.
                </video>
            </div>

            <div class="search-container">
                <input type="text"
                       class="search-input"
                       id="search_{player_id}"
                       placeholder="Search transcript...">
                <div class="search-results" id="search_results_{player_id}"></div>
            </div>

            <div class="controls">
                <button class="control-btn secondary" id="autoscroll_btn_{player_id}">
                    Auto-scroll: <span id="autoscroll_status_{player_id}">ON</span>
                </button>
            </div>

            <div class="transcript-container" id="transcript_{player_id}">
            </div>
        </div>

        <script>
            (function() {{
                'use strict';

                const video = document.getElementById('{player_id}');
                const transcriptContainer = document.getElementById('transcript_{player_id}');
                const searchInput = document.getElementById('search_{player_id}');
                const searchResults = document.getElementById('search_results_{player_id}');
                const autoscrollBtn = document.getElementById('autoscroll_btn_{player_id}');
                const autoscrollStatus = document.getElementById('autoscroll_status_{player_id}');

                let autoScroll = true;

                // Segment data passed safely from Python
                const segmentsData = {segments_json};

                // Build transcript using safe DOM methods
                function buildTranscript() {{
                    segmentsData.forEach(function(seg, index) {{
                        const segmentDiv = document.createElement('div');
                        segmentDiv.className = 'transcript-segment';
                        segmentDiv.dataset.start = seg.seconds;
                        segmentDiv.dataset.end = seg.end_seconds;
                        segmentDiv.dataset.index = index;

                        const timestampSpan = document.createElement('span');
                        timestampSpan.className = 'timestamp';
                        timestampSpan.textContent = '[' + seg.timestamp + ']';
                        timestampSpan.addEventListener('click', function() {{
                            video.currentTime = seg.seconds;
                            video.play();
                        }});

                        const textSpan = document.createElement('span');
                        textSpan.className = 'segment-text';
                        textSpan.textContent = seg.text;
                        textSpan.dataset.originalText = seg.text;

                        segmentDiv.appendChild(timestampSpan);
                        segmentDiv.appendChild(textSpan);
                        transcriptContainer.appendChild(segmentDiv);
                    }});
                }}

                // Toggle auto-scroll
                autoscrollBtn.addEventListener('click', function() {{
                    autoScroll = !autoScroll;
                    autoscrollStatus.textContent = autoScroll ? 'ON' : 'OFF';
                }});

                // Search transcript using safe text matching
                searchInput.addEventListener('input', function() {{
                    const query = this.value.toLowerCase().trim();
                    const segments = transcriptContainer.querySelectorAll('.transcript-segment');
                    let visibleCount = 0;

                    segments.forEach(function(segEl) {{
                        const textSpan = segEl.querySelector('.segment-text');
                        const originalText = textSpan.dataset.originalText;
                        const lowerText = originalText.toLowerCase();

                        if (query.length === 0) {{
                            // No search - show all, restore original text
                            segEl.classList.remove('hidden');
                            textSpan.textContent = originalText;
                            visibleCount++;
                        }} else if (lowerText.indexOf(query) !== -1) {{
                            // Match found - show and highlight safely
                            segEl.classList.remove('hidden');
                            highlightText(textSpan, originalText, query);
                            visibleCount++;
                        }} else {{
                            // No match - hide
                            segEl.classList.add('hidden');
                            textSpan.textContent = originalText;
                        }}
                    }});

                    if (query.length > 0) {{
                        searchResults.textContent = visibleCount + ' matching segment(s)';
                    }} else {{
                        searchResults.textContent = '';
                    }}
                }});

                // Safe highlight function using DOM manipulation
                function highlightText(element, text, query) {{
                    // Clear existing content
                    while (element.firstChild) {{
                        element.removeChild(element.firstChild);
                    }}

                    const lowerText = text.toLowerCase();
                    const lowerQuery = query.toLowerCase();
                    let lastIndex = 0;
                    let index = lowerText.indexOf(lowerQuery);

                    while (index !== -1) {{
                        // Add text before match
                        if (index > lastIndex) {{
                            element.appendChild(document.createTextNode(text.substring(lastIndex, index)));
                        }}

                        // Add highlighted match
                        const highlightSpan = document.createElement('span');
                        highlightSpan.className = 'highlight';
                        highlightSpan.textContent = text.substring(index, index + query.length);
                        element.appendChild(highlightSpan);

                        lastIndex = index + query.length;
                        index = lowerText.indexOf(lowerQuery, lastIndex);
                    }}

                    // Add remaining text
                    if (lastIndex < text.length) {{
                        element.appendChild(document.createTextNode(text.substring(lastIndex)));
                    }}
                }}

                // Highlight active segment during playback
                video.addEventListener('timeupdate', function() {{
                    const currentTime = video.currentTime;
                    const segments = transcriptContainer.querySelectorAll('.transcript-segment');
                    let activeSegment = null;

                    segments.forEach(function(segEl) {{
                        const start = parseFloat(segEl.dataset.start);
                        const end = parseFloat(segEl.dataset.end);

                        if (currentTime >= start && currentTime < end) {{
                            segEl.classList.add('active');
                            activeSegment = segEl;
                        }} else {{
                            segEl.classList.remove('active');
                        }}
                    }});

                    // Auto-scroll to active segment
                    if (autoScroll && activeSegment && !activeSegment.classList.contains('hidden')) {{
                        activeSegment.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                    }}
                }});

                // Initialize
                buildTranscript();
            }})();
        </script>
    </body>
    </html>
    """

    # Render the component
    components.html(html_content, height=900, scrolling=False)


def parse_transcript_with_timestamps(transcript_text: str) -> list:
    """
    Parse transcript text into segments with timestamps.

    Returns list of dicts with 'timestamp', 'seconds', 'text' keys.
    """
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
        match = re.match(r'\[(\d{2}:\d{2}:\d{2})\]\s*(.*)', line)
        if match:
            timestamp = match.group(1)
            text = match.group(2).strip()
            seconds = parse_timestamp_to_seconds(timestamp)

            if text:
                segments.append({
                    'timestamp': timestamp,
                    'seconds': seconds,
                    'text': text
                })
        elif line and not line.startswith('['):
            # Line without timestamp - append to previous segment
            if segments:
                segments[-1]['text'] += ' ' + line
            else:
                segments.append({
                    'timestamp': '00:00:00',
                    'seconds': 0,
                    'text': line
                })

    # Calculate end times based on next segment start
    for i, seg in enumerate(segments):
        if i < len(segments) - 1:
            seg['end_seconds'] = segments[i + 1]['seconds']
        else:
            seg['end_seconds'] = seg['seconds'] + 30  # Default 30 seconds for last segment

    return segments


def build_segments_json(segments: list) -> str:
    """Build safe JSON representation of segments for JavaScript."""
    import json

    # Create safe segment objects
    safe_segments = []
    for seg in segments:
        safe_segments.append({
            'timestamp': seg['timestamp'],
            'seconds': seg['seconds'],
            'end_seconds': seg.get('end_seconds', seg['seconds'] + 30),
            'text': seg['text']
        })

    return json.dumps(safe_segments, ensure_ascii=True)


def render_simple_transcript_with_timestamps(transcript_text: str):
    """
    Render a simpler clickable transcript that works with st.video().

    This version creates clickable timestamps that control the video element.
    """
    segments = parse_transcript_with_timestamps(transcript_text)

    # Build segment data as JSON
    segments_json = build_segments_json(segments)

    # Create HTML with safe DOM manipulation
    html_content = f"""
    <div id="simple_transcript_container" style="font-family: sans-serif;">
        <input type="text"
               id="simple_search"
               placeholder="Search transcript..."
               style="width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; margin-bottom: 10px; box-sizing: border-box;">
        <div id="simple_search_results" style="font-size: 0.9em; color: #666; margin-bottom: 5px;"></div>
        <div id="simple_transcript" style="max-height: 400px; overflow-y: auto; border: 1px solid #ddd; border-radius: 4px; padding: 10px;">
        </div>
    </div>

    <script>
        (function() {{
            'use strict';

            const container = document.getElementById('simple_transcript');
            const searchInput = document.getElementById('simple_search');
            const searchResults = document.getElementById('simple_search_results');
            const segmentsData = {segments_json};

            // Build transcript safely
            segmentsData.forEach(function(seg) {{
                const div = document.createElement('div');
                div.className = 'simple-segment';
                div.style.cssText = 'margin: 8px 0; padding: 8px; border-radius: 4px; cursor: pointer;';
                div.dataset.text = seg.text.toLowerCase();

                div.addEventListener('mouseover', function() {{ this.style.background = '#f0f0f0'; }});
                div.addEventListener('mouseout', function() {{ this.style.background = 'transparent'; }});
                div.addEventListener('click', function() {{
                    const video = document.querySelector('video');
                    if (video) {{
                        video.currentTime = seg.seconds;
                        video.play();
                    }}
                }});

                const timestamp = document.createElement('span');
                timestamp.style.cssText = 'background: #007bff; color: white; padding: 2px 6px; border-radius: 3px; font-family: monospace; font-size: 0.85em;';
                timestamp.textContent = '[' + seg.timestamp + ']';

                const text = document.createElement('span');
                text.className = 'seg-text';
                text.style.marginLeft = '8px';
                text.textContent = seg.text;
                text.dataset.original = seg.text;

                div.appendChild(timestamp);
                div.appendChild(text);
                container.appendChild(div);
            }});

            // Search functionality
            searchInput.addEventListener('input', function() {{
                const query = this.value.toLowerCase().trim();
                const segments = container.querySelectorAll('.simple-segment');
                let count = 0;

                segments.forEach(function(seg) {{
                    const textEl = seg.querySelector('.seg-text');
                    const original = textEl.dataset.original;

                    if (query.length === 0 || seg.dataset.text.indexOf(query) !== -1) {{
                        seg.style.display = '';
                        textEl.textContent = original;
                        count++;
                    }} else {{
                        seg.style.display = 'none';
                    }}
                }});

                searchResults.textContent = query.length > 0 ? count + ' result(s)' : '';
            }});
        }})();
    </script>
    """

    components.html(html_content, height=500, scrolling=False)


# Speaker color palette for visual distinction
SPEAKER_COLORS = [
    '#e74c3c',  # Red
    '#3498db',  # Blue
    '#2ecc71',  # Green
    '#9b59b6',  # Purple
    '#f39c12',  # Orange
    '#1abc9c',  # Teal
    '#e91e63',  # Pink
    '#00bcd4',  # Cyan
    '#8bc34a',  # Light Green
    '#ff5722',  # Deep Orange
]


def parse_transcript_with_speakers(transcript_text: str, speaker_names: dict = None) -> list:
    """
    Parse transcript text with speaker labels into segments.

    Expected format: [HH:MM:SS] [SPEAKER_XX] text
    or: [HH:MM:SS] [display_name] text

    Returns list of dicts with 'timestamp', 'seconds', 'text', 'speaker', 'display_name' keys.
    """
    speaker_names = speaker_names or {}
    segments = []
    lines = transcript_text.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip chunk markers
        if line.startswith('[CHUNK'):
            continue

        # Match timestamp and optional speaker pattern
        # Format: [HH:MM:SS] [SPEAKER_ID] text or [HH:MM:SS] text
        match = re.match(r'\[(\d{2}:\d{2}:\d{2})\]\s*(?:\[([^\]]+)\])?\s*(.*)', line)
        if match:
            timestamp = match.group(1)
            speaker_id = match.group(2)  # May be None if no speaker label
            text = match.group(3).strip()
            seconds = parse_timestamp_to_seconds(timestamp)

            if text:
                display_name = speaker_id
                if speaker_id and speaker_id in speaker_names:
                    display_name = speaker_names[speaker_id]

                segments.append({
                    'timestamp': timestamp,
                    'seconds': seconds,
                    'text': text,
                    'speaker': speaker_id,
                    'display_name': display_name
                })
        elif line and not line.startswith('['):
            # Line without timestamp - append to previous segment
            if segments:
                segments[-1]['text'] += ' ' + line

    return segments


def render_transcript_with_speakers(
    transcript_text: str,
    speaker_names: dict = None,
    height: int = 500
):
    """
    Render a clickable transcript with color-coded speaker labels.

    Args:
        transcript_text: The transcript text with timestamps and speaker labels
        speaker_names: Dict mapping speaker_id to display_name
        height: Height of the transcript container in pixels
    """
    speaker_names = speaker_names or {}
    segments = parse_transcript_with_speakers(transcript_text, speaker_names)

    # Build unique speakers list and assign colors
    unique_speakers = []
    for seg in segments:
        if seg.get('speaker') and seg['speaker'] not in unique_speakers:
            unique_speakers.append(seg['speaker'])

    # Create speaker-to-color mapping
    speaker_colors = {}
    for i, speaker in enumerate(unique_speakers):
        speaker_colors[speaker] = SPEAKER_COLORS[i % len(SPEAKER_COLORS)]

    # Build segments JSON with speaker info
    import json
    safe_segments = []
    for seg in segments:
        speaker = seg.get('speaker', '')
        display_name = seg.get('display_name', speaker) or ''
        color = speaker_colors.get(speaker, '#666666')

        safe_segments.append({
            'timestamp': seg['timestamp'],
            'seconds': seg['seconds'],
            'text': seg['text'],
            'speaker': speaker,
            'display_name': display_name,
            'color': color
        })

    segments_json = json.dumps(safe_segments)

    # Build speaker legend
    legend_items = []
    for speaker in unique_speakers:
        display_name = speaker_names.get(speaker, speaker)
        color = speaker_colors[speaker]
        legend_items.append(f'<span style="display: inline-block; margin-right: 12px;"><span style="display: inline-block; width: 12px; height: 12px; background: {color}; border-radius: 50%; margin-right: 4px;"></span>{escape_for_html(display_name)}</span>')

    legend_html = ''.join(legend_items) if legend_items else ''

    # Create HTML with speaker colors
    html_content = f"""
    <div id="speaker_transcript_container" style="font-family: sans-serif;">
        <div style="margin-bottom: 10px; padding: 8px; background: #f5f5f5; border-radius: 4px;">
            <strong>Speakers:</strong> {legend_html}
        </div>
        <input type="text"
               id="speaker_search"
               placeholder="Search transcript..."
               style="width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; margin-bottom: 10px; box-sizing: border-box;">
        <div id="speaker_search_results" style="font-size: 0.9em; color: #666; margin-bottom: 5px;"></div>
        <div id="speaker_transcript" style="max-height: {height - 100}px; overflow-y: auto; border: 1px solid #ddd; border-radius: 4px; padding: 10px;">
        </div>
    </div>

    <script>
        (function() {{
            'use strict';

            const container = document.getElementById('speaker_transcript');
            const searchInput = document.getElementById('speaker_search');
            const searchResults = document.getElementById('speaker_search_results');
            const segmentsData = {segments_json};

            // Build transcript safely with speaker colors
            segmentsData.forEach(function(seg) {{
                const div = document.createElement('div');
                div.className = 'speaker-segment';
                div.style.cssText = 'margin: 8px 0; padding: 8px; border-radius: 4px; cursor: pointer; border-left: 3px solid ' + seg.color + ';';
                div.dataset.text = seg.text.toLowerCase();
                div.dataset.speaker = seg.speaker ? seg.speaker.toLowerCase() : '';

                div.addEventListener('mouseover', function() {{ this.style.background = '#f0f0f0'; }});
                div.addEventListener('mouseout', function() {{ this.style.background = 'transparent'; }});
                div.addEventListener('click', function() {{
                    const video = document.querySelector('video');
                    if (video) {{
                        video.currentTime = seg.seconds;
                        video.play();
                    }}
                }});

                const timestamp = document.createElement('span');
                timestamp.style.cssText = 'background: #007bff; color: white; padding: 2px 6px; border-radius: 3px; font-family: monospace; font-size: 0.85em;';
                timestamp.textContent = '[' + seg.timestamp + ']';

                // Add speaker label if present
                if (seg.speaker) {{
                    const speakerLabel = document.createElement('span');
                    speakerLabel.style.cssText = 'background: ' + seg.color + '; color: white; padding: 2px 8px; border-radius: 3px; font-size: 0.85em; font-weight: bold; margin-left: 6px;';
                    speakerLabel.textContent = seg.display_name || seg.speaker;
                    div.appendChild(timestamp);
                    div.appendChild(speakerLabel);
                }} else {{
                    div.appendChild(timestamp);
                }}

                const text = document.createElement('span');
                text.className = 'seg-text';
                text.style.marginLeft = '8px';
                text.textContent = seg.text;
                text.dataset.original = seg.text;

                div.appendChild(text);
                container.appendChild(div);
            }});

            // Search functionality (includes speaker names)
            searchInput.addEventListener('input', function() {{
                const query = this.value.toLowerCase().trim();
                const segments = container.querySelectorAll('.speaker-segment');
                let count = 0;

                segments.forEach(function(seg) {{
                    const textEl = seg.querySelector('.seg-text');
                    const original = textEl.dataset.original;
                    const speakerMatch = seg.dataset.speaker.indexOf(query) !== -1;
                    const textMatch = seg.dataset.text.indexOf(query) !== -1;

                    if (query.length === 0 || speakerMatch || textMatch) {{
                        seg.style.display = '';
                        textEl.textContent = original;
                        count++;
                    }} else {{
                        seg.style.display = 'none';
                    }}
                }});

                searchResults.textContent = query.length > 0 ? count + ' result(s)' : '';
            }});
        }})();
    </script>
    """

    components.html(html_content, height=height, scrolling=False)
