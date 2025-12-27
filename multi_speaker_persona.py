"""
Multi-Speaker Persona Modes for Video Transcription
Enables separate AI personas for each speaker with individual and panel chat modes.
"""

import re
import json
import sqlite3
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import requests


@dataclass
class SpeakerProfile:
    """Profile for an identified speaker."""
    speaker_id: str
    display_name: str
    transcript_segments: List[Dict] = field(default_factory=list)
    word_count: int = 0
    total_time: float = 0.0
    topics: List[str] = field(default_factory=list)
    persona_prompt: str = ""

    def get_transcript_text(self) -> str:
        """Get all transcript text for this speaker."""
        return " ".join([seg.get('text', '') for seg in self.transcript_segments])


@dataclass
class PanelResponse:
    """Response from panel mode discussion."""
    responses: List[Dict]  # List of {speaker_name, response, timestamp_refs}
    interaction_type: str  # "agreement", "disagreement", "discussion"
    confidence: float


class MultiSpeakerPersona:
    """
    Manages multiple speaker personas from a diarized transcript.
    """

    def __init__(
        self,
        db_path: str = "transcription.db",
        ollama_base: str = "http://localhost:11434",
        model: str = "mistral:instruct"
    ):
        self.db_path = db_path
        self.ollama_base = ollama_base
        self.model = model
        self._create_tables()

    def _create_tables(self):
        """Create tables for speaker profiles."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS speaker_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transcription_id INTEGER NOT NULL,
                    speaker_id TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    segments_json TEXT,
                    word_count INTEGER DEFAULT 0,
                    total_time REAL DEFAULT 0.0,
                    topics_json TEXT,
                    persona_prompt TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (transcription_id) REFERENCES transcriptions (id),
                    UNIQUE (transcription_id, speaker_id)
                )
            ''')
            conn.commit()

    def parse_diarized_transcript(
        self,
        transcript: str
    ) -> Dict[str, SpeakerProfile]:
        """
        Parse a diarized transcript and extract speaker segments.

        Expected format: [HH:MM:SS] [Speaker Name] Text content

        Returns:
            Dict mapping speaker_id to SpeakerProfile
        """
        speakers = {}

        # Pattern for diarized transcript: [timestamp] [speaker] text
        # Matches: [00:00:00] [Speaker 1] Hello there
        pattern = re.compile(
            r'\[(\d{2}:\d{2}:\d{2})\]\s*\[([^\]]+)\]\s*(.*?)(?=\n\[|\Z)',
            re.DOTALL
        )

        for match in pattern.finditer(transcript):
            timestamp = match.group(1)
            speaker_id = match.group(2).strip()
            text = match.group(3).strip()

            if not text:
                continue

            # Normalize speaker ID
            normalized_id = speaker_id.lower().replace(' ', '_')

            if normalized_id not in speakers:
                speakers[normalized_id] = SpeakerProfile(
                    speaker_id=normalized_id,
                    display_name=speaker_id
                )

            # Parse timestamp to seconds
            parts = timestamp.split(':')
            seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

            # Add segment
            speakers[normalized_id].transcript_segments.append({
                'timestamp': timestamp,
                'seconds': seconds,
                'text': text
            })
            speakers[normalized_id].word_count += len(text.split())

        # Calculate total time for each speaker
        for speaker in speakers.values():
            if speaker.transcript_segments:
                # Estimate time based on segment timestamps
                times = [seg['seconds'] for seg in speaker.transcript_segments]
                speaker.total_time = len(times) * 5.0  # Rough estimate: 5 seconds per segment

        return speakers

    def generate_persona_prompt(
        self,
        speaker: SpeakerProfile,
        context: str = ""
    ) -> str:
        """
        Generate a persona prompt for a specific speaker based on their content.
        """
        transcript_text = speaker.get_transcript_text()

        # Use Ollama to analyze the speaker's communication style
        analysis_prompt = f"""Analyze this speaker's communication style and create a persona prompt.

SPEAKER: {speaker.display_name}
WORD COUNT: {speaker.word_count}
CONTEXT: {context}

SAMPLE OF THEIR SPEECH:
{transcript_text[:2000]}

Based on this sample, create a brief persona prompt (2-3 sentences) that captures:
1. Their speaking style (formal/casual, technical/accessible)
2. Their expertise areas based on what they discuss
3. Their general personality traits evident in their speech

Format as a single paragraph persona prompt for an AI to roleplay as this speaker."""

        try:
            response = requests.post(
                f"{self.ollama_base}/api/generate",
                json={
                    "model": self.model,
                    "prompt": analysis_prompt,
                    "stream": False,
                    "options": {"temperature": 0.7}
                },
                timeout=60
            )
            response.raise_for_status()
            persona = response.json().get("response", "").strip()

            # Wrap with speaker identity
            full_prompt = f"""You are {speaker.display_name}. {persona}

When answering questions, respond as {speaker.display_name} would, based on their actual statements in the video.
Always cite timestamps [HH:MM:SS] when referencing specific points you made.
If asked about something you didn't discuss, say "I didn't address that topic in this conversation."
"""
            return full_prompt

        except Exception as e:
            # Fallback to basic prompt
            return f"""You are {speaker.display_name}. Respond based on your statements in this video.
Always cite timestamps when referencing your points.
If asked about something you didn't discuss, say "I didn't address that topic in this conversation."
"""

    def extract_speaker_topics(
        self,
        speaker: SpeakerProfile
    ) -> List[str]:
        """Extract main topics discussed by a speaker."""
        transcript_text = speaker.get_transcript_text()

        if len(transcript_text) < 100:
            return []

        prompt = f"""Extract the 3-5 main topics this speaker discusses.
Return only a JSON array of topic strings.

SPEAKER TEXT:
{transcript_text[:3000]}

Return format: ["topic1", "topic2", "topic3"]"""

        try:
            response = requests.post(
                f"{self.ollama_base}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3}
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json().get("response", "")

            # Parse JSON array from response
            match = re.search(r'\[.*?\]', result, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass

        return []

    def save_speaker_profiles(
        self,
        transcription_id: int,
        speakers: Dict[str, SpeakerProfile]
    ):
        """Save speaker profiles to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            for speaker in speakers.values():
                cursor.execute('''
                    INSERT OR REPLACE INTO speaker_profiles
                    (transcription_id, speaker_id, display_name, segments_json,
                     word_count, total_time, topics_json, persona_prompt)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    transcription_id,
                    speaker.speaker_id,
                    speaker.display_name,
                    json.dumps(speaker.transcript_segments),
                    speaker.word_count,
                    speaker.total_time,
                    json.dumps(speaker.topics),
                    speaker.persona_prompt
                ))

            conn.commit()

    def load_speaker_profiles(
        self,
        transcription_id: int
    ) -> Dict[str, SpeakerProfile]:
        """Load speaker profiles from database."""
        speakers = {}

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT speaker_id, display_name, segments_json, word_count,
                       total_time, topics_json, persona_prompt
                FROM speaker_profiles
                WHERE transcription_id = ?
            ''', (transcription_id,))

            for row in cursor.fetchall():
                speaker = SpeakerProfile(
                    speaker_id=row[0],
                    display_name=row[1],
                    transcript_segments=json.loads(row[2]) if row[2] else [],
                    word_count=row[3],
                    total_time=row[4],
                    topics=json.loads(row[5]) if row[5] else [],
                    persona_prompt=row[6] or ""
                )
                speakers[speaker.speaker_id] = speaker

        return speakers

    def update_speaker_name(
        self,
        transcription_id: int,
        speaker_id: str,
        new_name: str
    ) -> bool:
        """Update display name for a speaker."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE speaker_profiles
                SET display_name = ?
                WHERE transcription_id = ? AND speaker_id = ?
            ''', (new_name, transcription_id, speaker_id))
            conn.commit()
            return cursor.rowcount > 0

    def has_speaker_profiles(self, transcription_id: int) -> bool:
        """Check if speaker profiles exist for a transcription."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT COUNT(*) FROM speaker_profiles
                WHERE transcription_id = ?
            ''', (transcription_id,))
            return cursor.fetchone()[0] > 0

    def generate_individual_response(
        self,
        speaker: SpeakerProfile,
        query: str
    ) -> Dict:
        """
        Generate a response from a single speaker's perspective.
        """
        # Find relevant segments for this query
        relevant_segments = self._find_relevant_segments(speaker, query)

        # Build context from speaker's transcript
        context = "\n".join([
            f"[{seg['timestamp']}] {seg['text']}"
            for seg in relevant_segments[:5]
        ])

        prompt = f"""{speaker.persona_prompt}

RELEVANT EXCERPTS FROM YOUR STATEMENTS:
{context}

USER QUESTION: {query}

Respond as {speaker.display_name}. Reference specific timestamps when citing your statements.
If you didn't discuss this topic, say so honestly."""

        try:
            response = requests.post(
                f"{self.ollama_base}/api/generate",
                json={
                    "model": self.model,
                    "system": speaker.persona_prompt,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.7}
                },
                timeout=120
            )
            response.raise_for_status()
            return {
                "speaker": speaker.display_name,
                "response": response.json().get("response", ""),
                "timestamp_refs": [seg['timestamp'] for seg in relevant_segments[:3]],
                "success": True
            }
        except Exception as e:
            return {
                "speaker": speaker.display_name,
                "response": f"Error generating response: {str(e)}",
                "timestamp_refs": [],
                "success": False
            }

    def generate_panel_response(
        self,
        speakers: Dict[str, SpeakerProfile],
        query: str,
        moderator_style: str = "balanced"
    ) -> PanelResponse:
        """
        Generate a panel discussion response from multiple speakers.

        Args:
            speakers: Dictionary of speaker profiles
            query: User's question
            moderator_style: "balanced" (equal turns), "debate" (opposing views),
                           "sequential" (one after another)
        """
        responses = []

        # Get individual responses from each speaker
        for speaker_id, speaker in speakers.items():
            result = self.generate_individual_response(speaker, query)
            if result["success"]:
                responses.append(result)

        # Determine interaction type based on content analysis
        interaction_type = self._analyze_interaction(responses)

        # Calculate overall confidence
        confidence = len([r for r in responses if r["success"]]) / max(len(speakers), 1)

        return PanelResponse(
            responses=responses,
            interaction_type=interaction_type,
            confidence=confidence
        )

    def _find_relevant_segments(
        self,
        speaker: SpeakerProfile,
        query: str,
        max_segments: int = 5
    ) -> List[Dict]:
        """Find segments most relevant to the query using keyword matching."""
        query_words = set(query.lower().split())
        scored_segments = []

        for segment in speaker.transcript_segments:
            text_words = set(segment['text'].lower().split())
            overlap = len(query_words & text_words)
            if overlap > 0:
                scored_segments.append((segment, overlap))

        # Sort by relevance score
        scored_segments.sort(key=lambda x: x[1], reverse=True)

        # Return top segments, maintaining chronological order if same score
        return [seg for seg, _ in scored_segments[:max_segments]]

    def _analyze_interaction(self, responses: List[Dict]) -> str:
        """Analyze the type of interaction between speakers."""
        if len(responses) < 2:
            return "monologue"

        # Simple heuristic - could be enhanced with NLP
        response_texts = [r['response'].lower() for r in responses]

        # Check for agreement words
        agreement_words = ["agree", "exactly", "right", "yes", "correct", "same"]
        disagreement_words = ["disagree", "however", "but", "different", "contrary", "no"]

        agreement_score = sum(
            1 for text in response_texts
            for word in agreement_words if word in text
        )
        disagreement_score = sum(
            1 for text in response_texts
            for word in disagreement_words if word in text
        )

        if disagreement_score > agreement_score:
            return "disagreement"
        elif agreement_score > 0:
            return "agreement"
        return "discussion"


def format_panel_response(panel_response: PanelResponse, show_timestamps: bool = True) -> str:
    """Format panel response for display."""
    lines = []

    # Add interaction type header
    type_icons = {
        "agreement": "🤝",
        "disagreement": "⚔️",
        "discussion": "💬",
        "monologue": "🎤"
    }
    icon = type_icons.get(panel_response.interaction_type, "💬")
    lines.append(f"### {icon} Panel Discussion\n")

    # Add each speaker's response
    for response in panel_response.responses:
        speaker_name = response.get('speaker', 'Unknown')
        text = response.get('response', '')

        lines.append(f"**{speaker_name}:**")
        lines.append(text)

        if show_timestamps and response.get('timestamp_refs'):
            refs = ", ".join([f"[{ts}]" for ts in response['timestamp_refs']])
            lines.append(f"\n*Referenced: {refs}*")

        lines.append("")  # Empty line between speakers

    # Add confidence indicator
    confidence = panel_response.confidence
    if confidence >= 0.8:
        conf_str = "🟢 High confidence"
    elif confidence >= 0.5:
        conf_str = "🟡 Medium confidence"
    else:
        conf_str = "🔴 Low confidence"

    lines.append(f"\n*{conf_str}*")

    return "\n".join(lines)


def get_multi_speaker_persona(db_path: str = "transcription.db") -> MultiSpeakerPersona:
    """Get or create a MultiSpeakerPersona instance."""
    return MultiSpeakerPersona(db_path=db_path)
