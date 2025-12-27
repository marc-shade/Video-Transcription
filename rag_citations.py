"""
RAG Citations for Video Transcription
Enables persona chat to cite specific timestamps from transcripts.
"""

import re
import os
import sqlite3
import json
import hashlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import requests


@dataclass
class TranscriptChunk:
    """Represents a chunk of transcript with metadata."""
    chunk_id: str
    text: str
    start_time: str  # HH:MM:SS format
    end_time: str    # HH:MM:SS format
    start_seconds: float
    end_seconds: float
    embedding: Optional[List[float]] = None


@dataclass
class Citation:
    """Represents a citation from the transcript."""
    timestamp: str
    seconds: float
    text: str
    relevance_score: float


@dataclass
class RAGResponse:
    """Response from RAG-augmented generation."""
    response: str
    citations: List[Citation]
    confidence: float  # 0.0 to 1.0
    used_general_knowledge: bool


class TranscriptRAG:
    """
    RAG (Retrieval-Augmented Generation) system for transcript citations.
    Chunks transcripts, creates embeddings, and retrieves relevant sections.
    """

    def __init__(
        self,
        db_path: str = "transcription.db",
        ollama_base: str = "http://localhost:11434",
        embedding_model: str = "nomic-embed-text"
    ):
        """
        Initialize the RAG system.

        Args:
            db_path: Path to SQLite database
            ollama_base: Ollama API base URL
            embedding_model: Model to use for embeddings
        """
        self.db_path = db_path
        self.ollama_base = ollama_base
        self.embedding_model = embedding_model
        self._create_tables()

    def _create_tables(self):
        """Create tables for storing chunks and embeddings."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS transcript_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    transcription_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    start_time TEXT,
                    end_time TEXT,
                    start_seconds REAL,
                    end_seconds REAL,
                    embedding_json TEXT,
                    FOREIGN KEY (transcription_id) REFERENCES transcriptions (id)
                )
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_chunks_transcription
                ON transcript_chunks (transcription_id)
            ''')
            conn.commit()

    def _generate_chunk_id(self, transcription_id: int, chunk_index: int) -> str:
        """Generate unique chunk ID."""
        return f"{transcription_id}_{chunk_index}"

    def _parse_timestamp(self, timestamp: str) -> float:
        """Convert timestamp string to seconds."""
        parts = timestamp.split(':')
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return 0.0

    def _format_timestamp(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def chunk_transcript(
        self,
        transcript: str,
        chunk_size: int = 3,
        overlap: int = 1
    ) -> List[TranscriptChunk]:
        """
        Split transcript into overlapping chunks based on timestamp lines.

        Args:
            transcript: Full transcript text with timestamps
            chunk_size: Number of lines per chunk
            overlap: Number of overlapping lines between chunks

        Returns:
            List of TranscriptChunk objects
        """
        # Parse lines with timestamps
        lines = []
        timestamp_pattern = re.compile(r'\[(\d{2}:\d{2}:\d{2})\]\s*(.*)')

        for line in transcript.strip().split('\n'):
            match = timestamp_pattern.match(line.strip())
            if match:
                timestamp = match.group(1)
                text = match.group(2).strip()
                if text:  # Only include non-empty lines
                    lines.append({
                        'timestamp': timestamp,
                        'seconds': self._parse_timestamp(timestamp),
                        'text': text
                    })

        if not lines:
            # No timestamps found, treat entire transcript as single chunk
            return [TranscriptChunk(
                chunk_id="0",
                text=transcript,
                start_time="00:00:00",
                end_time="00:00:00",
                start_seconds=0.0,
                end_seconds=0.0
            )]

        # Create overlapping chunks
        chunks = []
        step = max(1, chunk_size - overlap)

        for i in range(0, len(lines), step):
            chunk_lines = lines[i:i + chunk_size]
            if not chunk_lines:
                continue

            chunk_text = ' '.join([l['text'] for l in chunk_lines])
            start_time = chunk_lines[0]['timestamp']
            end_time = chunk_lines[-1]['timestamp']

            chunks.append(TranscriptChunk(
                chunk_id=str(len(chunks)),
                text=chunk_text,
                start_time=start_time,
                end_time=end_time,
                start_seconds=chunk_lines[0]['seconds'],
                end_seconds=chunk_lines[-1]['seconds']
            ))

        return chunks

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text using Ollama."""
        try:
            response = requests.post(
                f"{self.ollama_base}/api/embeddings",
                json={
                    "model": self.embedding_model,
                    "prompt": text
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json().get("embedding")
        except Exception as e:
            print(f"Embedding error: {e}")
            return None

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def index_transcript(self, transcription_id: int, transcript: str) -> int:
        """
        Index a transcript for RAG retrieval.

        Args:
            transcription_id: Database ID of the transcription
            transcript: Full transcript text

        Returns:
            Number of chunks indexed
        """
        # Delete existing chunks for this transcription
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'DELETE FROM transcript_chunks WHERE transcription_id = ?',
                (transcription_id,)
            )
            conn.commit()

        # Create chunks
        chunks = self.chunk_transcript(transcript)

        # Generate embeddings and store
        indexed = 0
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            for i, chunk in enumerate(chunks):
                embedding = self._get_embedding(chunk.text)
                chunk_id = self._generate_chunk_id(transcription_id, i)

                cursor.execute('''
                    INSERT INTO transcript_chunks
                    (chunk_id, transcription_id, chunk_index, text, start_time,
                     end_time, start_seconds, end_seconds, embedding_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    chunk_id,
                    transcription_id,
                    i,
                    chunk.text,
                    chunk.start_time,
                    chunk.end_time,
                    chunk.start_seconds,
                    chunk.end_seconds,
                    json.dumps(embedding) if embedding else None
                ))
                indexed += 1

            conn.commit()

        return indexed

    def retrieve_relevant_chunks(
        self,
        transcription_id: int,
        query: str,
        top_k: int = 5,
        min_score: float = 0.3
    ) -> List[Tuple[TranscriptChunk, float]]:
        """
        Retrieve chunks most relevant to a query.

        Args:
            transcription_id: Database ID of the transcription
            query: User's question
            top_k: Maximum number of chunks to return
            min_score: Minimum similarity score threshold

        Returns:
            List of (chunk, score) tuples sorted by relevance
        """
        # Get query embedding
        query_embedding = self._get_embedding(query)
        if not query_embedding:
            return []

        # Get all chunks for this transcription
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT chunk_id, text, start_time, end_time,
                       start_seconds, end_seconds, embedding_json
                FROM transcript_chunks
                WHERE transcription_id = ?
            ''', (transcription_id,))
            rows = cursor.fetchall()

        # Calculate similarities
        scored_chunks = []
        for row in rows:
            embedding_json = row[6]
            if not embedding_json:
                continue

            embedding = json.loads(embedding_json)
            score = self._cosine_similarity(query_embedding, embedding)

            if score >= min_score:
                chunk = TranscriptChunk(
                    chunk_id=row[0],
                    text=row[1],
                    start_time=row[2],
                    end_time=row[3],
                    start_seconds=row[4],
                    end_seconds=row[5],
                    embedding=embedding
                )
                scored_chunks.append((chunk, score))

        # Sort by score and return top-k
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]

    def is_indexed(self, transcription_id: int) -> bool:
        """Check if a transcription has been indexed."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT COUNT(*) FROM transcript_chunks WHERE transcription_id = ?',
                (transcription_id,)
            )
            return cursor.fetchone()[0] > 0

    def get_chunk_count(self, transcription_id: int) -> int:
        """Get number of indexed chunks for a transcription."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT COUNT(*) FROM transcript_chunks WHERE transcription_id = ?',
                (transcription_id,)
            )
            return cursor.fetchone()[0]


class RAGPersonaChat:
    """
    RAG-augmented persona chat that cites transcript sources.
    """

    def __init__(
        self,
        rag: TranscriptRAG,
        model: str = "mistral:instruct",
        api_base: str = "http://localhost:11434",
        options: Optional[Dict] = None
    ):
        """
        Initialize the RAG persona chat.

        Args:
            rag: TranscriptRAG instance
            model: Ollama model for generation
            api_base: Ollama API base URL
            options: Model options (temperature, etc.)
        """
        self.rag = rag
        self.model = model
        self.api_base = api_base
        self.options = options or {}

    def _generate_completion(self, system: str, user: str) -> str:
        """Generate completion using Ollama."""
        try:
            response = requests.post(
                f"{self.api_base}/api/generate",
                json={
                    "model": self.model,
                    "system": system,
                    "prompt": user,
                    "stream": False,
                    "options": self.options
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            print(f"Generation error: {e}")
            return ""

    def generate_response(
        self,
        transcription_id: int,
        persona_prompt: str,
        user_query: str,
        use_only_transcript: bool = True,
        top_k: int = 5
    ) -> RAGResponse:
        """
        Generate a response with citations from the transcript.

        Args:
            transcription_id: Database ID of the transcription
            persona_prompt: The persona's system prompt
            user_query: User's question
            use_only_transcript: If True, only use transcript content
            top_k: Number of relevant chunks to retrieve

        Returns:
            RAGResponse with response text, citations, and confidence
        """
        # Retrieve relevant chunks
        relevant_chunks = self.rag.retrieve_relevant_chunks(
            transcription_id, user_query, top_k
        )

        if not relevant_chunks:
            if use_only_transcript:
                return RAGResponse(
                    response="I couldn't find any relevant information about that in the transcript. Could you rephrase your question or ask about something discussed in the video?",
                    citations=[],
                    confidence=0.0,
                    used_general_knowledge=False
                )
            else:
                # Fall back to general knowledge
                response = self._generate_without_rag(persona_prompt, user_query)
                return RAGResponse(
                    response=response + "\n\n*Note: This response is based on general knowledge, not the video content.*",
                    citations=[],
                    confidence=0.5,
                    used_general_knowledge=True
                )

        # Build context from retrieved chunks
        context_parts = []
        for chunk, score in relevant_chunks:
            context_parts.append(
                f"[{chunk.start_time}] {chunk.text}"
            )

        context = "\n".join(context_parts)

        # Calculate confidence based on relevance scores
        avg_score = sum(s for _, s in relevant_chunks) / len(relevant_chunks)
        max_score = max(s for _, s in relevant_chunks)
        confidence = (avg_score + max_score) / 2

        # Build the enhanced prompt
        rag_system_prompt = f"""{persona_prompt}

IMPORTANT INSTRUCTIONS FOR CITATIONS:
You have access to specific excerpts from the video transcript below. When answering:
1. Base your response ONLY on the provided transcript excerpts
2. Include timestamp citations in [HH:MM:SS] format for every factual claim
3. If the information is not in the provided excerpts, say "I don't see information about that in the video"
4. Quote briefly from the transcript when relevant

TRANSCRIPT EXCERPTS:
{context}

Remember: Always cite timestamps when referencing information from the video."""

        user_prompt = f"""Based on the transcript excerpts provided, please answer this question:

{user_query}

Include [timestamp] citations for any information from the video."""

        # Generate response
        response = self._generate_completion(rag_system_prompt, user_prompt)

        # Extract citations from the response
        citations = self._extract_citations(response, relevant_chunks)

        return RAGResponse(
            response=response,
            citations=citations,
            confidence=confidence,
            used_general_knowledge=False
        )

    def _generate_without_rag(self, persona_prompt: str, user_query: str) -> str:
        """Generate response without RAG context."""
        return self._generate_completion(persona_prompt, user_query)

    def _extract_citations(
        self,
        response: str,
        chunks: List[Tuple[TranscriptChunk, float]]
    ) -> List[Citation]:
        """Extract citations from the response."""
        citations = []
        seen_timestamps = set()

        # Find all timestamp patterns in response
        timestamp_pattern = re.compile(r'\[(\d{2}:\d{2}:\d{2})\]')
        matches = timestamp_pattern.finditer(response)

        for match in matches:
            timestamp = match.group(1)
            if timestamp in seen_timestamps:
                continue
            seen_timestamps.add(timestamp)

            # Find the chunk that matches this timestamp
            seconds = self.rag._parse_timestamp(timestamp)
            for chunk, score in chunks:
                if chunk.start_seconds <= seconds <= chunk.end_seconds + 10:
                    citations.append(Citation(
                        timestamp=timestamp,
                        seconds=seconds,
                        text=chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                        relevance_score=score
                    ))
                    break

        # Sort by timestamp
        citations.sort(key=lambda c: c.seconds)
        return citations


def format_response_with_citations(rag_response: RAGResponse) -> str:
    """
    Format RAG response for display with citations section.

    Args:
        rag_response: RAGResponse object

    Returns:
        Formatted string with response and citations
    """
    output = [rag_response.response]

    if rag_response.citations:
        output.append("\n\n---\n**📍 Sources:**")
        for citation in rag_response.citations:
            output.append(f"- **[{citation.timestamp}]** \"{citation.text}\"")

    # Add confidence indicator
    confidence_level = ""
    if rag_response.confidence >= 0.7:
        confidence_level = "🟢 High confidence"
    elif rag_response.confidence >= 0.4:
        confidence_level = "🟡 Medium confidence"
    else:
        confidence_level = "🔴 Low confidence"

    output.append(f"\n\n*{confidence_level}*")

    if rag_response.used_general_knowledge:
        output.append("\n*⚠️ Response includes general knowledge, not just video content.*")

    return "\n".join(output)


def get_transcript_rag(db_path: str = "transcription.db") -> TranscriptRAG:
    """Get or create a TranscriptRAG instance."""
    return TranscriptRAG(db_path=db_path)
