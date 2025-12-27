"""
Batch Processor for Video Transcription
Handles background processing of multiple video uploads with job queue management.
"""

import threading
import queue
import time
import os
import tempfile
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import sqlite3


class JobStatus(Enum):
    """Status states for batch jobs."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchJob:
    """Represents a single batch processing job."""
    id: int
    filename: str
    file_path: str
    client_id: int
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    error_message: Optional[str] = None
    transcription_id: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    include_timestamps: bool = True


class BatchJobQueue:
    """Manages the job queue for batch video processing."""

    def __init__(self, db_path: str = "transcription.db"):
        self.db_path = db_path
        self._create_job_table()
        self._job_queue = queue.Queue()
        self._workers: List[threading.Thread] = []
        self._running = False
        self._max_concurrent = 1  # Default to sequential processing
        self._progress_callbacks: Dict[int, Callable] = {}
        self._completion_callbacks: List[Callable] = []
        self._lock = threading.Lock()

    def _create_job_table(self):
        """Create the job queue table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS batch_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    client_id INTEGER NOT NULL,
                    status TEXT DEFAULT 'pending',
                    progress REAL DEFAULT 0.0,
                    error_message TEXT,
                    transcription_id INTEGER,
                    include_timestamps INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    FOREIGN KEY (client_id) REFERENCES clients (id),
                    FOREIGN KEY (transcription_id) REFERENCES transcriptions (id)
                )
            ''')
            conn.commit()

    def add_job(self, filename: str, file_path: str, client_id: int,
                include_timestamps: bool = True) -> int:
        """Add a new job to the queue."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO batch_jobs (filename, file_path, client_id, include_timestamps)
                VALUES (?, ?, ?, ?)
            ''', (filename, file_path, client_id, 1 if include_timestamps else 0))
            job_id = cursor.lastrowid
            conn.commit()

        # Add to in-memory queue if workers are running
        if self._running:
            self._job_queue.put(job_id)

        return job_id

    def add_jobs(self, jobs: List[Dict]) -> List[int]:
        """Add multiple jobs at once."""
        job_ids = []
        for job in jobs:
            job_id = self.add_job(
                filename=job['filename'],
                file_path=job['file_path'],
                client_id=job['client_id'],
                include_timestamps=job.get('include_timestamps', True)
            )
            job_ids.append(job_id)
        return job_ids

    def get_job(self, job_id: int) -> Optional[BatchJob]:
        """Get job details by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, filename, file_path, client_id, status, progress,
                       error_message, transcription_id, include_timestamps,
                       created_at, started_at, completed_at
                FROM batch_jobs WHERE id = ?
            ''', (job_id,))
            row = cursor.fetchone()
            if row:
                return BatchJob(
                    id=row[0],
                    filename=row[1],
                    file_path=row[2],
                    client_id=row[3],
                    status=JobStatus(row[4]),
                    progress=row[5],
                    error_message=row[6],
                    transcription_id=row[7],
                    include_timestamps=bool(row[8]),
                    created_at=datetime.fromisoformat(row[9]) if row[9] else datetime.now(),
                    started_at=datetime.fromisoformat(row[10]) if row[10] else None,
                    completed_at=datetime.fromisoformat(row[11]) if row[11] else None
                )
        return None

    def get_all_jobs(self, client_id: Optional[int] = None,
                     status: Optional[JobStatus] = None) -> List[BatchJob]:
        """Get all jobs, optionally filtered by client or status."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            query = '''
                SELECT id, filename, file_path, client_id, status, progress,
                       error_message, transcription_id, include_timestamps,
                       created_at, started_at, completed_at
                FROM batch_jobs
            '''
            conditions = []
            params = []

            if client_id is not None:
                conditions.append("client_id = ?")
                params.append(client_id)

            if status is not None:
                conditions.append("status = ?")
                params.append(status.value)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY created_at DESC"

            cursor.execute(query, params)
            jobs = []
            for row in cursor.fetchall():
                jobs.append(BatchJob(
                    id=row[0],
                    filename=row[1],
                    file_path=row[2],
                    client_id=row[3],
                    status=JobStatus(row[4]),
                    progress=row[5],
                    error_message=row[6],
                    transcription_id=row[7],
                    include_timestamps=bool(row[8]),
                    created_at=datetime.fromisoformat(row[9]) if row[9] else datetime.now(),
                    started_at=datetime.fromisoformat(row[10]) if row[10] else None,
                    completed_at=datetime.fromisoformat(row[11]) if row[11] else None
                ))
            return jobs

    def get_pending_jobs(self) -> List[BatchJob]:
        """Get all pending jobs."""
        return self.get_all_jobs(status=JobStatus.PENDING)

    def update_job_status(self, job_id: int, status: JobStatus,
                          progress: float = None, error_message: str = None,
                          transcription_id: int = None):
        """Update job status and optionally progress/error."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            updates = ["status = ?"]
            params = [status.value]

            if progress is not None:
                updates.append("progress = ?")
                params.append(progress)

            if error_message is not None:
                updates.append("error_message = ?")
                params.append(error_message)

            if transcription_id is not None:
                updates.append("transcription_id = ?")
                params.append(transcription_id)

            if status == JobStatus.PROCESSING:
                updates.append("started_at = CURRENT_TIMESTAMP")
            elif status in [JobStatus.COMPLETE, JobStatus.FAILED, JobStatus.CANCELLED]:
                updates.append("completed_at = CURRENT_TIMESTAMP")

            params.append(job_id)

            cursor.execute(f'''
                UPDATE batch_jobs SET {", ".join(updates)} WHERE id = ?
            ''', params)
            conn.commit()

        # Notify progress callback if registered
        if job_id in self._progress_callbacks:
            try:
                self._progress_callbacks[job_id](job_id, status, progress)
            except Exception:
                pass

    def update_job_progress(self, job_id: int, progress: float):
        """Update just the progress of a job."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE batch_jobs SET progress = ? WHERE id = ?
            ''', (progress, job_id))
            conn.commit()

        # Notify progress callback
        if job_id in self._progress_callbacks:
            try:
                job = self.get_job(job_id)
                if job:
                    self._progress_callbacks[job_id](job_id, job.status, progress)
            except Exception:
                pass

    def cancel_job(self, job_id: int) -> bool:
        """Cancel a pending or processing job."""
        job = self.get_job(job_id)
        if job and job.status in [JobStatus.PENDING, JobStatus.PROCESSING]:
            self.update_job_status(job_id, JobStatus.CANCELLED)
            return True
        return False

    def retry_job(self, job_id: int) -> bool:
        """Retry a failed job."""
        job = self.get_job(job_id)
        if job and job.status == JobStatus.FAILED:
            self.update_job_status(job_id, JobStatus.PENDING, progress=0.0,
                                   error_message=None)
            if self._running:
                self._job_queue.put(job_id)
            return True
        return False

    def delete_job(self, job_id: int) -> bool:
        """Delete a job from the queue."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM batch_jobs WHERE id = ?', (job_id,))
            return cursor.rowcount > 0

    def clear_completed_jobs(self, client_id: Optional[int] = None):
        """Clear all completed jobs."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if client_id:
                cursor.execute('''
                    DELETE FROM batch_jobs
                    WHERE status IN ('complete', 'failed', 'cancelled') AND client_id = ?
                ''', (client_id,))
            else:
                cursor.execute('''
                    DELETE FROM batch_jobs
                    WHERE status IN ('complete', 'failed', 'cancelled')
                ''')
            conn.commit()

    def get_queue_stats(self, client_id: Optional[int] = None) -> Dict:
        """Get statistics about the job queue."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            base_query = "SELECT status, COUNT(*) FROM batch_jobs"
            if client_id:
                base_query += " WHERE client_id = ?"
                cursor.execute(base_query + " GROUP BY status", (client_id,))
            else:
                cursor.execute(base_query + " GROUP BY status")

            stats = {
                'pending': 0,
                'processing': 0,
                'complete': 0,
                'failed': 0,
                'cancelled': 0,
                'total': 0
            }

            for row in cursor.fetchall():
                stats[row[0]] = row[1]
                stats['total'] += row[1]

            return stats

    def register_progress_callback(self, job_id: int, callback: Callable):
        """Register a callback for job progress updates."""
        self._progress_callbacks[job_id] = callback

    def register_completion_callback(self, callback: Callable):
        """Register a callback for when any job completes."""
        self._completion_callbacks.append(callback)

    def set_max_concurrent(self, max_concurrent: int):
        """Set maximum concurrent processing jobs."""
        self._max_concurrent = max(1, max_concurrent)

    def start_workers(self, process_func: Callable):
        """Start background worker threads."""
        if self._running:
            return

        self._running = True
        self._process_func = process_func

        # Load pending jobs into queue
        for job in self.get_pending_jobs():
            self._job_queue.put(job.id)

        # Start worker threads
        for i in range(self._max_concurrent):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self._workers.append(worker)

    def stop_workers(self):
        """Stop all worker threads."""
        self._running = False
        # Add sentinel values to unblock workers
        for _ in self._workers:
            self._job_queue.put(None)
        self._workers.clear()

    def _worker_loop(self):
        """Worker thread main loop."""
        while self._running:
            try:
                job_id = self._job_queue.get(timeout=1.0)
                if job_id is None:  # Sentinel for shutdown
                    break

                job = self.get_job(job_id)
                if job and job.status == JobStatus.PENDING:
                    self._process_job(job)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker error: {e}")

    def _process_job(self, job: BatchJob):
        """Process a single job."""
        try:
            self.update_job_status(job.id, JobStatus.PROCESSING, progress=0.0)

            # Call the processing function
            result = self._process_func(
                job.file_path,
                job.filename,
                job.client_id,
                job.include_timestamps,
                lambda p: self.update_job_progress(job.id, p)
            )

            if result.get('success'):
                self.update_job_status(
                    job.id,
                    JobStatus.COMPLETE,
                    progress=100.0,
                    transcription_id=result.get('transcription_id')
                )
            else:
                self.update_job_status(
                    job.id,
                    JobStatus.FAILED,
                    error_message=result.get('error', 'Unknown error')
                )
        except Exception as e:
            self.update_job_status(
                job.id,
                JobStatus.FAILED,
                error_message=str(e)
            )

        # Notify completion callbacks
        for callback in self._completion_callbacks:
            try:
                callback(job.id, self.get_job(job.id))
            except Exception:
                pass

        # Clean up temp file if it exists
        try:
            if job.file_path and os.path.exists(job.file_path):
                if job.file_path.startswith(tempfile.gettempdir()):
                    os.remove(job.file_path)
        except Exception:
            pass


def process_video_job(file_path: str, filename: str, client_id: int,
                      include_timestamps: bool, progress_callback: Callable) -> Dict:
    """
    Process a video file for transcription.
    This function should be called by the worker threads.

    Args:
        file_path: Path to the video file
        filename: Original filename
        client_id: Client ID for the transcription
        include_timestamps: Whether to include timestamps
        progress_callback: Function to call with progress updates (0-100)

    Returns:
        Dict with 'success' and optionally 'transcription_id' or 'error'
    """
    # Import here to avoid circular imports
    from utils import extract_audio, transcribe_audio
    from database import TranscriptionDB

    try:
        db = TranscriptionDB()

        # Extract audio (0-40% progress)
        progress_callback(10)
        audio_path = extract_audio(file_path)
        progress_callback(40)

        # Transcribe (40-90% progress)
        def transcribe_progress(p):
            # Map transcription progress (0-100) to our range (40-90)
            progress_callback(40 + (p * 0.5))

        transcription_text = transcribe_audio(audio_path, include_timestamps)
        progress_callback(90)

        # Save to database (90-100% progress)
        transcription_id = db.add_transcription(
            client_id=client_id,
            original_filename=filename,
            transcription_text=transcription_text,
            include_timestamps=include_timestamps
        )
        progress_callback(100)

        # Clean up audio file
        try:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception:
            pass

        return {
            'success': True,
            'transcription_id': transcription_id
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


# Singleton instance for easy access
_batch_queue: Optional[BatchJobQueue] = None


def get_batch_queue(db_path: str = "transcription.db") -> BatchJobQueue:
    """Get or create the singleton batch job queue."""
    global _batch_queue
    if _batch_queue is None:
        _batch_queue = BatchJobQueue(db_path)
    return _batch_queue


def start_batch_processing(db_path: str = "transcription.db", max_concurrent: int = 1):
    """Start batch processing with the given configuration."""
    batch_queue = get_batch_queue(db_path)
    batch_queue.set_max_concurrent(max_concurrent)
    batch_queue.start_workers(process_video_job)
    return batch_queue


def stop_batch_processing():
    """Stop batch processing."""
    global _batch_queue
    if _batch_queue:
        _batch_queue.stop_workers()
