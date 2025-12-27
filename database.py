import sqlite3
from typing import Optional, List, Tuple
import os

class TranscriptionDB:
    def __init__(self, db_path: str = "transcription.db"):
        """Initialize the database connection."""
        self.db_path = db_path
        self._create_tables()
        self._migrate_database()

    def _migrate_database(self):
        """Handle database migrations."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if we need to migrate persona_prompts table
            cursor.execute("PRAGMA table_info(persona_prompts)")
            columns = {col[1] for col in cursor.fetchall()}
            
            if "persona_prompt" in columns and "system_prompt" not in columns:
                # Rename persona_prompt to system_prompt
                cursor.execute('''
                    CREATE TABLE persona_prompts_new (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        transcription_id INTEGER NOT NULL,
                        persona_name TEXT NOT NULL,
                        system_prompt TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (transcription_id) REFERENCES transcriptions (id)
                    )
                ''')
                
                # Copy data from old table to new table
                cursor.execute('''
                    INSERT INTO persona_prompts_new (transcription_id, persona_name, system_prompt, created_at)
                    SELECT transcription_id, persona_name, persona_prompt, created_at
                    FROM persona_prompts
                ''')
                
                # Drop old table and rename new table
                cursor.execute('DROP TABLE persona_prompts')
                cursor.execute('ALTER TABLE persona_prompts_new RENAME TO persona_prompts')
            
            # Check if we need to migrate transcriptions table
            cursor.execute("PRAGMA table_info(transcriptions)")
            columns = {col[1] for col in cursor.fetchall()}
            
            if "created_at" in columns and "timestamp" not in columns:
                # Create new transcriptions table with updated schema
                cursor.execute('''
                    CREATE TABLE transcriptions_new (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        client_id INTEGER,
                        filename TEXT NOT NULL,
                        original_text TEXT NOT NULL,
                        translated_text TEXT,
                        target_language TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (client_id) REFERENCES clients (id)
                    )
                ''')
                
                # Copy data from old table to new table
                cursor.execute('''
                    INSERT INTO transcriptions_new 
                    SELECT * FROM transcriptions
                ''')
                
                # Drop old table and rename new table
                cursor.execute('DROP TABLE transcriptions')
                cursor.execute('ALTER TABLE transcriptions_new RENAME TO transcriptions')
            
            conn.commit()

    def _create_tables(self):
        """Create necessary tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create clients table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS clients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create transcriptions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS transcriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_id INTEGER,
                    filename TEXT NOT NULL,
                    original_text TEXT NOT NULL,
                    translated_text TEXT,
                    target_language TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (client_id) REFERENCES clients (id)
                )
            ''')
            
            # Create persona_prompts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS persona_prompts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transcription_id INTEGER NOT NULL,
                    persona_name TEXT NOT NULL,
                    system_prompt TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (transcription_id) REFERENCES transcriptions (id)
                )
            ''')

            # Create generated_content table for task presets
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS generated_content (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transcription_id INTEGER NOT NULL,
                    task_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (transcription_id) REFERENCES transcriptions (id)
                )
            ''')

            conn.commit()

    def add_client(self, name: str, email: str) -> int:
        """Add a new client to the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO clients (name, email) VALUES (?, ?)',
                (name, email)
            )
            return cursor.lastrowid

    def get_client(self, client_id: int) -> Optional[Tuple[int, str, str]]:
        """Get client details by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT * FROM clients WHERE id = ?',
                (client_id,)
            )
            return cursor.fetchone()

    def get_all_clients(self) -> List[Tuple[int, str, str]]:
        """Get all clients."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM clients')
            return cursor.fetchall()

    def add_transcription(self, client_id: int, filename: str, 
                         original_text: str, translated_text: Optional[str] = None,
                         target_language: Optional[str] = None) -> int:
        """Add a new transcription to the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''INSERT INTO transcriptions 
                   (client_id, filename, original_text, translated_text, target_language)
                   VALUES (?, ?, ?, ?, ?)''',
                (client_id, filename, original_text, translated_text, target_language)
            )
            return cursor.lastrowid

    def get_transcription(self, transcription_id: int) -> Optional[Tuple]:
        """Get transcription details by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT * FROM transcriptions WHERE id = ?',
                (transcription_id,)
            )
            return cursor.fetchone()

    def get_client_transcriptions(self, client_id: int) -> List[Tuple]:
        """Get all transcriptions for a client."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''SELECT * FROM transcriptions 
                   WHERE client_id = ? 
                   ORDER BY created_at DESC''',
                (client_id,)
            )
            return cursor.fetchall()

    def add_persona_prompt(self, transcription_id: int, persona_name: str, system_prompt: str) -> int:
        """Add a new persona prompt to the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''INSERT INTO persona_prompts 
                   (transcription_id, persona_name, system_prompt)
                   VALUES (?, ?, ?)''',
                (transcription_id, persona_name, system_prompt)
            )
            return cursor.lastrowid

    def get_persona_prompt(self, transcription_id: int) -> Optional[Tuple[str, str]]:
        """Get persona prompt for a transcription."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''SELECT persona_name, system_prompt 
                   FROM persona_prompts 
                   WHERE transcription_id = ?''',
                (transcription_id,)
            )
            result = cursor.fetchone()
            return result  # This will return None if no result is found

    def get_all_persona_prompts(self) -> List[Tuple]:
        """Get all persona prompts."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''SELECT p.*, t.filename, c.name as client_name 
                   FROM persona_prompts p
                   JOIN transcriptions t ON p.transcription_id = t.id
                   JOIN clients c ON t.client_id = c.id
                   ORDER BY p.created_at DESC'''
            )
            return cursor.fetchall()

    def update_client(self, client_id: int, name: str, email: str) -> bool:
        """Update client information."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE clients SET name = ?, email = ? WHERE id = ?',
                (name, email, client_id)
            )
            return cursor.rowcount > 0

    def delete_client(self, client_id: int) -> bool:
        """Delete a client and all their associated transcriptions."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute('BEGIN TRANSACTION')
                cursor.execute('DELETE FROM transcriptions WHERE client_id = ?', (client_id,))
                cursor.execute('DELETE FROM clients WHERE id = ?', (client_id,))
                conn.commit()
                return True
            except Exception as e:
                conn.rollback()
                return False

    def get_all_clients(self) -> List[Tuple[int, str, str]]:
        """Get all clients."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, name, email FROM clients ORDER BY name')
            return cursor.fetchall()

    def get_client_by_id(self, client_id: int) -> Optional[Tuple[int, str, str]]:
        """Get client by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, name, email FROM clients WHERE id = ?', (client_id,))
            return cursor.fetchone()

    def add_transcription(self, client_id: int, original_filename: str, transcription_text: str, 
                         include_timestamps: bool, target_language: Optional[str] = None) -> int:
        """Add a new transcription record."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO transcriptions 
            (client_id, filename, original_text, translated_text, target_language)
            VALUES (?, ?, ?, ?, ?)
            ''', (client_id, original_filename, transcription_text, None, target_language))
            return cursor.lastrowid

    def delete_transcription(self, transcription_id: int) -> bool:
        """Delete a transcription record."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM transcriptions WHERE id = ?', (transcription_id,))
            return cursor.rowcount > 0

    def update_transcription_metadata(self, transcription_id: int, target_language: str) -> bool:
        """Update transcription metadata."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
            UPDATE transcriptions 
            SET target_language = ?
            WHERE id = ?
            ''', (target_language if target_language != "Original" else None, transcription_id))
            return cursor.rowcount > 0

    def get_client_transcriptions(self, client_id: int = None, email: str = None) -> List[Tuple]:
        """Get all transcriptions for a client by ID or email."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if client_id:
                cursor.execute('''
                SELECT t.* FROM transcriptions t
                WHERE t.client_id = ?
                ORDER BY t.created_at DESC
                ''', (client_id,))
            else:
                cursor.execute('''
                SELECT t.* FROM transcriptions t
                JOIN clients c ON t.client_id = c.id
                WHERE c.email = ?
                ORDER BY t.created_at DESC
                ''', (email,))
            return cursor.fetchall()

    def get_transcription(self, transcription_id: int) -> Optional[Tuple]:
        """Get a specific transcription by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM transcriptions WHERE id = ?', (transcription_id,))
            return cursor.fetchone()

    def get_transcription_by_id(self, transcription_id: int) -> Optional[Tuple]:
        """Get a transcription by its ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM transcriptions WHERE id = ?', (transcription_id,))
            result = cursor.fetchone()
            return result

    def add_persona_prompt(self, transcription_id: int, persona_name: str, system_prompt: str) -> int:
        """Add a new persona prompt to the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO persona_prompts (transcription_id, persona_name, system_prompt)
            VALUES (?, ?, ?)
            ''', (transcription_id, persona_name, system_prompt))
            return cursor.lastrowid

    def get_persona_prompt(self, transcription_id: int) -> Optional[Tuple[str, str]]:
        """Get the persona prompt for a specific transcription."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
            SELECT persona_name, system_prompt 
            FROM persona_prompts 
            WHERE transcription_id = ?
            ''', (transcription_id,))
            return cursor.fetchone()

    def get_all_client_transcriptions_text(self, client_id: int) -> List[Tuple]:
        """Get all transcriptions text for a client for bulk export."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute('''
                SELECT original_text, target_language, created_at 
                FROM transcriptions 
                WHERE client_id = ?
                ORDER BY created_at
                ''', (client_id,))
                return cursor.fetchall()
            finally:
                conn.close()

    def update_persona_prompt(self, transcription_id: int, persona_name: str, system_prompt: str) -> bool:
        """Update an existing persona prompt."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute('''
                    UPDATE persona_prompts 
                    SET persona_name = ?, system_prompt = ?
                    WHERE transcription_id = ?
                ''', (persona_name, system_prompt, transcription_id))
                return cursor.rowcount > 0
            except Exception as e:
                print(f"Error updating persona prompt: {str(e)}")
                return False

    def delete_transcript(self, transcript_id: int) -> bool:
        """
        Delete a specific transcript and its associated data.
        
        Args:
            transcript_id (int): The ID of the transcript to delete.
        
        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete associated persona prompts first
                cursor.execute('''
                    DELETE FROM persona_prompts 
                    WHERE transcription_id = ?
                ''', (transcript_id,))
                
                # Delete the transcript
                cursor.execute('''
                    DELETE FROM transcriptions 
                    WHERE id = ?
                ''', (transcript_id,))
                
                # Commit the transaction
                conn.commit()
                
                # Return True if at least one row was affected
                return cursor.rowcount > 0
        except sqlite3.Error as e:
            print(f"Error deleting transcript: {e}")
            return False

    def delete_client(self, client_id: int) -> bool:
        """
        Delete a specific client and all their associated transcripts.
        
        Args:
            client_id (int): The ID of the client to delete.
        
        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # First, find and delete all transcripts for this client
                cursor.execute('''
                    SELECT id FROM transcriptions 
                    WHERE client_id = ?
                ''', (client_id,))
                
                transcript_ids = [row[0] for row in cursor.fetchall()]
                
                # Delete associated persona prompts for these transcripts
                if transcript_ids:
                    cursor.execute('''
                        DELETE FROM persona_prompts 
                        WHERE transcription_id IN ({})
                    '''.format(','.join(map(str, transcript_ids)))
                    )
                
                # Delete all transcripts for this client
                cursor.execute('''
                    DELETE FROM transcriptions 
                    WHERE client_id = ?
                ''', (client_id,))
                
                # Delete the client
                cursor.execute('''
                    DELETE FROM clients 
                    WHERE id = ?
                ''', (client_id,))
                
                # Commit the transaction
                conn.commit()
                
                # Return True if at least one row was affected
                return cursor.rowcount > 0
        except sqlite3.Error as e:
            print(f"Error deleting client: {e}")
            return False

    def add_generated_content(self, transcription_id: int, task_type: str, content: str) -> int:
        """Add generated content from a task preset."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO generated_content (transcription_id, task_type, content)
                VALUES (?, ?, ?)
            ''', (transcription_id, task_type, content))
            return cursor.lastrowid

    def get_generated_content(self, transcription_id: int, task_type: str = None) -> List[Tuple]:
        """Get generated content for a transcription, optionally filtered by task type."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if task_type:
                cursor.execute('''
                    SELECT id, task_type, content, created_at
                    FROM generated_content
                    WHERE transcription_id = ? AND task_type = ?
                    ORDER BY created_at DESC
                ''', (transcription_id, task_type))
            else:
                cursor.execute('''
                    SELECT id, task_type, content, created_at
                    FROM generated_content
                    WHERE transcription_id = ?
                    ORDER BY created_at DESC
                ''', (transcription_id,))
            return cursor.fetchall()

    def delete_generated_content(self, content_id: int) -> bool:
        """Delete a specific generated content entry."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM generated_content WHERE id = ?', (content_id,))
            return cursor.rowcount > 0
