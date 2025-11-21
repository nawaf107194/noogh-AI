import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any

class ExperienceTracker:
    """
    Tracks experiences (queries, answers, etc.) in a SQLite database
    for reflection and learning.
    """
    def __init__(self, db_path: str = "data/reflection.db"):
        """
        Initializes the ExperienceTracker and connects to the database.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_table()
        self.session_start_time = datetime.now()

    def _create_table(self):
        """Creates the 'experiences' table if it doesn't exist."""
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS experiences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    question TEXT NOT NULL,
                    intent TEXT,
                    answer TEXT,
                    sources TEXT,
                    confidence REAL,
                    success BOOLEAN,
                    execution_time REAL,
                    handler TEXT,
                    used_web_search BOOLEAN
                )
                """)

    def track(
        self,
        question: str,
        intent: str,
        answer: str,
        sources: List[Dict[str, Any]],
        confidence: float,
        success: bool,
        execution_time: float,
        handler: str,
        used_web_search: bool = False
    ):
        """
        Logs an experience to the database.
        """
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO experiences (
                    timestamp, question, intent, answer, sources, 
                    confidence, success, execution_time, handler, used_web_search
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(),
                    question,
                    intent,
                    answer,
                    json.dumps(sources),
                    confidence,
                    success,
                    execution_time,
                    handler,
                    used_web_search,
                ),
            )

    def get_session_stats(self) -> Dict[str, Any]:
        """
        Calculates and returns statistics for the current session.
        "Session" is defined as the time since this ExperienceTracker was initialized.
        """
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT COUNT(*), AVG(confidence), AVG(execution_time) FROM experiences WHERE timestamp >= ?",
                (self.session_start_time,)
            )
            row = cursor.fetchone()
            
            if row and row[0] > 0:
                return {
                    "queries_in_session": row[0],
                    "average_confidence": row[1],
                    "average_execution_time": row[2],
                }
            else:
                return {
                    "queries_in_session": 0,
                    "average_confidence": 0.0,
                    "average_execution_time": 0.0,
                }

    def get_all_experiences(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieves all experiences from the database."""
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM experiences ORDER BY timestamp DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def __del__(self):
        """Ensures the database connection is closed when the object is destroyed."""
        if self.conn:
            self.conn.close()

if __name__ == '__main__':
    # Example Usage
    print("--- Testing ExperienceTracker ---")
    tracker = ExperienceTracker(db_path=":memory:") # Use in-memory DB for testing

    # Track a dummy experience
    tracker.track(
        question="What is the capital of France?",
        intent="question.web_search",
        answer="Paris",
        sources=[{"title": "Wikipedia", "link": "https://en.wikipedia.org/wiki/Paris"}],
        confidence=0.9,
        success=True,
        execution_time=0.123,
        handler="_handle_web_question",
        used_web_search=True
    )
    
    tracker.track(
        question="Hello",
        intent="chitchat",
        answer="Hello! How can I help you today?",
        sources=[],
        confidence=0.95,
        success=True,
        execution_time=0.01,
        handler="_handle_chitchat",
        used_web_search=False
    )

    # Get session stats
    stats = tracker.get_session_stats()
    print("\nSession Stats:")
    print(json.dumps(stats, indent=2))

    # Get all experiences
    experiences = tracker.get_all_experiences()
    print("\nAll Experiences (most recent first):")
    for exp in experiences:
        # Truncate long fields for display
        exp['answer'] = exp['answer'][:50] + '...' if len(exp['answer']) > 50 else exp['answer']
        exp['sources'] = exp['sources'][:50] + '...' if len(exp['sources']) > 50 else exp['sources']
        print(json.dumps(exp, indent=2))