import sqlite3
import hashlib
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "db.file")

def hash_password(password: str) -> str:
    """Basic SHA-256 hashing for passwords."""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def create_user(username: str, email: str, password: str) -> bool:
    """Creates a user. Returns True if successful, False if already exists."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
            (username, email, hash_password(password))
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(username: str, password: str) -> dict | None:
    """Verifies credentials. Returns user dict on success, None on failure."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, username, email FROM users WHERE username = ? AND password = ?", 
                   (username, hash_password(password)))
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return dict(user)
    return None

# Auto-initialize DB when imported
init_db()
