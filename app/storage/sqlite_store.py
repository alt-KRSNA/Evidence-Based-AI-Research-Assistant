
from contextlib import contextmanager
import hashlib
import json
import sqlite3

from pathlib import Path
from uuid import uuid4


DEFAULT_DB_PATH = Path("database/research_assistant.db")


@contextmanager
def get_connection(db_path=DEFAULT_DB_PATH):
    """Create and safely close a SQLite connection."""

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")

    try:
        yield connection
        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def calculate_file_hash(file_path: str | Path) -> str:
    """Calculate the SHA-256 hash of a file."""

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for block in iter(lambda: file.read(8192), b""):
            sha256.update(block)

    return sha256.hexdigest()


def initialize_database(db_path=DEFAULT_DB_PATH):
    """Create tables and apply database migrations."""

    with get_connection(db_path) as connection:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_path TEXT,
                file_hash TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Migrate existing databases.
        columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(documents)"
            ).fetchall()
        }

        if "file_hash" not in columns:
            connection.execute(
                "ALTER TABLE documents ADD COLUMN file_hash TEXT"
            )

        connection.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_file_hash
            ON documents(file_hash)
            WHERE file_hash IS NOT NULL
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                document_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                page_number INTEGER,
                content TEXT NOT NULL,
                token_count INTEGER,
                metadata TEXT DEFAULT '{}',

                PRIMARY KEY (document_id, chunk_id),

                FOREIGN KEY (document_id)
                REFERENCES documents(id)
                ON DELETE CASCADE
            )
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_document
            ON chunks(document_id)
        """)


def get_document_by_hash(
    file_hash: str,
    db_path=DEFAULT_DB_PATH,
):
    """Find an existing document using its SHA-256 hash."""

    with get_connection(db_path) as connection:

        result = connection.execute(
            """
            SELECT id, filename, file_path, file_hash, created_at
            FROM documents
            WHERE file_hash = ?
            """,
            (file_hash,),
        ).fetchone()

    return dict(result) if result else None


def save_document(
    filename,
    file_path=None,
    file_hash=None,
    db_path=DEFAULT_DB_PATH,
):
    """Register a document and return its ID.

    If the same file hash already exists, return its existing ID.
    """

    with get_connection(db_path) as connection:

        if file_hash:
            existing = connection.execute(
                """
                SELECT id
                FROM documents
                WHERE file_hash = ?
                """,
                (file_hash,),
            ).fetchone()

            if existing:
                return existing["id"]

        document_id = str(uuid4())

        connection.execute(
            """
            INSERT INTO documents (
                id, filename, file_path, file_hash
            )
            VALUES (?, ?, ?, ?)
            """,
            (document_id, filename, file_path, file_hash),
        )

    return document_id


def save_chunks(
    document_id,
    chunks,
    db_path=DEFAULT_DB_PATH,
):
    """Save extracted document chunks."""

    records = []

    for chunk in chunks:
        chunk_id = (
            f"p{chunk['page_number']}_c{chunk['chunk_index']}"
        )

        metadata = {
            "source": chunk["source"],
            "chunk_index": chunk["chunk_index"],
        }

        records.append((
            document_id,
            chunk_id,
            chunk["page_number"],
            chunk["text"],
            None,
            json.dumps(metadata),
        ))

    with get_connection(db_path) as connection:

        connection.executemany(
            """
            INSERT INTO chunks (
                document_id,
                chunk_id,
                page_number,
                content,
                token_count,
                metadata
            )
            VALUES (?, ?, ?, ?, ?, ?)

            ON CONFLICT(document_id, chunk_id)
            DO UPDATE SET
                page_number = excluded.page_number,
                content = excluded.content,
                token_count = excluded.token_count,
                metadata = excluded.metadata
            """,
            records,
        )

    return len(records)


def get_chunk(
    document_id,
    chunk_id,
    db_path=DEFAULT_DB_PATH,
):
    """Retrieve one chunk using its document and chunk IDs."""

    with get_connection(db_path) as connection:

        result = connection.execute(
            """
            SELECT *, content AS text
            FROM chunks
            WHERE document_id = ? AND chunk_id = ?
            """,
            (document_id, chunk_id),
        ).fetchone()

    if result is None:
        return None

    chunk = dict(result)
    chunk["metadata"] = json.loads(chunk["metadata"])

    return chunk


def get_document_chunks(
    document_id,
    db_path=DEFAULT_DB_PATH,
):
    """Retrieve all chunks belonging to a document."""

    with get_connection(db_path) as connection:

        results = connection.execute(
            """
            SELECT *
            FROM chunks
            WHERE document_id = ?
            ORDER BY page_number, chunk_id
            """,
            (document_id,),
        ).fetchall()

    chunks = []

    for result in results:
        chunk = dict(result)

        chunk["text"] = chunk.pop("content")
        chunk["metadata"] = json.loads(chunk["metadata"])

        chunks.append(chunk)

    return chunks
