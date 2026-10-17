-- Manual, one-time update for the ask-and-answer identity flow (16 October 2026).
--
-- WHAT IT CHANGES
--   1. item_embeddings.confirmed: marks a stored view a person confirmed (picked from a
--      question) as opposed to one the system stored on its own. Every row that exists now
--      becomes confirmed = 0 (automatic), which is true of all of them.
--   2. negative_examples: a new table of views a person said are not items (a hand, the
--      table). Nothing is read from it yet; it is only written to.
--
-- BEFORE YOU RUN IT
--   * Stop Project AUTO (and the diagnostics app).
--   * Make a backup copy of the database file. For example, in PowerShell from the
--     project root:
--         Copy-Item data\project_auto.db data\project_auto.db.bak
--     The write-ahead files (project_auto.db-wal / -shm), if present, belong with it.
--   * Run this against ONE database at a time. The diagnostics database
--     (data\diagnostics\project_auto_diagnostics.db) is separate and needs it too if it
--     already exists; if you would rather not migrate it, move it aside and let the
--     diagnostics app create a fresh one.
--
-- HOW TO RUN IT (Python's own sqlite3 is enough; the sqlite3 command-line tool is not needed)
--     .\.venv\Scripts\python.exe -c "import sqlite3; c = sqlite3.connect('data/project_auto.db'); c.executescript(open('docs/manual_schema_update_ask_answer.sql', encoding='utf-8').read()); print('done')"
--
-- DO NOT RUN IT TWICE. The ALTER TABLE below fails with "duplicate column name: confirmed"
-- on a database that already has it; that error means it is already updated, and the
-- transaction rolls back and changes nothing. The CREATE statements are safe to repeat
-- because Project AUTO's own startup may already have created the empty negative_examples
-- table (it creates missing tables, but never alters existing ones).
--
-- TO CHECK IT WORKED
--     select name from pragma_table_info('item_embeddings');   -- includes "confirmed"
--     select count(*) from item_embeddings where confirmed = 0; -- every existing view
--     select count(*) from negative_examples;                   -- 0
--
-- TO UNDO IT (only if nothing has been confirmed yet; needs SQLite 3.35 or newer)
--     ALTER TABLE item_embeddings DROP COLUMN confirmed;
--     DROP TABLE negative_examples;
-- or simply restore the backup copy.

PRAGMA foreign_keys = ON;
BEGIN IMMEDIATE;

ALTER TABLE item_embeddings ADD COLUMN confirmed BOOLEAN NOT NULL DEFAULT 0;

CREATE TABLE IF NOT EXISTS negative_examples (
    id INTEGER NOT NULL,
    model_name VARCHAR(200) NOT NULL,
    embedding JSON NOT NULL,
    aspect_ratio FLOAT,
    color_histogram JSON,
    object_image_path VARCHAR(1000),
    created_at DATETIME NOT NULL,
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS ix_negative_examples_model_name ON negative_examples (model_name);

COMMIT;
