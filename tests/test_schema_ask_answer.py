"""Schema tests for the ask-and-answer flow: confirmed views and negative examples.

The migration tests matter most. They build a database in the shape the live file has today
(before these columns), put rows in it, run the real manual SQL script against it, and check
that nothing was lost, that the result matches what the models create, and that a second run
does no damage. No test touches the real data/project_auto.db.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from project_auto.app import _require_current_schema
from project_auto.memory.models import Base, Item, ItemEmbedding
from project_auto.memory.store import DatabaseStore

SCRIPT = Path(__file__).resolve().parents[1] / "docs" / "manual_schema_update_ask_answer.sql"
MODEL = "test-model"


def make_store(path: Path) -> DatabaseStore:
    """A real file-backed store, so SQLite's own ALTER TABLE behaviour is what gets tested."""
    store = DatabaseStore(path)
    store.create_schema()
    return store


@pytest.fixture
def store(tmp_path: Path) -> DatabaseStore:
    database = make_store(tmp_path / "t.db")
    yield database
    database.engine.dispose()


def old_database(path: Path) -> Path:
    """Build a database in today's live shape: no confirmed column, no negatives table."""
    store = make_store(path)
    item = store.create_item("thing")
    store.save_item_embedding(item.id, [1.0, 0.0], MODEL, aspect_ratio=1.2)
    store.save_item_embedding(item.id, [0.0, 1.0], MODEL, color_histogram=[0.5, 0.5])
    store.engine.dispose()
    connection = sqlite3.connect(path)
    connection.executescript(
        "ALTER TABLE item_embeddings DROP COLUMN confirmed; DROP TABLE negative_examples;"
    )
    connection.close()
    return path


def columns(path: Path, table: str) -> list[tuple]:
    """(name, type, notnull, pk) for every column, in order."""
    connection = sqlite3.connect(path)
    try:
        return [
            (row[1], row[2], row[3], row[5])
            for row in connection.execute(f"pragma table_info({table})")
        ]
    finally:
        connection.close()


def run_script(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(SCRIPT.read_text(encoding="utf-8"))
    finally:
        connection.close()


class TestConfirmedViews:
    def test_a_view_is_automatic_unless_said_otherwise(self, store: DatabaseStore) -> None:
        item = store.create_item("thing")

        saved = store.save_item_embedding(item.id, [1.0, 0.0], MODEL)

        assert saved.confirmed is False
        assert store.count_item_embeddings(item.id, MODEL, confirmed_only=True) == 0

    def test_a_confirmed_view_is_recorded_as_confirmed(self, store: DatabaseStore) -> None:
        item = store.create_item("thing")

        store.save_item_embedding(item.id, [1.0, 0.0], MODEL, confirmed=True)
        store.save_item_embedding(item.id, [0.0, 1.0], MODEL)

        assert store.count_item_embeddings(item.id, MODEL) == 2
        assert store.count_item_embeddings(item.id, MODEL, confirmed_only=True) == 1

    def test_add_reference_if_needed_carries_the_flag(self, store: DatabaseStore) -> None:
        item = store.create_item("thing")

        saved, _ = store.add_reference_if_needed(item.id, [1.0, 0.0], MODEL, 4, confirmed=True)

        assert saved
        assert store.count_item_embeddings(item.id, MODEL, confirmed_only=True) == 1

    def test_a_confirmed_view_still_counts_towards_the_cap(self, store: DatabaseStore) -> None:
        """Confirming does not buy room: the per-item limit is a hard ceiling."""
        item = store.create_item("thing")
        store.add_reference_if_needed(item.id, [1.0, 0.0], MODEL, 1)

        saved, count = store.add_reference_if_needed(item.id, [0.0, 1.0], MODEL, 1, confirmed=True)

        assert not saved and count == 1

    def test_confirming_one_view_does_not_confirm_the_others(self, store: DatabaseStore) -> None:
        item = store.create_item("thing")
        store.save_item_embedding(item.id, [1.0, 0.0], MODEL)
        store.save_item_embedding(item.id, [0.0, 1.0], MODEL, confirmed=True)

        with store.session_factory() as session:
            flags = sorted(
                row.confirmed for row in session.query(ItemEmbedding).filter_by(item_id=item.id)
            )

        assert flags == [False, True]


class TestNegativeExamples:
    def test_a_negative_example_is_stored_and_counted(self, store: DatabaseStore) -> None:
        store.add_negative_example([3.0, 4.0], MODEL, aspect_ratio=0.8, color_histogram=[1.0, 1.0])

        assert store.count_negative_examples() == 1
        assert store.count_negative_examples(MODEL) == 1
        assert store.count_negative_examples("another-model") == 0

    def test_vectors_are_normalized_like_references(self, store: DatabaseStore) -> None:
        store.add_negative_example([3.0, 4.0], MODEL)

        loaded = store.load_negative_embeddings(MODEL)

        assert loaded.shape == (1, 2)
        assert loaded.dtype == np.float32
        assert np.allclose(loaded[0], [0.6, 0.8])

    def test_models_are_kept_apart(self, store: DatabaseStore) -> None:
        store.add_negative_example([1.0, 0.0], "model-a")
        store.add_negative_example([0.0, 1.0], "model-b")

        assert store.load_negative_embeddings("model-a").shape == (1, 2)
        assert np.allclose(store.load_negative_embeddings("model-b")[0], [0.0, 1.0])

    def test_none_stored_gives_an_empty_array(self, store: DatabaseStore) -> None:
        assert store.load_negative_embeddings(MODEL).shape == (0, 0)

    def test_it_belongs_to_no_item_and_touches_no_prototype(self, store: DatabaseStore) -> None:
        item = store.create_item("thing")
        store.save_item_embedding(item.id, [1.0, 0.0], MODEL)
        before = store.get_item(item.id).item_prototype

        store.add_negative_example([0.0, 1.0], MODEL)

        assert store.get_item(item.id).item_prototype == before
        assert store.count_item_embeddings(item.id, MODEL) == 1
        assert store.count_items() == 1

    @pytest.mark.parametrize("bad", [[], [float("nan"), 1.0], [float("inf"), 0.0]])
    def test_an_unusable_vector_is_refused(self, store: DatabaseStore, bad: list[float]) -> None:
        with pytest.raises(ValueError):
            store.add_negative_example(bad, MODEL)

        assert store.count_negative_examples() == 0

    def test_a_model_name_is_required(self, store: DatabaseStore) -> None:
        with pytest.raises(ValueError, match="model_name"):
            store.add_negative_example([1.0, 0.0], "  ")

    def test_deleting_an_item_never_removes_a_negative_example(self, store: DatabaseStore) -> None:
        store.add_negative_example([1.0, 0.0], MODEL)
        item = store.create_item("thing")

        with store.session_factory() as session:
            session.delete(session.get(Item, item.id))
            session.commit()

        assert store.count_negative_examples() == 1


class TestSchemaCheck:
    def test_a_current_database_reports_nothing_missing(self, store: DatabaseStore) -> None:
        assert store.missing_schema_columns() == []

    def test_an_old_database_reports_exactly_what_it_lacks(self, tmp_path: Path) -> None:
        path = old_database(tmp_path / "old.db")
        store = DatabaseStore(path)

        assert store.missing_schema_columns() == [("item_embeddings", "confirmed")]
        store.engine.dispose()

    def test_a_missing_table_is_not_reported_because_it_is_created_not_altered(
        self, tmp_path: Path
    ) -> None:
        path = old_database(tmp_path / "old.db")
        store = DatabaseStore(path)

        assert all(table != "negative_examples" for table, _ in store.missing_schema_columns())
        store.engine.dispose()

    def test_checking_never_changes_the_database(self, tmp_path: Path) -> None:
        path = old_database(tmp_path / "old.db")
        before = path.read_bytes()
        store = DatabaseStore(path)

        store.missing_schema_columns()
        store.engine.dispose()

        assert path.read_bytes() == before

    def test_the_app_stops_with_an_instruction_instead_of_a_bare_sql_error(
        self, tmp_path: Path
    ) -> None:
        path = old_database(tmp_path / "old.db")
        store = DatabaseStore(path)

        with pytest.raises(RuntimeError) as raised:
            _require_current_schema(store, path)
        store.engine.dispose()

        message = str(raised.value)
        assert "item_embeddings.confirmed" in message
        assert "manual_schema_update_ask_answer.sql" in message
        assert "backup" in message
        assert "Nothing was changed" in message

    def test_the_app_carries_on_with_a_current_database(self, store: DatabaseStore) -> None:
        _require_current_schema(store, Path("anything.db"))


class TestManualUpdateScript:
    def test_it_adds_the_column_and_keeps_every_existing_row(self, tmp_path: Path) -> None:
        path = old_database(tmp_path / "old.db")

        run_script(path)

        connection = sqlite3.connect(path)
        rows = connection.execute(
            "select item_id, model_name, aspect_ratio, confirmed from item_embeddings order by id"
        ).fetchall()
        connection.close()
        assert len(rows) == 2
        assert all(row[1] == MODEL for row in rows)
        assert rows[0][2] == 1.2, "an existing value must survive"
        assert [row[3] for row in rows] == [0, 0], "every existing view stays automatic"

    def test_the_result_matches_what_the_models_create(self, tmp_path: Path) -> None:
        """If the script and the models drift apart, a migrated database would differ from a new one."""
        migrated = old_database(tmp_path / "old.db")
        run_script(migrated)
        fresh = tmp_path / "fresh.db"
        make_store(fresh).engine.dispose()

        for table in ("item_embeddings", "negative_examples"):
            assert columns(migrated, table) == columns(fresh, table), table

    def test_a_migrated_database_works_with_the_store(self, tmp_path: Path) -> None:
        path = old_database(tmp_path / "old.db")
        run_script(path)
        store = DatabaseStore(path)

        assert store.missing_schema_columns() == []
        saved, _ = store.add_reference_if_needed(1, [1.0, 1.0], MODEL, 8, confirmed=True)
        store.add_negative_example([1.0, 0.0], MODEL)

        assert saved
        assert store.count_item_embeddings(1, MODEL) == 3
        assert store.count_item_embeddings(1, MODEL, confirmed_only=True) == 1
        assert store.count_negative_examples() == 1
        store.engine.dispose()

    def test_it_tolerates_the_negatives_table_already_existing(self, tmp_path: Path) -> None:
        """Startup creates missing tables before it notices the missing column."""
        path = old_database(tmp_path / "old.db")
        store = DatabaseStore(path)
        store.create_schema()  # what the app does first: creates negative_examples, adds no column
        store.engine.dispose()
        assert any(c[0] == "id" for c in columns(path, "negative_examples"))

        run_script(path)

        assert ("confirmed", "BOOLEAN", 1, 0) in columns(path, "item_embeddings")

    def test_running_it_twice_fails_loudly_and_changes_nothing(self, tmp_path: Path) -> None:
        path = old_database(tmp_path / "old.db")
        run_script(path)
        connection = sqlite3.connect(path)
        connection.execute(
            "insert into negative_examples (model_name, embedding, created_at) values ('m', '[1]', '2026-01-01')"
        )
        connection.commit()
        connection.close()
        before = columns(path, "item_embeddings")

        with pytest.raises(sqlite3.OperationalError, match="duplicate column name"):
            run_script(path)

        assert columns(path, "item_embeddings") == before
        connection = sqlite3.connect(path)
        assert connection.execute("select count(*) from negative_examples").fetchone()[0] == 1
        connection.close()

    def test_it_leaves_other_tables_alone(self, tmp_path: Path) -> None:
        path = old_database(tmp_path / "old.db")
        connection = sqlite3.connect(path)
        before = {
            table: connection.execute(f"select * from {table}").fetchall()
            for table in ("items", "item_events", "regions")
        }
        connection.close()

        run_script(path)

        connection = sqlite3.connect(path)
        after = {
            table: connection.execute(f"select * from {table}").fetchall()
            for table in ("items", "item_events", "regions")
        }
        connection.close()
        assert after == before

    def test_the_script_documents_its_own_backup_and_undo(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8").lower()

        assert "backup" in text
        assert "do not run it twice" in text
        assert "to undo" in text
        assert "stop project auto" in text


class TestModelsAndScriptAgree:
    def test_the_models_know_both_new_things(self) -> None:
        assert "confirmed" in {c.name for c in Base.metadata.tables["item_embeddings"].columns}
        assert "negative_examples" in Base.metadata.tables

    def test_the_default_is_set_in_the_database_not_only_in_python(self) -> None:
        """ALTER TABLE ... NOT NULL needs a database default, or existing rows would fail."""
        column = Base.metadata.tables["item_embeddings"].columns["confirmed"]

        assert column.server_default is not None
        assert column.nullable is False

    def test_the_default_applies_to_a_row_inserted_without_the_column(self) -> None:
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        with sessionmaker(bind=engine, class_=Session)() as session:
            item = Item(class_name="thing")
            session.add(item)
            session.commit()
            session.execute(
                ItemEmbedding.__table__.insert().values(
                    item_id=item.id, model_name="m", embedding=[1.0], created_at=item.created_at
                )
            )
            session.commit()
            assert session.query(ItemEmbedding).one().confirmed is False
        engine.dispose()

    def test_negative_examples_have_no_foreign_keys(self) -> None:
        assert not Base.metadata.tables["negative_examples"].foreign_keys
