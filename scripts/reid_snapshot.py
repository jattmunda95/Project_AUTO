"""Back up and restore the ReID diagnostic database between tuning blocks.

Usage (run only while the diagnostics app is stopped):
    python scripts/reid_snapshot.py backup post_enrolment
    python scripts/reid_snapshot.py restore post_enrolment
    python scripts/reid_snapshot.py list

The live database path is read from configs/reid_diagnostics.yaml (database_path);
snapshots live in a snapshots/ folder next to it. The normal project-auto database
is never touched.

Copies use SQLite's backup API rather than a file copy: the store runs in WAL mode,
so recent writes may sit in the -wal side file and a plain copy of the .db file
would silently lose them. restore first saves the live database it is about to
overwrite as snapshots/_before_restore.db, so one mistaken restore is recoverable.
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS_CONFIG_PATH = PROJECT_ROOT / "configs" / "reid_diagnostics.yaml"
BEFORE_RESTORE_LABEL = "_before_restore"
LABEL_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def live_database_path() -> Path:
    """Return the diagnostic database path configured in reid_diagnostics.yaml."""
    with DIAGNOSTICS_CONFIG_PATH.open(encoding="utf-8") as config_file:
        settings = yaml.safe_load(config_file)
    return PROJECT_ROOT / settings["database_path"]


def snapshot_path(live_path: Path, label: str) -> Path:
    """Return where the snapshot with this label is stored."""
    if not LABEL_PATTERN.match(label):
        raise SystemExit(f"Invalid label {label!r}: use letters, digits, '_' or '-' only.")
    return live_path.parent / "snapshots" / f"{label}.db"


def ensure_not_in_use(live_path: Path) -> None:
    """Abort if the diagnostics app still appears to hold the live database open.

    On Windows a file open in another process cannot be renamed, so a rename
    round-trip is a reliable probe there. Elsewhere this check cannot see other
    processes, so stopping the app first remains the actual rule.
    """
    if not live_path.exists():
        return
    probe_path = live_path.with_name(live_path.name + ".probe")
    try:
        os.rename(live_path, probe_path)
    except PermissionError:
        raise SystemExit(
            f"{live_path} is in use. Stop the diagnostics app (Ctrl+C) and try again."
        ) from None
    os.rename(probe_path, live_path)


def copy_database(source: Path, destination: Path) -> None:
    """Copy one SQLite database into another, including any unmerged WAL content."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_connection = sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True)
    destination_connection = sqlite3.connect(destination)
    try:
        source_connection.backup(destination_connection)
    finally:
        destination_connection.close()
        source_connection.close()


def backup(label: str, force: bool) -> None:
    """Save the live diagnostic database as a named snapshot."""
    live_path = live_database_path()
    if not live_path.exists():
        raise SystemExit(f"No diagnostic database at {live_path}; nothing to back up.")
    target = snapshot_path(live_path, label)
    if target.exists() and not force:
        raise SystemExit(f"Snapshot {label!r} already exists; pass --force to overwrite it.")
    ensure_not_in_use(live_path)
    if target.exists():
        target.unlink()
    copy_database(live_path, target)
    print(f"Backed up {live_path} -> {target}")


def restore(label: str, assume_yes: bool) -> None:
    """Overwrite the live diagnostic database with a named snapshot."""
    live_path = live_database_path()
    source = snapshot_path(live_path, label)
    if not source.exists():
        raise SystemExit(f"No snapshot {label!r} at {source}. Run 'list' to see snapshots.")
    ensure_not_in_use(live_path)
    if not assume_yes:
        answer = input(f"Overwrite {live_path} with snapshot {label!r}? [y/N] ")
        if answer.strip().lower() != "y":
            raise SystemExit("Restore cancelled.")

    if live_path.exists() and label != BEFORE_RESTORE_LABEL:
        safety_copy = snapshot_path(live_path, BEFORE_RESTORE_LABEL)
        if safety_copy.exists():
            safety_copy.unlink()
        copy_database(live_path, safety_copy)
        print(f"Saved the current live database as {safety_copy}")

    # Remove the old database and its WAL side files so no stale WAL content can
    # be replayed on top of the restored pages.
    for suffix in ("", "-wal", "-shm"):
        live_path.with_name(live_path.name + suffix).unlink(missing_ok=True)
    copy_database(source, live_path)
    print(f"Restored {source} -> {live_path}")


def list_snapshots() -> None:
    """Print every saved snapshot with its size and modification time."""
    snapshots_dir = live_database_path().parent / "snapshots"
    snapshots = sorted(snapshots_dir.glob("*.db")) if snapshots_dir.exists() else []
    if not snapshots:
        print(f"No snapshots in {snapshots_dir}")
        return
    for path in snapshots:
        stats = path.stat()
        modified = datetime.fromtimestamp(stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"{path.stem:<30} {stats.st_size / 1024:>10.1f} KB  {modified}")


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and run one snapshot command."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    backup_parser = commands.add_parser("backup", help="save the live database as a snapshot")
    backup_parser.add_argument("label")
    backup_parser.add_argument("--force", action="store_true", help="overwrite an existing snapshot")

    restore_parser = commands.add_parser("restore", help="overwrite the live database with a snapshot")
    restore_parser.add_argument("label")
    restore_parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")

    commands.add_parser("list", help="list saved snapshots")

    args = parser.parse_args(argv)
    if args.command == "backup":
        backup(args.label, args.force)
    elif args.command == "restore":
        restore(args.label, args.yes)
    else:
        list_snapshots()


if __name__ == "__main__":
    main(sys.argv[1:])
