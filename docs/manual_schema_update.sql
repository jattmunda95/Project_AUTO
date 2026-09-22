-- Manual, one-time update for the inspected Project AUTO database schema.
-- Stop Project AUTO and make a SQLite backup before explicitly applying this file.
-- Requires the existing regions table (created by create_schema).
-- Do not run twice: duplicate-column errors indicate a different/already updated schema.
-- Existing rows retain their values; new optional fields begin as SQL NULL.
PRAGMA foreign_keys = ON;
BEGIN IMMEDIATE;

ALTER TABLE item_embeddings ADD COLUMN aspect_ratio FLOAT;
ALTER TABLE item_embeddings ADD COLUMN color_histogram JSON;

ALTER TABLE items ADD COLUMN current_region_id INTEGER
    REFERENCES regions(id) ON DELETE SET NULL;
ALTER TABLE items ADD COLUMN current_box JSON;

ALTER TABLE item_events ADD COLUMN source_region_id INTEGER
    REFERENCES regions(id) ON DELETE SET NULL;
ALTER TABLE item_events ADD COLUMN destination_region_id INTEGER
    REFERENCES regions(id) ON DELETE SET NULL;
ALTER TABLE item_events ADD COLUMN source_box_area_fraction FLOAT;
ALTER TABLE item_events ADD COLUMN destination_box_area_fraction FLOAT;

CREATE INDEX ix_items_current_region_id ON items (current_region_id);
CREATE INDEX ix_item_events_source_region_id ON item_events (source_region_id);
CREATE INDEX ix_item_events_destination_region_id ON item_events (destination_region_id);

COMMIT;
