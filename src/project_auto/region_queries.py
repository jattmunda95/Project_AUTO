"""Temporary console queries; all persistence access stays in DatabaseStore."""

from project_auto.memory.store import DatabaseStore


def query_regions(store: DatabaseStore, key: int) -> list[int]:
    """Print item whereabouts or region contents/activity; return IDs to highlight."""
    try:
        if key == ord("w"):
            query = input("Item ID or exact name: ").strip()
            item = store.get_item(int(query)) if query.isdecimal() else None
            items = ([item] if item else []) if query.isdecimal() else store.find_items(query)
            if not items:
                print("No matching item.")
            highlights = []
            for item in items:
                region = store.get_item_current_region(item.id)
                location = region.name if region else (
                    "outside scene (removed)" if not item.is_present else "unknown/unassigned region"
                )
                print(f"#{item.id} {item.display_name or item.class_name}: {location}")
                if region:
                    highlights.append(region.id)
            return highlights
        if key == ord("r"):
            region = store.get_region_by_name(input("Region name: ").strip())
            if region is None:
                print("No matching region.")
                return []
            contents = store.get_items_in_region(region.id)
            print(f"{region.name} currently contains:")
            for item in contents:
                print(f"  #{item.id} {item.display_name or item.class_name}")
            if not contents:
                print("  (empty)")
            events = store.get_region_recent_events(region.id)
            print(f"Last activity: {events[0].occurred_at if events else 'none'}")
            for event in events:
                name = event.item.display_name or event.item.class_name
                print(f"  {event.occurred_at} {event.event_type.value.upper()} {name}: "
                      f"{event.source_region or 'unassigned/outside scene'} -> "
                      f"{event.destination_region or 'unassigned/outside scene'}")
            associated = store.get_items_associated_with_region(region.id)
            print("Historically associated items: " + (", ".join(
                f"#{item.id} {item.display_name or item.class_name}" for item in associated
            ) or "none"))
            return [region.id]
    except EOFError:
        print("Console query cancelled.")
    return []
