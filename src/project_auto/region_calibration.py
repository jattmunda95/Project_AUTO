"""Interactive polygon calibration using the application's camera and store."""

from pathlib import Path

import cv2
import numpy as np
import yaml
from sqlalchemy.exc import IntegrityError

from project_auto.capture.camera import Camera, CameraConfig
from project_auto.memory.regions import validate_polygon
from project_auto.memory.models import Region
from project_auto.memory.store import DatabaseStore
from project_auto.utils.drawing import draw_regions


def run_calibration() -> None:
    """Display live regions; click vertices, u undo, Enter/c save, q quit."""
    root = Path(__file__).resolve().parents[2]
    with (root / "configs/camera.yaml").open(encoding="utf-8") as config:
        camera_settings = yaml.safe_load(config)
    with (root / "configs/table.yaml").open(encoding="utf-8") as config:
        table_settings = yaml.safe_load(config)
    store = DatabaseStore(root / table_settings["database"]["path"])
    vertices: list[list[int]] = []
    window = "Project AUTO regions - click vertices, u undo, Enter/c save, q quit"
    accepting_vertices = True

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and accepting_vertices:
            vertices.append([x, y])

    try:
        store.create_schema()
        regions = store.list_regions()
        with Camera(CameraConfig(**camera_settings)) as camera:
            cv2.namedWindow(window)
            cv2.setMouseCallback(window, on_mouse)
            while True:
                output = draw_regions(camera.read(), regions)
                if vertices:
                    cv2.polylines(output, [np.array(vertices, dtype=np.int32)], False,
                                  (0, 220, 255), 2)
                    for point in vertices:
                        cv2.circle(output, tuple(point), 4, (0, 220, 255), -1)
                cv2.imshow(window, output)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("u") and vertices:
                    vertices.pop()
                if key in (10, 13, ord("c")):
                    try:
                        points = validate_polygon(vertices)
                    except ValueError as error:
                        print(f"Cannot save polygon: {error}")
                        continue
                    accepting_vertices = False
                    try:
                        preview = draw_regions(output, [Region(id=-1, name="New region", polygon=points)])
                        cv2.imshow(window, preview)
                        cv2.waitKey(1)
                        name = input("Region name (blank cancels): ").strip()
                        if name:
                            region = store.create_region(name, points)
                            regions.append(region)
                            vertices.clear()
                            print(f"Saved region #{region.id}: {region.name}")
                    except (ValueError, IntegrityError) as error:
                        print(f"Cannot save region (names must be unique): {error}")
                    except EOFError:
                        break
                    finally:
                        accepting_vertices = True
    finally:
        cv2.destroyAllWindows()
        store.engine.dispose()


if __name__ == "__main__":
    run_calibration()
