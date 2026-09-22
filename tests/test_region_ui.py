"""Drawing, console, and calibration interaction without a physical camera."""

import cv2
import numpy as np

from project_auto.memory.models import Region
from project_auto.memory.store import DatabaseStore
from project_auto.region_queries import query_regions
from project_auto.utils.drawing import draw_regions


def test_draw_regions_copies_frame_and_blends_fill():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    region = Region(id=1, name="tray", polygon=[[10, 30], [90, 30], [90, 90], [10, 90]])
    output = draw_regions(frame, [region], {1: (100, 150, 200)})
    assert not frame.any()
    assert tuple(output[60, 60]) == (20, 30, 40)
    assert tuple(output[90, 60]) == (100, 150, 200)
    assert not output[95, 95].any()


def test_console_item_and_region_queries(tmp_path, monkeypatch, capsys):
    store = DatabaseStore(tmp_path / "test.db")
    try:
        store.create_schema()
        region = store.create_region("tray", [[0, 0], [100, 0], [100, 100], [0, 100]])
        from project_auto.memory.models import ItemStatus
        item, _ = store.add_item_with_event("calculator", ItemStatus.PRESENT, 1, 0.9,
                                             destination_box=(10, 10, 20, 20))
        monkeypatch.setattr("builtins.input", lambda _: "calculator")
        assert query_regions(store, ord("w")) == [region.id]
        assert "tray" in capsys.readouterr().out
        monkeypatch.setattr("builtins.input", lambda _: "tray")
        assert query_regions(store, ord("r")) == [region.id]
        output = capsys.readouterr().out
        assert "currently contains" in output and "ADDED calculator" in output
        assert "Historically associated items" in output
        store.mark_removed(item.id)
        monkeypatch.setattr("builtins.input", lambda _: str(item.id))
        assert query_regions(store, ord("w")) == []
        assert "removed" in capsys.readouterr().out
    finally:
        store.engine.dispose()


def test_calibration_click_undo_invalid_and_save(tmp_path, monkeypatch):
    from project_auto import region_calibration as calibration

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/camera.yaml").write_text("usb_device: 0\n", encoding="utf-8")
    (tmp_path / "configs/table.yaml").write_text("database:\n  path: test.db\n", encoding="utf-8")
    monkeypatch.setattr(calibration, "__file__", str(tmp_path / "src/project_auto/region_calibration.py"))
    state = {"closed": False, "waits": 0, "inputs": 0}

    class FakeCamera:
        def __init__(self, config):
            assert config.usb_device == 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            state["closed"] = True

        def read(self):
            return np.zeros((100, 100, 3), dtype=np.uint8)

    def set_callback(window, callback):
        state["callback"] = callback

    def wait_key(delay):
        state["waits"] += 1
        callback = state["callback"]
        if state["waits"] == 1:
            callback(cv2.EVENT_LBUTTONDOWN, 10, 10, 0, None)
            return ord("c")  # insufficient vertices: no name prompt
        if state["waits"] == 2:
            callback(cv2.EVENT_LBUTTONDOWN, 50, 10, 0, None)
            callback(cv2.EVENT_LBUTTONDOWN, 99, 99, 0, None)
            return ord("u")
        if state["waits"] == 3:
            callback(cv2.EVENT_LBUTTONDOWN, 50, 50, 0, None)
            return 13
        if state["waits"] == 4:
            # Polygon is frozen while its closing preview and stdin prompt are active.
            callback(cv2.EVENT_LBUTTONDOWN, 80, 80, 0, None)
            return -1
        return ord("q")

    def name_prompt(prompt):
        state["inputs"] += 1
        return "tools"

    monkeypatch.setattr(calibration, "Camera", FakeCamera)
    monkeypatch.setattr(cv2, "namedWindow", lambda *args: None)
    monkeypatch.setattr(cv2, "setMouseCallback", set_callback)
    monkeypatch.setattr(cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(cv2, "waitKey", wait_key)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr("builtins.input", name_prompt)
    calibration.run_calibration()
    assert state["closed"]
    assert state["inputs"] == 1
    store = DatabaseStore(tmp_path / "test.db")
    try:
        assert store.get_region_by_name("tools").polygon == [[10, 10], [50, 10], [50, 50]]
    finally:
        store.engine.dispose()
