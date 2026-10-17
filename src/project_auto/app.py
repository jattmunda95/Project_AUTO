"""Compose and run Project AUTO; this file is the application coordinator.

Implemented responsibilities:
- Load configuration and construct camera, detector, tracker, store, and event engine.
- Construct the segmenter, ReID matcher, scene processor, and identity coordinator.
- Read frames, run detection/tracking, resolve identity, and draw the display.

The identity coordinator (project_auto.events.coordinator) requests scene processing
after tracker confirmation, retries PENDING identity requests on later visible frames,
routes NEW/EXISTING results through the event layer, and schedules spaced reference
captures toward the configured target count. This file only wires the collaborators
together; it does not decide identity itself.
"""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from time import monotonic

import cv2
import yaml

from project_auto.capture.camera import Camera, CameraConfig
from project_auto.events.coordinator import IdentityCoordinator
from project_auto.events.event_engine import EventEngine
from project_auto.events.identity_policy import IdentityPolicy, IdentityPolicyConfig
from project_auto.events.question_registry import Answer, Question, QuestionRegistry
from project_auto.events.removal_policy import RemovalConfig
from project_auto.memory.reid import ReidConfig, ReidMatcher
from project_auto.memory.store import DatabaseStore
from project_auto.perception.detector import Detection, YoloDetector
from project_auto.perception.scene_processor import SceneProcessor, SceneProcessorConfig
from project_auto.perception.segmenter import SamSegmenter, SegmenterConfig
from project_auto.perception.tracker import DetectionTracker
from project_auto.region_queries import query_regions
from project_auto.utils.drawing import draw_detections, draw_regions
from project_auto.utils.logging import Category, log_action
from project_auto.utils.question_overlay import QuestionOverlay, ThumbnailCache
from project_auto.utils.reference_images import ReferenceImageStore
from project_auto.utils.reid_diagnostics import ReidDiagnostics


def _build_reference_images(
    project_root: Path, scene_processor_settings: dict
) -> ReferenceImageStore | None:
    """Return the thumbnail store configured in scene_processor.yaml, or None if disabled.

    An empty or missing reference_image_dir turns thumbnails off; the path is relative to
    the project root. The folder is created on the first save, not here.
    """
    directory = scene_processor_settings.get("reference_image_dir")
    if not directory:
        return None
    return ReferenceImageStore(
        root=project_root / str(directory),
        max_edge=int(scene_processor_settings.get("reference_image_max_edge", 256)),
    )


def _build_identity_policy(
    project_root: Path, reid_config: ReidConfig, mode: str | None = None
) -> IdentityPolicy:
    """Return the ask-and-answer policy, configured from configs/identity_policy.yaml.

    The acceptance and margin thresholds are taken from the ReidConfig the matcher already
    uses, never re-read from the policy file, so the two can never disagree. mode, when
    given, overrides the file's mode.
    """
    config = IdentityPolicyConfig.from_yaml(project_root / "configs" / "identity_policy.yaml")
    if mode is not None:
        config = replace(config, mode=mode)
    return IdentityPolicy(
        config=config,
        acceptance_threshold=reid_config.acceptance_threshold,
        margin_threshold=reid_config.margin_threshold,
    )


def _require_current_schema(store: DatabaseStore, database_path: Path) -> None:
    """Stop with a clear instruction if the database lacks columns this version needs.

    create_all never alters an existing table, so a database made before a column was added
    would otherwise fail later with a bare "no such column". Nothing is changed here: the
    update is a manual SQL script, applied after a backup, never run automatically.
    """
    missing = store.missing_schema_columns()
    if not missing:
        return
    columns = ", ".join(f"{table}.{column}" for table, column in missing)
    raise RuntimeError(
        f"The database {database_path} is missing {columns}, which this version of "
        "Project AUTO needs. Nothing was changed. Stop the app, make a backup copy of the "
        "database file, then apply docs/manual_schema_update_ask_answer.sql (its header "
        "explains how). To start from an empty database instead, move this one aside."
    )


def _policy_mode_override(diagnostics: ReidDiagnostics | None) -> str | None:
    """Return the identity-policy mode a run must use, or None to use the configured one.

    A diagnostic run records ground truth for the normal flow, so questions and waits must
    not change what it records: the policy is switched off for it.
    """
    return "off" if diagnostics is not None else None


def _answer_open_question(
    answer_question: Callable[[int, Answer, float], Question | None],
    overlay: QuestionOverlay,
    open_questions: tuple[Question, ...],
    key: int,
    now: float,
) -> Answer | None:
    """Apply a key press to the oldest open question, and return the answer it gave.

    answer_question is the coordinator's method: it closes the question and carries the
    answer out (link, add, ignore or wait). Returns None when nothing is open, the key is
    not an answer (q, w and r are left to the caller), or the answer was stale and ignored.
    """
    if not open_questions:
        return None
    answer = overlay.answer_for_key(key, open_questions[0])
    if answer is None:
        return None
    applied = answer_question(open_questions[0].id, answer, now)
    return answer if applied is not None else None


def run_app(
    reid_config: ReidConfig | None = None,
    database_path: Path | None = None,
    diagnostics: ReidDiagnostics | None = None,
) -> None:
    """Run capture, perception, lifecycle decisions, persistence, and display.

    The normal entry point calls this with no arguments, which uses configs/ as-is
    and records no diagnostics. The optional arguments exist only for the separate
    diagnostic entry point (project_auto.reid_diagnostics_app), which supplies an
    adjusted ReID config, its own database, and a ReidDiagnostics recorder.
    """
    project_root = Path(__file__).resolve().parents[2]
    camera_config_path = project_root / "configs" / "camera.yaml"
    perception_config_path = project_root / "configs" / "perception.yaml"
    table_config_path = project_root / "configs" / "table.yaml"
    segmenter_config_path = project_root / "configs" / "segmenter.yaml"
    reid_config_path = project_root / "configs" / "reid.yaml"
    scene_processor_config_path = project_root / "configs" / "scene_processor.yaml"

    with camera_config_path.open(encoding="utf-8") as config_file:
        camera_settings = yaml.safe_load(config_file)
    with perception_config_path.open(encoding="utf-8") as config_file:
        perception_settings = yaml.safe_load(config_file)
    with table_config_path.open(encoding="utf-8") as config_file:
        table_settings = yaml.safe_load(config_file)
    with scene_processor_config_path.open(encoding="utf-8") as config_file:
        scene_processor_settings = yaml.safe_load(config_file)

    # The diagnostic entry point supplies its own database whose item ids overlap the real
    # ones, so it must never write into the real thumbnail folder.
    uses_default_database = database_path is None
    if database_path is None:
        database_path = project_root / table_settings["database"]["path"]
    store = DatabaseStore(database_path)

    try:
        store.create_schema()
        _require_current_schema(store, database_path)
        camera = Camera(CameraConfig(**camera_settings))
        detector = YoloDetector(perception_config_path)
        tracker = DetectionTracker(
            candidate_confirmation_seconds=float(
                perception_settings["candidate_confirmation_seconds"]
            ),
            max_candidate_missing_frames=int(perception_settings["max_candidate_missing_frames"]),
            removal_timeout_seconds=float(perception_settings["removal_timeout_seconds"]),
            movement_buffer_scale=float(perception_settings["movement_buffer_scale"]),
            movement_stop_tolerance_pixels=float(
                perception_settings["movement_stop_tolerance_pixels"]
            ),
            movement_stopped_confirmation_seconds=float(
                perception_settings["movement_stopped_confirmation_seconds"]
            ),
            candidate_stillness_tolerance_pixels=float(
                perception_settings["candidate_stillness_tolerance_pixels"]
            ),
        )
        event_engine = EventEngine(store)

        if reid_config is None:
            reid_config = ReidConfig.from_yaml(reid_config_path)
        segmenter = SamSegmenter(SegmenterConfig.from_yaml(segmenter_config_path))
        matcher = ReidMatcher(reid_config)
        scene_processor = SceneProcessor(
            SceneProcessorConfig.from_yaml(scene_processor_config_path),
            segmenter,
            matcher,
        )
        reference_images = (
            _build_reference_images(project_root, scene_processor_settings)
            if uses_default_database
            else None
        )
        # A diagnostic run records ground truth for the normal flow, so it must not be
        # changed by questions or waits: the policy is switched off for it.
        identity_policy = _build_identity_policy(
            project_root, reid_config, mode=_policy_mode_override(diagnostics)
        )
        question_registry = QuestionRegistry(identity_policy.config)
        question_overlay = QuestionOverlay(
            thumbnails=(
                ThumbnailCache(resolve=reference_images.resolve)
                if reference_images is not None
                else ThumbnailCache()
            )
        )
        coordinator = IdentityCoordinator(
            scene_processor=scene_processor,
            event_engine=event_engine,
            store=store,
            reid_model_name=reid_config.model_name,
            max_references_per_item=int(scene_processor_settings["max_references_per_item"]),
            bad_mask_cooldown_seconds=float(scene_processor_settings["bad_mask_cooldown_seconds"]),
            queue_full_retry_seconds=float(scene_processor_settings["queue_full_retry_seconds"]),
            min_box_area=int(scene_processor_settings["min_box_area"]),
            max_box_area_fraction=float(scene_processor_settings["max_box_area_fraction"]),
            min_detector_confidence=float(scene_processor_settings["min_detector_confidence"]),
            initial_reference_count=int(scene_processor_settings["initial_reference_count"]),
            initial_capture_spacing_frames=int(
                scene_processor_settings["initial_capture_spacing_frames"]
            ),
            move_candidate_delay_frames=int(
                scene_processor_settings["move_candidate_delay_frames"]
            ),
            candidate_retry_frames=int(scene_processor_settings["candidate_retry_frames"]),
            max_movement_reference_attempts=int(
                scene_processor_settings["max_movement_reference_attempts"]
            ),
            min_reference_sharpness=float(scene_processor_settings["min_reference_sharpness"]),
            min_reference_frame_visibility=float(
                scene_processor_settings["min_reference_frame_visibility"]
            ),
            min_mask_score=float(scene_processor_settings["min_mask_score"]),
            min_mask_occupancy=float(scene_processor_settings["min_mask_occupancy"]),
            reference_novelty_threshold=float(
                scene_processor_settings["reference_novelty_threshold"]
            ),
            min_reference_consistency=float(
                scene_processor_settings.get("min_reference_consistency", 0.0)
            ),
            job_queue_max_size=int(scene_processor_settings["job_queue_max_size"]),
            worker_torch_threads=int(scene_processor_settings["worker_torch_threads"]),
            removal_config=RemovalConfig(**scene_processor_settings.get("removal", {})),
            diagnostics=diagnostics,
            reference_images=reference_images,
            identity_policy=identity_policy,
            question_registry=question_registry,
        )

        with camera:
            log_action(Category.SYSTEM, event="camera_opened", device=camera.device)
            highlighted_regions = []
            highlight_frames_left = 0
            highlight_duration = int(table_settings.get("regions", {}).get("highlight_frames", 150))
            # TODO(perf): remove once frame dropout/latency is measured (TASKS #2b);
            # tied to the same task as freezing confidence/image_size/agnostic_nms.
            perf_window_frames = 60
            perf_window_started_at = monotonic()
            perf_detect_seconds = 0.0
            perf_track_seconds = 0.0
            perf_coordinator_seconds = 0.0
            perf_frames_in_window = 0
            while True:
                frame_started_at = monotonic()
                frame = camera.read()
                detections = detector.detect(frame)
                detect_finished_at = monotonic()
                signals = tracker.update(detections)
                track_finished_at = monotonic()
                detections_by_track_id: dict[int, Detection] = {
                    detection.track_id: detection
                    for detection in detections
                    if detection.track_id is not None
                }

                coordinator.handle_frame(frame, signals, detections_by_track_id)
                coordinator_finished_at = monotonic()

                perf_detect_seconds += detect_finished_at - frame_started_at
                perf_track_seconds += track_finished_at - detect_finished_at
                perf_coordinator_seconds += coordinator_finished_at - track_finished_at
                perf_frames_in_window += 1
                if perf_frames_in_window >= perf_window_frames:
                    elapsed = monotonic() - perf_window_started_at
                    log_action(
                        Category.SYSTEM,
                        event="frame_perf",
                        fps=perf_frames_in_window / elapsed if elapsed > 0 else 0.0,
                        detect_ms=1000 * perf_detect_seconds / perf_frames_in_window,
                        track_ms=1000 * perf_track_seconds / perf_frames_in_window,
                        coordinator_ms=1000 * perf_coordinator_seconds / perf_frames_in_window,
                        frame_ms=1000 * elapsed / perf_frames_in_window,
                    )
                    perf_window_started_at = monotonic()
                    perf_detect_seconds = 0.0
                    perf_track_seconds = 0.0
                    perf_coordinator_seconds = 0.0
                    perf_frames_in_window = 0

                debug_frame = draw_detections(frame, detections)
                if highlight_frames_left > 0:
                    debug_frame = draw_regions(debug_frame, highlighted_regions)
                    highlight_frames_left -= 1
                # The coordinator opens, expires and applies questions; the app only draws what
                # is open and forwards the key press. Idle, and free, while nothing is asked.
                question_now = monotonic()
                open_questions = question_registry.open_questions()
                debug_frame = question_overlay.draw(debug_frame, open_questions, question_now)
                cv2.imshow("Project AUTO - q quit, w where item, r inspect region", debug_frame)

                key = cv2.waitKey(1) & 0xFF
                _answer_open_question(
                    coordinator.answer_question, question_overlay, open_questions, key, monotonic()
                )
                if key == ord("q"):
                    break
                if key in (ord("w"), ord("r")):
                    region_ids = query_regions(store, key)
                    highlighted_regions = [
                        region for region in store.list_regions() if region.id in region_ids
                    ]
                    highlight_frames_left = highlight_duration
    finally:
        cv2.destroyAllWindows()
        if "coordinator" in locals():
            coordinator.shutdown()
        store.engine.dispose()
