# Technology stack and dependencies

## Stack by layer

| Layer | Technology | Project use |
| --- | --- | --- |
| Language and packaging | Python, setuptools, editable src-layout install | Local CLI application, typed data contracts |
| Camera and visualization | OpenCV | Camera capture, image operations, desktop debug window |
| Detection inference | Ultralytics YOLO11s + OpenVINO | Source weights exported to local OpenVINO format |
| Temporary tracking | Ultralytics BoT-SORT | Persistent IDs across frame detections |
| Segmentation | Transformers SAM2 + PyTorch | Box-prompted masks on CPU |
| Identity embeddings | Transformers DINOv2 + PyTorch | Normalized image embeddings on CPU |
| Numeric and image processing | NumPy, Pillow | Array operations, cosine similarity, RGB crops |
| Region geometry | Pure Python ray casting and shoelace formula | Fixed-frame polygon validation/resolution; no Shapely dependency |
| Storage | SQLAlchemy 2.x + SQLite | Local relational memory with JSON vectors, boxes and polygons |
| Configuration | YAML / PyYAML | Camera, models, timing, thresholds, database path |
| Concurrency | Python threading and queue | One background identification worker |
| Tests | pytest | Deterministic behavior and persistence checks |
| Documentation | Markdown, generated HTML/CSS/JavaScript | Offline portal; standard-library Python builder |

## Declared runtime dependencies

These are repository declarations, not an installed environment inventory or a lockfile.

| Dependency | Declared range | Why it is needed |
| --- | --- | --- |
| Python | >=3.10,<3.14 | Supported interpreter range in package metadata |
| opencv-python | >=4.10 | Capture and OpenCV display |
| openvino | >=2024.0 | Optimized detector runtime |
| PyYAML | >=6.0 | Configuration loading |
| ultralytics | >=8.4 | YOLO loading/export and BoT-SORT integration |
| SQLAlchemy | >=2.0,<3.0 | ORM, sessions, schema and database operations |
| torch | Unpinned, reid extra | SAM2 and DINOv2 inference |
| transformers | Unpinned, reid extra | Sam2Model/Sam2Processor and AutoModel/AutoImageProcessor |
| Pillow | Unpinned, reid extra | PIL images for preprocessing |
| setuptools | >=75, build dependency | setuptools.build_meta backend |

The full live application imports and constructs SAM2 and ReID unconditionally. Install `.[reid]` for the current live entry point even though packaging labels these dependencies optional. A base-only installation is not a complete live runtime.

NumPy is imported directly but is not explicitly declared in `pyproject.toml`; it currently relies on transitive installation. pytest is used by the test suite but has no declared test extra. Ruff settings exist, but Ruff is not declared as an installed development dependency. There is no dependency lockfile in the inspected repository; reproducibility requires recording a tested environment separately.

## Model and hardware dependencies

| Asset | Configured value | Acquisition / assumption |
| --- | --- | --- |
| Detector source | models/yolo11s.pt | Loaded through Ultralytics; may be downloaded if absent |
| Detector export | models/yolo11s_openvino_model | Reused if directory exists; otherwise exported dynamically |
| Segmentation model | facebook/sam2-hiera-base-plus | Transformers from_pretrained; needs compatible library APIs and cached/downloadable weights |
| Embedding model | facebook/dinov2-base | Transformers from_pretrained; same cache/download consideration |
| Camera | USB index 0, fallback webcam index 1 | Indices are machine-specific, not portable device identifiers |
| Target machine | Lenovo Yoga Pro 7i, Intel integrated graphics | Project notes' target; no minimum RAM/CPU or portable hardware qualification published |

All configured model devices are CPU. The detector uses OpenVINO while SAM2 and DINOv2 use PyTorch; the presence of Intel graphics does not imply that those models are GPU-accelerated.

## Environment assumptions

The supported workflow is an editable checkout with its adjacent configs and models. Packaged standalone distribution of those assets is not configured. Initial installation/model preparation can use network access; offline operation needs cached assets verified beforehand. Application data access depends on filesystem permissions, and the OpenCV window needs an interactive desktop session.

Library and model license review, exact transitive dependency inventory, and vulnerability auditing have not been performed as part of this documentation. No dependency versions were changed.

Sources: [pyproject.toml](../pyproject.toml), [detector](../src/project_auto/perception/detector.py), [segmenter](../src/project_auto/perception/segmenter.py), [ReID](../src/project_auto/memory/reid.py).
