# Model files

Project AUTO uses YOLO11s for object detection and OpenVINO for optimized local inference.

Generated runtime files:

- `yolo11s.pt`: source Ultralytics model downloaded on the first run;
- `yolo11s_openvino_model/`: exported OpenVINO representation used by the live detector.

The paths are configured in `configs/perception.yaml`. These are generated/downloaded model
artifacts rather than project source code and should not be edited manually.
