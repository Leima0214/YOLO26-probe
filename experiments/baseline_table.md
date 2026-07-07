# Japan Baseline Results

**Protocol**: Japan7 (7 classes: D00, D10, D20, D40, D43, D44, D50)  
**Config**: `configs/japan7_remote.yaml` | Training: epochs=100, imgsz=640, batch=32, seed=42

## Overall

| Model | Dataset | Epochs | Img | Batch | Params | FLOPs | P | R | mAP50 | mAP50-95 | Run |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| YOLOv8n | Japan7 | 100 | 640 | 32 | 3.007M | 8.1G | 0.647 | 0.606 | 0.642 | **0.353** | yolov8n_japan7_e100_img640_b32_seed422 |
| YOLO11n | Japan7 | 100 | 640 | 32 | 2.584M | 6.3G | 0.639 | **0.616** | 0.642 | 0.349 | yolo11n_japan7_e100_img640_b32_seed42 |
| YOLO26s | Japan7 | 100 | 640 | 32 | 9.468M | 20.5G | **0.678** | 0.591 | 0.630 | 0.347 | yolo26s_japan7_e100_img640_b32_seed42 |
| YOLO26n | Japan7 | 100 | 640 | 32 | **2.376M** | **5.2G** | 0.644 | 0.597 | 0.623 | 0.341 | yolo26n_japan7_e100_img640_b32_seed42 |

## Per-class mAP50-95

| Model | D00 | D10 | D20 | D40 | D43 | D44 | D50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8n | 0.219 | 0.167 | 0.338 | 0.268 | **0.570** | 0.448 | 0.459 |
| YOLO11n | 0.202 | 0.158 | 0.342 | 0.276 | 0.562 | **0.457** | 0.446 |
| YOLO26s | 0.173 | 0.150 | 0.338 | 0.289 | 0.584 | 0.439 | **0.454** |
| YOLO26n | 0.183 | 0.148 | 0.346 | 0.278 | 0.541 | 0.454 | 0.435 |

Full details: [`experiments/japan7_baseline_20260707/`](experiments/japan7_baseline_20260707/)
