# OpenVINO table detector on this chip

{'CPU': 'Intel(R) Core(TM) 5 210H', 'GPU.0': 'Intel(R) Graphics (iGPU)', 'GPU.1': 'NVIDIA GeForce RTX 4050 Laptop GPU (dGPU)'}

OpenVINO 2026.3.1-22476-759c5a6ab8c-releases/2026/3; detector `best.pt` (scene-trained YOLOv8n, 7 classes)

| precision | mAP50 | mAP50-95 | IR MB | device | hint | latency mean / p50 / p95 ms | FPS |
|---|---|---|---|---|---|---|---|
| fp32 | 0.980 | 0.917 | 12.36 | CPU | LATENCY | 45.537 / 44.44 / 53.935 | 22.0 |
| fp32 | 0.980 | 0.917 | 12.36 | CPU | THROUGHPUT (3 req) | | 39.4 |
| fp32 | 0.980 | 0.917 | 12.36 | GPU.0 | LATENCY | 13.449 / 13.109 / 15.816 | 74.4 |
| fp32 | 0.980 | 0.917 | 12.36 | GPU.0 | THROUGHPUT (32 req) | | 95.7 |
| fp16 | 0.979 | 0.916 | 6.38 | CPU | LATENCY | 45.365 / 44.452 / 50.94 | 22.0 |
| fp16 | 0.979 | 0.916 | 6.38 | CPU | THROUGHPUT (3 req) | | 39.0 |
| fp16 | 0.979 | 0.916 | 6.38 | GPU.0 | LATENCY | 14.5 / 14.38 / 18.258 | 69.0 |
| fp16 | 0.979 | 0.916 | 6.38 | GPU.0 | THROUGHPUT (32 req) | | 95.2 |
| int8 | 0.981 | 0.915 | 3.62 | CPU | LATENCY | 17.98 / 17.891 / 20.619 | 55.6 |
| int8 | 0.981 | 0.915 | 3.62 | CPU | THROUGHPUT (3 req) | | 97.9 |
| int8 | 0.981 | 0.915 | 3.62 | GPU.0 | LATENCY | 11.433 / 11.021 / 14.111 | 87.5 |
| int8 | 0.981 | 0.915 | 3.62 | GPU.0 | THROUGHPUT (64 req) | | 105.7 |
