# Third-party components

Cinegraph's own code is MIT. It uses (not bundles) these, each under its own licence — check upstream before redistributing:

| Component | Use | Licence (upstream) |
|---|---|---|
| FFmpeg (external binary) | decode/encode | LGPL/GPL depending on build (libx264/x265 are GPL) |
| OpenCV, NumPy, Pillow | imaging | Apache-2.0 / BSD / HPND |
| moderngl, trimesh, pygltflib | 3D | MIT |
| OpenColorIO (+ ACES studio config) | colour | BSD-3-Clause (config: ACES/ASWF) |
| OpenTimelineIO + cmx_3600 / fcp_xml adapters | interchange | Apache-2.0 |
| onnxruntime | AI inference | MIT |
| U²-Net / U²-Net-human / ISNet weights (via rembg releases) | cut-out | Apache-2.0 |
| sherpa-onnx + icefall Zipformer English model | speech-to-text | Apache-2.0 |
| Silero VAD | voice activity | MIT |
| faster-whisper (optional) | speech-to-text | MIT (Whisper weights: MIT) |

Model weights are downloaded on first use into `~/.cinegraph/models`; they are not shipped in this repository.
Test assets from the Khronos glTF Sample Assets (CesiumMan, Fox, RiggedSimple) are CC-BY 4.0 / CC0 and are fetched by
`scripts/fetch_test_assets.py`, not redistributed.
