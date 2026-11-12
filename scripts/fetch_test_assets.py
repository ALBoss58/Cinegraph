"""Download glTF sample models used by the 3D animation tests (Khronos glTF-Sample-Assets, CC-BY 4.0 / CC0)."""
import os
import urllib.request

BASE = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models"
OUT = os.path.join(os.path.dirname(__file__), "..", "examples", "media", "gltf")
os.makedirs(OUT, exist_ok=True)
for m in ("CesiumMan", "RiggedSimple", "Fox"):
    p = os.path.join(OUT, f"{m}.glb")
    if not os.path.exists(p):
        urllib.request.urlretrieve(f"{BASE}/{m}/glTF-Binary/{m}.glb", p)
        print("downloaded", p)
