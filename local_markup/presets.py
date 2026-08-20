from __future__ import annotations

from typing import Dict


PRESETS: Dict[str, Dict[str, str]] = {
    "remove_object": {
        "label": "Remove object",
        "instruction": "remove the selected object and naturally fill the area",
        "mode": "inpaint",
    },
    "change_clothing_color": {
        "label": "Change clothing color",
        "instruction": "change the selected clothing color while keeping fabric texture and shadows",
        "mode": "inpaint",
    },
    "clean_background": {
        "label": "Clean background",
        "instruction": "remove background clutter and keep the scene realistic",
        "mode": "inpaint",
    },
    "improve_face_lighting": {
        "label": "Improve face lighting",
        "instruction": "improve face lighting while keeping identity natural",
        "mode": "enhance",
    },
    "product_cleanup": {
        "label": "Product cleanup",
        "instruction": "clean product photo with sharp edges, realistic shadows, and no distractions",
        "mode": "enhance",
    },
}


def get_preset(name: str) -> Dict[str, str]:
    if name not in PRESETS:
        raise KeyError(f"unknown markup preset: {name}")
    return PRESETS[name]
