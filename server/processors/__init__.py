import importlib
import logging

logger = logging.getLogger(__name__)

from .base import BaseProcessor

# Lazy registry: maps feature_id -> (module_name, class_name)
_REGISTRY = {
    "fox_eye":       ("processors.fox_eye",       "FoxEyeProcessor"),
    "eyelid_lift":   ("processors.eyelid_lift",   "EyelidLiftProcessor"),
    "lips":          ("processors.lips",           "LipsProcessor"),
    "nose":          ("processors.nose",           "NoseProcessor"),
    "hair_overlay":  ("processors.hair_overlay",  "HairOverlayProcessor"),
    "hair_color":    ("processors.hair_color",     "HairColorProcessor"),
    "jaw":           ("processors.jaw",            "JawProcessor"),
    "scar":          ("processors.scar",           "ScarProcessor"),
    "sign_language": ("processors.sign_language",  "SignLanguageProcessor"),
}

FEATURES = [
    {"id": "fox_eye",       "name": "Fox Eyes",      "category": "Eyes",  "description": "Cat-eye lifting effect"},
    {"id": "eyelid_lift",   "name": "Eyelid Lift",   "category": "Eyes",  "description": "Natural eyelid enhancement"},
    {"id": "lips",          "name": "Lip Beautify",  "category": "Lips",  "description": "Lip color and shape enhancement"},
    {"id": "nose",          "name": "Nose Reshape",  "category": "Nose",  "description": "Virtual nose overlay and reshaping"},
    {"id": "hair_overlay",  "name": "Hair Overlay",  "category": "Hair",  "description": "Virtual hairstyle overlay"},
    {"id": "hair_color",    "name": "Hair Color",    "category": "Hair",  "description": "Change hair color"},
    {"id": "jaw",           "name": "Jaw V-Shape",   "category": "Face",  "description": "Jaw slimming and contouring"},
    {"id": "scar",          "name": "Scar Removal",  "category": "Skin",  "description": "Detect and conceal scars"},
    {"id": "sign_language", "name": "Sign Language", "category": "Hands", "description": "Hand sign landmark detection"},
]


def get_processor(feature_id: str) -> BaseProcessor:
    entry = _REGISTRY.get(feature_id)
    if entry is None:
        raise ValueError(f"Unknown feature: {feature_id}")
    module_name, class_name = entry
    try:
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
        return cls()
    except Exception as e:
        logger.error(f"Failed to load processor '{feature_id}': {e}")
        raise RuntimeError(f"Processor '{feature_id}' failed to load: {e}") from e
