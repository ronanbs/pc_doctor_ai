"""
categories.py
Defines the troubleshooting categories, their display labels, their raw
data folders, and their Chroma collection names. Central source of truth
used by both the ingestion pipeline and the Streamlit app.
"""

CATEGORIES = {
    "windows_bsod": {
        "label": "Windows / BSOD & Stop Codes",
        "raw_folder": "data/raw/windows_bsod",
        "collection_name": "windows_bsod",
        "description": (
            "Windows Blue Screen of Death (BSOD) errors, stop codes, "
            "boot problems, and driver-related crashes."
        ),
    },
    "gpu_drivers": {
        "label": "GPU & Drivers",
        "raw_folder": "data/raw/gpu_drivers",
        "collection_name": "gpu_drivers",
        "description": (
            "GPU driver installation, crashes, clean-reinstall procedures "
            "(e.g. DDU), and general graphics card troubleshooting."
        ),
    },
    "motherboard_bios": {
        "label": "Motherboard / BIOS / Power",
        "raw_folder": "data/raw/motherboard_bios",
        "collection_name": "motherboard_bios",
        "description": (
            "Motherboard BIOS/UEFI settings, XMP/DOCP memory profiles, "
            "power management, and hardware stability issues."
        ),
    },
    "linux_kernel": {
        "label": "Linux / Kernel Issues",
        "raw_folder": "data/raw/linux_kernel",
        "collection_name": "linux_kernel",
        "description": (
            "Linux kernel panics, NVIDIA driver issues on Linux, and "
            "kernel-module-level crashes."
        ),
    },
}


def get_category_keys() -> list[str]:
    return list(CATEGORIES.keys())


def get_category_labels() -> list[str]:
    return [v["label"] for v in CATEGORIES.values()]


def label_to_key(label: str) -> str | None:
    for key, val in CATEGORIES.items():
        if val["label"] == label:
            return key
    return None
