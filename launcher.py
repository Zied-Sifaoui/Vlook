"""
FaceAR Desktop Launcher
───────────────────────
Click a feature card to open the corresponding Python script.
All scripts use the laptop/PC webcam (cv2.VideoCapture(0)) directly.
No phone camera or Flask server required.
"""

import tkinter as tk
from tkinter import font as tkfont
import subprocess
import sys
import os

BASE = os.path.dirname(os.path.abspath(__file__))

# Map feature id → script path (relative to project root)
SCRIPTS = {
    "fox_eye":       os.path.join(BASE, "features", "eyes",  "foxeye.py"),
    "eyelid_lift":   os.path.join(BASE, "features", "eyes",  "eyelidlift.py"),
    "lips":          os.path.join(BASE, "features", "lips",  "lips4.py"),
    "nose":          os.path.join(BASE, "features", "nose",  "nose_combined.py"),
    "hair_overlay":  os.path.join(BASE, "features", "hair",  "hair_overlay.py"),
    "hair_color":    os.path.join(BASE, "features", "hair",  "haircolor.py"),
    "jaw":           os.path.join(BASE, "features", "jaw",   "jawlinevshape.py"),
    "scar":          os.path.join(BASE, "features", "scar",  "scardetection2.py"),
    "sign_language": os.path.join(BASE, "hands",             "signlan.py"),
}

FEATURES = [
    ("fox_eye",       "Fox Eyes",      "Eyes",  "Cat-eye lifting effect",           "👁️"),
    ("eyelid_lift",   "Eyelid Lift",   "Eyes",  "Natural eyelid enhancement",        "✨"),
    ("lips",          "Lip Beautify",  "Lips",  "Lip color and shape enhancement",   "💋"),
    ("nose",          "Nose Reshape",  "Nose",  "Virtual nose overlay",              "👃"),
    ("hair_overlay",  "Hair Overlay",  "Hair",  "Virtual hairstyle overlay",         "💇"),
    ("hair_color",    "Hair Color",    "Hair",  "Change your hair color",            "🎨"),
    ("jaw",           "Jaw V-Shape",   "Face",  "Jaw slimming & contouring",         "💎"),
    ("scar",          "Scar Removal",  "Skin",  "Detect and conceal scars",          "🌟"),
    ("sign_language", "Sign Language", "Hands", "Hand sign landmark detection",      "🤟"),
]

# Colours (dark theme matching Android app feel)
BG          = "#121212"
CARD_BG     = "#1E1E2E"
CARD_HOVER  = "#2A2A3E"
CARD_BORDER = "#3A3A5E"
ACCENT      = "#7C6AF7"
TEXT_MAIN   = "#FFFFFF"
TEXT_SUB    = "#AAAACC"
TEXT_CAT    = "#7C6AF7"
RUNNING_BG  = "#1A3A2A"
RUNNING_BDR = "#2ECC71"

running_procs: dict[str, subprocess.Popen] = {}


def launch(feature_id: str, btn_label: tk.Label):
    """Launch (or terminate) the feature script."""
    proc = running_procs.get(feature_id)

    # If already running, kill it
    if proc and proc.poll() is None:
        proc.terminate()
        del running_procs[feature_id]
        btn_label.config(text="▶  Open")
        return

    script = SCRIPTS[feature_id]
    if not os.path.exists(script):
        tk.messagebox.showerror("Script not found", f"Could not find:\n{script}")
        return

    proc = subprocess.Popen([sys.executable, script])
    running_procs[feature_id] = proc
    btn_label.config(text="■  Stop")


def make_card(parent, feature_id, name, category, desc, emoji, row, col):
    card = tk.Frame(parent, bg=CARD_BG, bd=0, padx=18, pady=14,
                    highlightbackground=CARD_BORDER, highlightthickness=1,
                    cursor="hand2")
    card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")

    # Emoji
    em = tk.Label(card, text=emoji, bg=CARD_BG, font=("Segoe UI Emoji", 26))
    em.pack(anchor="w")

    # Category chip
    cat = tk.Label(card, text=f"  {category}  ", bg=ACCENT, fg=TEXT_MAIN,
                   font=("Segoe UI", 8, "bold"), padx=4, pady=1)
    cat.pack(anchor="w", pady=(4, 0))

    # Feature name
    tk.Label(card, text=name, bg=CARD_BG, fg=TEXT_MAIN,
             font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(6, 0))

    # Description
    tk.Label(card, text=desc, bg=CARD_BG, fg=TEXT_SUB,
             font=("Segoe UI", 9), wraplength=170, justify="left").pack(anchor="w", pady=(3, 8))

    # Open button
    btn_label = tk.Label(card, text="▶  Open", bg=ACCENT, fg=TEXT_MAIN,
                         font=("Segoe UI", 9, "bold"), padx=10, pady=5, cursor="hand2")
    btn_label.pack(anchor="w")

    # Bind click on entire card and button
    def on_click(e=None):
        launch(feature_id, btn_label)

    def on_enter(e=None):
        card.config(bg=CARD_HOVER)
        for w in card.winfo_children():
            if isinstance(w, tk.Label) and w.cget("bg") not in (ACCENT,):
                w.config(bg=CARD_HOVER)

    def on_leave(e=None):
        card.config(bg=CARD_BG)
        for w in card.winfo_children():
            if isinstance(w, tk.Label) and w.cget("bg") not in (ACCENT,):
                w.config(bg=CARD_BG)

    for widget in [card, em, cat, btn_label] + list(card.winfo_children()):
        widget.bind("<Button-1>", on_click)
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)


def poll_procs(root: tk.Tk):
    """Periodically clean up finished processes."""
    for fid in list(running_procs.keys()):
        if running_procs[fid].poll() is not None:
            del running_procs[fid]
    root.after(1000, lambda: poll_procs(root))


def main():
    root = tk.Tk()
    root.title("FaceAR — Desktop Launcher")
    root.configure(bg=BG)
    root.resizable(True, True)

    # Header
    header = tk.Frame(root, bg=BG, pady=20)
    header.pack(fill="x", padx=20)

    tk.Label(header, text="FaceAR", bg=BG, fg=ACCENT,
             font=("Segoe UI", 22, "bold")).pack(side="left")
    tk.Label(header, text="  Laptop Camera Mode", bg=BG, fg=TEXT_SUB,
             font=("Segoe UI", 11)).pack(side="left", pady=(6, 0))

    # Info bar
    info = tk.Frame(root, bg="#1A1A2E", pady=8)
    info.pack(fill="x", padx=20)
    tk.Label(info, text="  Using webcam (device 0)  —  click a card to start, click again to stop",
             bg="#1A1A2E", fg=TEXT_SUB, font=("Segoe UI", 9)).pack(side="left", padx=10)

    # Grid
    grid_frame = tk.Frame(root, bg=BG)
    grid_frame.pack(padx=20, pady=10)

    COLS = 3
    for i, (fid, name, cat, desc, emoji) in enumerate(FEATURES):
        row, col = divmod(i, COLS)
        grid_frame.columnconfigure(col, weight=1)
        make_card(grid_frame, fid, name, cat, desc, emoji, row, col)

    poll_procs(root)
    root.mainloop()

    # Kill any still-running scripts on exit
    for proc in running_procs.values():
        if proc.poll() is None:
            proc.terminate()


if __name__ == "__main__":
    import tkinter.messagebox  # ensure available
    main()
