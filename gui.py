"""
╔══════════════════════════════════════════════════════════════╗
║         POTHOLE DETECTION SYSTEM - Desktop GUI               ║
║         Model: YOLO11m | Classes: 3 Pothole Types            ║
║         University of Agriculture Peshawar - FYP 2026        ║
╚══════════════════════════════════════════════════════════════╝

Requirements:
    pip install ultralytics opencv-python pillow

Run:
    python pothole_detection_gui.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import time
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
APP_TITLE   = "Pothole Detection System"
APP_VERSION = "v1.0 | YOLO11m | UAP FYP 2026"
WIN_WIDTH   = 1100
WIN_HEIGHT  = 720

# Class colors in BGR (for OpenCV) and hex (for Tkinter)
CLASS_CONFIG = {
    "medium-pothole": {"bgr": (0, 140, 255),  "hex": "#FF8C00"},   # orange
    "risk-pothole":   {"bgr": (0,   0, 220),  "hex": "#E02020"},   # red
    "safe-pothole":   {"bgr": (0, 200,  50),  "hex": "#20C040"},   # green
}

CONF_DEFAULT = 0.25
IOU_DEFAULT  = 0.45

# ─────────────────────────────────────────────────────────────
# THEME COLORS
# ─────────────────────────────────────────────────────────────
BG_DARK     = "#0f1117"
BG_PANEL    = "#1a1d27"
BG_CARD     = "#23263a"
ACCENT      = "#4f8ef7"
ACCENT2     = "#7c3aed"
TEXT_MAIN   = "#e8eaf0"
TEXT_DIM    = "#7a7f94"
TEXT_WHITE  = "#ffffff"
BORDER      = "#2e3150"
SUCCESS     = "#20C040"
WARNING     = "#FF8C00"
DANGER      = "#E02020"


# ─────────────────────────────────────────────────────────────
# MAIN APPLICATION CLASS
# ─────────────────────────────────────────────────────────────
class PotholeDetectionApp:

    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry(f"{WIN_WIDTH}x{WIN_HEIGHT}")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(True, True)
        self.root.minsize(900, 600)

        # State variables
        self.model           = None
        self.model_path      = tk.StringVar(value="")
        self.model_loaded    = False
        self.input_path      = tk.StringVar(value="")
        self.output_dir      = tk.StringVar(value=os.path.join(os.path.expanduser("~"), "pothole_results"))
        self.conf_thresh     = tk.DoubleVar(value=CONF_DEFAULT)
        self.iou_thresh      = tk.DoubleVar(value=IOU_DEFAULT)
        self.mode            = tk.StringVar(value="image")   # "image" or "video"
        self.is_processing   = False
        self.video_cap       = None
        self.video_playing   = False

        # Stats counters
        self.stat_total      = tk.StringVar(value="0")
        self.stat_medium     = tk.StringVar(value="0")
        self.stat_risk       = tk.StringVar(value="0")
        self.stat_safe       = tk.StringVar(value="0")
        self.stat_fps        = tk.StringVar(value="—")
        self.stat_status     = tk.StringVar(value="No model loaded")

        self._build_ui()

    # ─────────────────────────────────────────────────────────
    # UI BUILDER
    # ─────────────────────────────────────────────────────────
    def _build_ui(self):
        """Build the full UI layout."""
        self._style_ttk()
        self._build_header()

        # Main content: left panel + right canvas
        main = tk.Frame(self.root, bg=BG_DARK)
        main.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self._build_left_panel(main)
        self._build_right_canvas(main)
        self._build_status_bar()

    def _style_ttk(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TScale",
                         background=BG_PANEL,
                         troughcolor=BG_CARD,
                         sliderthickness=14)
        style.configure("Horizontal.TProgressbar",
                         troughcolor=BG_CARD,
                         background=ACCENT,
                         thickness=6)

    def _build_header(self):
        header = tk.Frame(self.root, bg=BG_PANEL, height=62)
        header.pack(fill="x")
        header.pack_propagate(False)

        # Left: title
        left = tk.Frame(header, bg=BG_PANEL)
        left.pack(side="left", padx=18, pady=10)

        tk.Label(left, text="🕳  POTHOLE DETECTION SYSTEM",
                 font=("Courier New", 14, "bold"),
                 fg=ACCENT, bg=BG_PANEL).pack(anchor="w")
        tk.Label(left, text=APP_VERSION,
                 font=("Courier New", 8),
                 fg=TEXT_DIM, bg=BG_PANEL).pack(anchor="w")

        # Right: model status pill
        right = tk.Frame(header, bg=BG_PANEL)
        right.pack(side="right", padx=18, pady=12)

        self.model_pill = tk.Label(right, text="● MODEL NOT LOADED",
                                   font=("Courier New", 9, "bold"),
                                   fg=DANGER, bg=BG_PANEL)
        self.model_pill.pack()

        # Separator line
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x")

    def _build_left_panel(self, parent):
        """Left control panel."""
        panel = tk.Frame(parent, bg=BG_PANEL, width=310)
        panel.pack(side="left", fill="y", padx=(0, 8), pady=8)
        panel.pack_propagate(False)

        # ── Section: Load Model ──────────────────────────────
        self._section_label(panel, "① LOAD MODEL")

        model_row = tk.Frame(panel, bg=BG_PANEL)
        model_row.pack(fill="x", padx=12, pady=(0, 4))

        self.model_entry = tk.Entry(model_row, textvariable=self.model_path,
                                    bg=BG_CARD, fg=TEXT_DIM,
                                    insertbackground=TEXT_MAIN,
                                    relief="flat", font=("Courier New", 8),
                                    bd=0)
        self.model_entry.pack(side="left", fill="x", expand=True,
                               ipady=6, ipadx=6)
        self.model_entry.insert(0, "Select best.pt ...")

        self._btn(model_row, "Browse", self._browse_model,
                  side="right", width=7)

        self._btn(panel, "  Load Model  ", self._load_model,
                  bg=ACCENT2, fg=TEXT_WHITE, fullwidth=True)

        self._divider(panel)

        # ── Section: Mode Selection ──────────────────────────
        self._section_label(panel, "② SELECT MODE")

        mode_row = tk.Frame(panel, bg=BG_PANEL)
        mode_row.pack(fill="x", padx=12, pady=(0, 8))

        for label, val in [("🖼  Image", "image"), ("🎬  Video", "video")]:
            rb = tk.Radiobutton(mode_row, text=label, variable=self.mode,
                                value=val, command=self._on_mode_change,
                                bg=BG_PANEL, fg=TEXT_MAIN,
                                selectcolor=BG_CARD,
                                activebackground=BG_PANEL,
                                font=("Courier New", 10, "bold"),
                                indicatoron=False, relief="flat",
                                padx=10, pady=6, bd=0,
                                cursor="hand2")
            rb.pack(side="left", expand=True, fill="x", padx=3)

        self._divider(panel)

        # ── Section: Input File ──────────────────────────────
        self._section_label(panel, "③ INPUT FILE")

        input_row = tk.Frame(panel, bg=BG_PANEL)
        input_row.pack(fill="x", padx=12, pady=(0, 4))

        self.input_entry = tk.Entry(input_row, textvariable=self.input_path,
                                     bg=BG_CARD, fg=TEXT_DIM,
                                     insertbackground=TEXT_MAIN,
                                     relief="flat", font=("Courier New", 8),
                                     bd=0)
        self.input_entry.pack(side="left", fill="x", expand=True,
                               ipady=6, ipadx=6)
        self.input_entry.insert(0, "Select image or video ...")

        self._btn(input_row, "Browse", self._browse_input,
                  side="right", width=7)

        self._divider(panel)

        # ── Section: Thresholds ──────────────────────────────
        self._section_label(panel, "④ THRESHOLDS")

        self._slider_row(panel, "Confidence", self.conf_thresh, 0.05, 0.95,
                         "conf_val_lbl")
        self._slider_row(panel, "IoU (NMS)",  self.iou_thresh,  0.05, 0.95,
                         "iou_val_lbl")

        self._divider(panel)

        # ── Section: Output Dir ──────────────────────────────
        self._section_label(panel, "⑤ OUTPUT FOLDER")

        out_row = tk.Frame(panel, bg=BG_PANEL)
        out_row.pack(fill="x", padx=12, pady=(0, 8))

        tk.Entry(out_row, textvariable=self.output_dir,
                 bg=BG_CARD, fg=TEXT_DIM,
                 insertbackground=TEXT_MAIN,
                 relief="flat", font=("Courier New", 8),
                 bd=0).pack(side="left", fill="x", expand=True,
                             ipady=6, ipadx=6)
        self._btn(out_row, "Browse", self._browse_output,
                  side="right", width=7)

        self._divider(panel)

        # ── RUN BUTTON ───────────────────────────────────────
        self.run_btn = self._btn(panel, "  ▶  RUN DETECTION  ",
                                  self._run_detection,
                                  bg=ACCENT, fg=TEXT_WHITE,
                                  fullwidth=True, font_size=11)

        # ── STATS PANEL ──────────────────────────────────────
        self._divider(panel)
        self._section_label(panel, "DETECTION STATS")
        self._build_stats(panel)

    def _build_stats(self, parent):
        stats_frame = tk.Frame(parent, bg=BG_CARD, pady=8)
        stats_frame.pack(fill="x", padx=12, pady=(0, 8))

        rows = [
            ("Total Detected",   self.stat_total,  TEXT_MAIN),
            ("Medium Pothole",   self.stat_medium,  WARNING),
            ("Risk Pothole",     self.stat_risk,    DANGER),
            ("Safe Pothole",     self.stat_safe,    SUCCESS),
            ("FPS",              self.stat_fps,     ACCENT),
        ]

        for label, var, color in rows:
            row = tk.Frame(stats_frame, bg=BG_CARD)
            row.pack(fill="x", padx=10, pady=2)
            tk.Label(row, text=label, font=("Courier New", 8),
                     fg=TEXT_DIM, bg=BG_CARD).pack(side="left")
            tk.Label(row, textvariable=var, font=("Courier New", 9, "bold"),
                     fg=color, bg=BG_CARD).pack(side="right")

        # Progress bar (for video)
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(parent,
                                             variable=self.progress_var,
                                             maximum=100,
                                             style="Horizontal.TProgressbar")
        self.progress_bar.pack(fill="x", padx=12, pady=(4, 2))

        self.progress_label = tk.Label(parent, text="",
                                        font=("Courier New", 7),
                                        fg=TEXT_DIM, bg=BG_PANEL)
        self.progress_label.pack()

    def _build_right_canvas(self, parent):
        """Right side: image/video preview canvas."""
        right = tk.Frame(parent, bg=BG_DARK)
        right.pack(side="left", fill="both", expand=True, pady=8)

        # Canvas header
        canvas_header = tk.Frame(right, bg=BG_PANEL, height=36)
        canvas_header.pack(fill="x")
        canvas_header.pack_propagate(False)

        tk.Label(canvas_header, text="DETECTION OUTPUT PREVIEW",
                 font=("Courier New", 9, "bold"),
                 fg=TEXT_DIM, bg=BG_PANEL).pack(side="left", padx=12,
                                                  pady=8)

        self.save_btn = self._btn(canvas_header, "💾 Save Result",
                                   self._save_result,
                                   side="right", padx=8,
                                   bg=BG_CARD, fg=TEXT_MAIN)

        # Canvas
        self.canvas = tk.Canvas(right, bg="#0a0c14",
                                 highlightthickness=0,
                                 cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)

        # Placeholder text on canvas
        self._canvas_placeholder()

        # Current result frame (for saving)
        self.current_frame    = None
        self.current_img_path = None

    def _build_status_bar(self):
        bar = tk.Frame(self.root, bg=BG_PANEL, height=26)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        tk.Frame(bar, bg=BORDER, width=1).pack(side="left", fill="y")
        tk.Label(bar, textvariable=self.stat_status,
                 font=("Courier New", 8),
                 fg=TEXT_DIM, bg=BG_PANEL).pack(side="left", padx=10,
                                                  pady=4)

    # ─────────────────────────────────────────────────────────
    # UI HELPERS
    # ─────────────────────────────────────────────────────────
    def _section_label(self, parent, text):
        tk.Label(parent, text=text,
                 font=("Courier New", 8, "bold"),
                 fg=ACCENT, bg=BG_PANEL).pack(anchor="w", padx=12,
                                               pady=(10, 4))

    def _divider(self, parent):
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x",
                                                    padx=12, pady=4)

    def _btn(self, parent, text, command, side=None, width=None,
             bg=BG_CARD, fg=TEXT_MAIN, fullwidth=False,
             padx=0, font_size=9):
        btn = tk.Button(parent, text=text, command=command,
                        bg=bg, fg=fg,
                        activebackground=ACCENT,
                        activeforeground=TEXT_WHITE,
                        relief="flat", bd=0,
                        font=("Courier New", font_size, "bold"),
                        cursor="hand2",
                        pady=6,
                        width=width if width else 0)
        if fullwidth:
            btn.pack(fill="x", padx=12, pady=4)
        elif side:
            btn.pack(side=side, padx=(4, 0) if side == "right" else padx)
        else:
            btn.pack()

        # Hover effect
        btn.bind("<Enter>", lambda e, b=btn, c=bg: b.config(bg=ACCENT))
        btn.bind("<Leave>", lambda e, b=btn, c=bg: b.config(bg=c))
        return btn

    def _slider_row(self, parent, label, var, from_, to, lbl_attr):
        row = tk.Frame(parent, bg=BG_PANEL)
        row.pack(fill="x", padx=12, pady=2)

        tk.Label(row, text=label, font=("Courier New", 8),
                 fg=TEXT_DIM, bg=BG_PANEL, width=12,
                 anchor="w").pack(side="left")

        val_lbl = tk.Label(row, text=f"{var.get():.2f}",
                           font=("Courier New", 8, "bold"),
                           fg=TEXT_MAIN, bg=BG_PANEL, width=5)
        val_lbl.pack(side="right")
        setattr(self, lbl_attr, val_lbl)

        slider = ttk.Scale(row, from_=from_, to=to,
                           variable=var, orient="horizontal",
                           command=lambda v, l=val_lbl,
                                          sv=var: l.config(
                               text=f"{float(v):.2f}"))
        slider.pack(side="left", fill="x", expand=True, padx=6)

    def _canvas_placeholder(self):
        self.canvas.update_idletasks()
        cx = self.canvas.winfo_width()  // 2 or 380
        cy = self.canvas.winfo_height() // 2 or 280
        self.canvas.delete("placeholder")
        self.canvas.create_text(cx, cy - 20,
                                 text="🕳",
                                 font=("Arial", 48),
                                 fill="#1e2235",
                                 tags="placeholder")
        self.canvas.create_text(cx, cy + 40,
                                 text="Load model → Select file → Run Detection",
                                 font=("Courier New", 10),
                                 fill="#2a2f4a",
                                 tags="placeholder")

    # ─────────────────────────────────────────────────────────
    # BROWSE HANDLERS
    # ─────────────────────────────────────────────────────────
    def _browse_model(self):
        path = filedialog.askopenfilename(
            title="Select YOLO model weights",
            filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")])
        if path:
            self.model_path.set(path)

    def _browse_input(self):
        if self.mode.get() == "image":
            path = filedialog.askopenfilename(
                title="Select input image",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                            ("All files", "*.*")])
        else:
            path = filedialog.askopenfilename(
                title="Select input video",
                filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"),
                            ("All files", "*.*")])
        if path:
            self.input_path.set(path)

    def _browse_output(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_dir.set(path)

    def _on_mode_change(self):
        self.input_path.set("")
        self._canvas_placeholder()
        self.stat_status.set(f"Mode: {self.mode.get().upper()}")

    # ─────────────────────────────────────────────────────────
    # MODEL LOADING
    # ─────────────────────────────────────────────────────────
    def _load_model(self):
        path = self.model_path.get().strip()
        if not path or not os.path.isfile(path):
            messagebox.showerror("Error", "Please select a valid .pt model file.")
            return

        self.stat_status.set("Loading model ...")
        self.model_pill.config(text="● LOADING ...", fg=WARNING)
        self.root.update()

        try:
            self.model        = YOLO(path)
            self.model_loaded = True
            self.model_pill.config(
                text=f"● MODEL LOADED  {os.path.basename(path)}",
                fg=SUCCESS)
            self.stat_status.set(f"Model loaded: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Model Load Error", str(e))
            self.model_pill.config(text="● LOAD FAILED", fg=DANGER)
            self.stat_status.set("Model load failed.")

    # ─────────────────────────────────────────────────────────
    # RUN DETECTION
    # ─────────────────────────────────────────────────────────
    def _run_detection(self):
        if not self.model_loaded:
            messagebox.showwarning("No Model", "Please load a model first.")
            return

        src = self.input_path.get().strip()
        if not src or not os.path.isfile(src):
            messagebox.showerror("Error", "Please select a valid input file.")
            return

        if self.is_processing:
            messagebox.showinfo("Busy", "Detection is already running.")
            return

        os.makedirs(self.output_dir.get(), exist_ok=True)

        if self.mode.get() == "image":
            threading.Thread(target=self._detect_image,
                             args=(src,), daemon=True).start()
        else:
            threading.Thread(target=self._detect_video,
                             args=(src,), daemon=True).start()

    # ─────────────────────────────────────────────────────────
    # IMAGE DETECTION
    # ─────────────────────────────────────────────────────────
    def _detect_image(self, path):
        self.is_processing = True
        self._set_status("Running detection on image ...")

        try:
            frame = cv2.imread(path)
            if frame is None:
                raise ValueError(f"Cannot read image: {path}")

            t0      = time.time()
            results = self.model.predict(
                source=path,
                conf=self.conf_thresh.get(),
                iou=self.iou_thresh.get(),
                imgsz=640,
                verbose=False
            )
            fps = 1.0 / (time.time() - t0 + 1e-6)

            result    = results[0]
            annotated = self._draw_boxes(frame.copy(), result)

            # Save output
            out_name = f"detected_{os.path.basename(path)}"
            out_path = os.path.join(self.output_dir.get(), out_name)
            cv2.imwrite(out_path, annotated)
            self.current_img_path = out_path
            self.current_frame    = annotated.copy()

            # Update stats
            counts = self._count_classes(result)
            self.root.after(0, self._update_stats, counts, fps)

            # Show on canvas
            self.root.after(0, self._show_image_on_canvas, annotated)
            self._set_status(f"Done! Saved → {out_path}")

        except Exception as e:
            self.root.after(0, messagebox.showerror, "Detection Error", str(e))
            self._set_status("Detection failed.")

        finally:
            self.is_processing = False

    # ─────────────────────────────────────────────────────────
    # VIDEO DETECTION
    # ─────────────────────────────────────────────────────────
    def _detect_video(self, path):
        self.is_processing = True
        self._set_status("Processing video ...")

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self.root.after(0, messagebox.showerror,
                            "Error", f"Cannot open video: {path}")
            self.is_processing = False
            return

        fps_in       = cap.get(cv2.CAP_PROP_FPS) or 25
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_name   = f"detected_{os.path.basename(path)}"
        out_path   = os.path.join(self.output_dir.get(), out_name)
        fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
        writer     = cv2.VideoWriter(out_path, fourcc, fps_in, (width, height))

        frame_num     = 0
        total_counts  = {"medium-pothole": 0, "risk-pothole": 0,
                          "safe-pothole": 0}
        fps_list      = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_num += 1
                t0 = time.time()

                results = self.model.predict(
                    source=frame,
                    conf=self.conf_thresh.get(),
                    iou=self.iou_thresh.get(),
                    imgsz=640,
                    verbose=False
                )

                result    = results[0]
                fps_frame = 1.0 / (time.time() - t0 + 1e-6)
                fps_list.append(fps_frame)

                counts = self._count_classes(result)
                for k in total_counts:
                    total_counts[k] += counts[k]

                # Draw + overlay info
                annotated = self._draw_boxes(frame.copy(), result)
                annotated = self._draw_video_overlay(
                    annotated, fps_frame, counts, frame_num, total_frames)
                writer.write(annotated)
                self.current_frame = annotated.copy()

                # Update canvas every 5 frames (smooth preview)
                if frame_num % 5 == 0:
                    self.root.after(0, self._show_image_on_canvas, annotated)

                # Update stats + progress
                progress = (frame_num / total_frames) * 100
                avg_fps  = sum(fps_list[-30:]) / len(fps_list[-30:])
                self.root.after(0, self._update_stats,
                                 total_counts, avg_fps, progress, frame_num,
                                 total_frames)

        except Exception as e:
            self.root.after(0, messagebox.showerror, "Error", str(e))

        finally:
            cap.release()
            writer.release()
            self.is_processing = False
            avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
            self._set_status(
                f"Video done! {frame_num} frames @ {avg_fps:.1f} FPS "
                f"→ {out_path}")
            self.root.after(0, self._update_stats,
                             total_counts, avg_fps, 100,
                             frame_num, total_frames)

    # ─────────────────────────────────────────────────────────
    # DRAWING HELPERS
    # ─────────────────────────────────────────────────────────
    def _draw_boxes(self, frame, result):
        """Draw detection boxes on frame."""
        names = result.names
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            label  = names[cls_id]
            color  = CLASS_CONFIG.get(label, {}).get("bgr", (200, 200, 200))

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label background
            text = f"{label}  {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame,
                           (x1, y1 - th - 10), (x1 + tw + 6, y1),
                           color, -1)
            cv2.putText(frame, text, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 2)
        return frame

    def _draw_video_overlay(self, frame, fps, counts, frame_num, total):
        """Draw semi-transparent stats overlay on video frame."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (240, 110), (10, 10, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 220, 255), 2)
        cv2.putText(frame, f"Frame: {frame_num}/{total}",
                    (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (160, 160, 160), 1)

        y = 66
        colors_bgr = {
            "medium-pothole": (0, 140, 255),
            "risk-pothole":   (0, 0, 220),
            "safe-pothole":   (0, 200, 50),
        }
        for cls, cnt in counts.items():
            short = cls.replace("-pothole", "")
            cv2.putText(frame, f"{short}: {cnt}",
                        (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        colors_bgr.get(cls, (200, 200, 200)), 2)
            y += 18
        return frame

    def _count_classes(self, result):
        counts = {"medium-pothole": 0, "risk-pothole": 0, "safe-pothole": 0}
        for box in result.boxes:
            name = result.names[int(box.cls[0])]
            if name in counts:
                counts[name] += 1
        return counts

    # ─────────────────────────────────────────────────────────
    # CANVAS DISPLAY
    # ─────────────────────────────────────────────────────────
    def _show_image_on_canvas(self, cv_img):
        """Convert OpenCV BGR image and display on Tkinter canvas."""
        rgb     = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # Fit to canvas size
        cw = max(self.canvas.winfo_width(),  400)
        ch = max(self.canvas.winfo_height(), 300)
        pil_img.thumbnail((cw, ch), Image.LANCZOS)

        tk_img = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("all")
        cx = cw // 2
        cy = ch // 2
        self.canvas.create_image(cx, cy, image=tk_img, anchor="center")
        self.canvas.image = tk_img   # keep reference

    # ─────────────────────────────────────────────────────────
    # STATS UPDATE
    # ─────────────────────────────────────────────────────────
    def _update_stats(self, counts, fps,
                       progress=None, frame_num=None, total=None):
        total_det = sum(counts.values())
        self.stat_total.set(str(total_det))
        self.stat_medium.set(str(counts.get("medium-pothole", 0)))
        self.stat_risk.set(str(counts.get("risk-pothole", 0)))
        self.stat_safe.set(str(counts.get("safe-pothole", 0)))
        self.stat_fps.set(f"{fps:.1f}")

        if progress is not None:
            self.progress_var.set(progress)
            self.progress_label.config(
                text=f"Frame {frame_num}/{total}  ({progress:.1f}%)")

    def _set_status(self, msg):
        self.root.after(0, self.stat_status.set, msg)

    # ─────────────────────────────────────────────────────────
    # SAVE RESULT
    # ─────────────────────────────────────────────────────────
    def _save_result(self):
        if self.current_frame is None:
            messagebox.showinfo("Nothing to Save",
                                "Run detection first to get a result.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")],
            title="Save annotated result")

        if path:
            cv2.imwrite(path, self.current_frame)
            messagebox.showinfo("Saved", f"Result saved to:\n{path}")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = PotholeDetectionApp(root)
    root.mainloop()