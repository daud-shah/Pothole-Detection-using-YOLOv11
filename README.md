# Pothole Detection using YOLOv11

Short, self-contained instructions to set up and run the Pothole Detection GUI from a fresh Windows environment.

## Project Overview

- Purpose: Desktop GUI for detecting potholes using a YOLOv11 model (3 classes).
- Main script: `gui.py` (launches the Tkinter GUI).
- Example model weights: `best.pt` (included in the repository root).

## Features

- Load a trained `.pt` model and run detection on images or videos.
- Adjustable confidence and IoU thresholds from the GUI.
- Save detection outputs to an output folder.

## Repository Structure

- `gui.py` – Main GUI application.
- `best.pt` – Example model weights (place your trained weights here).
- `REQUIREMENTS.txt` – Python dependencies (pip install -r REQUIREMENTS.txt).
- `HOW_TO_RUN.txt` – Step-by-step run guide (also included here).
- `Model coding and data/` – Notebooks and dataset artifacts.

## Prerequisites

- Windows 10/11 (or similar)
- Python 3.8+ (3.10–3.14 recommended)
- Optional: NVIDIA GPU and CUDA for faster inference

## Quick Setup (Windows)

1. Open PowerShell and change into the project folder:

```powershell
cd "D:\MY WORK\FYP_project\Pothole Detection using YOLOv11"
```

2. Create a virtual environment (one-time):

```powershell
python -m venv venv
```

3. Activate the virtual environment:

PowerShell:
```powershell
.\venv\Scripts\Activate.ps1
```

Command Prompt (cmd.exe):
```cmd
venv\Scripts\activate.bat
```

4. Upgrade pip and install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r REQUIREMENTS.txt
```

If you prefer to install manually:

```powershell
python -m pip install ultralytics opencv-python pillow torch torchvision numpy
```

## Run the GUI

With the virtual environment activated, run:

```powershell
python gui.py
```

If you don't activate the venv, you can run directly with the venv python:

```powershell
.\venv\Scripts\python.exe gui.py
```

## Using the GUI

1. Click the **Browse** button next to "Load Model" and select your `.pt` weights (e.g., `best.pt`).
2. Choose `Image` or `Video` mode.
3. Click the input browse button to choose an image or video file.
4. Adjust confidence and IoU sliders as needed.
5. Click **RUN DETECTION** to start. Outputs are saved to the configured output folder.

## Troubleshooting

- ModuleNotFoundError: No module named 'cv2' — ensure the venv is activated and run:

```powershell
python -m pip install opencv-python
```

- No module named 'ultralytics' — install with:

```powershell
python -m pip install ultralytics
```

- No module named 'tkinter' — tkinter is bundled with standard Windows Python installers. Reinstall Python if missing.

## Model & Training Notes

- If you trained your own model, place the `.pt` file in the project root or select it through the GUI.
- Training scripts and notebooks are in `Model coding and data/`.

## Optional: Run detection from command line (headless)

If you want to run detection without the GUI, consider using the ultralytics CLI or a small script that loads the model and runs `model.predict()`.

## License & Contact

- License: MIT (or change as desired)
- Author / FYP: University of Agriculture Peshawar — FYP 2026

---

If you'd like, I can also:
- Add a small example script to run detections headless, or
- Add a screenshot and GIF to this README showing the GUI in action.

