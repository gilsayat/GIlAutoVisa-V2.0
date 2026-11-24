# US Visa Photo Automator

A self-contained, 100% offline Smart Canvas desktop & web studio utility that automatically segments backgrounds with edge de-fringing, levels and centers portraits using 478-point facial landmarks, checks live U.S. Visa compliance, and exports 600 DPI lab print sheets with clean outer crop marks. Created by Gil.

## Building the Portable No-Install Windows `.exe` via GitHub

1. Push this repository to GitHub (branch `main` or `master`), or trigger **Actions → Build Portable Windows EXE → Run workflow**.
2. Once the workflow finishes, download the **`USVisaPhotoAutomator-Windows-Portable`** artifact.
3. Extract and double-click **`USVisaPhotoAutomator.exe`** — no installation or internet connection required.

## Local Offline AI Models

Both neural network models are already downloaded in `public/mediapipe/models/` (~3.9 MB total) and bundled directly inside the portable `.exe`:

- `face_landmarker.task` (3.6 MB) — 478-point facial landmarks & automatic eye-leveling
- `selfie_segmenter.tflite` (244 KB) — portrait foreground matting

You can also place a `Models/` folder directly next to `USVisaPhotoAutomator.exe` containing `face_landmarker.task` and `selfie_segmenter.tflite`; the desktop app checks the local `Models/` folder next to the `.exe` first and falls back to its built-in models automatically.

## Running Locally (Web Dev Mode)

```bash
npm install
npm run dev
```

To build the portable `.exe` locally on Windows:

```bash
npm install
npm install --no-save electron electron-builder
npm run build:exe
```
