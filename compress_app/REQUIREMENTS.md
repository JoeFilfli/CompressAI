# Image Compression Application — Requirements Specification

## Overview

An application that uses CompressAI pretrained models to compress and decompress folders of images. It exposes a local Streamlit web UI (`localhost:8501`) for interactive use, backed by a Python pipeline that handles the compression and decompression logic.

---

## Part 1 — Functional Requirements (Backend)

---

### 1. Model Configuration

- User can select a pretrained model from the following options:
  - `bmshj2018-factorized`
  - `bmshj2018-hyperprior`
  - `mbt2018`
  - `mbt2018-mean`
  - `cheng2020-anchor`
  - `cheng2020-attn`
- User can set the quality level (integer from 1 to 8).
- Pretrained weights are downloaded automatically from the CompressAI model zoo if not already cached locally.

---

### 2. Compression

#### Input
- Accepts a folder path as input.
- Scans for supported image formats: PNG and JPEG.
- Supports recursive scanning (default) or flat scanning via a flag.
- Skips unsupported files gracefully with a per-file warning; does not abort the batch.
- Validates that each image is in RGB mode; converts automatically if not (e.g. RGBA, grayscale).

#### Processing
- Pads each image to the nearest multiple of 64 pixels before encoding (required by the model architecture).
- Compresses each image to a `.bin` bitstream file using the selected model.

#### Output
- Stores compressed `.bin` files in a user-specified output folder.
- Preserves the source folder structure inside the output folder.
- Reports per-image statistics:
  - Original file size (KB)
  - Compressed file size (KB)
  - Compression ratio
  - Bits per pixel (BPP)

#### Source File Handling
- Original images are **not deleted by default**.
- Deletion of source files after successful compression is available via an explicit option.
- Source files are **never deleted** if compression fails for that file.

---

### 3. Decompression

#### Input
- Accepts a folder of `.bin` bitstream files as input.
- Scans the folder (recursively by default) for `.bin` files.

#### Processing
- Reads the model identifier and quality level from each bitstream's header — no need for the user to specify these manually.
- Reconstructs the image from the bitstream.

#### Output
- Saves each reconstructed image as a JPEG file.
- Stores output files in a user-specified output folder.
- Preserves the source folder structure inside the output folder.

---

### 4. Reporting & Metrics

#### Summary (always shown)
After every run, display a summary containing:
- Total number of images processed successfully
- Total number of images skipped or failed
- Total input size (MB)
- Total output size (MB)
- Overall compression ratio

#### Quality Metrics (optional)
- Enabled via a toggle; requires access to the original image.
- Computes and reports per-image:
  - PSNR (Peak Signal-to-Noise Ratio, dB)
  - MS-SSIM (Multi-Scale Structural Similarity Index)
- Metrics are included in the per-run summary when enabled.

#### Error Logging
- Errors are logged per file with the file path and reason.
- A failed file does not abort processing of remaining files.
- At the end of the run, all failed files are listed in the summary.

---

### 5. Non-Functional Requirements (Backend)

- **Safety**: Never delete source files on compression failure.
- **Graceful degradation**: A single bad file must not crash the application.
- **Portability**: Runs on CPU without requiring a GPU.
- **Dependency**: Built on top of the CompressAI library; model weights are managed by the CompressAI model zoo.

---

## Part 2 — Frontend Requirements (Streamlit UI)

---

### 6. Layout

#### 6.1 Page Structure
- A **sidebar** on the left for all configuration (model, quality, options).
- A **main area** on the right split into two tabs:
  - `Compress`
  - `Decompress`
- The active tab determines which configuration options are shown in the sidebar.

#### 6.2 Sidebar — Compress Tab
| Control | Type | Details |
|---|---|---|
| Model | Selectbox | Options: `bmshj2018-factorized`, `bmshj2018-hyperprior`, `mbt2018`, `mbt2018-mean`, `cheng2020-anchor`, `cheng2020-attn` |
| Quality | Slider | Integer, range 1–8, default 4 |
| Device | Radio | `CPU` / `CUDA` — CUDA option disabled (greyed out) if no GPU is detected |
| Recursive scan | Toggle | On by default |
| Compute metrics | Toggle | Off by default; shows PSNR and MS-SSIM per image when enabled |
| Delete source files | Toggle | Off by default; shows a red warning message when turned on |

#### 6.3 Sidebar — Decompress Tab
| Control | Type | Details |
|---|---|---|
| Recursive scan | Toggle | On by default |

---

### 7. Compress Tab

#### 7.1 Path Inputs
- **Input folder** — text field for the local folder path containing images.
- **Output folder** — text field for the destination folder for `.bin` files.
- Both fields show a red inline error if the path does not exist when the user clicks Run.

#### 7.2 Run Button
- Label: `Compress`
- Disabled while a run is already in progress.
- Triggers folder scan and compression pipeline.

#### 7.3 Progress Display
Shown only during an active run:
- A progress bar showing `images done / total images`.
- A status line below it showing the filename currently being processed (e.g. `Processing: photos/trip/img_001.png`).

#### 7.4 Per-Image Results Table
Updated live as each image completes. Columns:
| Column | Description |
|---|---|
| File | Relative path from the input folder |
| Original size | In KB |
| Compressed size | In KB |
| Ratio | `original / compressed` |
| BPP | Bits per pixel |
| PSNR (dB) | Only shown if metrics are enabled |
| MS-SSIM | Only shown if metrics are enabled |
| Status | `OK`, `Skipped`, or `Failed` |

#### 7.5 Image Preview (optional, shown when metrics are enabled)
- A side-by-side image comparison: **Original** on the left, **Reconstructed** on the right.
- Updates to show the most recently processed image.
- Displayed below the results table.

#### 7.6 Run Summary
Shown after the run completes:
- Total processed / skipped / failed
- Total input size (MB)
- Total compressed size (MB)
- Overall compression ratio
- Average PSNR and MS-SSIM (if metrics were enabled)
- A collapsible **Failed Files** section listing each failed file and its error reason.

---

### 8. Decompress Tab

#### 8.1 Path Inputs
- **Input folder** — text field for the folder containing `.bin` files.
- **Output folder** — text field for the destination folder for reconstructed JPEG images.
- Same inline validation as the Compress tab.

#### 8.2 Run Button
- Label: `Decompress`
- Disabled while a run is in progress.

#### 8.3 Progress Display
Same pattern as compression:
- Progress bar (`files done / total files`).
- Status line showing the current `.bin` file being processed.

#### 8.4 Per-File Results Table
Updated live as each file completes. Columns:
| Column | Description |
|---|---|
| File | Relative path from the input folder |
| Compressed size | In KB |
| Output size | In KB |
| Model | Auto-detected from bitstream header |
| Quality | Auto-detected from bitstream header |
| Status | `OK`, `Skipped`, or `Failed` |

#### 8.5 Run Summary
Shown after the run completes:
- Total processed / skipped / failed
- Total compressed input size (MB)
- Total reconstructed output size (MB)
- A collapsible **Failed Files** section.

---

### 9. Warnings & Validation

| Trigger | Message |
|---|---|
| `Delete source files` toggle turned on | Red warning: "Source images will be permanently deleted after successful compression. This cannot be undone." |
| Input path does not exist | Red inline error under the field: "Folder not found." |
| Output path does not exist | Yellow inline warning: "Folder does not exist and will be created." |
| No supported images found in input folder | Warning banner: "No PNG or JPEG files found in the selected folder." |
| CUDA selected but no GPU available | CUDA option is greyed out; tooltip: "No CUDA-capable GPU detected." |

---

### 10. State & Session Behavior

- Configuration (model, quality, device, toggles) persists within the session via Streamlit session state.
- Path inputs reset to empty when switching between tabs.
- Results tables and summaries clear when a new run is started.
- The app does not persist any state across browser refreshes.

---

### 11. Non-Functional Requirements (Frontend)

- **Responsiveness**: The UI must not freeze during compression; long-running operations run in a background thread or via a generator pattern so the progress bar stays live.
- **Error isolation**: An unhandled exception in the backend must show an error banner in the UI, not crash the Streamlit server.
- **No external network calls**: The UI is fully local; no telemetry, no CDN assets.
- **Startup time**: The app should be usable within 5 seconds of `streamlit run app.py`; model weights are loaded on demand (when the user clicks Run), not at startup.
