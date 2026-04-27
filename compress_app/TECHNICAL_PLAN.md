# Image Compression Application — Technical Plan

## 1. Module Structure

```
compress_app/
├── REQUIREMENTS.md
├── TECHNICAL_PLAN.md
├── app.py           # Streamlit UI — layout, session state, progress rendering
├── compress.py      # Compression pipeline
├── decompress.py    # Decompression pipeline
└── utils.py         # Shared: model loading, file scanning, image I/O, padding, metrics
```

No CLI. Streamlit is the only interface, which eliminates the need for argparse and keeps the entry point to a single file.

---

## 2. Module Responsibilities

### `utils.py`
- **Model loading**: `load_model(name, quality, device)` — returns a cached model.  
  Uses a module-level dict `_MODEL_CACHE: dict[tuple, nn.Module]` so the same model is not reloaded across files in a run.
- **File scanning**: `scan_images(folder, recursive)` → list of `Path`; `scan_bins(folder, recursive)` → list of `Path`.
- **Image I/O**: `load_image(path)` → RGB PIL Image; `save_jpeg(tensor, path)`.
- **Padding**: `pad_to_multiple(tensor, multiple=64)` → padded tensor + original `(H, W)`; `crop_to_original(tensor, H, W)` → tensor.
- **Metrics**: `compute_psnr(original, reconstructed)` → float; `compute_msssim(original, reconstructed)` → float.

### `compress.py`
Single public function:
```python
def compress_folder(
    input_folder: Path,
    output_folder: Path,
    model_name: str,
    quality: int,
    device: str,
    recursive: bool,
    metrics: bool,
    delete_source: bool,
) -> Iterator[dict]
```
Yields one result dict per image:
```python
{
    "file": str,          # relative path
    "status": "ok" | "skipped" | "failed",
    "error": str | None,
    "orig_kb": float,
    "comp_kb": float,
    "ratio": float,
    "bpp": float,
    "psnr": float | None,
    "msssim": float | None,
    "orig_image": PIL.Image | None,   # only when metrics=True
    "recon_image": PIL.Image | None,  # only when metrics=True
}
```

**Per-image steps:**
1. Load image → convert to RGB → to tensor `[0, 1]`
2. Pad to multiple of 64; record original `(H, W)`
3. `model.compress(tensor)` → `strings`, `shape`
4. Write `.bin`: 4-byte header length + JSON metadata + raw bytes (see Section 4)
5. If `metrics=True`: decompress immediately, crop, compute PSNR / MS-SSIM
6. If `delete_source=True` and status is `ok`: delete source file
7. Yield result dict

### `decompress.py`
Single public function:
```python
def decompress_folder(
    input_folder: Path,
    output_folder: Path,
    recursive: bool,
) -> Iterator[dict]
```
Yields one result dict per `.bin` file:
```python
{
    "file": str,
    "status": "ok" | "failed",
    "error": str | None,
    "comp_kb": float,
    "out_kb": float,
    "model": str,
    "quality": int,
}
```

**Per-file steps:**
1. Parse `.bin` header → `model_name`, `quality`, `orig_H`, `orig_W`, compressed strings
2. Load/cache model (no device choice — inferred from header or defaulted to CPU)
3. `model.decompress(strings, shape)` → reconstructed tensor
4. Crop to `(orig_H, orig_W)`
5. Save as JPEG
6. Yield result dict

### `app.py`
- Renders sidebar controls and two tabs using `st.tabs`.
- Calls `compress_folder(...)` or `decompress_folder(...)` inside a `for` loop, updating a `st.progress` bar and a `st.dataframe` (held in `st.session_state`) after each yielded result.
- Renders the run summary and failed-files expander after the loop ends.
- Renders side-by-side image preview when metrics are enabled.
- No business logic — delegates everything to `compress.py` / `decompress.py`.

---

## 3. Data Flow

### Compression
```
app.py (user clicks Compress)
  → compress_folder() generator
      → utils.scan_images()
      → for each image:
          utils.load_image() → utils.pad_to_multiple()
          → model.compress()
          → write .bin (Section 4)
          → [optional] model.decompress() → utils.compute_psnr/msssim()
          → yield result dict
  → app.py updates progress bar + table row
  → app.py renders summary
```

### Decompression
```
app.py (user clicks Decompress)
  → decompress_folder() generator
      → utils.scan_bins()
      → for each .bin:
          parse header
          → utils.load_model()
          → model.decompress()
          → utils.crop_to_original()
          → utils.save_jpeg()
          → yield result dict
  → app.py updates progress bar + table row
  → app.py renders summary
```

---

## 4. Bitstream Format

Each `.bin` file is structured as:

```
[4 bytes: uint32 header_length]
[header_length bytes: UTF-8 JSON]
[remaining bytes: concatenated compressed string bytes]
```

JSON header fields:
```json
{
  "model": "bmshj2018-factorized",
  "quality": 4,
  "orig_h": 512,
  "orig_w": 768,
  "shape": [8, 12],
  "string_lengths": [1024, 256]
}
```

`string_lengths` records the byte length of each compressed string so the binary payload can be split correctly on decompression. This avoids any dependency on the `codec.py` format and keeps the file self-describing.

---

## 5. Model Cache

```python
# utils.py
_MODEL_CACHE: dict[tuple[str, int, str], torch.nn.Module] = {}

def load_model(name, quality, device):
    key = (name, quality, device)
    if key not in _MODEL_CACHE:
        model = image_models[name](quality=quality, pretrained=True)
        model.eval().to(device)
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]
```

Cache lives for the duration of the Streamlit server process. A single run over 500 images with the same model loads weights exactly once.

---

## 6. Live Progress in Streamlit

Streamlit reruns the script on every interaction. To stream per-image updates without blocking the UI:

```python
# app.py (simplified)
results = []
progress = st.progress(0)
status = st.empty()
table = st.empty()

for i, result in enumerate(compress_folder(...)):
    results.append(result)
    progress.progress((i + 1) / total)
    status.text(f"Processing: {result['file']}")
    table.dataframe(build_df(results))
```

The generator in `compress.py` yields after each image, giving Streamlit a chance to re-render. No threads needed.

---

## 7. Dependencies

| Package | Purpose |
|---|---|
| `compressai` | Models, compress/decompress API |
| `streamlit` | UI |
| `torch` | Tensor ops, model inference |
| `Pillow` | Image load/save |
| `pytorch-msssim` | MS-SSIM metric |

All are pip-installable. `pytorch-msssim` is the only addition beyond what CompressAI already requires.

---

## 8. What Is Explicitly Out of Scope

- No CLI interface (Streamlit only)
- No parallel processing
- No authentication or multi-user support
- No persistent storage of results between sessions
- No support for video or formats beyond PNG / JPEG input
