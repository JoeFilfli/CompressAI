import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

from compress import compress_folder
from decompress import decompress_folder
from utils import (
    BUILTIN_MODELS,
    CUSTOM_BASE_MODELS,
    infer_checkpoint_base,
    list_custom_checkpoints,
)

APP_DIR = Path(__file__).parent
cuda_available = torch.cuda.is_available()

CUSTOM_MODEL_LABEL = "Custom checkpoint (.pth)"

st.set_page_config(page_title="CompressAI", layout="wide")
st.title("Image Compression")

# ── Folder picker ─────────────────────────────────────────────────────────────

def _pick_folder(session_key: str, initial: Path) -> None:
    if sys.platform == "darwin":
        script = (
            'tell application "System Events" to activate\n'
            f'set chosen to POSIX path of (choose folder with prompt "Select Folder" default location POSIX file "{initial}")\n'
            'return chosen'
        )
        result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    elif sys.platform == "win32":
        ps_script = (
            "Add-Type -AssemblyName System.Windows.Forms;"
            "$f = New-Object System.Windows.Forms.FolderBrowserDialog;"
            f'$f.SelectedPath = "{initial}";'
            "$f.ShowDialog() | Out-Null;"
            "Write-Output $f.SelectedPath"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            capture_output=True, text=True,
        )
    else:
        result = subprocess.run(
            ["zenity", "--file-selection", "--directory",
             "--title", "Select Folder",
             "--filename", str(initial) + "/"],
            capture_output=True, text=True,
        )
    folder = result.stdout.strip()
    if folder:
        st.session_state[session_key] = folder


def _folder_row(label: str, session_key: str, browse_key: str, initial: Path) -> str:
    col_label, col_path, col_btn = st.columns([2, 5, 1])
    col_label.markdown(f"**{label}**")
    path = st.session_state.get(session_key, "")
    col_path.markdown(path or "_Not selected_")
    if col_btn.button("Browse…", key=browse_key):
        _pick_folder(session_key, initial)
        st.rerun()
    return path


# ── Table helpers ─────────────────────────────────────────────────────────────

def _compress_df(results: list[dict], show_metrics: bool) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {
            "File":              r.get("file"),
            "Original (KB)":    round(r["orig_kb"], 2) if r.get("orig_kb") is not None else None,
            "Compressed (KB)":  round(r["comp_kb"], 2) if r.get("comp_kb") is not None else None,
            "Ratio":            round(r["ratio"], 2)   if r.get("ratio")   is not None else None,
            "BPP":              round(r["bpp"], 3)     if r.get("bpp")     is not None else None,
            "Status":           r.get("status"),
        }
        if show_metrics:
            row["PSNR (dB)"] = round(r["psnr"], 2)   if r.get("psnr")   is not None else None
            row["MS-SSIM"]   = round(r["msssim"], 4) if r.get("msssim") is not None else None
        rows.append(row)
    return pd.DataFrame(rows)


def _decompress_df(results: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "File":               r.get("file"),
            "Compressed (KB)":   round(r["comp_kb"], 2) if r.get("comp_kb") is not None else None,
            "Output (KB)":       round(r["out_kb"], 2)  if r.get("out_kb")  is not None else None,
            "Model":             r.get("model"),
            "Quality":           r.get("quality"),
            "Checkpoint":        r.get("checkpoint"),
            "Status":            r.get("status"),
        }
        for r in results
    ])


# ── Summary helpers ───────────────────────────────────────────────────────────

def _compress_summary(results: list[dict], show_metrics: bool, elapsed: float) -> None:
    ok = sum(1 for r in results if r["status"] == "ok")
    failed = [r for r in results if r["status"] == "failed"]
    total_in = sum(r.get("orig_kb") or 0 for r in results) / 1024
    total_out = sum(r.get("comp_kb") or 0 for r in results if r.get("comp_kb")) / 1024
    ratio = total_in / total_out if total_out > 0 else 0.0
    avg_s = elapsed / ok if ok > 0 else 0.0

    st.markdown("---")
    st.subheader("Summary")

    comp_times   = [r["comp_time"]   for r in results if r.get("comp_time")   is not None]
    decomp_times = [r["decomp_time"] for r in results if r.get("decomp_time") is not None]
    avg_comp_s   = sum(comp_times)   / len(comp_times)   if comp_times   else None
    avg_decomp_s = sum(decomp_times) / len(decomp_times) if decomp_times else None

    card1, card2 = st.columns(2)

    with card1:
        with st.container(border=True):
            st.caption("BATCH")
            r = st.columns(2)
            r[0].metric("Processed", ok)
            r[1].metric("Failed", len(failed))
            r2 = st.columns(2)
            r2[0].metric("Total Time", f"{elapsed:.1f}s")
            r2[1].metric("Avg per Image", f"{avg_s:.2f}s")
            r3 = st.columns(2)
            r3[0].metric("Avg Compress", f"{avg_comp_s:.2f}s" if avg_comp_s is not None else "—")
            r3[1].metric("Avg Decompress", f"{avg_decomp_s:.2f}s" if avg_decomp_s is not None else "—")

    with card2:
        with st.container(border=True):
            st.caption("SIZE & COMPRESSION")
            r = st.columns(2)
            r[0].metric("Input (MB)", f"{total_in:.2f}")
            r[1].metric("Compressed (MB)", f"{total_out:.2f}")
            st.metric("Compression Ratio", f"{ratio:.2f}×")

    if show_metrics:
        psnr_vals = [r["psnr"] for r in results if r.get("psnr") is not None]
        ms_vals = [r["msssim"] for r in results if r.get("msssim") is not None]
        if psnr_vals or ms_vals:
            with st.container(border=True):
                st.caption("QUALITY")
                r = st.columns(2)
                r[0].metric("Avg PSNR (dB)", f"{sum(psnr_vals) / len(psnr_vals):.2f}" if psnr_vals else "—")
                r[1].metric("Avg MS-SSIM", f"{sum(ms_vals) / len(ms_vals):.4f}" if ms_vals else "—")

    total_jpeg = sum(r.get("jpeg_kb") or 0 for r in results if r.get("jpeg_kb")) / 1024
    if total_jpeg > 0:
        saved = total_jpeg - total_out
        pct = (saved / total_jpeg) * 100
        if saved > 0:
            st.success(
                f"You saved **{saved:.2f} MB ({pct:.1f}%)** compared to "
                f"standard JPEG compression."
            )
        else:
            st.info(
                f"At this quality level, output is **{abs(saved):.2f} MB ({abs(pct):.1f}%) larger** "
                f"than standard JPEG compression."
            )

    if failed:
        with st.expander(f"Failed Files ({len(failed)})"):
            for r in failed:
                st.text(f"{r['file']}: {r['error']}")


def _decompress_summary(results: list[dict], elapsed: float) -> None:
    ok = sum(1 for r in results if r["status"] == "ok")
    failed = [r for r in results if r["status"] == "failed"]
    total_in = sum(r.get("comp_kb") or 0 for r in results) / 1024
    total_out = sum(r.get("out_kb") or 0 for r in results if r.get("out_kb")) / 1024
    avg_s = elapsed / ok if ok > 0 else 0.0
    decomp_times = [r["decomp_time"] for r in results if r.get("decomp_time") is not None]
    avg_decomp_s = sum(decomp_times) / len(decomp_times) if decomp_times else None

    st.markdown("---")
    st.subheader("Summary")

    card1, card2 = st.columns(2)

    with card1:
        with st.container(border=True):
            st.caption("BATCH")
            r = st.columns(2)
            r[0].metric("Processed", ok)
            r[1].metric("Failed", len(failed))
            r2 = st.columns(2)
            r2[0].metric("Total Time", f"{elapsed:.1f}s")
            r2[1].metric("Avg per File", f"{avg_s:.2f}s")
            r3 = st.columns(2)
            r3[0].metric("Avg Decompress", f"{avg_decomp_s:.2f}s" if avg_decomp_s is not None else "—")
            r3[1].write("")

    with card2:
        with st.container(border=True):
            st.caption("SIZE")
            r = st.columns(2)
            r[0].metric("Compressed Input (MB)", f"{total_in:.2f}")
            r[1].metric("Reconstructed Output (MB)", f"{total_out:.2f}")

    if failed:
        with st.expander(f"Failed Files ({len(failed)})"):
            for r in failed:
                st.text(f"{r['file']}: {r['error']}")


compress_tab, decompress_tab = st.tabs(["Compress", "Decompress"])

# ── Compress Tab ──────────────────────────────────────────────────────────────
with compress_tab:
    checkpoint_path = None
    checkpoint_quality = None
    checkpoint_base = None

    cfg, _ = st.columns([1, 2])
    with cfg:
        model_choice = st.selectbox(
            "Model",
            BUILTIN_MODELS + [CUSTOM_MODEL_LABEL],
            key="c_model",
        )
        quality = st.slider("Quality", 1, 8, 4, key="c_quality")
        device_opts = ["CPU"] + (["CUDA"] if cuda_available else [])
        device_label = st.radio(
            "Device",
            device_opts,
            key="c_device",
            help="No CUDA-capable GPU detected." if not cuda_available else "",
        )
        device = "cuda" if device_label == "CUDA" else "cpu"
        recursive = st.toggle("Recursive scan", value=True, key="c_recursive")
        metrics = st.toggle("Compute metrics (PSNR / MS-SSIM)", value=False, key="c_metrics")
        delete_source = st.toggle("Delete source files", value=False, key="c_delete")

        if model_choice == CUSTOM_MODEL_LABEL:
            checkpoints = list_custom_checkpoints(APP_DIR)
            if not checkpoints:
                st.warning("No .pth checkpoints found in compress_app.")
            else:
                labels = [c["label"] for c in checkpoints]
                selected_label = st.selectbox(
                    "Checkpoint (.pth)",
                    labels,
                    key="c_checkpoint",
                )
                selected = checkpoints[labels.index(selected_label)]
                checkpoint_path = selected["path"]
                checkpoint_quality = selected["quality"]
                if selected.get("teacher"):
                    st.caption(f"Distilled from: {selected['teacher']}")
                if checkpoint_quality is not None:
                    st.caption(
                        f"Checkpoint quality: q{checkpoint_quality} (slider ignored)."
                    )
                base_default = (
                    selected["student"]
                    if selected["student"] in CUSTOM_BASE_MODELS
                    else "tiny-hyperprior"
                )
                detected = infer_checkpoint_base(checkpoint_path)
                detected_base = detected.get("base")
                detected_matches = detected.get("matches") or []
                if detected.get("n") is not None and detected.get("m") is not None:
                    st.caption(
                        f"Detected channels: N={detected['n']}, M={detected['m']}"
                    )

                if detected_base:
                    st.caption(f"Detected base model: {detected_base}")
                    override = st.toggle(
                        "Override detected base model",
                        value=False,
                        key="c_override_base",
                    )
                    if override:
                        checkpoint_base = st.selectbox(
                            "Checkpoint base model",
                            CUSTOM_BASE_MODELS,
                            index=CUSTOM_BASE_MODELS.index(detected_base),
                            key="c_custom_base",
                        )
                    else:
                        checkpoint_base = detected_base
                else:
                    if detected_matches:
                        base_default = detected_matches[0]
                        st.warning(
                            "Multiple base models match this checkpoint. "
                            "Please choose one."
                        )
                    else:
                        st.warning(
                            "Could not detect a base model from the checkpoint. "
                            "Please choose one."
                        )
                    checkpoint_base = st.selectbox(
                        "Checkpoint base model",
                        CUSTOM_BASE_MODELS,
                        index=CUSTOM_BASE_MODELS.index(base_default),
                        key="c_custom_base",
                    )

    if model_choice == CUSTOM_MODEL_LABEL:
        model_name = checkpoint_base or "tiny-hyperprior"
        model_quality = checkpoint_quality if checkpoint_quality is not None else quality
    else:
        model_name = model_choice
        model_quality = quality

    if delete_source:
        st.error(
            "Source images will be permanently deleted after successful compression. "
            "This cannot be undone."
        )

    st.markdown("---")
    input_str = _folder_row("Input folder", "c_input", "c_browse_input", Path.home())
    output_loc_str = _folder_row("Output location", "c_output_loc", "c_browse_output", APP_DIR)

    # Derive the actual output folder name from the input folder
    input_path = Path(input_str) if input_str else None
    output_loc = Path(output_loc_str) if output_loc_str else APP_DIR
    output_path = output_loc / f"{input_path.name}_compressed" if input_path else None

    if output_path:
        st.caption(f"Compressed files will be saved to: `{output_path}`")

    run_btn = st.button("Compress", key="c_run")

    if run_btn:
        errors = []
        if not input_path or not input_path.is_dir():
            errors.append("Please select a valid input folder.")
        if model_choice == CUSTOM_MODEL_LABEL and not checkpoint_path:
            errors.append("Please select a .pth checkpoint.")

        for e in errors:
            st.error(e)

        if not errors:
            total, gen = compress_folder(
                input_path,
                output_path,
                model_name,
                model_quality,
                device,
                recursive,
                metrics,
                delete_source,
                checkpoint_path=checkpoint_path,
            )

            if total == 0:
                st.warning("No PNG or JPEG files found in the selected folder.")
            else:
                st.info(f"Found {total} image(s). Compressing…")
                progress = st.progress(0)
                status = st.empty()
                table_ph = st.empty()
                preview_ph = st.empty()
                results = []
                t_start = time.time()

                for i, result in enumerate(gen):
                    results.append(result)
                    progress.progress((i + 1) / total)
                    status.text(f"Processing: {result['file']}")
                    table_ph.dataframe(
                        _compress_df(results, show_metrics=metrics),
                        use_container_width=True,
                    )
                    if metrics and result.get("orig_image") and result.get("recon_image"):
                        with preview_ph.container():
                            c1, c2 = st.columns(2)
                            c1.image(result["orig_image"], caption="Original")
                            c2.image(result["recon_image"], caption="Reconstructed")

                status.text("Done.")
                _compress_summary(results, show_metrics=metrics, elapsed=time.time() - t_start)


# ── Decompress Tab ────────────────────────────────────────────────────────────
with decompress_tab:
    dcfg, _ = st.columns([1, 2])
    with dcfg:
        d_recursive = st.toggle("Recursive scan", value=True, key="d_recursive")

    st.markdown("---")
    d_input_str = _folder_row("Input folder (.bin files)", "d_input", "d_browse_input", Path.home())
    d_output_loc_str = _folder_row("Output location", "d_output_loc", "d_browse_output", APP_DIR)

    d_input_path = Path(d_input_str) if d_input_str else None
    d_output_loc = Path(d_output_loc_str) if d_output_loc_str else APP_DIR
    d_output_path = d_output_loc / f"{d_input_path.name}_decompressed" if d_input_path else None

    if d_output_path:
        st.caption(f"Reconstructed files will be saved to: `{d_output_path}`")

    d_run_btn = st.button("Decompress", key="d_run")

    if d_run_btn:
        d_errors = []
        if not d_input_path or not d_input_path.is_dir():
            d_errors.append("Please select a valid input folder.")

        for e in d_errors:
            st.error(e)

        if not d_errors:
            total, gen = decompress_folder(d_input_path, d_output_path, d_recursive)

            if total == 0:
                st.warning("No .bin files found in the selected folder.")
            else:
                st.info(f"Found {total} file(s). Decompressing…")
                progress = st.progress(0)
                status = st.empty()
                table_ph = st.empty()
                results = []
                t_start = time.time()

                for i, result in enumerate(gen):
                    results.append(result)
                    progress.progress((i + 1) / total)
                    status.text(f"Processing: {result['file']}")
                    table_ph.dataframe(
                        _decompress_df(results),
                        use_container_width=True,
                    )

                status.text("Done.")
                _decompress_summary(results, elapsed=time.time() - t_start)
