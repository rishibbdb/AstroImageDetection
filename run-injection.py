"""
drips_injection_pipeline.py
----------------------------
Blob-detection pipeline for HAWC injection-study runs.

Reads:
  - model_dir  : injected-model YAML files  (model_<N>_roi_<RA>_<Dec>.yaml)
  - fit_dir    : ALPS fit results            (FinalRefit yml or curModel.model)
  - map FITS   : significance maps           (fits/model_<N>/initialMap/maps/map.fits)

Writes per run:
  - blob_detection_results.yaml  (injected sources + ALPS fits + detected blobs)
  - per_run PDF diagnostics

Requirements:
    pip install astropy matplotlib numpy pandas pyyaml scikit-image scipy
    (plus your local helpers.py with load_hawc_data, make_plots, etc.)

Usage:
    python drips_injection_pipeline.py
    python drips_injection_pipeline.py --runs 1 9 42   # specific runs only
    python drips_injection_pipeline.py --max_runs 20   # first N runs
"""

import argparse
import os
import re
import sys
import warnings
import yaml

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as mpdf

from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.ndimage import gaussian_filter
from skimage.filters import difference_of_gaussians
import scipy.ndimage as ndi
from skimage.feature import blob_dog
from astropy.stats import sigma_clip

warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(".."))
from helpers import (
    load_hawc_data, make_plots, blob_filter_intensity,
    radius_to_sigma, find_peak, soft_floor,
    setupMilagroColormap,
)

# ── Directory / path config ────────────────────────────────────────────────────

MODEL_DIR = (
    "/lustre/hawcz01/scratch/userspace/sgroetsch/source_fitting/"
    "4HWCv5/injectionStudy/ModelsMaps/finishedModels2/"
)
FIT_DIR = (
    "/lustre/hawcz01/scratch/userspace/sgroetsch/source_fitting/"
    "4HWCv5/injectionStudy/fits/"
)
OUTPUT_DIR  = os.path.join(os.getcwd(), "drips_results")
YAML_OUTPUT = os.path.join(OUTPUT_DIR, "blob_detection_results.yaml")
MODEL_LIST  = os.path.join(OUTPUT_DIR, "injected-model-list.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "per_run"), exist_ok=True)

# ── Pipeline settings ──────────────────────────────────────────────────────────

RUN_NUMBERS   = np.arange(1, 250, dtype=int)   # 1..249
SMEAR_RADII   = [0.2, 0.25, 0.3, 0.4]          # degrees
SIG_THRESHOLD = 5.0
XLENGTH       = 11
YLENGTH       = 11
COORD_SYS     = "C"

# ── File-name / YAML helpers ───────────────────────────────────────────────────

def build_model_list(model_dir, out_path):
    """Write a sorted list of all files in model_dir and return it."""
    files = sorted(
        f for f in os.listdir(model_dir)
        if os.path.isfile(os.path.join(model_dir, f))
    )
    with open(out_path, "w") as fh:
        for f in files:
            fh.write(f + "\n")
    return files


def extract_ra_dec(filename):
    """
    Pull RA and Dec from a filename like
    model_9_roi_83.61_22.00.yaml
    Returns (ra_str, dec_str).
    """
    # m = re.search(r"roi_([\d.]+)_([-\d.]+)", filename)
    m = re.search(r"roi_([0-9]+(?:\.[0-9]+)?)_([-0-9]+(?:\.[0-9]+)?)", filename)
    if m:
        return m.group(1), m.group(2)
    raise ValueError(f"Cannot parse RA/Dec from filename: {filename}")


def extract_run(filename):
    """Pull the run number from model_<N>_roi_....yaml"""
    m = re.search(r"model_(\d+)", filename)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot parse run number from filename: {filename}")


def parse_yaml_file(filepath):
    with open(filepath, 'r') as file:
        data = yaml.safe_load(file)
    
    sources = []
    for source, properties in data.items():
        if 'Gaussian_on_sphere' in properties:
            lon0 = properties['Gaussian_on_sphere']['lon0']['value']
            lat0 = properties['Gaussian_on_sphere']['lat0']['value']
            sigma = properties['Gaussian_on_sphere'].get('sigma', {}).get('value', None)
            sources.append({
                'source_name': source,
                'lon0': lon0,
                'lat0': lat0,
                'sigma': sigma,
            })
        elif 'position' in properties:
            lon0 = properties['position']['ra']['value']
            lat0 = properties['position']['dec']['value']
            sigma = 0
            sources.append({
                'source_name': source,
                'lon0': lon0,
                'lat0': lat0,
                'sigma': sigma,
            })
        else:
            pass
    return sources

def parse_model_file(path):
    """
    Minimal parser for ALPS .model plain-text files.
    Each source block has lines like  key = value.
    Returns a list of dicts.
    """
    sources, current = [], {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                key, val = key.strip(), val.strip()
                try:
                    val = float(val)
                except ValueError:
                    pass
                current[key] = val
            elif line.lower().startswith("source"):
                if current:
                    sources.append(current)
                current = {}
    if current:
        sources.append(current)
    return sources


# ── Per-run file-path helpers ──────────────────────────────────────────────────

def map_fits_path(run_number):
    return os.path.join(
        FIT_DIR,
        f"model_{run_number}/initialMap/maps/map.fits",
    )


def final_refit_path(run_number):
    return os.path.join(
        FIT_DIR,
        f"model_{run_number}/fitResults/FinalRefit/"
        f"model_{run_number}_modelFit.yml",
    )


def cur_model_path(run_number):
    return os.path.join(
        FIT_DIR,
        f"model_{run_number}/models/curModel.model",
    )


def model_yaml_path(run_number, ra_center, dec_center):
    return os.path.join(
        MODEL_DIR,
        f"model_{run_number}_roi_{ra_center:.2f}_{dec_center:.2f}.yaml",
    )


# ── Source-index builders ──────────────────────────────────────────────────────

def load_injected_sources(run_number, ra_center, dec_center):
    """Return a DataFrame of injected (true) sources for this run."""
    path = model_yaml_path(run_number, ra_center, dec_center)
    print(path)
    sources = parse_yaml_file(path)
    # print(sources)
    return pd.DataFrame({
        "name": ["Injected Source"+str(s) for s in range(len(sources))],
        "ra":   [s["lon0"]        for s in sources],
        "dec":  [s["lat0"]        for s in sources],
        "ext":  [s.get("sigma", 0.0) for s in sources],
    })


def load_alps_sources(run_number):
    """Return a DataFrame of ALPS-fitted sources, trying FinalRefit then curModel."""
    try:
        fitted = parse_yaml_file(final_refit_path(run_number))
        return pd.DataFrame({
            "name": [s["source_name"] for s in fitted],
            "ra":   [s["lon0"]        for s in fitted],
            "dec":  [s["lat0"]        for s in fitted],
            "ext":  [s.get("sigma", 0.0) for s in fitted],
        }), "FinalRefit"
    except FileNotFoundError:
        pass

    try:
        fitted = parse_model_file(cur_model_path(run_number))
        return pd.DataFrame({
            "name": [s.get("name", f"src{i}") for i, s in enumerate(fitted)],
            "ra":   [s["ra"]                   for s in fitted],
            "dec":  [s["dec"]                  for s in fitted],
            "ext":  [s.get("sigma", 0.0)       for s in fitted],
        }), "curModel"
    except FileNotFoundError:
        return pd.DataFrame(columns=["name", "ra", "dec", "ext"]), "none"


# ── Blob-detection helpers (unchanged from original) ──────────────────────────

def estimate_background_sigma(image, sigma=3, maxiters=5):
    clipped = sigma_clip(image, sigma=sigma, maxiters=maxiters)
    return float(np.std(clipped.data[~clipped.mask]))


def remove_overlapping_blobs(blobs, coords, radii, overlap_threshold=0.5):
    if len(blobs) == 0:
        return blobs, coords, radii

    order         = np.argsort(blobs[:, 2])[::-1]
    sorted_blobs  = blobs[order]
    sorted_coords = [coords[i] for i in order]
    sorted_radii  = [radii[i]  for i in order]
    keep = np.ones(len(sorted_blobs), dtype=bool)

    for i in range(len(sorted_blobs)):
        if not keep[i]:
            continue
        dist      = np.hypot(sorted_blobs[i, 0] - sorted_blobs[:, 0],
                             sorted_blobs[i, 1] - sorted_blobs[:, 1])
        radii_sum = sorted_blobs[i, 2] + sorted_blobs[:, 2]
        overlap   = (radii_sum - dist) / np.minimum(sorted_blobs[i, 2],
                                                     sorted_blobs[:, 2])
        keep[overlap > overlap_threshold] = False
        keep[i] = True

    return (
        sorted_blobs[keep],
        [c for i, c in enumerate(sorted_coords) if keep[i]],
        [r for i, r in enumerate(sorted_radii)  if keep[i]],
    )


def convert_sources(coords, radii, name_prefix="DripPS", source_type="PS"):
    cols = ["name", "ra", "dec", "circle_radius", "sigma_radius", "type"]
    if len(coords) == 0:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame({
        "name":          [f"{name_prefix}{i}" for i in range(len(coords))],
        "ra":            [c.ra.deg  for c in coords],
        "dec":           [c.dec.deg for c in coords],
        "circle_radius": radii,
        "sigma_radius":  [radius_to_sigma(r) for r in radii],
        "type":          source_type,
    })


def df_to_records(df):
    """Convert a DataFrame to a plain list of dicts for YAML serialisation."""
    if df is None or df.empty:
        return []
    out = []
    for _, row in df.iterrows():
        out.append({k: (float(v) if isinstance(v, (np.floating, float)) else
                        int(v)   if isinstance(v, (np.integer, int)) else
                        str(v))
                    for k, v in row.items()})
    return out


# ── Run-centre lookup from model-list ─────────────────────────────────────────

def build_run_index(model_list_path):
    """
    Return a dict  run_number -> (ra_center_float, dec_center_float)
    by parsing the model-list text file.
    """
    index = {}
    with open(model_list_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                run = extract_run(line)
                ra_s, dec_s = extract_ra_dec(line)
                index[run] = (float(ra_s), float(dec_s))
            except ValueError:
                print("Cannot extract")
                continue
    return index


# ── Single-run pipeline ────────────────────────────────────────────────────────

def run_one(run_number, run_index, master_pdf, results):
    """
    Process a single injection-study run.
    Appends diagnostics to master_pdf.
    Writes results entry in-place.
    """
    print(f"\n{'='*60}")
    print(f"Run {run_number}")

    # ── Centre coordinates ────────────────────────────────────────────────────
    if run_number not in run_index:
        print(f"  [SKIP] run {run_number} not found in model list.")
        results[run_number] = {"status": "not_in_model_list"}
        return

    ra_center, dec_center = run_index[run_number]
    print(f"  ROI centre  RA={ra_center:.2f}  Dec={dec_center:.2f}")

    # ── Per-run output folder + PDF ───────────────────────────────────────────
    run_out_dir  = os.path.join(OUTPUT_DIR, "per_run", f"run_{run_number:04d}")
    os.makedirs(run_out_dir, exist_ok=True)
    run_pdf_path = os.path.join(run_out_dir, f"run_{run_number:04d}_diagnostics.pdf")
    run_pdf      = mpdf.PdfPages(run_pdf_path)

    def _save(fig, tag):
        master_pdf.savefig(fig)
        run_pdf.savefig(fig)
        plt.savefig(os.path.join(run_out_dir, tag + ".png"),
                    dpi=120, bbox_inches="tight")
        plt.close(fig)

    # ── Load injected sources ─────────────────────────────────────────────────
    # try:
    inj_df = load_injected_sources(run_number, ra_center, dec_center)
    print(f"  Injected sources : {len(inj_df)}")
    # except Exception as exc:
    #     print(f"  [WARN] Could not load injected model: {exc}")
    #     inj_df = pd.DataFrame(columns=["name", "ra", "dec", "ext"])

    # Build hotspot overlay DataFrame for make_plots
    hotspots_df = inj_df.rename(columns={"name": "Name", "ext": "ext"}) \
                        if not inj_df.empty else None

    # ── Load ALPS fitted sources ──────────────────────────────────────────────
    alps_df, alps_source = load_alps_sources(run_number)
    print(f"  ALPS source      : {alps_source}  ({len(alps_df)} sources)")

    # ── Load significance map ─────────────────────────────────────────────────
    map_path = map_fits_path(run_number)
    if not os.path.exists(map_path):
        print(f"  [SKIP] Map not found: {map_path}")
        run_pdf.close()
        results[run_number] = {
            "status":           "map_not_found",
            "ra_center":        ra_center,
            "dec_center":       dec_center,
            "injected_sources": df_to_records(inj_df),
            "alps_sources":     df_to_records(alps_df),
            "alps_fit_source":  alps_source,
            "blobs":            [],
        }
        return

    try:
        array, _, wcs, xnum, ynum, pixel_size = load_hawc_data(
            map_path, ra_center, dec_center,
            XLENGTH, YLENGTH, COORD_SYS,
        )
    except Exception as exc:
        print(f"  [ERROR] load_hawc_data failed: {exc}")
        run_pdf.close()
        results[run_number] = {
            "status":           "load_error",
            "ra_center":        ra_center,
            "dec_center":       dec_center,
            "injected_sources": df_to_records(inj_df),
            "alps_sources":     df_to_records(alps_df),
            "alps_fit_source":  alps_source,
            "blobs":            [],
            "note":             str(exc),
        }
        return

    print(f"  Map shape  : {ynum} × {xnum}  |  pixel size: {pixel_size:.4f}°")
    max_sig = float(np.max(array))
    print(f"  Max signif : {max_sig:.2f}σ")

    # ── Title page ────────────────────────────────────────────────────────────
    fig_title, ax_t = plt.subplots(figsize=(10, 3))
    ax_t.axis("off")
    ax_t.text(
        0.5, 0.5,
        (f"Run {run_number}   RA={ra_center:.2f}  Dec={dec_center:.2f}\n"
         f"Injected: {len(inj_df)}  |  ALPS ({alps_source}): {len(alps_df)}  "
         f"|  Max sig: {max_sig:.2f}σ"),
        fontsize=12, ha="center", va="center",
    )
    _save(fig_title, "00_title")

    # ── Raw significance map ──────────────────────────────────────────────────
    fig, _ = make_plots(
        array, wcs, pixel_size, coordsys="G",
        threshold=3, vmin=-5, vmax=8, contour=True,
        title=f"Significance Map — Run {run_number}",
        hotspots=hotspots_df,
        cmap="ult", figsize=(10, 6),
        labels=["4hawc"],
    )
    _save(fig, "01_raw_sigmap")

    # ── Below-threshold check ─────────────────────────────────────────────────
    if max_sig < SIG_THRESHOLD:
        print(f"  ↳ Below {SIG_THRESHOLD}σ — skipping blob detection.")
        fig_b, ax_b = plt.subplots(figsize=(8.5, 4))
        ax_b.axis("off")
        ax_b.text(0.5, 0.5,
                  f"Run {run_number}: max significance {max_sig:.2f}σ < {SIG_THRESHOLD}σ\n"
                  "Blob detection skipped.",
                  fontsize=12, ha="center", va="center")
        _save(fig_b, "01b_below_threshold")
        run_pdf.close()
        results[run_number] = {
            "status":           "below_threshold",
            "ra_center":        ra_center,
            "dec_center":       dec_center,
            "max_significance": max_sig,
            "injected_sources": df_to_records(inj_df),
            "alps_sources":     df_to_records(alps_df),
            "alps_fit_source":  alps_source,
            "blobs":            [],
        }
        return

    # ── Optional soft floor ───────────────────────────────────────────────────
    if np.min(array) < -5:
        array = soft_floor(array, floor_min=-6, scale=1.0)

    # ── Normalise ─────────────────────────────────────────────────────────────
    norm_image = (array - np.min(array)) / (np.max(array) - np.min(array))

    # ── Multi-radius blob detection ───────────────────────────────────────────
    all_blobs, all_coords, all_radii = [], [], []
    radius_results = {}   # stored in YAML per radius

    for radius in SMEAR_RADII:
        print(f"  Radius {radius:.2f}°  ({radius/pixel_size:.1f} px)")

        dog_image     = difference_of_gaussians(norm_image, 1, radius / pixel_size)
        dog_smooth    = ndi.gaussian_filter(dog_image, sigma=1.0)
        sigma_resid   = estimate_background_sigma(dog_image)

        # for noise_mult, noise_tag in [(3, ""), (2, "_lownoise")]:
        for noise_mult, noise_tag in [(2, "_lownoise"), (3, "")]:
            dog_final = np.where(dog_smooth > noise_mult * sigma_resid, dog_smooth, 0)

            # DoG diagnostic
            try:
                vmin_d, vmax_d = np.min(dog_final), np.max(dog_final)
                fig, _ = make_plots(
                    dog_final, wcs, pixel_size, coordsys="G",
                    threshold=1e-6, vmin=vmin_d, vmax=vmax_d,
                    title=f"DoG  r={radius}°{noise_tag}",
                    cmap="ult",
                )
            except Exception:
                fig, _ = make_plots(
                    dog_final, wcs, pixel_size, coordsys="G",
                    threshold=1e-6, vmin=0, vmax=0.5,
                    title=f"DoG  r={radius}°{noise_tag}",
                    cmap="ult",
                )
            _save(fig, f"03_dog_r{radius:.2f}{noise_tag}")

            raw_blobs = blob_dog(
                dog_final,
                min_sigma=0.15 / pixel_size,
                max_sigma=0.80 / pixel_size,
                threshold=0.01,
                exclude_border=50,
                overlap=0.7,
            )
            print(f"    [{noise_tag or 'normal'}] Raw blobs: {len(raw_blobs)}")

            if len(raw_blobs) > 0 or noise_tag == "":
                # Raw detection overlay
                blobs_dict = {"psblobs": raw_blobs} if len(raw_blobs) else {}
                fig, _ = make_plots(
                    dog_image, wcs, pixel_size, coordsys="G",
                    threshold=0.005,
                    blobs=blobs_dict,
                    vmin=np.min(dog_image), vmax=np.max(dog_image),
                    title=f"Raw Blobs  r={radius}°{noise_tag}",
                    cmap="ult", labels=["4hawc"],
                )
                _save(fig, f"04_raw_blobs_r{radius:.2f}{noise_tag}")

            if len(raw_blobs) > 0:
                filt_blobs, filt_coords, filt_radii = blob_filter_intensity(
                    raw_blobs, array, 5, wcs, pixel_size
                )
                print(f"    After intensity filter: {len(filt_blobs)}")

                if len(filt_blobs) > 0:
                    all_blobs.append(filt_blobs)
                    all_coords.extend(filt_coords)
                    all_radii.extend(filt_radii)

                radius_results[radius] = {
                    "n_raw":              int(len(raw_blobs)),
                    "n_after_intensity":  int(len(filt_blobs)),
                    "noise_threshold":    noise_mult,
                }
                break   # don't retry with lower threshold if we found blobs

            if noise_tag == "":
                continue   # try low-noise threshold
            radius_results[radius] = {
                "n_raw":             0,
                "n_after_intensity": 0,
                "noise_threshold":   noise_mult,
            }

    # ── Combine & deduplicate ─────────────────────────────────────────────────
    if all_blobs:
        combined_blobs  = np.vstack(all_blobs)
        combined_coords = all_coords
        combined_radii  = all_radii
    else:
        combined_blobs  = np.empty((0, 3))
        combined_coords = []
        combined_radii  = []

    final_blobs, final_coords, final_radii = remove_overlapping_blobs(
        combined_blobs, combined_coords, combined_radii
    )
    print(f"  Combined {len(combined_blobs)} → deduped {len(final_blobs)}")

    # ── Final-detection plot ──────────────────────────────────────────────────
    blobs_dict = {"psblobs": final_blobs} if len(final_blobs) > 0 else {}
    fig, _ = make_plots(
        array, wcs, pixel_size, coordsys="G",
        blobs=blobs_dict, hotspots=hotspots_df,
        title=f"Final Detections — Run {run_number}",
        cmap="ult", labels=["4hawc"],
    )
    if len(final_blobs) == 0:
        fig.axes[0].set_title(fig.axes[0].get_title() +
                              "\n(no blobs passed all cuts)", fontsize=9, color="red")
    _save(fig, "05_final_detections")

    # ── ALPS overlay plot ─────────────────────────────────────────────────────
    if not alps_df.empty:
        alps_hotspots = alps_df.rename(columns={"name": "Name"})
        fig, _ = make_plots(
            array, wcs, pixel_size, coordsys="G",
            blobs=blobs_dict, hotspots=alps_hotspots,
            title=f"ALPS ({alps_source}) Overlay — Run {run_number}",
            cmap="ult", labels=["4hawc"],
        )
        _save(fig, "06_alps_overlay")

    # ── Source table ──────────────────────────────────────────────────────────
    blob_df = convert_sources(final_coords, final_radii,
                              name_prefix="Drip", source_type="blob")

    def _table_page(df, title):
        fig_t, ax_t = plt.subplots(figsize=(12, max(2, len(df) * 0.45 + 2)))
        ax_t.axis("off")
        if not df.empty:
            tbl = ax_t.table(
                cellText=df.round(4).values,
                colLabels=df.columns,
                cellLoc="center", loc="center",
            )
            tbl.scale(1, 1.8)
        else:
            ax_t.text(0.5, 0.5, f"No sources\n{title}",
                      fontsize=11, ha="center", va="center")
        ax_t.set_title(title, fontsize=9)
        return fig_t

    _save(_table_page(blob_df,   f"Detected Blobs — Run {run_number}"), "07_blob_table")
    _save(_table_page(inj_df,    f"Injected Sources — Run {run_number}"), "08_injected_table")
    _save(_table_page(alps_df,   f"ALPS Sources ({alps_source}) — Run {run_number}"), "09_alps_table")

    run_pdf.close()

    # ── YAML record ───────────────────────────────────────────────────────────
    results[run_number] = {
        "status":           "detected" if len(final_blobs) > 0 else "no_blobs",
        "ra_center":        ra_center,
        "dec_center":       dec_center,
        "max_significance": max_sig,
        "alps_fit_source":  alps_source,
        # ── The three source lists ──────────────────────────────────────────
        "injected_sources": df_to_records(inj_df),
        "alps_sources":     df_to_records(alps_df),
        "blobs": [
            {
                "name":          row["name"],
                "ra":            row["ra"],
                "dec":           row["dec"],
                "circle_radius": row["circle_radius"],
                "sigma_radius":  row["sigma_radius"],
            }
            for _, row in blob_df.iterrows()
        ] if not blob_df.empty else [],
        # ── Per-radius breakdown ────────────────────────────────────────────
        "radius_results":   {str(k): v for k, v in radius_results.items()},
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main(run_numbers=None, max_runs=None):
    # Build / refresh model list
    print(f"Scanning {MODEL_DIR} …")
    model_files = build_model_list(MODEL_DIR, MODEL_LIST)
    print(f"  {len(model_files)} model files written to {MODEL_LIST}")

    # Count completed fits
    n_fit_dirs = sum(
        os.path.isdir(os.path.join(FIT_DIR, d))
        for d in os.listdir(FIT_DIR)
    )
    print(f"  {n_fit_dirs} fit directories found in {FIT_DIR}")

    # Build run → (RA, Dec) lookup
    run_index = build_run_index(MODEL_LIST)
    print(f"  {len(run_index)} runs indexed from model list")

    # Decide which runs to process
    if run_numbers:
        runs = [int(r) for r in run_numbers]
    else:
        runs = sorted(run_index.keys())
        if max_runs is not None:
            runs = runs[:max_runs]

    print(f"\nProcessing {len(runs)} runs: {runs[:10]}{'…' if len(runs)>10 else ''}")

    results      = {}
    master_pdf_path = os.path.join(OUTPUT_DIR, "all_diagnostics.pdf")

    with mpdf.PdfPages(master_pdf_path) as master_pdf:
        for run_number in runs:
            run_one(run_number, run_index, master_pdf, results)

            # Checkpoint YAML after every run
            with open(YAML_OUTPUT, "w") as fh:
                yaml.dump(results, fh, default_flow_style=False, sort_keys=True)

    # Final write
    with open(YAML_OUTPUT, "w") as fh:
        yaml.dump(results, fh, default_flow_style=False, sort_keys=True)

    print(f"\n{'='*60}")
    print(f"Master PDF  → {master_pdf_path}")
    print(f"Results YAML→ {YAML_OUTPUT}")

    status_counts = {}
    for v in results.values():
        s = v.get("status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1
    for s, n in sorted(status_counts.items()):
        print(f"  {s:25s}: {n}")
    print("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", type=int, default=None,
                   help="Specific run numbers to process (e.g. --runs 1 9 42)")
    p.add_argument("--max_runs", type=int, default=None,
                   help="Cap total number of runs (useful for testing)")
    args = p.parse_args()
    main(run_numbers=args.runs, max_runs=args.max_runs)