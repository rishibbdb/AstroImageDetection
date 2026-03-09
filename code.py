"""
Blob Detection Pipeline for Gamma-Ray Maps
==========================================
Loops over all FITS files in a directory, runs blob detection,
saves diagnostic plots, and writes results to a YAML file.

Output YAML structure per file:
  filename:
    source_type: PS | EXT
    true_extension: float (degrees)
    flux_fraction: float
    status: detected | below_threshold | no_blobs
    detected_sources:
      - name: DripPS0
        ra: float
        dec: float
        extension_deg: float
        circle_radius: float
        sigma_radius: float
        type: PS
      ...
    # OR if below threshold:
    detected_sources: null
    note: "Max significance below 5σ"
"""

import os
import re
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive backend for batch runs
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as mpdf
import scipy.ndimage as ndi

from astropy.stats import sigma_clip
from skimage.feature import blob_dog
from skimage.filters import difference_of_gaussians

# ── Import your project helpers ──────────────────────────────────────────────
# Adjust these imports to match your actual module structure
from your_module import (
    load_hawc_data,
    make_plots,
    find_peak,
    gal_to_cel,
    soft_floor,
    blob_filter_intensity,
    radius_to_sigma,
)

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

model_dir   = '/lustre/hawcz01/scratch/userspace/igherzog/IMAGE-analysis/sim-sources/fits-files'
output_dir  = '/lustre/hawcz01/scratch/userspace/igherzog/IMAGE-analysis/sim-sources/results'
yaml_output = os.path.join(output_dir, 'blob_detection_results.yaml')

os.makedirs(output_dir, exist_ok=True)

# Map centre & display settings
RA_CENTER  = 83.61
DEC_CENTER = 22.0
COORD_SYS  = 'C'
XLENGTH    = 4
YLENGTH    = 4

# Blob-detection radii to sweep (degrees)
SMEAR_RADII = [0.2, 0.25, 0.3, 0.4]

# Significance threshold to proceed
SIG_THRESHOLD = 5.0

# ═════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def parse_filename(fname):
    """
    Extract source type, true extension, and flux fraction from filename.

    Patterns handled
    ----------------
    PS-<flux_frac>-flux-frac-index-<idx>.fits
    EXT-<ext>-flux-frac-<flux_frac>-index-<idx>.fits
    """
    basename = os.path.basename(fname)

    # ---- Point Source -------------------------------------------------------
    ps_match = re.match(
        r'PS-([0-9.]+)-flux-frac-index-([0-9.]+)\.fits', basename
    )
    if ps_match:
        return {
            'source_type':    'PS',
            'true_extension': 0.01,            # your convention for PS
            'flux_fraction':  float(ps_match.group(1)),
            'spectral_index': float(ps_match.group(2)),
        }

    # ---- Extended Source ----------------------------------------------------
    ext_match = re.match(
        r'EXT-([0-9.]+)-flux-frac-([0-9.]+)-index-([0-9.]+)\.fits', basename
    )
    if ext_match:
        return {
            'source_type':    'EXT',
            'true_extension': float(ext_match.group(1)),
            'flux_fraction':  float(ext_match.group(2)),
            'spectral_index': float(ext_match.group(3)),
        }

    # Fallback
    return {
        'source_type':    'UNKNOWN',
        'true_extension': None,
        'flux_fraction':  None,
        'spectral_index': None,
    }


def estimate_background_sigma(image, sigma=3, maxiters=5):
    """Sigma-clipped RMS of the DoG residual map."""
    clipped = sigma_clip(image, sigma=sigma, maxiters=maxiters)
    return float(np.std(clipped.data[~clipped.mask]))


def remove_overlapping_blobs(blobs, coords, radii, overlap_threshold=0.5):
    """
    Remove duplicate blobs, keeping the one with the larger pixel radius.

    Parameters
    ----------
    blobs : ndarray (N, 3)  – [y, x, radius_px]
    coords : list of SkyCoord
    radii  : list of float  – angular radii (degrees)
    overlap_threshold : float

    Returns
    -------
    filtered_blobs, filtered_coords, filtered_radii
    """
    if len(blobs) == 0:
        return blobs, coords, radii

    # Sort largest-radius first so we always keep the bigger detection
    order          = np.argsort(blobs[:, 2])[::-1]
    sorted_blobs   = blobs[order]
    sorted_coords  = [coords[i] for i in order]
    sorted_radii   = [radii[i]  for i in order]

    keep = np.ones(len(sorted_blobs), dtype=bool)

    for i in range(len(sorted_blobs)):
        if not keep[i]:
            continue
        dist      = np.sqrt((sorted_blobs[i, 0] - sorted_blobs[:, 0])**2 +
                            (sorted_blobs[i, 1] - sorted_blobs[:, 1])**2)
        radii_sum = sorted_blobs[i, 2] + sorted_blobs[:, 2]
        overlap   = (radii_sum - dist) / np.minimum(sorted_blobs[i, 2],
                                                     sorted_blobs[:, 2])
        keep[overlap > overlap_threshold] = False
        keep[i] = True   # always keep the reference blob itself

    filtered_blobs  = sorted_blobs[keep]
    filtered_coords = [c for i, c in enumerate(sorted_coords) if keep[i]]
    filtered_radii  = [r for i, r in enumerate(sorted_radii)  if keep[i]]
    return filtered_blobs, filtered_coords, filtered_radii


def convert_sources(coords, radii, name_prefix="DripPS", source_type="PS"):
    """
    Build a DataFrame from detected SkyCoord list + radii.
    Returns an empty DataFrame (with correct columns) when coords is empty.
    """
    cols = ['Names', 'ra', 'dec', 'Circle Radius', 'Sigma Radius', 'Type']
    if len(coords) == 0:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame({
        'Names':         [f'{name_prefix}{i}' for i in range(len(coords))],
        'ra':            [c.ra.deg  for c in coords],
        'dec':           [c.dec.deg for c in coords],
        'Circle Radius': radii,
        'Sigma Radius':  [radius_to_sigma(r) for r in radii],
        'Type':          source_type,
    })
    return df


def sources_to_yaml_list(df):
    """Convert a source DataFrame to a plain-Python list suitable for yaml.dump."""
    records = []
    for _, row in df.iterrows():
        records.append({
            'name':          str(row['Names']),
            'ra':            float(row['ra']),
            'dec':           float(row['dec']),
            'circle_radius': float(row['Circle Radius']),
            'sigma_radius':  float(row['Sigma Radius']),
            'type':          str(row['Type']),
        })
    return records


# ═════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═════════════════════════════════════════════════════════════════════════════

def run_pipeline(max_files=None):
    """
    Loop over FITS files, run blob detection, save diagnostics and YAML.

    Parameters
    ----------
    max_files : int or None
        Cap the number of files processed (useful for testing).
        Pass None to process all files.
    """

    # ── Collect files ────────────────────────────────────────────────────────
    all_files = sorted([
        f for f in os.listdir(model_dir)
        if os.path.isfile(os.path.join(model_dir, f)) and f.endswith('.fits')
    ])
    if max_files is not None:
        all_files = all_files[:max_files]

    print(f"Total files to process: {len(all_files)}")

    results = {}   # dict keyed by filename, collected into YAML at the end

    # ── Per-file PDF for diagnostic plots ────────────────────────────────────
    master_pdf_path = os.path.join(output_dir, 'all_diagnostics.pdf')
    master_pdf      = mpdf.PdfPages(master_pdf_path)

    for run_num, fname in enumerate(all_files):

        filepath  = os.path.join(model_dir, fname)
        file_info = parse_filename(fname)

        print(f"\n{'='*60}")
        print(f"[{run_num+1}/{len(all_files)}] Processing: {fname}")
        print(f"  Type={file_info['source_type']}  "
              f"Ext={file_info['true_extension']}°  "
              f"Flux frac={file_info['flux_fraction']}")

        # Per-file output sub-folder and individual PDF
        file_stem    = os.path.splitext(fname)[0]
        file_out_dir = os.path.join(output_dir, 'per_file', file_stem)
        os.makedirs(file_out_dir, exist_ok=True)
        file_pdf_path = os.path.join(file_out_dir, f'{file_stem}_diagnostics.pdf')
        file_pdf      = mpdf.PdfPages(file_pdf_path)

        # ── Build source index for overlays ──────────────────────────────────
        src_index = pd.DataFrame({
            'Name': 'Sim Source',
            'ra':   RA_CENTER,
            'dec':  DEC_CENTER,
            'ext':  float(file_info['true_extension'] or 0),
        }, index=[0])

        # ── Load FITS ────────────────────────────────────────────────────────
        try:
            array, _, wcs, xnum, ynum, pixel_size = load_hawc_data(
                filepath, RA_CENTER, DEC_CENTER, XLENGTH, YLENGTH, COORD_SYS
            )
        except Exception as exc:
            print(f"  ERROR loading {fname}: {exc}")
            results[fname] = {
                **file_info,
                'status':           'load_error',
                'detected_sources': None,
                'note':             str(exc),
            }
            file_pdf.close()
            continue

        print(f"  Map shape: {ynum} × {xnum}  |  pixel size: {pixel_size:.4f}°")
        print(f"  Significance range: [{np.min(array):.2f}, {np.max(array):.2f}]")

        # ── Diagnostic plot 1 – raw significance map ─────────────────────────
        fig, ax = make_plots(
            array, wcs, pixel_size, coordsys='G',
            threshold=3, vmin=-5, vmax=6, contour=True,
            title=f'SkyMap  {fname}',
            hotspots=src_index,
            save_dir=None, pdf=False,
            cmap='ult', figsize=(10, 6),
            labels=['4afgl', 'pulsar', '4hawc', '1lhaaso'],
        )
        fig.suptitle(f'[{run_num+1}] {fname}', fontsize=8, y=0.01)
        master_pdf.savefig(fig)
        file_pdf.savefig(fig)
        plt.savefig(os.path.join(file_out_dir, '01_raw_sigmap.png'),
                    dpi=120, bbox_inches='tight')
        plt.close(fig)

        # ── Check significance threshold ──────────────────────────────────────
        if np.max(array) < SIG_THRESHOLD:
            print(f"  ↳ Below threshold ({np.max(array):.2f} < {SIG_THRESHOLD}σ). Skipping.")

            fig_blank, ax_blank = plt.subplots(figsize=(8.5, 4))
            ax_blank.axis('off')
            ax_blank.text(
                0.5, 0.5,
                f"No data > {SIG_THRESHOLD}σ in {fname}\n"
                f"(Max = {np.max(array):.2f}σ — algorithm did not proceed)",
                fontsize=12, ha='center', va='center', wrap=True,
            )
            master_pdf.savefig(fig_blank)
            file_pdf.savefig(fig_blank)
            plt.close(fig_blank)
            file_pdf.close()

            results[fname] = {
                **file_info,
                'status':           'below_threshold',
                'max_significance': float(np.max(array)),
                'detected_sources': None,
                'note':             f'Max significance {np.max(array):.2f}σ < {SIG_THRESHOLD}σ threshold',
            }
            continue

        # ── Optional soft floor ───────────────────────────────────────────────
        if np.min(array) < -5:
            print("  Image softly floored to -5σ")
            array = soft_floor(array, floor_min=-6, scale=1.0)

        # ── Normalise 0–1 ────────────────────────────────────────────────────
        image      = array
        norm_image = (image - np.min(image)) / (np.max(image) - np.min(image))

        fig, ax = make_plots(
            norm_image, wcs, pixel_size, coordsys='C',
            threshold=0.4, vmin=0, vmax=1, contour=False,
            title='Normalised Image',
            hotspots=None, save_dir=None, pdf=False,
            cmap='ult', figsize=(10, 6),
        )
        master_pdf.savefig(fig)
        file_pdf.savefig(fig)
        plt.savefig(os.path.join(file_out_dir, '02_norm_image.png'),
                    dpi=120, bbox_inches='tight')
        plt.close(fig)

        # ── Multi-radius blob detection ───────────────────────────────────────
        all_ps_blobs   = []
        all_pscoords   = []
        all_psradii    = []

        for radius in SMEAR_RADII:
            smear_radius = radius
            print(f"  Radius={smear_radius:.2f}°  "
                  f"({smear_radius/pixel_size:.1f} px)")

            dog_image    = difference_of_gaussians(norm_image, 1,
                                                   smear_radius / pixel_size)
            dog_low_noise = ndi.gaussian_filter(dog_image, sigma=1.0)
            sigma_resid   = estimate_background_sigma(dog_image)
            dog_final     = np.where(dog_low_noise > 3 * sigma_resid,
                                     dog_low_noise, 0)

            # DoG diagnostic plot
            fig, ax = make_plots(
                dog_final, wcs, pixel_size, coordsys='G',
                threshold=0.0000005,
                vmin=np.min(dog_final), vmax=np.max(dog_final),
                title=f'DoG Image  r={smear_radius}°',
                cmap='ult',
            )
            master_pdf.savefig(fig)
            file_pdf.savefig(fig)
            plt.savefig(
                os.path.join(file_out_dir,
                             f'03_dog_r{smear_radius:.2f}.png'),
                dpi=120, bbox_inches='tight',
            )
            plt.close(fig)

            # Blob detection
            ps_blobs = blob_dog(
                dog_final,
                min_sigma=0.15 / pixel_size,
                max_sigma=0.8  / pixel_size,
                threshold=0.01,
                exclude_border=50,
                overlap=0.7,
            )
            print(f"    Raw blobs found: {len(ps_blobs)}")

            # Raw-detection diagnostic plot
            blobs_dict = {'psblobs': ps_blobs} if len(ps_blobs) > 0 else {}
            fig, ax = make_plots(
                dog_image, wcs, pixel_size, coordsys='G',
                threshold=0.005,
                hotspots=None, blobs=blobs_dict,
                vmin=np.min(dog_image), vmax=np.max(dog_image),
                title=f'Raw Blob Detection  r={smear_radius}°',
                cmap='ult', labels=['4hawc'],
            )
            master_pdf.savefig(fig)
            file_pdf.savefig(fig)
            plt.savefig(
                os.path.join(file_out_dir,
                             f'04_raw_blobs_r{smear_radius:.2f}.png'),
                dpi=120, bbox_inches='tight',
            )
            plt.close(fig)

            # Intensity filter (≥5σ peak inside blob)
            ps_filt_blobs, ps_filt_coords, ps_filt_radii = (
                blob_filter_intensity(ps_blobs, array, 5, wcs, pixel_size)
                if len(ps_blobs) > 0
                else (np.array([]).reshape(0, 3), [], [])
            )
            print(f"    After intensity filter: {len(ps_filt_blobs)}")

            if len(ps_filt_blobs) > 0:
                all_ps_blobs.append(ps_filt_blobs)
                all_pscoords.append(ps_filt_coords)
                all_psradii.append(ps_filt_radii)

        # ── Combine across radii and remove overlaps ──────────────────────────
        if all_ps_blobs:
            combined_blobs  = np.vstack(all_ps_blobs)
            combined_coords = [c for sub in all_pscoords for c in sub]
            combined_radii  = [r for sub in all_psradii  for r in sub]
        else:
            combined_blobs  = np.array([]).reshape(0, 3)
            combined_coords = []
            combined_radii  = []

        final_blobs, final_coords, final_radii = remove_overlapping_blobs(
            combined_blobs, combined_coords, combined_radii
        )
        print(f"  Combined: {len(combined_blobs)} → "
              f"after dedup: {len(final_blobs)}")

        # ── Final-result diagnostic plot ──────────────────────────────────────
        if len(final_blobs) > 0:
            blobs_dict = {'psblobs': final_blobs}
            fig, ax = make_plots(
                array, wcs, pixel_size, coordsys='G',
                blobs=blobs_dict, hotspots=src_index,
                title='Final Detections (overlaps removed)',
                cmap='ult', labels=['4hawc'],
            )
        else:
            fig, ax = make_plots(
                array, wcs, pixel_size, coordsys='G',
                hotspots=src_index,
                title='Final Map – No Blobs Detected',
                cmap='ult',
            )
            ax.set_title(ax.get_title() + '\n(no blobs passed all cuts)',
                         fontsize=9, color='red')

        master_pdf.savefig(fig)
        file_pdf.savefig(fig)
        plt.savefig(os.path.join(file_out_dir, '05_final_detections.png'),
                    dpi=120, bbox_inches='tight')
        plt.close(fig)

        # ── Summary table diagnostic plot ─────────────────────────────────────
        ps_df = convert_sources(
            final_coords, final_radii,
            name_prefix="DripPS", source_type="PS",
        )
        fig_tbl, ax_tbl = plt.subplots(figsize=(10, max(2, len(ps_df) * 0.5 + 1.5)))
        ax_tbl.axis('off')
        if len(ps_df) > 0:
            tbl = ax_tbl.table(
                cellText=ps_df.round(4).values,
                colLabels=ps_df.columns,
                cellLoc='center', loc='center',
            )
            tbl.scale(1, 2)
            ax_tbl.set_title(
                f'Detected Sources — {fname}\n'
                f'True ext={file_info["true_extension"]}°  '
                f'flux_frac={file_info["flux_fraction"]}',
                fontsize=9,
            )
        else:
            ax_tbl.text(
                0.5, 0.5,
                f'No sources detected\n{fname}',
                fontsize=12, ha='center', va='center',
            )

        master_pdf.savefig(fig_tbl)
        file_pdf.savefig(fig_tbl)
        plt.savefig(os.path.join(file_out_dir, '06_summary_table.png'),
                    dpi=120, bbox_inches='tight')
        plt.close(fig_tbl)

        file_pdf.close()

        # ── Build YAML record for this file ───────────────────────────────────
        status = 'detected' if len(final_blobs) > 0 else 'no_blobs'
        results[fname] = {
            **file_info,
            'status':           status,
            'max_significance': float(np.max(array)),
            'detected_sources': sources_to_yaml_list(ps_df) if len(ps_df) > 0 else None,
        }

    # ── Finalise ──────────────────────────────────────────────────────────────
    master_pdf.close()
    print(f"\nMaster diagnostic PDF → {master_pdf_path}")

    # Write YAML
    with open(yaml_output, 'w') as fh:
        yaml.dump(results, fh, default_flow_style=False, sort_keys=True)
    print(f"Results YAML          → {yaml_output}")

    # Quick summary
    detected   = sum(1 for v in results.values() if v.get('status') == 'detected')
    no_blobs   = sum(1 for v in results.values() if v.get('status') == 'no_blobs')
    below_thr  = sum(1 for v in results.values() if v.get('status') == 'below_threshold')
    errors     = sum(1 for v in results.values() if v.get('status') == 'load_error')

    print(f"\n{'='*60}")
    print(f"Summary over {len(all_files)} files:")
    print(f"  Detected   : {detected}")
    print(f"  No blobs   : {no_blobs}")
    print(f"  Below {SIG_THRESHOLD}σ : {below_thr}")
    print(f"  Load error : {errors}")
    print('='*60)

    return results


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    # Set max_files=None to process all; use a small number for quick tests
    run_pipeline(max_files=100)