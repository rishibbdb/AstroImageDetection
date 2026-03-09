import numpy as np
import sys, os, re
sys.path.append(os.path.abspath(".."))

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import matplotlib.backends.backend_pdf as mpdf

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.stats import sigma_clip
import astropy.wcs.utils as astropy_utils

from scipy.ndimage import gaussian_filter
from skimage.filters import difference_of_gaussians
from skimage.feature import blob_dog
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import yaml
import pandas as pd
import time
from helpers2 import *

milagrotextcolor, milagro   = setupMilagroColormap(-3, 15, 2, 256)
milagrotextcolor2, milagro2 = setupMilagroColormap(0.2, 1, 2, 256)


# ── Paths & pipeline constants ────────────────────────────────────────────────
MODEL_DIR   = '/lustre/hawcz01/scratch/userspace/igherzog/IMAGE-analysis/sim-sources/fits-files'
OUTPUT_DIR  = '//lustre/hawcz01/scratch/userspace/rbabu/astroimage/AstroImageDetection/new_results'

os.makedirs(OUTPUT_DIR, exist_ok=True)

RA_CENTER     = 83.61
DEC_CENTER    = 22.0
COORD_SYS     = 'C'
XLENGTH       = 2
YLENGTH       = 2
SMEAR_RADII   = [0.25, 0.3, 0.4, 0.5]
SIG_THRESHOLD = 5.0
DIAGNOSTICS   = False
SAVE_PDF      = True


# ── Helper functions ──────────────────────────────────────────────────────────
def parse_filename(fname):
    """
    Extract source type, true extension, and flux fraction from filename.

    Patterns handled
    ----------------
    PS-<flux_frac>-flux-frac-index-<idx>.fits
    EXT-<ext>-flux-frac-<flux_frac>-index-<idx>.fits
    """
    basename = os.path.basename(fname)

    ps_match = re.match(r'PS-([0-9.]+)-flux-frac-index-([0-9.]+)\.fits', basename)
    if ps_match:
        return {
            'source_type':    'PS',
            'true_extension': 0.01,
            'flux_fraction':  float(ps_match.group(1)),
            'spectral_index': float(ps_match.group(2)),
        }

    ext_match = re.match(r'EXT-([0-9.]+)-flux-frac-([0-9.]+)-index-([0-9.]+)\.fits', basename)
    if ext_match:
        return {
            'source_type':    'EXT',
            'true_extension': float(ext_match.group(1)),
            'flux_fraction':  float(ext_match.group(2)),
            'spectral_index': float(ext_match.group(3)),
        }

    return {'source_type': 'UNKNOWN', 'true_extension': None,
            'flux_fraction': None,    'spectral_index': None}


def estimate_background_sigma(image, sigma=3, maxiters=5):
    """Sigma-clipped RMS of the DoG residual map."""
    clipped = sigma_clip(image, sigma=sigma, maxiters=maxiters)
    return float(np.std(clipped.data[~clipped.mask]))


def combine_blobs(all_blobs, all_coords, all_radii):
    """Flatten per-radius blob lists into single combined arrays."""
    if all_blobs:
        return (
            np.vstack(all_blobs),
            [c for sub in all_coords for c in sub],
            [r for sub in all_radii  for r in sub],
        )
    return np.empty((0, 3)), [], []


def run_ps(dog_final, pixel_size, threshold_val):
    return blob_dog(dog_final,
                    min_sigma=0.15 / pixel_size, max_sigma=0.39 / pixel_size,
                    threshold=threshold_val, exclude_border=50, overlap=0.7)


def run_ext(extmap, pixel_size, threshold_val):
    return blob_dog(extmap,
                    min_sigma=0.5 / pixel_size, max_sigma=1.0 / pixel_size,
                    threshold=threshold_val, exclude_border=50, overlap=0.7)


def compute_bright_frac(image, ly, lx, lr):
    """Fraction of pixels within blob circle brighter than the center pixel."""
    y_min, y_max = int(max(ly - lr, 0)), int(min(ly + lr, image.shape[0]))
    x_min, x_max = int(max(lx - lr, 0)), int(min(lx + lr, image.shape[1]))
    yy, xx       = np.mgrid[y_min:y_max, x_min:x_max]
    mask_circle  = np.sqrt((yy - ly)**2 + (xx - lx)**2) <= lr
    mask_bright  = mask_circle & (image[y_min:y_max, x_min:x_max] > 5)
    return mask_bright.sum() / mask_circle.sum()


def overlap_fraction(ly, lx, lr, sy, sx, sr):
    """Fraction of smaller blob area overlapping with the larger blob."""
    dist = np.sqrt((sy - ly)**2 + (sx - lx)**2)
    if dist >= lr + sr:
        return 0.0
    if dist + sr <= lr:
        return 1.0
    r, R, d  = sr, lr, dist
    alpha    = np.arccos(np.clip((d**2 + r**2 - R**2) / (2*d*r), -1, 1))
    beta     = np.arccos(np.clip((d**2 + R**2 - r**2) / (2*d*R), -1, 1))
    intersection = (r**2 * alpha + R**2 * beta
                    - 0.5 * (r**2 * np.sin(2*alpha) + R**2 * np.sin(2*beta)))
    return intersection / (np.pi * r**2)


def calculate_separation(coord1, coord2):
    """Angular separation in degrees between two SkyCoord objects."""
    return coord1.separation(coord2).deg


def circle_overlap(coord1, r1, coord2, r2, pixel_size):
    """Classify geometric relationship between two sky circles."""
    dist      = calculate_separation(coord1, coord2)
    radii_sum = (r1 + r2) * pixel_size
    radii_dif = abs(r1 - r2) * pixel_size
    if dist > radii_sum and dist > radii_dif:
        return 0.0    # disjoint
    elif dist < radii_dif:
        return 1.0    # one inside the other
    return None       # partial overlap


def blob_to_yaml_record(blob, array, wcs, pixel_size, label):
    """Convert a single blob row to a serialisable dict."""
    y, x, r = blob
    y, x    = int(y), int(x)
    coord   = astropy_utils.pixel_to_skycoord(x, y, wcs=wcs).icrs
    return {
        'label':      label,
        'x_px':       x,
        'y_px':       y,
        'radius_px':  float(r),
        'radius_deg': float(r * pixel_size),
        'l_deg':      float(coord.ra.deg),
        'b_deg':      float(coord.dec.deg),
        'center_ts':  float(array[y, x]),
    }


def serialise_group(group, array, wcs, pixel_size, label):
    """Convert a blob group list to YAML-ready records."""
    records = []
    for item in group:
        try:
            blob = item if (hasattr(item, '__len__') and len(item) == 3
                            and not hasattr(item[0], '__len__')) else item[0][0]
        except (TypeError, ValueError, IndexError):
            blob = item
        records.append(blob_to_yaml_record(blob, array, wcs, pixel_size, label))
    return records


def plot_blob_map(array, wcs, xnum, ynum, kept_ext, kept_ps, ax_title):
    """Overlay blobs on the significance map and return the figure."""
    fig = plt.figure(figsize=(12, 8))
    ax  = fig.add_subplot(111, projection=wcs)
    im  = ax.imshow(array, cmap='magma', vmin=-5, vmax=15)
    plt.colorbar(im, ax=ax, label=r'Significance ($\sigma$)',
                 fraction=0.046, pad=0.04)

    for lb in kept_ext:
        ly, lx, lr = lb
        ly, lx = int(ly), int(lx)
        ax.add_patch(plt.Circle((lx, ly), lr, fill=False,
                                edgecolor='white', linewidth=2, linestyle='--'))
        ax.scatter(lx, ly, s=80, c='black', marker='*', zorder=5)
        ax.text(lx, ly + lr + 10, f'TS={array[ly,lx]:.1f}',
                color='white', fontsize=9, ha='center')

    for sb in kept_ps:
        try:
            sy, sx, sr = sb
        except (TypeError, ValueError):
            sy, sx, sr = sb[0][0]
        sy, sx = int(sy), int(sx)
        ax.add_patch(plt.Circle((sx, sy), sr, fill=False,
                                edgecolor='cyan', linewidth=2, linestyle='-'))
        ax.text(sx + sr + 20, sy + sr + 10, f'TS={array[sy,sx]:.1f}',
                color='black', fontsize=9, ha='center')

    ax.legend(handles=[
        Line2D([0],[0], color='white', linewidth=2,   linestyle='--', label='Extended blob'),
        Line2D([0],[0], color='cyan',  linewidth=1.5, linestyle='-',  label='Point source blob'),
    ], fontsize=8, loc='upper right')
    ax.set_xlim(0, xnum);  ax.set_ylim(0, ynum)
    ax.set_title(ax_title, fontsize=9)
    ax.set_xlabel('X (px)');  ax.set_ylabel('Y (px)')
    plt.tight_layout()
    return fig

def deduplicate_ps_group(ps_filtered_group, wcs, sep_threshold_deg=0.4):
    """
    Remove smaller PS blobs from ps_filtered_group when two blobs are
    within sep_threshold_deg of each other, keeping the larger one.

    Parameters
    ----------
    ps_filtered_group : list of (y, x, r) tuples
    array             : significance map (for TS lookup)
    wcs               : WCS object
    pixel_size        : degrees per pixel
    sep_threshold_deg : angular separation threshold in degrees

    Returns
    -------
    kept    : list of blobs that survived
    removed : list of blobs that were culled
    """
    if len(ps_filtered_group) <= 1:
        return list(ps_filtered_group), []

    # Normalise entries — ps_filtered_group can contain raw tuples or
    # nested [(blob, frac)] lists from the grouping loop
    blobs = []
    for item in ps_filtered_group:
        try:
            y, x, r = item
            blobs.append((y, x, r))
        except (TypeError, ValueError):
            y, x, r = item[0][0]
            blobs.append((y, x, r))

    # Sort largest-radius first so we always keep the bigger detection
    blobs = sorted(blobs, key=lambda b: b[2], reverse=True)

    keep_mask = [True] * len(blobs)

    for i in range(len(blobs)):
        if not keep_mask[i]:
            continue
        yi, xi, ri = blobs[i]
        coord_i    = astropy_utils.pixel_to_skycoord(int(xi), int(yi), wcs=wcs).galactic

        for j in range(i + 1, len(blobs)):
            if not keep_mask[j]:
                continue
            yj, xj, rj = blobs[j]
            coord_j     = astropy_utils.pixel_to_skycoord(int(xj), int(yj), wcs=wcs).galactic
            sep         = calculate_separation(coord_i, coord_j)
            print(f" Separation between points = {sep}")
            if sep < sep_threshold_deg:
                print(f"  PS dedup: removing smaller blob @ ({xj},{yj}) r={rj:.1f}px "
                      f"sep={sep:.3f}° < {sep_threshold_deg}° threshold")
                keep_mask[j] = False

    kept    = [b for b, k in zip(blobs, keep_mask) if     k]
    removed = [b for b, k in zip(blobs, keep_mask) if not k]
    print(f"  PS dedup: {len(blobs)} → {len(kept)} kept, {len(removed)} removed")
    return kept, removed

# ── Per-file processing ───────────────────────────────────────────────────────
def process_file(fname, run_num, n_total, master_pdf):
    """
    Run the full blob-detection and grouping pipeline on a single FITS file.

    Returns a tuple of (kept_record, removed_record) dicts for YAML output,
    or None for both on load failure.
    """
    filepath  = os.path.join(MODEL_DIR, fname)
    file_info = parse_filename(fname)

    print(f"\n{'='*60}")
    print(f"[{run_num+1}/{n_total}] {fname}")
    print(f"  Type={file_info['source_type']}  "
          f"Ext={file_info['true_extension']}°  "
          f"Flux frac={file_info['flux_fraction']}")

    # Per-file output folder + PDF
    file_stem     = os.path.splitext(fname)[0]
    file_out_dir  = os.path.join(OUTPUT_DIR, 'per_file', file_stem)
    os.makedirs(file_out_dir, exist_ok=True)
    file_pdf      = mpdf.PdfPages(os.path.join(file_out_dir,
                                               f'{file_stem}_diagnostics.pdf'))

    def _save(fig, name):
        master_pdf.savefig(fig)
        file_pdf.savefig(fig)
        plt.savefig(os.path.join(file_out_dir, name), dpi=120, bbox_inches='tight')
        plt.close(fig)

    # ── Load FITS ─────────────────────────────────────────────────────────────
    try:
        array, _, wcs, xnum, ynum, pixel_size = load_hawc_data(
            filepath, RA_CENTER, DEC_CENTER, XLENGTH, YLENGTH, COORD_SYS
        )
    except Exception as exc:
        print(f"  ERROR loading {fname}: {exc}")
        file_pdf.close()
        return (
            {**file_info, 'status': 'load_error', 'ps_kept':  None, 'ext_kept':  None, 'note': str(exc)},
            {**file_info, 'status': 'load_error', 'ps_removed': None, 'ext_removed': None, 'note': str(exc)},
        )

    print(f"  Map shape  : {ynum} × {xnum}  |  pixel size: {pixel_size:.4f}°")
    print(f"  Sig range  : [{np.min(array):.2f}, {np.max(array):.2f}]")

    src_index = pd.DataFrame({
        'Name': 'Sim Source', 'ra': RA_CENTER,
        'dec': DEC_CENTER, 'ext': float(file_info['true_extension'] or 0),
    }, index=[0])

    # ── Raw map plot ──────────────────────────────────────────────────────────
    max_signif = find_peak(array, wcs)
    fig, _ = make_plots(
        array, wcs, pixel_size, coordsys='G',
        threshold=3, vmin=-5, vmax=15, contour=True,
        title=f'SkyMap (Max Sig {max_signif})',
        hotspots=src_index, save_dir=None, pdf=False,
        cmap='ult', figsize=(10, 6),
        labels=['4afgl', 'pualsar', '4hawc', '1lhaaaso'],
    )
    _save(fig, '01_raw_sigmap.png')

    # ── Significance threshold check ──────────────────────────────────────────
    if np.max(array) < SIG_THRESHOLD:
        print(f"    Below threshold ({np.max(array):.2f} < {SIG_THRESHOLD}sigma) — skipping.")
        fig_blank, ax_blank = plt.subplots(figsize=(8.5, 4))
        ax_blank.axis('off')
        ax_blank.text(0.5, 0.5,
                      f"No data > {SIG_THRESHOLD}sigma in {fname}\n"
                      f"(Max = {np.max(array):.2f}sigma)",
                      fontsize=12, ha='center', va='center')
        _save(fig_blank, '01b_below_threshold.png')
        file_pdf.close()
        record = {**file_info, 'status': 'below_threshold',
                  'max_significance': float(np.max(array))}
        return ({**record, 'ps_kept': None, 'ext_kept': None},
                {**record, 'ps_removed': None, 'ext_removed': None})

    # ── Optional soft floor & normalise ───────────────────────────────────────
    if np.min(array) < -5:
        print("  Image softly floored to -5sigma")
        array = soft_floor(array, floor_min=-6, scale=1.0)

    norm_image = (array - array.min()) / (array.max() - array.min())

    fig, _ = make_plots(
        norm_image, wcs, pixel_size, coordsys='C',
        threshold=0.2, vmin=0, vmax=1, contour=False,
        title='Normalised Image', cmap='ult', figsize=(10, 6),
    )
    _save(fig, '02_norm_image.png')

    # ── Multi-radius blob detection ───────────────────────────────────────────
    all_ps_blobs,  all_ps_coords,  all_ps_radii  = [], [], []
    all_ext_blobs, all_ext_coords, all_ext_radii = [], [], []
    threshold_val = 0.001 if np.max(array) > 100 else 0.01

    for radius in tqdm(SMEAR_RADII, desc=f'  Radii [{file_stem[:20]}]', leave=False):
        pixel_smear = radius / pixel_size
        dog_image   = difference_of_gaussians(norm_image, 1, pixel_smear)
        sigma_resid = estimate_background_sigma(dog_image)
        dog_final   = np.where(dog_image > 2.5 * sigma_resid, dog_image, 0)
        dog_norm    = (dog_final - dog_final.min()) / (dog_final.max() - dog_final.min() + 1e-12)
        extmap      = gaussian_filter(norm_image - dog_norm, sigma=0.3 / pixel_size)

        with ThreadPoolExecutor(max_workers=2) as ex:
            f_ps  = ex.submit(run_ps,  dog_final, pixel_size, threshold_val)
            f_ext = ex.submit(run_ext, extmap,    pixel_size, threshold_val)
        ps_blobs, ext_blobs = f_ps.result(), f_ext.result()

        if DIAGNOSTICS:
            fig, _ = make_plots(
                dog_final, wcs, pixel_size, coordsys='G', threshold=0.005,
                blobs={'psblobs': ps_blobs, 'extblobs': ext_blobs},
                vmin=dog_image.min(), vmax=dog_image.max(),
                title=f'Raw detection r={radius}°', cmap='ult',
            )
            _save(fig, f'03_dog_r{radius:.2f}.png')

        ps_filt, ps_coords, ps_radii = (
            blob_filter_intensity(ps_blobs,  array, 5, wcs, pixel_size)
            if len(ps_blobs)  > 0 else (np.empty((0,3)), [], [])
        )
        ext_filt, ext_coords, ext_radii = (
            blob_filter_intensity(ext_blobs, array, 5, wcs, pixel_size)
            if len(ext_blobs) > 0 else (np.empty((0,3)), [], [])
        )

        if len(ps_filt)  > 0:
            all_ps_blobs.append(ps_filt);   all_ps_coords.append(ps_coords);   all_ps_radii.append(ps_radii)
        if len(ext_filt) > 0:
            all_ext_blobs.append(ext_filt); all_ext_coords.append(ext_coords); all_ext_radii.append(ext_radii)

    # ── Combine & dedup ───────────────────────────────────────────────────────
    combined_ps_blobs,  combined_ps_coords,  combined_ps_radii  = combine_blobs(all_ps_blobs,  all_ps_coords,  all_ps_radii)
    combined_ext_blobs, combined_ext_coords, combined_ext_radii = combine_blobs(all_ext_blobs, all_ext_coords, all_ext_radii)

    final_ps_blobs,  final_ps_coords,  final_ps_radii,  *_ = remove_overlapping_blobs(combined_ps_blobs,  combined_ps_coords,  combined_ps_radii)
    final_ext_blobs, final_ext_coords, final_ext_radii, *_ = remove_overlapping_blobs(combined_ext_blobs, combined_ext_coords, combined_ext_radii)

    final_ps_blobs,  final_ps_coords,  final_ps_radii  = blob_filter_intensity(final_ps_blobs,  array, 5, wcs, pixel_size)
    final_ext_blobs, final_ext_coords, final_ext_radii = blob_filter_intensity(final_ext_blobs, array, 5, wcs, pixel_size)

    print(f"  PS  after all cuts: {len(final_ps_blobs):>4}")
    print(f"  EXT after all cuts: {len(final_ext_blobs):>4}")

    # ── Generate Overlap Plots ────────────────────────────────────────


    # ── Group PS blobs under EXT blobs ────────────────────────────────────────
    groups, matched_ps = [], set()
    ps_plot  = final_ps_blobs  if len(final_ps_blobs)  > 0 else np.zeros((1, 3))
    ext_plot = final_ext_blobs if len(final_ext_blobs) > 0 else np.zeros((1, 3))
    all_blobs = np.vstack([ps_plot, ext_plot])

    fig, ax = plt.subplots(figsize=(8, 14))

    for blob in final_ext_blobs:
        cy, cx, r = blob
        ax.add_patch(patches.Circle((cx, cy), r, linewidth=1.5,
                                    edgecolor='steelblue', facecolor='steelblue', alpha=0.3))

    for blob in final_ps_blobs:
        cy, cx, r = blob
        ax.add_patch(patches.Circle((cx, cy), r, linewidth=1.5,
                                    edgecolor='tomato', facecolor='tomato', alpha=0.5))

    legend_handles = [
        patches.Patch(facecolor='steelblue', edgecolor='steelblue', alpha=0.5,
                    label=f'Larger blobs (n={len(final_ext_blobs)})'),
        patches.Patch(facecolor='tomato',    edgecolor='tomato',    alpha=0.5,
                    label=f'Smaller blobs (n={len(final_ps_blobs)})'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=9)

    margin = 150
    ax.set_ylim(all_blobs[:, 0].min() - margin, all_blobs[:, 0].max() + margin)
    ax.set_xlim(all_blobs[:, 1].min() - margin, all_blobs[:, 1].max() + margin)
    ax.set_aspect('equal')
    ax.set_xlabel('Center X (px)')
    ax.set_ylabel('Center Y (px)')
    ax.set_title('Overlap Visualization — Smaller (red) vs Larger (blue)')
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    _save(fig, 'blob_overlap.png')

    for lb in final_ext_blobs:
        ly, lx, lr = lb
        matched = []
        for i, sb in enumerate(final_ps_blobs):
            sy, sx, sr = sb
            if overlap_fraction(ly, lx, lr, sy, sx, sr) > 0.1:
                matched.append((sb, overlap_fraction(ly, lx, lr, sy, sx, sr)))
                matched_ps.add(i)
        groups.append((lb, matched))
    print(f"Matched {(groups)} groups of overlapping PS and EXT")
    for i, sb in enumerate(final_ps_blobs):
        if i not in matched_ps:
            groups.append((None, [(sb, 0.0)]))

    ps_filtered_group, ext_filtered_group = [], []
    ps_removed_group,  ext_removed_group  = [], []


    for lb, sbs in groups:
        tag_ps     = 0
        tag_ex     = 0
        ps_flagged = []
        if lb is None:
            for sb, _ in sbs:
                ps_filtered_group.append(sb)
            continue

        ly, lx, lr = lb
        ly, lx     = int(ly), int(lx)

        bright_frac = compute_bright_frac(array, ly, lx, lr)
        print(f"Intensity Fraction of pixels greater than 5 sigma detection threshold = {100*bright_frac:.1f}%")

        if bright_frac < 0.5:
            print("Larger blob is artifact of blob detection on subtracted map")
            tag_ex += 1
            for sb, _ in sbs:
                ps_flagged.append(sb)
        else:
            coord_lb   = astropy_utils.pixel_to_skycoord(lx, ly, wcs=wcs).galactic
            if len(sbs) == 0:
                ext_filtered_group.append(lb)
                continue
            print(f"Larger blob coord = {coord_lb.l.deg, coord_lb.b.deg}")
            if len(sbs) > 0:
                ext_filtered_group.append(lb)
                continue
            for sb, _ in sbs:
                sy, sx, sr  = sb
                sy, sx      = int(sy), int(sx)
                coord_sb    = astropy_utils.pixel_to_skycoord(sx, sy, wcs=wcs).galactic
                sep_deg     = calculate_separation(coord_lb, coord_sb)
                ovl         = circle_overlap(coord_lb, lr, coord_sb, sr, pixel_size)
                delta_ts    = float(array[ly, lx]) - float(array[sy, sx])

                if sep_deg < 0.1:
                    tag_ex +=1
                    ps_flagged.append(sb)
                    continue
                if ovl is None:
                    if sep_deg > 0.3:
                        tag_ps += 1
                        ps_flagged.append(sb)
                    else:
                        if delta_ts < 5:
                            tag_ex+=1
                            ps_flagged.append(sb)
                        else:
                            ps_removed_group.append(sb)
                elif ovl == 1.0:
                    if sep_deg > 0.3 and delta_ts >= 9:
                        tag_ps += 1;  tag_ex += 1;  ps_flagged.append(sb)
                    elif delta_ts >= 9:
                        tag_ps += 1;  ps_flagged.append(sb)
                    else:
                        ps_removed_group.append(sb)

        if tag_ex > 0:
            ext_removed_group.append(lb)
        else:
            ext_filtered_group.append(lb)
        ps_filtered_group.extend(ps_flagged)

    print(f"  Kept   — PS: {len(ps_filtered_group)}  EXT: {len(ext_filtered_group)}")
    print(f"  Removed— PS: {len(ps_removed_group)}  EXT: {len(ext_removed_group)}")

    ps_filtered_group, ps_dedup_removed = deduplicate_ps_group(
    ps_filtered_group, wcs, sep_threshold_deg=0.3
    )
    ps_removed_group.extend(ps_dedup_removed)

    print(f"Final — PS kept: {len(ps_filtered_group)}  EXT kept: {len(ext_filtered_group)}")
    print(f"        PS removed: {len(ps_removed_group)}  EXT removed: {len(ext_removed_group)}")
    # ── Diagnostic plots ──────────────────────────────────────────────────────
    fig_kept    = plot_blob_map(array, wcs, xnum, ynum,
                                ext_filtered_group, ps_filtered_group,
                                ax_title=f'Kept blobs — {fname}')
    _save(fig_kept,    '04_kept_blobs.png')
    fig_removed = plot_blob_map(array, wcs, xnum, ynum,
                                ext_removed_group, ps_removed_group,
                                ax_title=f'Removed blobs — {fname}')
    _save(fig_removed, '05_removed_blobs.png')

    file_pdf.close()

    # ── Build YAML records ────────────────────────────────────────────────────
    kept_record = {
        **file_info,
        'status':        'detected' if (ps_filtered_group or ext_filtered_group) else 'no_blobs',
        'max_significance': float(np.max(array)),
        'ps_kept':       serialise_group(ps_filtered_group,  array, wcs, pixel_size, 'PS'),
        'ext_kept':      serialise_group(ext_filtered_group, array, wcs, pixel_size, 'EXT'),
    }
    removed_record = {
        **file_info,
        'status':        kept_record['status'],
        'max_significance': float(np.max(array)),
        'ps_removed':    serialise_group(ps_removed_group,  array, wcs, pixel_size, 'PS'),
        'ext_removed':   serialise_group(ext_removed_group, array, wcs, pixel_size, 'EXT'),
    }
    return kept_record, removed_record


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(start_file=0, max_files=None):
    """
    Loop over FITS files, run blob detection, save diagnostics and YAML.

    Parameters
    ----------
    start_file : int
        Index of the first file to process (default 0).
    max_files : int or None
        Index of the last file to process, exclusive (default: all files).
        E.g. start_file=10, max_files=20 processes files 10–19.
    """
    print(f"Scanning model directory: {MODEL_DIR}")
    all_files = sorted([
        f for f in os.listdir(MODEL_DIR)
        if os.path.isfile(os.path.join(MODEL_DIR, f)) and f.endswith('.fits')
    ])
    if max_files is not None:
        try:
            all_files = all_files[start_file:start_file+max_files]
        except:
            print(f"Max number files to process exceed total file length, processing from start file to end")
            all_files = all_files[start_file:]
    print(f"Total files to process: {len(all_files)}  "
          f"(indices {start_file}-{(max_files or len(all_files)+start_file)-1})")

    master_pdf_path = os.path.join(OUTPUT_DIR, 'all_diagnostics.pdf')
    master_pdf      = mpdf.PdfPages(master_pdf_path)

    kept_results    = {}
    removed_results = {}
    total_time = 0.0
    for run_num, fname in enumerate(all_files):
        file_start = time.perf_counter()
        kept_rec, removed_rec = process_file(
            fname, run_num, len(all_files), master_pdf
        )
        kept_results[fname]    = kept_rec
        removed_results[fname] = removed_rec

        # Write YAML incrementally so progress is never lost
        with open(os.path.join(OUTPUT_DIR, 'blob_detection_kept.yaml'),    'w') as fh:
            yaml.dump(kept_results,    fh, default_flow_style=False, sort_keys=False)
        with open(os.path.join(OUTPUT_DIR, 'blob_detection_removed.yaml'), 'w') as fh:
            yaml.dump(removed_results, fh, default_flow_style=False, sort_keys=False)
        file_end = time.perf_counter()
        print(f"  Completed in {file_end - file_start:.1f} seconds")
        total_time += file_end - file_start
    master_pdf.close()

    # ── Final summary ─────────────────────────────────────────────────────────
    statuses  = [v.get('status') for v in kept_results.values()]
    print(f"\n{'='*60}")
    print(f"Total processing time: {total_time:.1f} seconds")
    print(f"Summary over {len(all_files)} files:")
    print(f"  Detected       : {statuses.count('detected')}")
    print(f"  No blobs       : {statuses.count('no_blobs')}")
    print(f"  Below {SIG_THRESHOLD}sigma      : {statuses.count('below_threshold')}")
    print(f"  Load errors    : {statuses.count('load_error')}")
    print(f"  Master PDF     → {master_pdf_path}")
    print(f"  Kept YAML      → {os.path.join(OUTPUT_DIR, 'blob_detection_kept.yaml')}")
    print(f"  Removed YAML   → {os.path.join(OUTPUT_DIR, 'blob_detection_removed.yaml')}")
    
    print('='*60)

    return kept_results, removed_results


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Examples:
    #   run_pipeline()                  → all files
    #   run_pipeline(max_files=10)      → first 10 files
    #   run_pipeline(start_file=50, max_files=60)  → files 50–59
    # run_pipeline(start_file=40, max_files=100)
    run_pipeline(start_file=100)#, max_files=100)