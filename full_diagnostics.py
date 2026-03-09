"""
drips_diagnostics.py
--------------------
Generate diagnostic plots from the DRIPS injection-study blob-detection results.

Reads:  blob_detection_results.yaml (output from drips_injection_pipeline.py)
Writes: Multi-panel diagnostic PNGs + merged PDF

Usage:
    python drips_diagnostics.py
    python drips_diagnostics.py --yaml custom_results.yaml
"""

import argparse
import os
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_pdf import PdfPages
import yaml

warnings.filterwarnings("ignore")

# ── Colour scheme ──────────────────────────────────────────────────────────────
BG      = "#0d1117"
FG      = "#dde2f0"
C_DET   = "#69f0ae"   # detected
C_BEL   = "#ff6b6b"   # below threshold
C_NOB   = "#ffb74d"   # no blobs (>5σ but failed detection)
C_IDX2  = "#42a5f5"   # spectral index 2.0
C_IDX3  = "#ab47bc"   # spectral index 3.0
C_INJ   = "#ffd54f"   # injected sources
C_ALPS  = "#26c6da"   # ALPS fitted sources

plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor":   BG,
    "axes.facecolor":     "#161b27",
    "axes.edgecolor":     "#21273a",
    "axes.labelcolor":    FG,
    "xtick.color":        FG,
    "ytick.color":        FG,
    "text.color":         FG,
    "grid.color":         "#21273a",
    "grid.linestyle":     ":",
    "legend.facecolor":   "#1a1e2e",
    "legend.edgecolor":   "#21273a",
    "legend.framealpha":  0.8,
})

# ── Load & parse YAML ──────────────────────────────────────────────────────────

def load_results(yaml_path):
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    
    records = []
    all_blobs    = []
    all_injected = []
    all_alps     = []
    
    for run_num, val in data.items():
        status = val.get("status", "unknown")
        max_sig = val.get("max_significance", np.nan)
        ra_c    = val.get("ra_center", np.nan)
        dec_c   = val.get("dec_center", np.nan)
        
        # Extract injected sources
        inj_list = val.get("injected_sources", [])
        for src in inj_list:
            all_injected.append({
                "run":  run_num,
                "name": src.get("name", ""),
                "ra":   src.get("ra", np.nan),
                "dec":  src.get("dec", np.nan),
                "ext":  src.get("ext", 0.0),
            })
        
        # Extract ALPS sources
        alps_list = val.get("alps_sources", [])
        for src in alps_list:
            all_alps.append({
                "run":  run_num,
                "name": src.get("name", ""),
                "ra":   src.get("ra", np.nan),
                "dec":  src.get("dec", np.nan),
                "ext":  src.get("ext", 0.0),
            })
        
        # Extract detected blobs
        blob_list = val.get("blobs", [])
        n_detected = len(blob_list)
        for blob in blob_list:
            all_blobs.append({
                "run":           run_num,
                "name":          blob.get("name", ""),
                "ra":            blob.get("ra", np.nan),
                "dec":           blob.get("dec", np.nan),
                "circle_radius": blob.get("circle_radius", 0.0),
                "sigma_radius":  blob.get("sigma_radius", 0.0),
                "status":        status,
                "max_sig":       max_sig,
                "ra_center":     ra_c,
                "dec_center":    dec_c,
            })
        
        # Per-run summary record
        records.append({
            "run":             run_num,
            "status":          status,
            "max_significance": max_sig,
            "ra_center":       ra_c,
            "dec_center":      dec_c,
            "n_injected":      len(inj_list),
            "n_alps":          len(alps_list),
            "n_detected":      n_detected,
        })
    
    return (pd.DataFrame(records),
            pd.DataFrame(all_blobs)    if all_blobs    else pd.DataFrame(),
            pd.DataFrame(all_injected) if all_injected else pd.DataFrame(),
            pd.DataFrame(all_alps)     if all_alps     else pd.DataFrame())


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Overview & detection statistics
# ══════════════════════════════════════════════════════════════════════════════

def make_fig1(df_records, df_blobs):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("DRIPS Injection Study — Overview",
                 fontsize=14, fontweight="bold", color=FG, y=0.98)
    fig.patch.set_facecolor(BG)
    
    det   = df_records[df_records["status"] == "detected"]
    below = df_records[df_records["status"] == "below_threshold"]
    nob   = df_records[df_records["status"] == "no_blobs"]
    
    # ── 1a: Max significance distribution by status ───────────────────────────
    ax = axes[0, 0]
    bins = np.linspace(0, df_records["max_significance"].max() + 2, 25)
    if not below.empty:
        ax.hist(below["max_significance"], bins=bins,
                color=C_BEL, alpha=0.75, label="Below 5σ",
                edgecolor=BG, lw=0.8)
    if not nob.empty:
        ax.hist(nob["max_significance"], bins=bins,
                color=C_NOB, alpha=0.75, label="No blobs (>5σ)",
                edgecolor=BG, lw=0.8)
    if not det.empty:
        ax.hist(det["max_significance"], bins=bins,
                color=C_DET, alpha=0.75, label="Detected",
                edgecolor=BG, lw=0.8)
    ax.axvline(5.0, color="white", lw=1.4, ls="--", label="5σ threshold", zorder=5)
    ax.set_xlabel("Max Significance (σ)")
    ax.set_ylabel("Count")
    ax.set_title("Significance Distribution by Status")
    ax.legend()
    ax.grid(True, ls=":")
    
    # ── 1b: Status breakdown bar chart ────────────────────────────────────────
    ax = axes[0, 1]
    labels = ["Detected", "Below 5σ", "No Blobs", "Other"]
    counts = [
        len(det),
        len(below),
        len(nob),
        len(df_records) - len(det) - len(below) - len(nob),
    ]
    colors = [C_DET, C_BEL, C_NOB, "#78909c"]
    bars = ax.bar(labels, counts, color=colors, width=0.6,
                  edgecolor=BG, linewidth=1.5, zorder=3)
    for bar, cnt in zip(bars, counts):
        if cnt > 0:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5,
                    str(cnt), ha="center", va="bottom",
                    color=FG, fontsize=10, fontweight="bold")
    ax.set_ylabel("Number of Runs")
    ax.set_title(f"Status Breakdown  (N={len(df_records)} runs)")
    ax.set_ylim(0, max(counts) * 1.25)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, axis="y", ls=":")
    
    # ── 1c: Number of blobs detected per run ──────────────────────────────────
    ax = axes[1, 0]
    n_det_vals = df_records["n_detected"].values
    bins_n = np.arange(-0.5, n_det_vals.max() + 1.5, 1)
    ax.hist(n_det_vals, bins=bins_n, color=C_DET, alpha=0.85,
            edgecolor=BG, lw=1.0)
    ax.set_xlabel("N blobs detected")
    ax.set_ylabel("N runs")
    ax.set_title("Distribution of Blob Count per Run")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, axis="y", ls=":")
    
    # ── 1d: Detection rate vs n_injected ──────────────────────────────────────
    ax = axes[1, 1]
    n_inj_vals = sorted(df_records["n_injected"].unique())
    det_frac = []
    for n in n_inj_vals:
        sub = df_records[df_records["n_injected"] == n]
        detected = (sub["status"] == "detected").sum()
        det_frac.append(detected / len(sub) if len(sub) > 0 else 0)
    ax.plot(n_inj_vals, det_frac, marker="o", color=C_DET,
            lw=2, ms=8, zorder=3)
    ax.fill_between(n_inj_vals, det_frac, alpha=0.15, color=C_DET)
    ax.set_xlabel("N injected sources")
    ax.set_ylabel("Detection fraction")
    ax.set_title("Detection Rate vs. Number of Injected Sources")
    ax.set_ylim(-0.05, 1.15)
    ax.axhline(1.0, color=FG, lw=0.7, ls=":", alpha=0.4)
    ax.axhline(0.5, color=FG, lw=0.7, ls=":", alpha=0.4)
    ax.grid(True, ls=":")
    
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Detected blob properties
# ══════════════════════════════════════════════════════════════════════════════

def make_fig2(df_records, df_blobs, df_injected):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Detected Blob Properties",
                 fontsize=14, fontweight="bold", color=FG, y=0.98)
    fig.patch.set_facecolor(BG)
    
    # ── 2a: RA/Dec offset from injection centre ──────────────────────────────
    ax = axes[0, 0]
    if not df_blobs.empty:
        # Compute offset in arcmin
        df_blobs["dra_arcmin"]  = ((df_blobs["ra"]  - df_blobs["ra_center"]) *
                                    np.cos(np.radians(df_blobs["dec_center"])) * 60)
        df_blobs["ddec_arcmin"] = (df_blobs["dec"] - df_blobs["dec_center"]) * 60
        
        sc = ax.scatter(df_blobs["dra_arcmin"], df_blobs["ddec_arcmin"],
                        c=df_blobs["max_sig"], cmap="plasma",
                        s=100, edgecolors="white", linewidths=0.4,
                        zorder=3, alpha=0.9)
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label("Max Significance (σ)", color=FG)
        cb.ax.yaxis.set_tick_params(color=FG)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=FG)
    
    ax.axhline(0, color="white", lw=0.8, ls=":")
    ax.axvline(0, color="white", lw=0.8, ls=":")
    ax.plot(0, 0, marker="+", color="lime", ms=14, mew=2, zorder=5,
            label="Injection ROI centre")
    ax.set_xlabel("ΔRA cos(dec) [arcmin]")
    ax.set_ylabel("ΔDec [arcmin]")
    ax.set_title("Blob Centroid Offset from ROI Centre")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    ax.grid(True, ls=":")
    
    # ── 2b: Detected circle radius distribution ───────────────────────────────
    ax = axes[0, 1]
    if not df_blobs.empty:
        bins = np.linspace(0, df_blobs["circle_radius"].max() + 0.05, 20)
        ax.hist(df_blobs["circle_radius"], bins=bins,
                color=C_DET, alpha=0.85, edgecolor=BG, lw=1.0)
        # Add vertical lines for each unique injected extension
        if not df_injected.empty:
            for ext in df_injected["ext"].unique():
                if ext > 0:
                    ax.axvline(ext, color=C_INJ, lw=1.2, ls="--",
                               alpha=0.6, label=f"Inj. ext {ext}°")
    ax.set_xlabel("Circle Radius (°)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Detected Blob Radii")
    if not df_injected.empty:
        ax.legend()
    ax.grid(True, axis="y", ls=":")
    
    # ── 2c: sigma_radius vs circle_radius (conversion check) ──────────────────
    ax = axes[1, 0]
    if not df_blobs.empty and len(df_blobs) > 1:
        ax.scatter(df_blobs["circle_radius"], df_blobs["sigma_radius"],
                   c=df_blobs["max_sig"], cmap="plasma",
                   s=90, edgecolors="white", linewidths=0.4,
                   zorder=3, alpha=0.9)
        # Fit line
        cr = df_blobs["circle_radius"].values
        sr = df_blobs["sigma_radius"].values
        m, b = np.polyfit(cr, sr, 1)
        xs = np.linspace(cr.min(), cr.max(), 100)
        ax.plot(xs, m*xs + b, color="white", lw=1.5, ls="--",
                label=f"Fit: σ = {m:.3f}·r + {b:.4f}")
        ax.legend()
    ax.set_xlabel("Circle Radius (°)")
    ax.set_ylabel("Sigma Radius (°)")
    ax.set_title("radius_to_sigma Conversion Check")
    ax.grid(True, ls=":")
    
    # ── 2d: Blobs per run vs injected sources per run ─────────────────────────
    ax = axes[1, 1]
    ax.scatter(df_records["n_injected"], df_records["n_detected"],
               s=80, color=C_DET, edgecolors="white", linewidths=0.5,
               alpha=0.75, zorder=3)
    # 1:1 line
    max_val = max(df_records["n_injected"].max(),
                  df_records["n_detected"].max())
    ax.plot([0, max_val], [0, max_val], color="white", lw=1.2, ls="--",
            alpha=0.5, label="1:1 line")
    ax.set_xlabel("N injected sources")
    ax.set_ylabel("N detected blobs")
    ax.set_title("Detected Blobs vs. Injected Sources")
    ax.legend()
    ax.grid(True, ls=":")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Sky map of all detections + injected/ALPS
# ══════════════════════════════════════════════════════════════════════════════

def make_fig3(df_blobs, df_injected, df_alps):
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor("#161b27")
    
    # ── Injected sources ──────────────────────────────────────────────────────
    if not df_injected.empty:
        ax.scatter(df_injected["ra"], df_injected["dec"],
                   s=140, marker="*", color=C_INJ,
                   edgecolors="white", linewidths=0.6,
                   label=f"Injected ({len(df_injected)})", zorder=4, alpha=0.9)
        # Draw extension circles
        for _, row in df_injected.iterrows():
            if row["ext"] > 0:
                circle = plt.Circle((row["ra"], row["dec"]), row["ext"],
                                    fill=False, edgecolor=C_INJ,
                                    lw=0.6, alpha=0.3, zorder=2)
                ax.add_patch(circle)
    
    # ── ALPS fitted sources ───────────────────────────────────────────────────
    if not df_alps.empty:
        ax.scatter(df_alps["ra"], df_alps["dec"],
                   s=100, marker="s", color=C_ALPS,
                   edgecolors="white", linewidths=0.5,
                   label=f"ALPS fitted ({len(df_alps)})", zorder=5, alpha=0.85)
    
    # ── Detected blobs ────────────────────────────────────────────────────────
    if not df_blobs.empty:
        sc = ax.scatter(df_blobs["ra"], df_blobs["dec"],
                        c=df_blobs["max_sig"], cmap="viridis",
                        s=120, edgecolors="white", linewidths=0.5,
                        label=f"Blobs ({len(df_blobs)})", zorder=6, alpha=0.9)
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label("Max Significance (σ)", color=FG)
        cb.ax.yaxis.set_tick_params(color=FG)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=FG)
        
        # Draw blob circles
        for _, row in df_blobs.iterrows():
            circle = plt.Circle((row["ra"], row["dec"]), row["circle_radius"],
                                fill=False, edgecolor="white",
                                lw=0.5, alpha=0.2, zorder=3)
            ax.add_patch(circle)
    
    ax.set_xlabel("RA (°)", labelpad=6)
    ax.set_ylabel("Dec (°)", labelpad=6)
    ax.set_title("Sky Positions — Injected · ALPS · Detected Blobs\n"
                 "Stars=injected · Squares=ALPS · Circles=blobs (colour=sig)",
                 fontsize=11)
    ax.invert_xaxis()
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, ls=":", alpha=0.5)
    
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — ALPS vs Injected comparison
# ══════════════════════════════════════════════════════════════════════════════

def make_fig4(df_records, df_injected, df_alps):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("ALPS vs. Injected Sources — Comparison",
                 fontsize=14, fontweight="bold", color=FG, y=0.98)
    fig.patch.set_facecolor(BG)
    
    # ── 4a: N ALPS vs N injected per run ──────────────────────────────────────
    ax = axes[0, 0]
    ax.scatter(df_records["n_injected"], df_records["n_alps"],
               s=80, color=C_ALPS, edgecolors="white", linewidths=0.5,
               alpha=0.75, zorder=3)
    max_val = max(df_records["n_injected"].max(),
                  df_records["n_alps"].max())
    ax.plot([0, max_val], [0, max_val], color="white", lw=1.2, ls="--",
            alpha=0.5, label="1:1 line")
    ax.set_xlabel("N injected sources")
    ax.set_ylabel("N ALPS fitted sources")
    ax.set_title("ALPS Recovery: N sources per run")
    ax.legend()
    ax.grid(True, ls=":")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # ── 4b: Extension distribution: injected vs ALPS ──────────────────────────
    ax = axes[0, 1]
    if not df_injected.empty:
        bins = np.linspace(0, max(df_injected["ext"].max(),
                                  df_alps["ext"].max() if not df_alps.empty else 0) + 0.1,
                           20)
        ax.hist(df_injected["ext"], bins=bins, color=C_INJ, alpha=0.7,
                label=f"Injected ({len(df_injected)})",
                edgecolor=BG, lw=1.0)
    if not df_alps.empty:
        ax.hist(df_alps["ext"], bins=bins, color=C_ALPS, alpha=0.7,
                label=f"ALPS ({len(df_alps)})",
                edgecolor=BG, lw=1.0)
    ax.set_xlabel("Extension (°)")
    ax.set_ylabel("Count")
    ax.set_title("Extension Distribution: Injected vs ALPS")
    ax.legend()
    ax.grid(True, axis="y", ls=":")
    
    # ── 4c: Run summary table (top 10 by n_detected) ──────────────────────────
    ax = axes[1, 0]
    ax.axis("off")
    top10 = df_records.nlargest(10, "n_detected")[
        ["run", "status", "n_injected", "n_alps", "n_detected", "max_significance"]
    ].round(2)
    if not top10.empty:
        tbl = ax.table(
            cellText=top10.values,
            colLabels=top10.columns,
            cellLoc="center", loc="center",
        )
        tbl.scale(1, 1.8)
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
    ax.set_title("Top 10 Runs by N Detected Blobs", fontsize=10, pad=10)
    
    # ── 4d: placeholder or additional metric ──────────────────────────────────
    ax = axes[1, 1]
    # Example: detection efficiency (n_detected / n_injected) histogram
    df_records["efficiency"] = np.where(
        df_records["n_injected"] > 0,
        df_records["n_detected"] / df_records["n_injected"],
        np.nan
    )
    eff = df_records["efficiency"].dropna()
    if not eff.empty:
        bins = np.linspace(0, eff.max() + 0.1, 20)
        ax.hist(eff, bins=bins, color=C_DET, alpha=0.85,
                edgecolor=BG, lw=1.0)
    ax.axvline(1.0, color="white", lw=1.2, ls="--", alpha=0.6, label="100%")
    ax.set_xlabel("Detection Efficiency (n_detected / n_injected)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Detection Efficiency per Run")
    ax.legend()
    ax.grid(True, axis="y", ls=":")
    
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--yaml", default="drips_results/blob_detection_results.yaml",
                   help="Path to blob_detection_results.yaml")
    p.add_argument("--out_pdf", default="drips_diagnostics.pdf",
                   help="Output PDF path")
    args = p.parse_args()
    
    print(f"Loading results from {args.yaml} …")
    df_records, df_blobs, df_injected, df_alps = load_results(args.yaml)
    
    print(f"  Runs:     {len(df_records)}")
    print(f"  Blobs:    {len(df_blobs)}")
    print(f"  Injected: {len(df_injected)}")
    print(f"  ALPS:     {len(df_alps)}")
    
    print("\nGenerating figures…")
    fig1 = make_fig1(df_records, df_blobs)
    fig2 = make_fig2(df_records, df_blobs, df_injected)
    fig3 = make_fig3(df_blobs, df_injected, df_alps)
    fig4 = make_fig4(df_records, df_injected, df_alps)
    
    # Save individual PNGs
    out_dir = os.path.dirname(args.out_pdf) or "."
    os.makedirs(out_dir, exist_ok=True)
    
    fig1.savefig(os.path.join(out_dir, "diag_fig1_overview.png"),
                 dpi=150, bbox_inches="tight", facecolor=BG)
    fig2.savefig(os.path.join(out_dir, "diag_fig2_blobs.png"),
                 dpi=150, bbox_inches="tight", facecolor=BG)
    fig3.savefig(os.path.join(out_dir, "diag_fig3_skymap.png"),
                 dpi=150, bbox_inches="tight", facecolor=BG)
    fig4.savefig(os.path.join(out_dir, "diag_fig4_alps_vs_inj.png"),
                 dpi=150, bbox_inches="tight", facecolor=BG)
    
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)
    
    print(f"  → PNG files saved to {out_dir}")
    
    # Merge into PDF
    print(f"\nMerging into {args.out_pdf} …")
    with PdfPages(args.out_pdf) as pdf:
        for figpath, title in [
            ("diag_fig1_overview.png",      "Fig 1 · Overview & Detection Statistics"),
            ("diag_fig2_blobs.png",         "Fig 2 · Detected Blob Properties"),
            ("diag_fig3_skymap.png",        "Fig 3 · Sky Map: Injected · ALPS · Blobs"),
            ("diag_fig4_alps_vs_inj.png",   "Fig 4 · ALPS vs Injected Comparison"),
        ]:
            full_path = os.path.join(out_dir, figpath)
            if not os.path.exists(full_path):
                continue
            import matplotlib.image as mpimg
            img = mpimg.imread(full_path)
            fig, ax = plt.subplots(figsize=(14, 10))
            fig.patch.set_facecolor(BG)
            ax.imshow(img)
            ax.axis("off")
            fig.text(0.5, 0.01, title, ha="center", fontsize=8,
                     color="#888", fontfamily="monospace")
            pdf.savefig(fig, facecolor=BG, bbox_inches="tight")
            plt.close(fig)
    
    print(f"PDF saved → {args.out_pdf}")
    print("Done.")


if __name__ == "__main__":
    main()