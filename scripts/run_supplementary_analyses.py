#!/usr/bin/env python
"""Supplementary analyses for JGP submission.

Generates:
1. Grid convergence table (dx sweep at 3 resolutions)
2. Eta sensitivity sweep table + figure
3. Wavefront snapshot figure
4. Paired statistical tests on existing grid data
5. Formatted supplementary results table

Outputs to outputs/supplementary/
"""
from __future__ import annotations

import csv
import dataclasses as dc
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "figure.titlesize": 14,
})

from csd_sulcus.analysis import (
    electrode_speed_mm_min,
    fixed_scale_from_control,
    arrival_speed_map_mm_min,
    compartment_field_medians,
)
from csd_sulcus.model import Params, run_simulation

E1 = (0.43, 0.50)
E2 = (0.62, 0.50)
R_MM = 1.0
REP_WIDTH = 4.0
REP_GMIN = 0.75

# Physical domain of the manuscript-scale simulation
LX_MM = 19.9  # (200-1)*0.1
LY_MM = 13.9  # (140-1)*0.1


def _save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------------ #
# 1. Grid Convergence
# ------------------------------------------------------------------ #
def grid_convergence(output_dir: Path) -> list[dict]:
    """Run the representative case at multiple spatial resolutions."""
    print("\n=== Grid Convergence Study ===")
    resolutions = [
        {"dx": 0.20, "dt": 0.004},
        {"dx": 0.10, "dt": 0.002},
        {"dx": 0.05, "dt": 0.001},
    ]
    results = []
    for res in resolutions:
        dx = res["dx"]
        dt_val = res["dt"]
        nx = int(round(LX_MM / dx)) + 1
        ny = int(round(LY_MM / dx)) + 1
        cfl = 0.01 * dt_val / (dx ** 2)
        print(f"\n  dx={dx:.3f} mm  (grid {nx}x{ny}, dt={dt_val}, CFL={cfl:.4f})")

        base = Params(
            nx=nx, ny=ny, dx=dx, dt=dt_val,
            final_t_end=300.0,
            sulcus_width_mm=REP_WIDTH,
            g_sulcus_min=REP_GMIN,
            g_profile="flat",
            g_smooth_mm=1.2,
        )

        t0 = time.time()
        # Control
        control = run_simulation(base, dipole_on=False)
        fs = fixed_scale_from_control(control, base, E1, E2, R_MM)

        # Scalar
        p_scalar = dc.replace(base, diffusion_mode="scalar")
        scalar_out = run_simulation(p_scalar, dipole_on=True)
        scalar_meas = electrode_speed_mm_min(
            scalar_out.arr, p_scalar, E1, E2, R_MM, fixed_scale=fs
        )

        # Tensor
        p_tensor = dc.replace(base, diffusion_mode="tensor")
        tensor_out = run_simulation(p_tensor, dipole_on=True)
        tensor_meas = electrode_speed_mm_min(
            tensor_out.arr, p_tensor, E1, E2, R_MM, fixed_scale=fs
        )
        elapsed = time.time() - t0

        delta = tensor_meas.scaled_speed_mm_min - scalar_meas.scaled_speed_mm_min
        row = {
            "dx_mm": dx,
            "dt_s": dt_val,
            "nx": nx,
            "ny": ny,
            "CFL": round(cfl, 5),
            "scalar_speed_mm_min": round(scalar_meas.scaled_speed_mm_min, 4),
            "tensor_speed_mm_min": round(tensor_meas.scaled_speed_mm_min, 4),
            "anisotropy_gain_mm_min": round(delta, 4),
            "elapsed_s": round(elapsed, 1),
        }
        results.append(row)
        print(
            f"    scalar={row['scalar_speed_mm_min']:.4f}, "
            f"tensor={row['tensor_speed_mm_min']:.4f}, "
            f"Δ={row['anisotropy_gain_mm_min']:.4f} mm/min  "
            f"({elapsed:.1f}s)"
        )

    _save_csv(output_dir / "grid_convergence.csv", results)

    # Convergence plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    dxs = [r["dx_mm"] for r in results]

    axes[0].plot(dxs, [r["scalar_speed_mm_min"] for r in results], "s-", label="Scalar flat")
    axes[0].plot(dxs, [r["tensor_speed_mm_min"] for r in results], "o-", label="Tensor flat")
    axes[0].set_xlabel("Spatial resolution dx (mm)", fontsize=12)
    axes[0].set_ylabel("Speed (mm/min)", fontsize=12)
    axes[0].set_title("Speed convergence", fontsize=14)
    axes[0].invert_xaxis()
    axes[0].legend(fontsize=11)
    axes[0].tick_params(axis="both", labelsize=10)

    axes[1].plot(dxs, [r["anisotropy_gain_mm_min"] for r in results], "D-", color="tab:green")
    axes[1].set_xlabel("Spatial resolution dx (mm)", fontsize=12)
    axes[1].set_ylabel("Tensor − Scalar (mm/min)", fontsize=12)
    axes[1].set_title("Anisotropy gain convergence", fontsize=14)
    axes[1].tick_params(axis="both", labelsize=10)
    axes[1].invert_xaxis()

    fig.savefig(output_dir / "fig_grid_convergence.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved grid_convergence.csv + fig_grid_convergence.png")
    return results


# ------------------------------------------------------------------ #
# 2. Eta Sensitivity Sweep
# ------------------------------------------------------------------ #
def eta_sensitivity(output_dir: Path) -> list[dict]:
    """Sweep the tangential attenuation ratio eta."""
    print("\n=== Eta Sensitivity Sweep ===")
    etas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    base = Params(
        sulcus_width_mm=REP_WIDTH,
        g_sulcus_min=REP_GMIN,
        g_profile="flat",
    )

    # Control + scalar baseline (run once)
    control = run_simulation(base, dipole_on=False)
    fs = fixed_scale_from_control(control, base, E1, E2, R_MM)

    p_scalar = dc.replace(base, diffusion_mode="scalar")
    scalar_out = run_simulation(p_scalar, dipole_on=True)
    scalar_meas = electrode_speed_mm_min(
        scalar_out.arr, p_scalar, E1, E2, R_MM, fixed_scale=fs
    )
    scalar_speed = scalar_meas.scaled_speed_mm_min

    # Also compute local speed for scalar
    scalar_arr_s = scalar_out.arr / fs
    scalar_speed_map = arrival_speed_map_mm_min(scalar_arr_s, p_scalar)
    scalar_sulcus_local = compartment_field_medians(
        scalar_speed_map, scalar_out.phi, scalar_out.sulc_mask, REP_WIDTH
    )["sulcus"]

    print(f"  Scalar baseline: {scalar_speed:.4f} mm/min (local sulcus: {scalar_sulcus_local:.4f})")

    results = []
    for eta in etas:
        t0 = time.time()
        p_tensor = dc.replace(
            base,
            diffusion_mode="tensor",
            tensor_tangent_attenuation_ratio=eta,
        )
        tensor_out = run_simulation(p_tensor, dipole_on=True)
        tensor_meas = electrode_speed_mm_min(
            tensor_out.arr, p_tensor, E1, E2, R_MM, fixed_scale=fs
        )

        # Local speed in sulcus
        tensor_arr_s = tensor_out.arr / fs
        tensor_speed_map = arrival_speed_map_mm_min(tensor_arr_s, p_tensor)
        tensor_sulcus_local = compartment_field_medians(
            tensor_speed_map, tensor_out.phi, tensor_out.sulc_mask, REP_WIDTH
        )["sulcus"]

        delta = tensor_meas.scaled_speed_mm_min - scalar_speed
        elapsed = time.time() - t0

        row = {
            "eta": eta,
            "tensor_speed_mm_min": round(tensor_meas.scaled_speed_mm_min, 4),
            "scalar_speed_mm_min": round(scalar_speed, 4),
            "anisotropy_gain_mm_min": round(delta, 4),
            "tensor_sulcus_local_mm_min": round(tensor_sulcus_local, 4),
            "scalar_sulcus_local_mm_min": round(scalar_sulcus_local, 4),
        }
        results.append(row)
        print(
            f"  η={eta:.1f}: tensor={row['tensor_speed_mm_min']:.4f}, "
            f"Δ={row['anisotropy_gain_mm_min']:.4f}, "
            f"local sulcus={row['tensor_sulcus_local_mm_min']:.4f}  "
            f"({elapsed:.1f}s)"
        )

    _save_csv(output_dir / "eta_sensitivity.csv", results)

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    eta_vals = [r["eta"] for r in results]
    tensor_speeds = [r["tensor_speed_mm_min"] for r in results]
    tensor_locals = [r["tensor_sulcus_local_mm_min"] for r in results]

    # Panel A: Virtual-electrode speed
    axes[0].plot(eta_vals, tensor_speeds, "o-", color="tab:blue", label="Tensor", lw=2)
    axes[0].axhline(scalar_speed, color="tab:red", ls="--", lw=1.5, label="Scalar (η = 1)")
    axes[0].axhline(3.0, color="gray", ls=":", lw=1, label="Control")
    axes[0].set_xlabel("Tangential attenuation ratio η", fontsize=12)
    axes[0].set_ylabel("Virtual-electrode speed (mm/min)", fontsize=12)
    #axes[0].set_title("Electrode speed vs η")
    axes[0].legend(fontsize=11)
    axes[0].tick_params(axis="both", labelsize=10)

    # Panel B: Local sulcus speed
    axes[1].plot(eta_vals, tensor_locals, "s-", color="tab:blue", label="Tensor", lw=2)
    axes[1].axhline(scalar_sulcus_local, color="tab:red", ls="--", lw=1.5, label="Scalar")
    axes[1].set_xlabel("Tangential attenuation ratio η", fontsize=12)
    axes[1].set_ylabel("Median sulcal local speed (mm/min)", fontsize=12)
    #axes[1].set_title("Sulcal local speed vs η")
    axes[1].legend(fontsize=11)
    axes[1].tick_params(axis="both", labelsize=10)

    fig.savefig(output_dir / "fig_eta_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved eta_sensitivity.csv + fig_eta_sensitivity.png")
    return results


# ------------------------------------------------------------------ #
# 3. Wavefront Snapshots
# ------------------------------------------------------------------ #
def wavefront_snapshots(output_dir: Path) -> None:
    """Generate wavefront snapshot figure for control, scalar-flat, tensor-flat."""
    print("\n=== Wavefront Snapshots ===")
    snapshot_times = [12.0, 18.0, 24.0]

    base = Params(
        sulcus_width_mm=REP_WIDTH,
        g_sulcus_min=REP_GMIN,
        g_profile="flat",
    )

    configs = [
        ("Control", base, False),
        ("Scalar flat", dc.replace(base, diffusion_mode="scalar"), True),
        ("Tensor flat", dc.replace(base, diffusion_mode="tensor"), True),
    ]

    outputs = []
    for label, p, dipole_on in configs:
        t0 = time.time()
        out = run_simulation(p, dipole_on=dipole_on, snapshot_times=snapshot_times)
        elapsed = time.time() - t0
        outputs.append((label, out))
        print(f"  {label}: {elapsed:.1f}s")

    nrows = len(snapshot_times)
    ncols = len(configs)
    fig, ax = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(4.5 * ncols, 3.2 * nrows),
        constrained_layout=True,
    )

    for col, (label, out) in enumerate(outputs):
        for row in range(nrows):
            field = out.snapshot_fields[row].T
            im = ax[row, col].imshow(field, origin="lower", vmin=0, vmax=1, cmap="viridis")
            ax[row, col].contour(
                out.sulc_mask.T, levels=[0.5], colors="white", linewidths=0.8
            )
            if row == 0:
                ax[row, col].set_title(label, fontsize=14, fontweight="bold")
            if col == 0:
                ax[row, col].set_ylabel(
                    f"t = {out.snapshot_times[row]:.0f} s", fontsize=13
                )
            ax[row, col].set_xticks([])
            ax[row, col].set_yticks([])

    cbar = fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.85, pad=0.02)
    cbar.set_label("Activator u", rotation=90, fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # Scale bar on bottom-left panel
    bar_mm = 5.0
    bar_px = int(bar_mm / base.dx)
    x0, y0 = 10, 8
    ax[-1, 0].plot([x0, x0 + bar_px], [y0, y0], "w-", lw=3)
    ax[-1, 0].text(
        x0 + bar_px / 2, y0 + 5, f"{bar_mm:.0f} mm",
        color="white", ha="center", va="bottom", fontsize=11,
    )

    fig.savefig(output_dir / "fig_wavefront_snapshots.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_wavefront_snapshots.png")


# ------------------------------------------------------------------ #
# 4. Statistical Tests
# ------------------------------------------------------------------ #
def statistical_tests(output_dir: Path) -> dict:
    """Paired statistical tests on the full-grid CSV data."""
    print("\n=== Statistical Tests ===")
    from scipy import stats

    csv_path = ROOT / "outputs" / "extended_full" / "extended_results.csv"
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    results = {}
    for profile in ["flat", "gaussian"]:
        scalar_rows = sorted(
            [r for r in rows if r["diffusion_mode"] == "scalar" and r["base_profile"] == profile],
            key=lambda r: (float(r["sulcus_width_mm"]), float(r["g_sulcus_min"])),
        )
        tensor_rows = sorted(
            [r for r in rows if r["diffusion_mode"] == "tensor" and r["base_profile"] == profile],
            key=lambda r: (float(r["sulcus_width_mm"]), float(r["g_sulcus_min"])),
        )

        scalar_speeds = np.array([float(r["electrode_speed_mm_min"]) for r in scalar_rows])
        tensor_speeds = np.array([float(r["electrode_speed_mm_min"]) for r in tensor_rows])
        differences = tensor_speeds - scalar_speeds

        n = len(differences)
        mean_diff = float(np.mean(differences))
        sd_diff = float(np.std(differences, ddof=1))

        # Wilcoxon signed-rank test (one-sided: tensor > scalar)
        stat_w, p_wilcoxon = stats.wilcoxon(differences, alternative="greater")
        # Paired t-test as well
        stat_t, p_ttest = stats.ttest_rel(tensor_speeds, scalar_speeds, alternative="greater")

        result = {
            "profile": profile,
            "n_pairs": n,
            "mean_difference_mm_min": round(mean_diff, 4),
            "sd_difference_mm_min": round(sd_diff, 4),
            "min_difference_mm_min": round(float(np.min(differences)), 4),
            "max_difference_mm_min": round(float(np.max(differences)), 4),
            "wilcoxon_statistic": float(stat_w),
            "wilcoxon_p_value": float(p_wilcoxon),
            "paired_t_statistic": round(float(stat_t), 4),
            "paired_t_p_value": float(p_ttest),
            "all_positive": bool(np.all(differences > 0)),
        }
        results[profile] = result
        print(
            f"  {profile.title()} family: "
            f"Δ = {mean_diff:.4f} ± {sd_diff:.4f} mm/min, "
            f"range [{np.min(differences):.4f}, {np.max(differences):.4f}], "
            f"Wilcoxon p = {p_wilcoxon:.4e}, "
            f"paired t p = {p_ttest:.4e}, "
            f"all positive = {np.all(differences > 0)}"
        )

    (output_dir / "statistical_tests.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print("  Saved statistical_tests.json")
    return results


# ------------------------------------------------------------------ #
# 5. Supplementary Table (LaTeX)
# ------------------------------------------------------------------ #
def supplementary_table(output_dir: Path) -> None:
    """Format the full results as a LaTeX table."""
    print("\n=== Supplementary LaTeX Table ===")
    csv_path = ROOT / "outputs" / "extended_full" / "extended_results.csv"
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Complete sensitivity-grid results for all model families. "
        r"Columns show virtual-electrode speed, speed change relative to control, "
        r"median sulcal compartment delay, and median local sulcal speed.}",
        r"\label{tab:full_grid}",
        r"\small",
        r"\begin{tabular}{@{}llrrrrr@{}}",
        r"\toprule",
        r"Family & $w$ (mm) & $g_{\mathrm{sulcus}}$ & Speed & $\Delta v$ & Sulcal delay & Sulcal local \\",
        r" & & & (mm/min) & (mm/min) & (s) & (mm/min) \\",
        r"\midrule",
    ]

    for row in rows:
        label = row["family_label"].replace("-", " ").title()
        w = float(row["sulcus_width_mm"])
        g = float(row["g_sulcus_min"])
        speed = float(row["electrode_speed_mm_min"])
        delta = float(row["delta_speed_mm_min"])
        delay = float(row["delay_sulcus_s"])
        local = float(row["dipole_sulcus_local_speed_mm_min"])
        lines.append(
            f"{label} & {w:.1f} & {g:.2f} & {speed:.3f} & {delta:.3f} & {delay:.1f} & {local:.3f} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    tex_path = output_dir / "supplementary_table.tex"
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Saved {tex_path.name}")


# ------------------------------------------------------------------ #
# 6. Biophysical Grid Convergence
# ------------------------------------------------------------------ #
def biophysical_grid_convergence(output_dir: Path) -> list[dict]:
    """Run the calibrated K-buffer representative case at multiple resolutions."""
    print("\n=== Biophysical (K-buffer) Grid Convergence ===")
    resolutions = [
        {"dx": 0.20, "dt": 0.004},
        {"dx": 0.10, "dt": 0.002},
        {"dx": 0.05, "dt": 0.001},
    ]
    results = []
    for res in resolutions:
        dx = res["dx"]
        dt_val = res["dt"]
        nx = int(round(LX_MM / dx)) + 1
        ny = int(round(LY_MM / dx)) + 1
        cfl = 0.01 * dt_val / (dx ** 2)
        print(f"\n  dx={dx:.3f} mm  (grid {nx}x{ny}, dt={dt_val}, CFL={cfl:.4f})")

        base = Params(
            kinetics_model="potassium_buffer",
            k_release_rate=0.16,
            k_clearance_rate=0.040,
            k_threshold=9.0,
            k_arrival_threshold=10.0,
            nx=nx, ny=ny, dx=dx, dt=dt_val,
            final_t_end=120.0,
            sulcus_width_mm=REP_WIDTH,
            g_sulcus_min=REP_GMIN,
            g_profile="flat",
            g_smooth_mm=1.2,
        )

        t0 = time.time()
        # Control
        control = run_simulation(base, dipole_on=False)
        fs = fixed_scale_from_control(control, base, E1, E2, R_MM)

        # Scalar
        p_scalar = dc.replace(base, diffusion_mode="scalar")
        scalar_out = run_simulation(p_scalar, dipole_on=True)
        scalar_meas = electrode_speed_mm_min(
            scalar_out.arr, p_scalar, E1, E2, R_MM, fixed_scale=fs
        )

        # Tensor
        p_tensor = dc.replace(base, diffusion_mode="tensor")
        tensor_out = run_simulation(p_tensor, dipole_on=True)
        tensor_meas = electrode_speed_mm_min(
            tensor_out.arr, p_tensor, E1, E2, R_MM, fixed_scale=fs
        )
        elapsed = time.time() - t0

        delta = tensor_meas.scaled_speed_mm_min - scalar_meas.scaled_speed_mm_min
        row = {
            "dx_mm": dx,
            "dt_s": dt_val,
            "nx": nx,
            "ny": ny,
            "CFL": round(cfl, 5),
            "scalar_speed_mm_min": round(scalar_meas.scaled_speed_mm_min, 4),
            "tensor_speed_mm_min": round(tensor_meas.scaled_speed_mm_min, 4),
            "anisotropy_gain_mm_min": round(delta, 4),
            "elapsed_s": round(elapsed, 1),
        }
        results.append(row)
        print(
            f"    scalar={row['scalar_speed_mm_min']:.4f}, "
            f"tensor={row['tensor_speed_mm_min']:.4f}, "
            f"Δ={row['anisotropy_gain_mm_min']:.4f} mm/min  "
            f"({elapsed:.1f}s)"
        )

    _save_csv(output_dir / "biophysical_grid_convergence.csv", results)

    # Convergence plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    dxs = [r["dx_mm"] for r in results]

    axes[0].plot(dxs, [r["scalar_speed_mm_min"] for r in results], "s-", label="Scalar flat (K-buffer)")
    axes[0].plot(dxs, [r["tensor_speed_mm_min"] for r in results], "o-", label="Tensor flat (K-buffer)")
    axes[0].set_xlabel("Spatial resolution dx (mm)", fontsize=12)
    axes[0].set_ylabel("Speed (mm/min)", fontsize=12)
    axes[0].set_title("K-buffer speed convergence", fontsize=14)
    axes[0].invert_xaxis()
    axes[0].legend(fontsize=11)
    axes[0].tick_params(axis="both", labelsize=10)

    axes[1].plot(dxs, [r["anisotropy_gain_mm_min"] for r in results], "D-", color="tab:green")
    axes[1].set_xlabel("Spatial resolution dx (mm)", fontsize=12)
    axes[1].set_ylabel("Tensor − Scalar (mm/min)", fontsize=12)
    axes[1].set_title("K-buffer anisotropy gain convergence", fontsize=14)
    axes[1].tick_params(axis="both", labelsize=10)
    axes[1].invert_xaxis()

    fig.savefig(output_dir / "fig_biophysical_grid_convergence.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved biophysical_grid_convergence.csv + fig_biophysical_grid_convergence.png")
    return results


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main() -> None:
    output_dir = ROOT / "outputs" / "supplementary"
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # 1. Statistical tests (fast, no simulation needed)
    stat_results = statistical_tests(output_dir)

    # 2. Supplementary table (fast, reads existing CSV)
    supplementary_table(output_dir)

    # 3. Eta sensitivity sweep
    eta_results = eta_sensitivity(output_dir)

    # 4. Wavefront snapshots
    wavefront_snapshots(output_dir)

    # 5. Grid convergence — Barkley (slowest single block)
    convergence_results = grid_convergence(output_dir)

    # 6. Grid convergence — Biophysical (K-buffer)
    bio_convergence_results = biophysical_grid_convergence(output_dir)

    # Summary
    elapsed_total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"All supplementary analyses complete in {elapsed_total:.0f}s")
    print(f"Outputs in: {output_dir}")
    print(f"{'='*60}")

    summary = {
        "statistical_tests": stat_results,
        "eta_sensitivity": eta_results,
        "grid_convergence": convergence_results,
        "biophysical_grid_convergence": bio_convergence_results,
        "total_elapsed_s": round(elapsed_total, 1),
    }
    (output_dir / "supplementary_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
