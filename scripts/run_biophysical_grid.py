"""Run the calibrated potassium-buffer model across the same width × g_sulcus
grid as the Barkley extended study (run_extended_study.py).

Produces directly comparable CSV, heatmaps, and statistical tests.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses as dc
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from csd_sulcus.analysis import (  # noqa: E402
    arrival_speed_map_mm_min,
    compare_against_reference_control,
    compartment_field_medians,
    fixed_scale_from_control,
)
from csd_sulcus.model import Params, run_simulation  # noqa: E402
from csd_sulcus.plotting import plot_local_speed_triptych, plot_profile_heatmaps  # noqa: E402


# Electrode positions (match Barkley study exactly)
E1 = (0.43, 0.50)
E2 = (0.62, 0.50)
R_MM = 1.0

# Grid parameters — identical to run_extended_study.py
DEFAULT_GMINS = [0.95, 0.85, 0.75]
DEFAULT_WIDTHS = [2.5, 4.0, 5.5]
DEFAULT_PROFILES = ["flat", "gaussian"]
DEFAULT_DIFFUSION_MODES = ["scalar", "tensor"]
REP_WIDTH = 4.0
REP_GMIN = 0.75
REP_PROFILE = "flat"


def build_base_params() -> Params:
    """Calibrated potassium-buffer base parameters (full resolution)."""
    return Params(
        kinetics_model="potassium_buffer",
        # Calibrated kinetics (acceptance score 5/5)
        k_release_rate=0.16,
        k_clearance_rate=0.040,
        k_threshold=9.0,
        k_arrival_threshold=10.0,
        # Domain (same as Barkley)
        nx=200,
        ny=140,
        dx=0.10,
        dt=0.002,
        final_t_end=120.0,
        # Default geometry
        sulcus_width_mm=4.0,
        g_sulcus_min=0.75,
        g_smooth_mm=1.2,
        g_profile="flat",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the calibrated K-buffer model on the Barkley-equivalent grid."
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--gmins", type=float, nargs="*", default=DEFAULT_GMINS)
    parser.add_argument("--widths", type=float, nargs="*", default=DEFAULT_WIDTHS)
    parser.add_argument("--profiles", nargs="*", default=DEFAULT_PROFILES)
    parser.add_argument("--diffusion-modes", nargs="*", default=DEFAULT_DIFFUSION_MODES)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def representative_delta(
    representative: dict[str, dict[str, float | str]], profile: str
) -> float:
    scalar_key = f"scalar-{profile}"
    tensor_key = f"tensor-{profile}"
    if scalar_key not in representative or tensor_key not in representative:
        return float("nan")
    return float(representative[tensor_key]["electrode_speed_mm_min"]) - float(
        representative[scalar_key]["electrode_speed_mm_min"]
    )


def write_summary(path: Path, summary: dict[str, object]) -> None:
    representative = summary["representative_cases"]
    worst_case = summary["worst_case"]
    lines = [
        "# Biophysical Grid Sweep Summary",
        "",
        f"- Mode: {summary['mode']}",
        f"- Kinetics: potassium_buffer (calibrated)",
        f"- Control speed: {summary['control_speed_mm_min']:.3f} mm/min",
        f"- Representative scalar-flat case: {representative['scalar-flat']['electrode_speed_mm_min']:.3f} mm/min",
        f"- Representative tensor-flat case: {representative['tensor-flat']['electrode_speed_mm_min']:.3f} mm/min",
        f"- Tensor minus scalar at the flat representative case: {summary['representative_tensor_minus_scalar_flat_mm_min']:.3f} mm/min",
        f"- Representative scalar-gaussian case: {representative['scalar-gaussian']['electrode_speed_mm_min']:.3f} mm/min",
        f"- Representative tensor-gaussian case: {representative['tensor-gaussian']['electrode_speed_mm_min']:.3f} mm/min",
        f"- Tensor minus scalar at the gaussian representative case: {summary['representative_tensor_minus_scalar_gaussian_mm_min']:.3f} mm/min",
        f"- Worst-case slowing in grid: {worst_case['delta_speed_mm_min']:.3f} mm/min for family={worst_case['family_label']}, width={worst_case['sulcus_width_mm']:.1f} mm, g={worst_case['g_sulcus_min']:.2f}",
        f"- Mean flat anisotropy gain (tensor - scalar): {summary['mean_flat_anisotropy_mm_min']:.3f} mm/min",
        f"- Mean gaussian anisotropy gain (tensor - scalar): {summary['mean_gaussian_anisotropy_mm_min']:.3f} mm/min",
        f"- All tensor > scalar (flat): {summary['all_tensor_gt_scalar_flat']}",
        f"- All tensor > scalar (gaussian): {summary['all_tensor_gt_scalar_gaussian']}",
    ]
    if "statistical_tests" in summary:
        for profile, stats in summary["statistical_tests"].items():
            lines.extend([
                f"",
                f"## Statistical tests ({profile})",
                f"- N pairs: {stats['n_pairs']}",
                f"- Mean difference: {stats['mean_difference_mm_min']:.4f} mm/min",
                f"- SD: {stats['sd_difference_mm_min']:.4f} mm/min",
                f"- Wilcoxon p: {stats['wilcoxon_p_value']:.4e}",
                f"- Paired t p: {stats['paired_t_p_value']:.4e}",
                f"- All positive: {stats['all_positive']}",
            ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_statistical_tests(
    rows: list[dict[str, float | str]], profiles: list[str]
) -> dict[str, dict]:
    """Paired tests on tensor vs scalar for each profile."""
    from scipy import stats as sp_stats

    results = {}
    for profile in profiles:
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

        stat_w, p_wilcoxon = sp_stats.wilcoxon(differences, alternative="greater")
        stat_t, p_ttest = sp_stats.ttest_rel(tensor_speeds, scalar_speeds, alternative="greater")

        results[profile] = {
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
        print(
            f"  {profile.title()} family: "
            f"Δ = {mean_diff:.4f} ± {sd_diff:.4f} mm/min, "
            f"range [{np.min(differences):.4f}, {np.max(differences):.4f}], "
            f"Wilcoxon p = {p_wilcoxon:.4e}, "
            f"paired t p = {p_ttest:.4e}, "
            f"all positive = {np.all(differences > 0)}"
        )
    return results


def main() -> None:
    args = parse_args()
    base_params = build_base_params()
    output_root = args.output_root or ROOT / "outputs/biophysical_grid_full"
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    print("=" * 60)
    print("Biophysical (K-buffer) grid sweep")
    print(f"Grid: widths={args.widths}, gmins={args.gmins}")
    print(f"Profiles: {args.profiles}, Modes: {args.diffusion_modes}")
    print("=" * 60)

    # Control run
    print("\nRunning control...")
    t0 = time.time()
    control = run_simulation(base_params, dipole_on=False)
    fixed_scale = fixed_scale_from_control(control, base_params, E1, E2, R_MM)
    control_arr_s = control.arr / fixed_scale
    control_speed_map = arrival_speed_map_mm_min(control_arr_s, base_params)
    control_speed_mm_min = base_params.target_gyrus_mm_min
    print(f"  Control done in {time.time() - t0:.1f}s, speed = {control_speed_mm_min:.3f} mm/min")

    rows: list[dict[str, float | str]] = []
    representative_maps: dict[str, object] = {}
    family_labels = [
        f"{mode}-{profile}"
        for mode in args.diffusion_modes
        for profile in args.profiles
    ]

    n_total = len(args.diffusion_modes) * len(args.profiles) * len(args.widths) * len(args.gmins)
    i_sim = 0

    for mode in args.diffusion_modes:
        for profile in args.profiles:
            family_label = f"{mode}-{profile}"
            for width in args.widths:
                for gmin in args.gmins:
                    i_sim += 1
                    t0 = time.time()
                    case_params = dc.replace(
                        base_params,
                        diffusion_mode=mode,
                        g_profile=profile,
                        sulcus_width_mm=float(width),
                        g_sulcus_min=float(gmin),
                    )
                    comparison = compare_against_reference_control(
                        control,
                        base_params,
                        case_params,
                        E1,
                        E2,
                        radius_mm=R_MM,
                    )

                    dipole_arr_s = comparison.dipole.arr / comparison.fixed_scale
                    dipole_speed_map = arrival_speed_map_mm_min(dipole_arr_s, case_params)
                    control_compartment = compartment_field_medians(
                        control_speed_map,
                        comparison.dipole.phi,
                        comparison.dipole.sulc_mask,
                        case_params.sulcus_width_mm,
                    )
                    dipole_compartment = compartment_field_medians(
                        dipole_speed_map,
                        comparison.dipole.phi,
                        comparison.dipole.sulc_mask,
                        case_params.sulcus_width_mm,
                    )

                    row = {
                        "family_label": family_label,
                        "diffusion_mode": mode,
                        "g_profile": family_label,
                        "base_profile": profile,
                        "sulcus_width_mm": float(width),
                        "g_sulcus_min": float(gmin),
                        "electrode_speed_mm_min": float(comparison.dipole_measurement.scaled_speed_mm_min),
                        "delta_speed_mm_min": float(comparison.dipole_measurement.scaled_speed_mm_min - control_speed_mm_min),
                        "delay_sulcus_s": float(comparison.delay_sulcus_s),
                        "delay_upstream_s": float(comparison.delay_upstream_s),
                        "delay_downstream_s": float(comparison.delay_downstream_s),
                        "peak_delay_s": float(comparison.peak_delay_s),
                        "control_sulcus_local_speed_mm_min": float(control_compartment["sulcus"]),
                        "dipole_sulcus_local_speed_mm_min": float(dipole_compartment["sulcus"]),
                        "delta_sulcus_local_speed_mm_min": float(dipole_compartment["sulcus"] - control_compartment["sulcus"]),
                        "control_upstream_local_speed_mm_min": float(control_compartment["upstream"]),
                        "dipole_upstream_local_speed_mm_min": float(dipole_compartment["upstream"]),
                        "control_downstream_local_speed_mm_min": float(control_compartment["downstream"]),
                        "dipole_downstream_local_speed_mm_min": float(dipole_compartment["downstream"]),
                        "sulcus_tangent_min": float(np.nanmin(comparison.dipole.g_tangent[comparison.dipole.sulc_mask])),
                        "sulcus_normal_min": float(np.nanmin(comparison.dipole.g_normal[comparison.dipole.sulc_mask])),
                    }
                    rows.append(row)
                    elapsed = time.time() - t0

                    if width == REP_WIDTH and gmin == REP_GMIN:
                        representative_maps[family_label] = dipole_speed_map
                        representative_maps[f"{family_label}_row"] = row
                        representative_maps[f"{family_label}_mask"] = comparison.dipole.sulc_mask

                    print(
                        f"  [{i_sim}/{n_total}] {family_label} w={width:.1f} g={gmin:.2f}: "
                        f"speed={row['electrode_speed_mm_min']:.3f}, "
                        f"Δ={row['delta_speed_mm_min']:.3f} mm/min  "
                        f"({elapsed:.1f}s)"
                    )

    # Sort and write CSV
    rows.sort(key=lambda item: (str(item["family_label"]), float(item["sulcus_width_mm"]), float(item["g_sulcus_min"])))
    write_csv(output_root / "biophysical_grid_results.csv", rows)

    # Anisotropy gain table
    anisotropy_rows: list[dict[str, float | str]] = []
    for profile in args.profiles:
        for width in args.widths:
            for gmin in args.gmins:
                scalar_row = next(
                    row for row in rows
                    if row["diffusion_mode"] == "scalar" and row["base_profile"] == profile
                    and row["sulcus_width_mm"] == float(width) and row["g_sulcus_min"] == float(gmin)
                )
                tensor_row = next(
                    row for row in rows
                    if row["diffusion_mode"] == "tensor" and row["base_profile"] == profile
                    and row["sulcus_width_mm"] == float(width) and row["g_sulcus_min"] == float(gmin)
                )
                anisotropy_rows.append({
                    "g_profile": profile,
                    "sulcus_width_mm": float(width),
                    "g_sulcus_min": float(gmin),
                    "anisotropy_delta_mm_min": float(tensor_row["electrode_speed_mm_min"] - scalar_row["electrode_speed_mm_min"]),
                    "anisotropy_sulcus_local_delta_mm_min": float(tensor_row["dipole_sulcus_local_speed_mm_min"] - scalar_row["dipole_sulcus_local_speed_mm_min"]),
                })
    write_csv(output_root / "anisotropy_gain.csv", anisotropy_rows)

    # ---- Heatmaps ----
    print("\nGenerating heatmaps...")
    plot_profile_heatmaps(
        rows,
        profiles=family_labels,
        widths=[float(x) for x in args.widths],
        gmins=[float(x) for x in args.gmins],
        metric_key="delta_speed_mm_min",
        value_label="Delta speed (mm/min)",
        output_path=figure_dir / "fig_biophysical_delta_speed_heatmaps.png",
        cmap="coolwarm_r",
    )
    plot_profile_heatmaps(
        rows,
        profiles=family_labels,
        widths=[float(x) for x in args.widths],
        gmins=[float(x) for x in args.gmins],
        metric_key="dipole_sulcus_local_speed_mm_min",
        value_label="Sulcus local speed (mm/min)",
        output_path=figure_dir / "fig_biophysical_sulcus_speed_heatmaps.png",
        cmap="viridis",
    )
    plot_profile_heatmaps(
        anisotropy_rows,
        profiles=list(args.profiles),
        widths=[float(x) for x in args.widths],
        gmins=[float(x) for x in args.gmins],
        metric_key="anisotropy_delta_mm_min",
        value_label="Tensor - scalar speed (mm/min)",
        output_path=figure_dir / "fig_biophysical_anisotropy_gain_heatmaps.png",
        cmap="coolwarm",
    )

    if "scalar-flat" in representative_maps and "tensor-flat" in representative_maps:
        plot_local_speed_triptych(
            control_speed_map,
            representative_maps["scalar-flat"],
            representative_maps["tensor-flat"],
            representative_maps["scalar-flat_mask"],
            output_path=figure_dir / "fig_biophysical_local_speed_triptych.png",
            labels=("Control local speed", "Scalar flat local speed", "Tensor flat local speed"),
        )

    # ---- Statistical tests ----
    print("\n=== Statistical Tests (K-buffer grid) ===")
    stat_results = run_statistical_tests(rows, list(args.profiles))

    # ---- Summary ----
    representative_cases = {
        label: representative_maps[f"{label}_row"]
        for label in family_labels
        if f"{label}_row" in representative_maps
    }
    worst_case = min(rows, key=lambda row: float(row["delta_speed_mm_min"]))
    flat_anisotropy = [row for row in anisotropy_rows if row["g_profile"] == "flat"]
    gaussian_anisotropy = [row for row in anisotropy_rows if row["g_profile"] == "gaussian"]

    all_flat_positive = all(
        float(row["anisotropy_delta_mm_min"]) > 0 for row in flat_anisotropy
    )
    all_gaussian_positive = all(
        float(row["anisotropy_delta_mm_min"]) > 0 for row in gaussian_anisotropy
    )

    summary = {
        "mode": "full",
        "kinetics_model": "potassium_buffer",
        "calibration": {
            "k_release_rate": 0.16,
            "k_clearance_rate": 0.040,
            "k_threshold": 9.0,
            "k_arrival_threshold": 10.0,
        },
        "control_speed_mm_min": control_speed_mm_min,
        "representative_cases": representative_cases,
        "representative_tensor_minus_scalar_flat_mm_min": representative_delta(representative_cases, "flat"),
        "representative_tensor_minus_scalar_gaussian_mm_min": representative_delta(representative_cases, "gaussian"),
        "worst_case": worst_case,
        "mean_flat_anisotropy_mm_min": sum(float(row["anisotropy_delta_mm_min"]) for row in flat_anisotropy) / len(flat_anisotropy),
        "mean_gaussian_anisotropy_mm_min": sum(float(row["anisotropy_delta_mm_min"]) for row in gaussian_anisotropy) / len(gaussian_anisotropy),
        "all_tensor_gt_scalar_flat": all_flat_positive,
        "all_tensor_gt_scalar_gaussian": all_gaussian_positive,
        "statistical_tests": stat_results,
    }
    (output_root / "biophysical_grid_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_summary(output_root / "biophysical_grid_summary.md", summary)

    elapsed_total = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"Biophysical grid sweep complete in {elapsed_total:.0f}s ({elapsed_total / 60:.1f} min)")
    print(f"Outputs in: {output_root}")
    print(f"Control speed: {control_speed_mm_min:.3f} mm/min")
    print(
        f"Worst-case delta speed: {float(worst_case['delta_speed_mm_min']):.3f} mm/min "
        f"({worst_case['family_label']}, width={float(worst_case['sulcus_width_mm']):.1f}, "
        f"g={float(worst_case['g_sulcus_min']):.2f})"
    )
    print(f"All tensor > scalar (flat): {all_flat_positive}")
    print(f"All tensor > scalar (gaussian): {all_gaussian_positive}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
