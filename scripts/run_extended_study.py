from __future__ import annotations

import argparse
import csv
import dataclasses as dc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from csd_sulcus.analysis import (
    arrival_speed_map_mm_min,
    compare_against_reference_control,
    compartment_field_medians,
    fixed_scale_from_control,
)
from csd_sulcus.model import Params, run_simulation
from csd_sulcus.plotting import plot_local_speed_triptych, plot_profile_heatmaps


E1 = (0.43, 0.50)
E2 = (0.62, 0.50)
R_MM = 1.0
DEFAULT_GMINS = [0.95, 0.85, 0.75]
DEFAULT_WIDTHS = [2.5, 4.0, 5.5]
DEFAULT_PROFILES = ["flat", "gaussian"]
DEFAULT_DIFFUSION_MODES = ["scalar", "tensor"]
REP_WIDTH = 4.0
REP_GMIN = 0.75
REP_PROFILE = "flat"


def build_params(quick: bool) -> Params:
    p = Params()
    if not quick:
        return p
    return dc.replace(
        p,
        nx=84,
        ny=58,
        final_t_end=50.0,
        sulcus_width_mm=4.0,
        g_smooth_mm=0.8,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the expanded CSD sulcus study.")
    parser.add_argument("--quick", action="store_true", help="Use the reduced extended-study grid for faster iteration.")
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


def representative_delta(representative: dict[str, dict[str, float | str]], profile: str) -> float:
    scalar_key = f"scalar-{profile}"
    tensor_key = f"tensor-{profile}"
    if scalar_key not in representative or tensor_key not in representative:
        return float("nan")
    return float(representative[tensor_key]["electrode_speed_mm_min"]) - float(representative[scalar_key]["electrode_speed_mm_min"])


def write_summary(path: Path, summary: dict[str, object]) -> None:
    representative = summary["representative_cases"]
    worst_case = summary["worst_case"]
    lines = [
        "# Extended Study Summary",
        "",
        f"- Mode: {summary['mode']}",
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
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_params = build_params(args.quick)
    output_root = args.output_root or ROOT / ("outputs/extended_quick" if args.quick else "outputs/extended_full")
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    control = run_simulation(base_params, dipole_on=False)
    fixed_scale = fixed_scale_from_control(control, base_params, E1, E2, R_MM)
    control_arr_s = control.arr / fixed_scale
    control_speed_map = arrival_speed_map_mm_min(control_arr_s, base_params)
    control_speed_mm_min = base_params.target_gyrus_mm_min

    rows: list[dict[str, float | str]] = []
    representative_maps: dict[str, object] = {}

    family_labels = [f"{mode}-{profile}" for mode in args.diffusion_modes for profile in args.profiles]

    for mode in args.diffusion_modes:
        for profile in args.profiles:
            family_label = f"{mode}-{profile}"
            for width in args.widths:
                for gmin in args.gmins:
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

                    if width == REP_WIDTH and gmin == REP_GMIN:
                        representative_maps[family_label] = dipole_speed_map
                        representative_maps[f"{family_label}_row"] = row
                        representative_maps[f"{family_label}_mask"] = comparison.dipole.sulc_mask

    rows.sort(key=lambda item: (str(item["family_label"]), float(item["sulcus_width_mm"]), float(item["g_sulcus_min"])))
    write_csv(output_root / "extended_results.csv", rows)

    anisotropy_rows: list[dict[str, float | str]] = []
    for profile in args.profiles:
        for width in args.widths:
            for gmin in args.gmins:
                scalar_row = next(
                    row for row in rows if row["diffusion_mode"] == "scalar" and row["base_profile"] == profile and row["sulcus_width_mm"] == float(width) and row["g_sulcus_min"] == float(gmin)
                )
                tensor_row = next(
                    row for row in rows if row["diffusion_mode"] == "tensor" and row["base_profile"] == profile and row["sulcus_width_mm"] == float(width) and row["g_sulcus_min"] == float(gmin)
                )
                anisotropy_rows.append(
                    {
                        "g_profile": profile,
                        "sulcus_width_mm": float(width),
                        "g_sulcus_min": float(gmin),
                        "anisotropy_delta_mm_min": float(tensor_row["electrode_speed_mm_min"] - scalar_row["electrode_speed_mm_min"]),
                        "anisotropy_sulcus_local_delta_mm_min": float(tensor_row["dipole_sulcus_local_speed_mm_min"] - scalar_row["dipole_sulcus_local_speed_mm_min"]),
                    }
                )
    write_csv(output_root / "anisotropy_gain.csv", anisotropy_rows)

    plot_profile_heatmaps(
        rows,
        profiles=family_labels,
        widths=[float(x) for x in args.widths],
        gmins=[float(x) for x in args.gmins],
        metric_key="delta_speed_mm_min",
        value_label="Delta speed (mm/min)",
        output_path=figure_dir / "fig_extended_delta_speed_heatmaps.png",
        cmap="coolwarm_r",
    )
    plot_profile_heatmaps(
        rows,
        profiles=family_labels,
        widths=[float(x) for x in args.widths],
        gmins=[float(x) for x in args.gmins],
        metric_key="dipole_sulcus_local_speed_mm_min",
        value_label="Sulcus local speed (mm/min)",
        output_path=figure_dir / "fig_extended_sulcus_speed_heatmaps.png",
        cmap="viridis",
    )
    plot_profile_heatmaps(
        anisotropy_rows,
        profiles=list(args.profiles),
        widths=[float(x) for x in args.widths],
        gmins=[float(x) for x in args.gmins],
        metric_key="anisotropy_delta_mm_min",
        value_label="Tensor - scalar speed (mm/min)",
        output_path=figure_dir / "fig_extended_anisotropy_gain_heatmaps.png",
        cmap="coolwarm",
    )

    if "scalar-flat" in representative_maps and "tensor-flat" in representative_maps:
        plot_local_speed_triptych(
            control_speed_map,
            representative_maps["scalar-flat"],
            representative_maps["tensor-flat"],
            representative_maps["scalar-flat_mask"],
            output_path=figure_dir / "fig_extended_local_speed_triptych.png",
            labels=("Control local speed", "Scalar flat local speed", "Tensor flat local speed"),
        )

    representative_cases = {
        label: representative_maps[f"{label}_row"]
        for label in family_labels
        if f"{label}_row" in representative_maps
    }
    worst_case = min(rows, key=lambda row: float(row["delta_speed_mm_min"]))
    flat_anisotropy = [row for row in anisotropy_rows if row["g_profile"] == "flat"]
    gaussian_anisotropy = [row for row in anisotropy_rows if row["g_profile"] == "gaussian"]

    summary = {
        "mode": "quick" if args.quick else "full",
        "control_speed_mm_min": control_speed_mm_min,
        "representative_cases": representative_cases,
        "representative_tensor_minus_scalar_flat_mm_min": representative_delta(representative_cases, "flat"),
        "representative_tensor_minus_scalar_gaussian_mm_min": representative_delta(representative_cases, "gaussian"),
        "worst_case": worst_case,
        "mean_flat_anisotropy_mm_min": sum(float(row["anisotropy_delta_mm_min"]) for row in flat_anisotropy) / len(flat_anisotropy),
        "mean_gaussian_anisotropy_mm_min": sum(float(row["anisotropy_delta_mm_min"]) for row in gaussian_anisotropy) / len(gaussian_anisotropy),
    }
    (output_root / "extended_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary(output_root / "extended_summary.md", summary)

    print(f"Extended outputs written to {output_root}")
    print(f"Control speed: {control_speed_mm_min:.3f} mm/min")
    print(
        "Worst-case delta speed: "
        f"{float(worst_case['delta_speed_mm_min']):.3f} mm/min "
        f"({worst_case['family_label']}, width={float(worst_case['sulcus_width_mm']):.1f}, g={float(worst_case['g_sulcus_min']):.2f})"
    )


if __name__ == "__main__":
    import numpy as np

    main()
