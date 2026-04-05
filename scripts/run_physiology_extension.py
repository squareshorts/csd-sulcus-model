from __future__ import annotations

import argparse
import csv
import dataclasses as dc
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from csd_sulcus.analysis import (  # noqa: E402
    arrival_speed_map_mm_min,
    compare_against_reference_control,
    compartment_field_medians,
    fixed_scale_from_control,
)
from csd_sulcus.model import Params, resolve_tensor_tangent_attenuation_ratio, run_simulation  # noqa: E402


E1 = (0.43, 0.50)
E2 = (0.62, 0.50)
R_MM = 1.0
DEFAULT_TARGET_RATIOS = [1.05, 1.10, 1.15, 1.20, 1.25, 1.30]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the physiology-constrained CSD extension study.")
    parser.add_argument("--quick", action="store_true", help="Use a smaller grid and shorter run time.")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--target-ratios", type=float, nargs="*", default=DEFAULT_TARGET_RATIOS)
    return parser.parse_args()


def build_params(kinetics_model: str, quick: bool) -> Params:
    base = Params(
        kinetics_model=kinetics_model,
        sulcus_width_mm=4.0,
        g_sulcus_min=0.75,
        g_profile="flat",
    )
    if kinetics_model == "potassium_buffer":
        base = dc.replace(base, final_t_end=120.0, tensor_constraint_mode="cortical_microstructure")
    else:
        base = dc.replace(base, tensor_constraint_mode="cortical_microstructure")

    if not quick:
        return base

    return dc.replace(
        base,
        nx=120 if kinetics_model == "barkley" else 72,
        ny=84 if kinetics_model == "barkley" else 52,
        final_t_end=120.0 if kinetics_model == "barkley" else 70.0,
        sulcus_width_mm=3.0,
        g_smooth_mm=0.6,
    )


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def representative_rows(quick: bool) -> tuple[list[dict[str, float | str]], dict[str, float]]:
    rows: list[dict[str, float | str]] = []
    anisotropy_gains: dict[str, float] = {}

    for kinetics_model in ["barkley", "potassium_buffer"]:
        base = build_params(kinetics_model, quick)
        control = run_simulation(base, dipole_on=False)
        fixed_scale = fixed_scale_from_control(control, base, E1, E2, R_MM)
        control_speed_map = arrival_speed_map_mm_min(control.arr / fixed_scale, base)
        control_compartments = compartment_field_medians(control_speed_map, control.phi, control.sulc_mask, base.sulcus_width_mm)

        case_params = [
            ("scalar", dc.replace(base, diffusion_mode="scalar", tensor_constraint_mode="manual")),
            ("tensor_manual", dc.replace(base, diffusion_mode="tensor", tensor_constraint_mode="manual", tensor_tangent_attenuation_ratio=0.40)),
            ("tensor_microstructure", dc.replace(base, diffusion_mode="tensor", tensor_constraint_mode="cortical_microstructure")),
        ]

        scalar_speed = None
        for label, params in case_params:
            comparison = compare_against_reference_control(control, base, params, E1, E2, radius_mm=R_MM)
            dipole_speed_map = arrival_speed_map_mm_min(comparison.dipole.arr / fixed_scale, params)
            dipole_compartments = compartment_field_medians(
                dipole_speed_map,
                comparison.dipole.phi,
                comparison.dipole.sulc_mask,
                params.sulcus_width_mm,
            )
            if params.diffusion_mode == "tensor":
                achieved_ratio = float(
                    np.nanmin(
                        comparison.dipole.g_tangent[comparison.dipole.sulc_mask]
                        / comparison.dipole.g_normal[comparison.dipole.sulc_mask]
                    )
                )
            else:
                achieved_ratio = 1.0

            row = {
                "kinetics_model": kinetics_model,
                "case_label": label,
                "diffusion_mode": params.diffusion_mode,
                "tensor_constraint_mode": params.tensor_constraint_mode,
                "resolved_eta": resolve_tensor_tangent_attenuation_ratio(params) if params.diffusion_mode == "tensor" else 1.0,
                "achieved_tangent_normal_ratio": achieved_ratio,
                "electrode_speed_mm_min": float(comparison.dipole_measurement.scaled_speed_mm_min),
                "delta_speed_mm_min": float(comparison.dipole_measurement.scaled_speed_mm_min - base.target_gyrus_mm_min),
                "delay_sulcus_s": float(comparison.delay_sulcus_s),
                "delay_upstream_s": float(comparison.delay_upstream_s),
                "delay_downstream_s": float(comparison.delay_downstream_s),
                "sulcus_local_speed_mm_min": float(dipole_compartments["sulcus"]),
                "control_sulcus_local_speed_mm_min": float(control_compartments["sulcus"]),
            }
            rows.append(row)
            if label == "scalar":
                scalar_speed = row["electrode_speed_mm_min"]
            elif scalar_speed is not None:
                anisotropy_gains[f"{kinetics_model}:{label}"] = float(row["electrode_speed_mm_min"] - scalar_speed)

    return rows, anisotropy_gains


def microstructure_sweep_rows(quick: bool, target_ratios: list[float]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for kinetics_model in ["barkley", "potassium_buffer"]:
        base = build_params(kinetics_model, quick)
        control = run_simulation(base, dipole_on=False)
        fixed_scale = fixed_scale_from_control(control, base, E1, E2, R_MM)
        scalar_params = dc.replace(base, diffusion_mode="scalar", tensor_constraint_mode="manual")
        scalar = run_simulation(scalar_params, dipole_on=True)
        scalar_speed = float(fixed_scale * ((E2[0] * (scalar_params.nx - 1) * scalar_params.dx - E1[0] * (scalar_params.nx - 1) * scalar_params.dx) / 1.0))
        # Use the measured scalar speed rather than a closed-form estimate.
        from csd_sulcus.analysis import electrode_speed_mm_min  # local import keeps the script self-contained
        scalar_speed = float(electrode_speed_mm_min(scalar.arr, scalar_params, E1, E2, R_MM, fixed_scale).scaled_speed_mm_min)

        for target_ratio in target_ratios:
            params = dc.replace(
                base,
                diffusion_mode="tensor",
                tensor_constraint_mode="cortical_microstructure",
                microstructure_target_tangent_normal_ratio=float(target_ratio),
            )
            out = run_simulation(params, dipole_on=True)
            measurement = electrode_speed_mm_min(out.arr, params, E1, E2, R_MM, fixed_scale)
            rows.append(
                {
                    "kinetics_model": kinetics_model,
                    "target_tangent_normal_ratio": float(target_ratio),
                    "resolved_eta": resolve_tensor_tangent_attenuation_ratio(params),
                    "electrode_speed_mm_min": float(measurement.scaled_speed_mm_min),
                    "anisotropy_gain_mm_min": float(measurement.scaled_speed_mm_min - scalar_speed),
                }
            )
    return rows


def plot_representative(rows: list[dict[str, float | str]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    kinetics_order = ["barkley", "potassium_buffer"]
    labels = ["scalar", "tensor_manual", "tensor_microstructure"]
    pretty = {
        "scalar": "Scalar",
        "tensor_manual": "Tensor (manual)",
        "tensor_microstructure": "Tensor (microstructure)",
    }
    colors = {
        "scalar": "tab:red",
        "tensor_manual": "tab:blue",
        "tensor_microstructure": "tab:green",
    }
    width = 0.22
    x = np.arange(len(kinetics_order))

    for offset, label in enumerate(labels):
        vals_speed = []
        vals_delay = []
        for kinetics in kinetics_order:
            row = next(r for r in rows if r["kinetics_model"] == kinetics and r["case_label"] == label)
            vals_speed.append(float(row["electrode_speed_mm_min"]))
            vals_delay.append(float(row["delay_sulcus_s"]))
        axes[0].bar(x + (offset - 1) * width, vals_speed, width=width, color=colors[label], label=pretty[label])
        axes[1].bar(x + (offset - 1) * width, vals_delay, width=width, color=colors[label], label=pretty[label])

    axes[0].axhline(3.0, color="0.5", ls=":", lw=1.5)
    axes[0].set_xticks(x, ["Barkley", "K-buffer"])
    axes[0].set_ylabel("Virtual-electrode speed (mm/min)")
    #axes[0].set_title("Representative ordering")

    axes[1].set_xticks(x, ["Barkley", "K-buffer"])
    axes[1].set_ylabel("Median sulcal delay (s)")
    #axes[1].set_title("Sulcal delay")
    axes[1].legend(loc="upper right")

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_microstructure_sweep(rows: list[dict[str, float | str]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for kinetics_model, marker, color in [("barkley", "o", "tab:blue"), ("potassium_buffer", "s", "tab:green")]:
        subset = [row for row in rows if row["kinetics_model"] == kinetics_model]
        subset.sort(key=lambda row: float(row["target_tangent_normal_ratio"]))
        ratios = [float(row["target_tangent_normal_ratio"]) for row in subset]
        speeds = [float(row["electrode_speed_mm_min"]) for row in subset]
        gains = [float(row["anisotropy_gain_mm_min"]) for row in subset]
        axes[0].plot(ratios, speeds, marker=marker, color=color, lw=2, label=kinetics_model.replace("_", " ").title())
        axes[1].plot(ratios, gains, marker=marker, color=color, lw=2, label=kinetics_model.replace("_", " ").title())

    axes[0].set_xlabel("Target tangential/normal ratio")
    axes[0].set_ylabel("Virtual-electrode speed (mm/min)")
    #axes[0].set_title("Microstructure-constrained tensor speed")
    axes[0].legend()

    axes[1].set_xlabel("Target tangential/normal ratio")
    axes[1].set_ylabel("Tensor - scalar (mm/min)")
    #axes[1].set_title("Ordering across cortex-like anisotropy")
    axes[1].legend()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary(path: Path, representative: list[dict[str, float | str]], sweep: list[dict[str, float | str]], anisotropy_gains: dict[str, float]) -> None:
    def _row(kinetics: str, case: str) -> dict[str, float | str]:
        return next(row for row in representative if row["kinetics_model"] == kinetics and row["case_label"] == case)

    barkley_scalar = _row("barkley", "scalar")
    barkley_tensor = _row("barkley", "tensor_microstructure")
    potassium_scalar = _row("potassium_buffer", "scalar")
    potassium_tensor = _row("potassium_buffer", "tensor_microstructure")

    lines = [
        "# Physiology Extension Summary",
        "",
        "- Tensor constraint mode: cortical_microstructure",
        "- Cortex-like tangential/normal ratio bounds: 1.05 to 1.30",
        f"- Default target tangential/normal ratio: {build_params('barkley', False).microstructure_target_tangent_normal_ratio:.2f}",
        f"- Representative Barkley scalar speed: {float(barkley_scalar['electrode_speed_mm_min']):.3f} mm/min",
        f"- Representative Barkley microstructure-tensor speed: {float(barkley_tensor['electrode_speed_mm_min']):.3f} mm/min",
        f"- Representative potassium-buffer scalar speed: {float(potassium_scalar['electrode_speed_mm_min']):.3f} mm/min",
        f"- Representative potassium-buffer microstructure-tensor speed: {float(potassium_tensor['electrode_speed_mm_min']):.3f} mm/min",
        f"- Barkley microstructure anisotropy gain: {anisotropy_gains['barkley:tensor_microstructure']:.3f} mm/min",
        f"- Potassium-buffer microstructure anisotropy gain: {anisotropy_gains['potassium_buffer:tensor_microstructure']:.3f} mm/min",
        f"- Largest Barkley target-ratio sweep gain: {max(float(row['anisotropy_gain_mm_min']) for row in sweep if row['kinetics_model'] == 'barkley'):.3f} mm/min",
        f"- Largest potassium-buffer target-ratio sweep gain: {max(float(row['anisotropy_gain_mm_min']) for row in sweep if row['kinetics_model'] == 'potassium_buffer'):.3f} mm/min",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = args.output_root or ROOT / ("outputs/physiology_extension_quick" if args.quick else "outputs/physiology_extension")
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    representative, anisotropy_gains = representative_rows(args.quick)
    sweep_rows = microstructure_sweep_rows(args.quick, [float(x) for x in args.target_ratios])

    write_csv(output_root / "physiology_representative.csv", representative)
    write_csv(output_root / "physiology_microstructure_sweep.csv", sweep_rows)
    plot_representative(representative, figure_dir / "fig_physiology_ordering.png")
    plot_microstructure_sweep(sweep_rows, figure_dir / "fig_microstructure_ratio_sweep.png")

    summary = {
        "mode": "quick" if args.quick else "full",
        "representative_rows": representative,
        "microstructure_sweep": sweep_rows,
        "anisotropy_gains": anisotropy_gains,
    }
    (output_root / "physiology_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary(output_root / "physiology_summary.md", representative, sweep_rows, anisotropy_gains)
    print(f"Physiology extension outputs written to {output_root}")


if __name__ == "__main__":
    main()

