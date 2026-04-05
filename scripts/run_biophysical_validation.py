from __future__ import annotations

import argparse
import csv
import dataclasses as dc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from csd_sulcus.analysis import (  # noqa: E402
    arrival_speed_map_mm_min,
    compare_against_reference_control,
    compartment_field_medians,
    fixed_scale_from_control,
)
from csd_sulcus.model import Params, run_simulation  # noqa: E402


E1 = (0.43, 0.50)
E2 = (0.62, 0.50)
R_MM = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and calibrate the reduced biophysical SD model.")
    parser.add_argument("--quick", action="store_true", help="Use a reduced domain for parameter search.")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--release-rates", type=float, nargs="*", default=[0.10, 0.12, 0.14, 0.16])
    parser.add_argument("--clearance-rates", type=float, nargs="*", default=[0.03, 0.04, 0.05])
    parser.add_argument("--thresholds", type=float, nargs="*", default=[7.0, 8.0, 9.0])
    parser.add_argument("--arrival-thresholds", type=float, nargs="*", default=[10.0, 11.0, 12.0])
    return parser.parse_args()


def build_params(quick: bool) -> Params:
    base = Params(
        kinetics_model="potassium_buffer",
        diffusion_mode="scalar",
        tensor_constraint_mode="manual",
        sulcus_width_mm=4.0,
        g_sulcus_min=0.75,
        g_profile="flat",
        final_t_end=120.0,
    )
    if not quick:
        return base
    return dc.replace(
        base,
        nx=84,
        ny=58,
        final_t_end=90.0,
        sulcus_width_mm=3.0,
        g_smooth_mm=0.8,
    )


def write_csv(path: Path, rows: list[dict[str, float | str | bool]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def bool_score(value: bool) -> int:
    return 1 if value else 0


def evaluate_candidate(
    p: Params,
    release_rate: float,
    clearance_rate: float,
    threshold: float,
    arrival_threshold: float,
) -> dict[str, float | str | bool]:
    scalar_params = dc.replace(
        p,
        diffusion_mode="scalar",
        tensor_constraint_mode="manual",
        k_release_rate=release_rate,
        k_clearance_rate=clearance_rate,
        k_threshold=threshold,
        k_arrival_threshold=arrival_threshold,
    )
    tensor_params = dc.replace(
        scalar_params,
        diffusion_mode="tensor",
        tensor_constraint_mode="cortical_microstructure",
    )

    control = run_simulation(scalar_params, dipole_on=False)
    fixed_scale = fixed_scale_from_control(control, scalar_params, E1, E2, R_MM)
    control_speed_map = arrival_speed_map_mm_min(control.arr / fixed_scale, scalar_params)
    control_compartments = compartment_field_medians(
        control_speed_map,
        control.phi,
        control.sulc_mask,
        scalar_params.sulcus_width_mm,
    )

    scalar = compare_against_reference_control(control, scalar_params, scalar_params, E1, E2, radius_mm=R_MM)
    tensor = compare_against_reference_control(control, scalar_params, tensor_params, E1, E2, radius_mm=R_MM)

    scalar_speed_map = arrival_speed_map_mm_min(scalar.dipole.arr / fixed_scale, scalar_params)
    tensor_speed_map = arrival_speed_map_mm_min(tensor.dipole.arr / fixed_scale, tensor_params)
    scalar_compartments = compartment_field_medians(
        scalar_speed_map,
        scalar.dipole.phi,
        scalar.dipole.sulc_mask,
        scalar_params.sulcus_width_mm,
    )
    tensor_compartments = compartment_field_medians(
        tensor_speed_map,
        tensor.dipole.phi,
        tensor.dipole.sulc_mask,
        tensor_params.sulcus_width_mm,
    )

    scalar_slows_electrode = scalar.dipole_measurement.scaled_speed_mm_min < scalar.control_measurement.scaled_speed_mm_min
    tensor_faster_electrode = tensor.dipole_measurement.scaled_speed_mm_min > scalar.dipole_measurement.scaled_speed_mm_min
    scalar_slows_sulcus = scalar_compartments["sulcus"] < control_compartments["sulcus"]
    tensor_faster_sulcus = tensor_compartments["sulcus"] > scalar_compartments["sulcus"]
    downstream_delay_positive = scalar.delay_downstream_s > 0.0

    score = sum(
        [
            bool_score(scalar_slows_electrode),
            bool_score(tensor_faster_electrode),
            bool_score(scalar_slows_sulcus),
            bool_score(tensor_faster_sulcus),
            bool_score(downstream_delay_positive),
        ]
    )

    return {
        "k_release_rate": float(release_rate),
        "k_clearance_rate": float(clearance_rate),
        "k_threshold": float(threshold),
        "k_arrival_threshold": float(arrival_threshold),
        "control_speed_mm_min": float(scalar.control_measurement.scaled_speed_mm_min),
        "scalar_speed_mm_min": float(scalar.dipole_measurement.scaled_speed_mm_min),
        "tensor_speed_mm_min": float(tensor.dipole_measurement.scaled_speed_mm_min),
        "scalar_delta_mm_min": float(scalar.dipole_measurement.scaled_speed_mm_min - scalar.control_measurement.scaled_speed_mm_min),
        "tensor_minus_scalar_mm_min": float(tensor.dipole_measurement.scaled_speed_mm_min - scalar.dipole_measurement.scaled_speed_mm_min),
        "control_sulcus_local_speed_mm_min": float(control_compartments["sulcus"]),
        "scalar_sulcus_local_speed_mm_min": float(scalar_compartments["sulcus"]),
        "tensor_sulcus_local_speed_mm_min": float(tensor_compartments["sulcus"]),
        "scalar_sulcus_delay_s": float(scalar.delay_sulcus_s),
        "scalar_downstream_delay_s": float(scalar.delay_downstream_s),
        "tensor_sulcus_delay_s": float(tensor.delay_sulcus_s),
        "scalar_slows_electrode": scalar_slows_electrode,
        "tensor_faster_electrode": tensor_faster_electrode,
        "scalar_slows_sulcus": scalar_slows_sulcus,
        "tensor_faster_sulcus": tensor_faster_sulcus,
        "downstream_delay_positive": downstream_delay_positive,
        "acceptance_score": score,
        "accepted": score == 5,
    }


def write_summary(path: Path, rows: list[dict[str, float | str | bool]]) -> None:
    lines = ["# Biophysical Validation Summary", ""]
    if not rows:
        lines.append("- No candidate rows were generated.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    best = max(rows, key=lambda row: (int(row["acceptance_score"]), float(row["tensor_minus_scalar_mm_min"])))
    accepted = [row for row in rows if bool(row["accepted"])]
    lines.extend(
        [
            f"- Candidates evaluated: {len(rows)}",
            f"- Fully accepted candidates: {len(accepted)}",
            f"- Best candidate acceptance score: {best['acceptance_score']} / 5",
            f"- Best candidate release rate: {best['k_release_rate']:.3f}",
            f"- Best candidate clearance rate: {best['k_clearance_rate']:.3f}",
            f"- Best candidate K threshold: {best['k_threshold']:.2f}",
            f"- Best candidate arrival threshold: {best['k_arrival_threshold']:.2f}",
            f"- Best candidate scalar speed: {best['scalar_speed_mm_min']:.3f} mm/min",
            f"- Best candidate tensor speed: {best['tensor_speed_mm_min']:.3f} mm/min",
            f"- Best candidate tensor-minus-scalar: {best['tensor_minus_scalar_mm_min']:.3f} mm/min",
            f"- Best candidate scalar downstream delay: {best['scalar_downstream_delay_s']:.3f} s",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = args.output_root or ROOT / ("outputs/biophysical_validation_quick" if args.quick else "outputs/biophysical_validation")
    output_root.mkdir(parents=True, exist_ok=True)

    base = build_params(args.quick)
    rows: list[dict[str, float | str | bool]] = []
    for release_rate in args.release_rates:
        for clearance_rate in args.clearance_rates:
            for threshold in args.thresholds:
                for arrival_threshold in args.arrival_thresholds:
                    try:
                        result = evaluate_candidate(
                            base,
                            release_rate=float(release_rate),
                            clearance_rate=float(clearance_rate),
                            threshold=float(threshold),
                            arrival_threshold=float(arrival_threshold),
                        )
                        rows.append(result)
                        print(f"Tested: release={release_rate}, clearance={clearance_rate}, threshold={threshold}, score={result['acceptance_score']}")
                    except ValueError as e:
                        print(f"Skipped: release={release_rate}, clearance={clearance_rate}, threshold={threshold} - {e}")
                        continue

    if not rows:
        print("No candidates successfully completed.")
        return

    rows.sort(key=lambda row: (-int(row["acceptance_score"]), -float(row["tensor_minus_scalar_mm_min"]), float(row["scalar_delta_mm_min"])))
    write_csv(output_root / "biophysical_validation.csv", rows)
    write_summary(output_root / "biophysical_validation.md", rows)
    (output_root / "biophysical_validation.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Biophysical validation outputs written to {output_root}")


if __name__ == "__main__":
    main()
