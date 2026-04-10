from __future__ import annotations

import argparse
import csv
import dataclasses as dc
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from csd_sulcus.analysis import compare_control_vs_dipole, multi_seed_robustness, sweep_g_sulcus
from csd_sulcus.model import Params
from csd_sulcus.plotting import (
    plot_coupling_and_arrivals,
    plot_theory_vs_observed_sweep,
    plot_velocity_vs_coupling,
    plot_virtual_electrode_arrivals,
    plot_wavefront_snapshots,
)


E1 = (0.43, 0.50)
E2 = (0.62, 0.50)
R_MM = 1.0
DEFAULT_GMINS = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75]


def build_params(quick: bool) -> Params:
    p = Params()
    if not quick:
        return p
    return dc.replace(
        p,
        nx=120,
        ny=84,
        final_t_end=80.0,
        sulcus_width_mm=3.0,
        g_smooth_mm=0.8,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CSD sulcus study workflow.")
    parser.add_argument("--quick", action="store_true", help="Use a smaller domain and shorter runtime for fast iteration.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory where figures and summary files are written.",
    )
    parser.add_argument(
        "--snapshot-times",
        type=float,
        nargs="*",
        default=[12.0, 18.0, 24.0],
        help="Wavefront snapshot times in seconds.",
    )
    parser.add_argument(
        "--gmins",
        type=float,
        nargs="*",
        default=DEFAULT_GMINS,
        help="Values used for the g_sulcus sweep.",
    )
    parser.add_argument("--num-seeds", type=int, default=5, help="Number of seeds used in the reproducibility sweep.")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_markdown(path: Path, summary: dict[str, float | str]) -> None:
    lines = [
        "# Study Summary",
        "",
        f"- Mode: {summary['mode']}",
        f"- Fixed scale: {summary['fixed_scale']:.7f}",
        f"- Control speed: {summary['control_speed_mm_min']:.3f} mm/min",
        f"- Dipole speed: {summary['dipole_speed_mm_min']:.3f} mm/min",
        f"- Delta speed: {summary['delta_speed_mm_min']:.4f} mm/min ({summary['delta_speed_percent']:.2f}%)",
        f"- Median sulcal delay: {summary['delay_sulcus_s']:.3f} s",
        f"- Median upstream delay: {summary['delay_upstream_s']:.3f} s",
        f"- Median downstream delay: {summary['delay_downstream_s']:.3f} s",
        f"- Peak delay: {summary['peak_delay_s']:.3f} s",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    params = build_params(args.quick)
    snapshot_times = [time for time in args.snapshot_times if time <= params.final_t_end]

    output_root = args.output_root
    if output_root is None:
        output_root = ROOT / ("outputs/quick" if args.quick else "outputs/full")

    figure_dir = output_root / "figures"
    data_dir = output_root / "data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    comparison = compare_control_vs_dipole(
        params,
        E1,
        E2,
        radius_mm=R_MM,
        snapshot_times=snapshot_times,
    )
    control_speed = comparison.control_measurement.scaled_speed_mm_min
    dipole_speed = comparison.dipole_measurement.scaled_speed_mm_min
    delta_speed = dipole_speed - control_speed

    sweep_rows = sweep_g_sulcus(
        params,
        E1,
        E2,
        radius_mm=R_MM,
        fixed_scale=comparison.fixed_scale,
        gmins=args.gmins,
        control_speed_mm_min=control_speed,
    )
    seed_rows = multi_seed_robustness(
        params,
        E1,
        E2,
        radius_mm=R_MM,
        fixed_scale=comparison.fixed_scale,
        num_seeds=args.num_seeds,
    )

    plot_coupling_and_arrivals(comparison, figure_dir / "fig1_diffusion_arrival.png")
    if snapshot_times:
        plot_wavefront_snapshots(comparison.control, comparison.dipole, figure_dir / "fig2_wavefront_snapshots.png")
    plot_virtual_electrode_arrivals(comparison, E1, E2, R_MM, figure_dir / "fig3_virtual_electrodes.png")
    plot_velocity_vs_coupling(sweep_rows, control_speed, figure_dir / "fig4_velocity_vs_g.png")
    plot_theory_vs_observed_sweep(sweep_rows, figure_dir / "fig5_theory_vs_observed.png")

    delay_map_s = comparison.dipole.arr / comparison.fixed_scale - comparison.control.arr / comparison.fixed_scale
    np.savez_compressed(
        data_dir / "core_fields.npz",
        control_arr=comparison.control.arr,
        dipole_arr=comparison.dipole.arr,
        g_field=comparison.dipole.g_field,
        sulcus_mask=comparison.dipole.sulc_mask,
        delay_map_s=delay_map_s,
        snapshot_times=np.asarray(snapshot_times, dtype=float),
        control_snapshots=comparison.control.snapshot_fields,
        dipole_snapshots=comparison.dipole.snapshot_fields,
    )

    summary = {
        "mode": "quick" if args.quick else "full",
        "fixed_scale": float(comparison.fixed_scale),
        "control_speed_mm_min": float(control_speed),
        "dipole_speed_mm_min": float(dipole_speed),
        "delta_speed_mm_min": float(delta_speed),
        "delta_speed_percent": float(100.0 * delta_speed / control_speed),
        "delay_sulcus_s": float(comparison.delay_sulcus_s),
        "delay_upstream_s": float(comparison.delay_upstream_s),
        "delay_downstream_s": float(comparison.delay_downstream_s),
        "peak_delay_s": float(comparison.peak_delay_s),
        "params": dc.asdict(params),
        "electrodes": {"E1": E1, "E2": E2, "radius_mm": R_MM},
    }

    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary_markdown(output_root / "summary.md", summary)
    write_csv(output_root / "velocity_sweep.csv", sweep_rows)
    write_csv(output_root / "multi_seed.csv", seed_rows)

    print(f"Outputs written to {output_root}")
    print(f"Control speed: {control_speed:.3f} mm/min")
    print(f"Dipole speed:  {dipole_speed:.3f} mm/min")
    print(f"Delta speed:   {delta_speed:.4f} mm/min")


if __name__ == "__main__":
    main()
