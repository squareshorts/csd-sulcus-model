from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from csd_sulcus.analysis import summarize_reference_band  # noqa: E402


# ------------------------------------------------------------------ #
# Reference bands
#
#  Band 1 (broad)  : 2.0-5.0 mm/min — general experimental cortex
#                    Lauritzen et al. 2011, Physiol Rev
#  Band 2 (broad)  : 1.7-9.2 mm/min — human malignant stroke
#                    Woitzik et al. 2013, Ann Neurol
#  Band 3 (narrow) : 2.5-4.5 mm/min — controlled animal and human CSD
#                    recordings with known electrode geometry
#                    Ayata & Lauritzen 2015; Dreier et al. 2017 COSBID
#
# Delay reference  : 10-30 s inter-electrode delay across 3-5 mm
#                    tissue separations in human neocortex, from the
#                    COSBID guidelines (Dreier et al. 2017)
# ------------------------------------------------------------------ #

REFERENCE_BANDS = [
    {
        "key": "experimental_cortex",
        "label": "Experimental cortex (broad)",
        "lower_mm_min": 2.0,
        "upper_mm_min": 5.0,
        "citation": "Lauritzen et al. 2011",
    },
    {
        "key": "human_malignant_stroke",
        "label": "Human malignant stroke (broad)",
        "lower_mm_min": 1.7,
        "upper_mm_min": 9.2,
        "citation": "Woitzik et al. 2013",
    },
    {
        "key": "controlled_recordings_narrow",
        "label": "Controlled recordings (narrow)",
        "lower_mm_min": 2.5,
        "upper_mm_min": 4.5,
        "citation": "Ayata & Lauritzen 2015; Dreier et al. 2017",
    },
]

# Published downstream delay range for sulcus-scale electrode separations
# Dreier et al. 2017, J Cereb Blood Flow Metab 37:1595–1625
# Representative inter-electrode delays for 3–5 mm separations: 10–30 s
DELAY_BAND_LOWER_S = 10.0
DELAY_BAND_UPPER_S = 30.0
DELAY_CITATION = "Dreier et al. 2017"

METRICS = [
    ("control_speed_mm_min", "Control", "tab:gray"),
    ("scalar_speed_mm_min", "Scalar", "tab:red"),
    ("tensor_speed_mm_min", "Tensor", "tab:blue"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize accepted biophysical-validation candidates against published CSD speed bands and delay ranges."
    )
    parser.add_argument(
        "--validation-json",
        type=Path,
        default=ROOT / "outputs/biophysical_validation/biophysical_validation.json",
        help="Path to the JSON output produced by scripts/run_biophysical_validation.py",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/physiology_anchor",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_rows(path: Path) -> list[dict[str, object]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list of validation rows in {path}")
    return rows


def metric_values(rows: list[dict[str, object]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if key in row and row[key] is not None]


def summarize_metrics(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for key, label, _ in METRICS:
        values = metric_values(rows, key)
        summaries.append(
            {
                "metric_key": key,
                "metric_label": label,
                "range_min_mm_min": float(np.min(values)),
                "range_max_mm_min": float(np.max(values)),
                "mean_mm_min": float(np.mean(values)),
                "sd_mm_min": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
        )

    gains = metric_values(rows, "tensor_minus_scalar_mm_min")
    summaries.append(
        {
            "metric_key": "tensor_minus_scalar_mm_min",
            "metric_label": "Tensor \u2212 scalar",
            "range_min_mm_min": float(np.min(gains)),
            "range_max_mm_min": float(np.max(gains)),
            "mean_mm_min": float(np.mean(gains)),
            "sd_mm_min": float(np.std(gains, ddof=1)) if len(gains) > 1 else 0.0,
        }
    )
    return summaries


def summarize_bands(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for band in REFERENCE_BANDS:
        for key, label, _ in METRICS:
            band_summary = summarize_reference_band(
                metric_values(rows, key),
                label=f"{band['key']}:{key}",
                lower_mm_min=float(band["lower_mm_min"]),
                upper_mm_min=float(band["upper_mm_min"]),
            )
            summaries.append(
                {
                    "band_key": band["key"],
                    "band_label": band["label"],
                    "citation": band["citation"],
                    "lower_mm_min": float(band["lower_mm_min"]),
                    "upper_mm_min": float(band["upper_mm_min"]),
                    "metric_key": key,
                    "metric_label": label,
                    "n_total": band_summary.n_total,
                    "n_within": band_summary.n_within,
                    "all_within": band_summary.n_within == band_summary.n_total,
                }
            )
    return summaries


def summarize_delays(rows: list[dict[str, object]]) -> dict[str, object]:
    """Compare scalar downstream delays against the published COSBID range."""
    delays = metric_values(rows, "scalar_downstream_delay_s")
    n_total = len(delays)
    n_within = sum(1 for d in delays if DELAY_BAND_LOWER_S <= d <= DELAY_BAND_UPPER_S)
    return {
        "n_total": n_total,
        "n_within_delay_band": n_within,
        "delay_band_lower_s": DELAY_BAND_LOWER_S,
        "delay_band_upper_s": DELAY_BAND_UPPER_S,
        "delay_citation": DELAY_CITATION,
        "delay_min_s": float(np.min(delays)),
        "delay_max_s": float(np.max(delays)),
        "delay_mean_s": float(np.mean(delays)),
        "delay_sd_s": float(np.std(delays, ddof=1)) if len(delays) > 1 else 0.0,
    }


def write_table(
    path: Path,
    metric_summary: list[dict[str, object]],
    band_summary: list[dict[str, object]],
    delay_summary: dict[str, object],
) -> None:

    def cov(metric_key: str, band_key: str) -> str:
        row = next(
            item for item in band_summary
            if item["metric_key"] == metric_key and item["band_key"] == band_key
        )
        return f"{row['n_within']}/{row['n_total']}"

    lines = [
        r"\begin{tabular}{@{}llllll@{}}",
        r"\toprule",
        r"Metric & Range (mm/min) & Mean & "
        r"Within 2.0--5.0 & Within 1.7--9.2 & Within 2.5--4.5 \\",
        r"\midrule",
    ]

    for row in metric_summary:
        key = str(row["metric_key"])
        label = str(row["metric_label"])
        lo = float(row["range_min_mm_min"])
        hi = float(row["range_max_mm_min"])
        mean = float(row["mean_mm_min"])
        if key == "tensor_minus_scalar_mm_min":
            c_exp = "--"
            c_human = "--"
            c_narrow = "--"
        else:
            c_exp = cov(key, "experimental_cortex")
            c_human = cov(key, "human_malignant_stroke")
            c_narrow = cov(key, "controlled_recordings_narrow")
        lines.append(
            rf"{label} & {lo:.3f}--{hi:.3f} & {mean:.3f} & {c_exp} & {c_human} & {c_narrow} \\"
        )

    # Delay row
    n_w = delay_summary["n_within_delay_band"]
    n_t = delay_summary["n_total"]
    d_lo = delay_summary["delay_min_s"]
    d_hi = delay_summary["delay_max_s"]
    d_mean = delay_summary["delay_mean_s"]
    lines.append(r"\midrule")
    lines.append(
        rf"Scalar downstream delay (s) & {d_lo:.1f}--{d_hi:.1f} s & {d_mean:.1f} s & "
        rf"\multicolumn{{3}}{{l}}{{Within 10--30 s band: {n_w}/{n_t}}} \\"
    )

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_anchor(
    rows: list[dict[str, object]],
    delay_summary: dict[str, object],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)

    # ---- Panel A: speed distributions with all three bands ----
    left = axes[0]
    left.axhspan(1.7, 9.2, color="#d9d9d9", alpha=0.35, label="Human 1.7–9.2")
    left.axhspan(2.0, 5.0, color="#9ecae1", alpha=0.40, label="Experimental 2.0–5.0")
    left.axhspan(2.5, 4.5, color="#fdae6b", alpha=0.45, label="Narrow 2.5–4.5")

    for idx, (key, label, color) in enumerate(METRICS, start=1):
        values = np.sort(np.asarray(metric_values(rows, key), dtype=float))
        jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else np.array([0.0])
        left.scatter(
            np.full(len(values), idx, dtype=float) + jitter,
            values, s=24, color=color, alpha=0.75, edgecolor="none",
        )
        left.hlines(np.mean(values), idx - 0.22, idx + 0.22, color="black", lw=2)

    left.set_xlim(0.5, len(METRICS) + 0.5)
    left.set_xticks([1, 2, 3], [label for _, label, _ in METRICS])
    left.set_ylabel("Speed (mm/min)")
    left.legend(loc="upper right", frameon=False, fontsize=8)

    # ---- Panel B: tensor-minus-scalar gains ----
    mid = axes[1]
    gains = np.sort(np.asarray(metric_values(rows, "tensor_minus_scalar_mm_min"), dtype=float))
    rank = np.arange(1, len(gains) + 1)
    mid.axhline(0.0, color="0.35", lw=1.2, ls=":")
    mid.fill_between([1, len(gains)], 0.0, float(np.max(gains)) * 1.08,
                     color="#c7e9c0", alpha=0.35)
    mid.plot(rank, gains, color="tab:green", lw=1.8)
    mid.scatter(rank, gains, color="tab:green", s=22)
    mid.set_xlim(1, len(gains))
    mid.set_xlabel("Candidate rank")
    mid.set_ylabel("Tensor \u2212 scalar (mm/min)")

    # ---- Panel C: scalar downstream delay vs. COSBID band ----
    right = axes[2]
    delays = np.sort(np.asarray(metric_values(rows, "scalar_downstream_delay_s"), dtype=float))
    rank_d = np.arange(1, len(delays) + 1)
    right.axhspan(
        DELAY_BAND_LOWER_S, DELAY_BAND_UPPER_S,
        color="#fdae6b", alpha=0.35, label=f"COSBID {DELAY_BAND_LOWER_S:.0f}–{DELAY_BAND_UPPER_S:.0f} s",
    )
    right.plot(rank_d, delays, color="tab:red", lw=1.8)
    right.scatter(rank_d, delays, color="tab:red", s=22)
    right.set_xlim(1, len(delays))
    right.set_xlabel("Candidate rank")
    right.set_ylabel("Scalar downstream delay (s)")
    right.legend(loc="upper left", frameon=False, fontsize=9)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    path: Path,
    rows: list[dict[str, object]],
    metric_summary: list[dict[str, object]],
    band_summary: list[dict[str, object]],
    delay_summary: dict[str, object],
) -> None:
    best = max(rows, key=lambda row: float(row["tensor_minus_scalar_mm_min"]))
    metric_lookup = {row["metric_key"]: row for row in metric_summary}

    def cov(metric_key: str, band_key: str) -> str:
        row = next(item for item in band_summary
                   if item["metric_key"] == metric_key and item["band_key"] == band_key)
        return f"{row['n_within']}/{row['n_total']}"

    n_w = delay_summary["n_within_delay_band"]
    n_t = delay_summary["n_total"]

    lines = [
        "# Physiology Anchor Summary (upgraded)",
        "",
        f"- Accepted validation candidates: {len(rows)}",
        "",
        "## Speed distributions",
        f"- Control speed range:   {metric_lookup['control_speed_mm_min']['range_min_mm_min']:.3f}–{metric_lookup['control_speed_mm_min']['range_max_mm_min']:.3f} mm/min",
        f"- Scalar speed range:    {metric_lookup['scalar_speed_mm_min']['range_min_mm_min']:.3f}–{metric_lookup['scalar_speed_mm_min']['range_max_mm_min']:.3f} mm/min",
        f"- Tensor speed range:    {metric_lookup['tensor_speed_mm_min']['range_min_mm_min']:.3f}–{metric_lookup['tensor_speed_mm_min']['range_max_mm_min']:.3f} mm/min",
        f"- Mean tensor−scalar:    {metric_lookup['tensor_minus_scalar_mm_min']['mean_mm_min']:.3f} ± {metric_lookup['tensor_minus_scalar_mm_min']['sd_mm_min']:.3f} mm/min",
        "",
        "## Speed-band coverage",
        f"  Broad band 2.0-5.0 mm/min (Lauritzen 2011):",
        f"    Scalar:  {cov('scalar_speed_mm_min', 'experimental_cortex')}",
        f"    Tensor:  {cov('tensor_speed_mm_min', 'experimental_cortex')}",
        f"  Broad band 1.7-9.2 mm/min (Woitzik 2013):",
        f"    Scalar:  {cov('scalar_speed_mm_min', 'human_malignant_stroke')}",
        f"    Tensor:  {cov('tensor_speed_mm_min', 'human_malignant_stroke')}",
        f"  NARROW band 2.5-4.5 mm/min (Ayata & Lauritzen 2015; Dreier 2017):",
        f"    Scalar:  {cov('scalar_speed_mm_min', 'controlled_recordings_narrow')}",
        f"    Tensor:  {cov('tensor_speed_mm_min', 'controlled_recordings_narrow')}",
        "",
        "## Downstream delay comparison",
        f"- Scalar downstream delay range: {delay_summary['delay_min_s']:.1f}–{delay_summary['delay_max_s']:.1f} s",
        f"- Mean delay: {delay_summary['delay_mean_s']:.1f} ± {delay_summary['delay_sd_s']:.1f} s",
        f"- Published COSBID range (Dreier 2017): {DELAY_BAND_LOWER_S:.0f}–{DELAY_BAND_UPPER_S:.0f} s",
        f"- Candidates within COSBID delay band: {n_w}/{n_t}",
        "",
        "## Best accepted candidate",
        f"- Tensor−scalar gain:   {float(best['tensor_minus_scalar_mm_min']):.3f} mm/min",
        f"- Scalar speed:         {float(best['scalar_speed_mm_min']):.3f} mm/min",
        f"- Tensor speed:         {float(best['tensor_speed_mm_min']):.3f} mm/min",
        f"- Downstream delay:     {float(best['scalar_downstream_delay_s']):.1f} s",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.validation_json)
    accepted = [row for row in rows if bool(row.get("accepted"))]
    if not accepted:
        raise ValueError(f"No accepted candidates found in {args.validation_json}")

    metric_summary = summarize_metrics(accepted)
    band_summary = summarize_bands(accepted)
    delay_summary = summarize_delays(accepted)

    write_csv(output_root / "physiology_anchor_metrics.csv", metric_summary)
    write_csv(output_root / "physiology_anchor_band_coverage.csv", band_summary)
    write_table(
        output_root / "physiology_anchor_table.tex",
        metric_summary,
        band_summary,
        delay_summary,
    )
    plot_anchor(accepted, delay_summary, figure_dir / "fig_physiology_anchor.png")
    write_summary(
        output_root / "physiology_anchor_summary.md",
        accepted,
        metric_summary,
        band_summary,
        delay_summary,
    )

    payload = {
        "accepted_candidates": len(accepted),
        "metric_summary": metric_summary,
        "band_summary": band_summary,
        "delay_summary": delay_summary,
    }
    (output_root / "physiology_anchor_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"Physiology anchor outputs written to {output_root}")
    print(f"  Narrow band (2.5-4.5 mm/min) scalar coverage: {next(x for x in band_summary if x['band_key']=='controlled_recordings_narrow' and x['metric_key']=='scalar_speed_mm_min')['n_within']}/{next(x for x in band_summary if x['band_key']=='controlled_recordings_narrow' and x['metric_key']=='scalar_speed_mm_min')['n_total']}")
    print(f"  Downstream delay in COSBID band: {delay_summary['n_within_delay_band']}/{delay_summary['n_total']}")


if __name__ == "__main__":
    main()
