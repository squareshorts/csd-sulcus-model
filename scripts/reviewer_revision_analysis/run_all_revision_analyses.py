"""Reproduce the isolated reviewer-revision computational analyses.

This driver intentionally writes only below results/reviewer_revision_analysis.
It imports, but never modifies, the study implementation and archived outputs.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses as dc
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
from scipy import sparse
from scipy.signal import savgol_filter
from scipy.sparse import csgraph

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
OUT = ROOT / "results" / "reviewer_revision_analysis"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SCRIPTS))

from csd_sulcus.surface_io import generate_folded_strip_mesh
from csd_sulcus.surface_mechanistic import (
    MechanisticSurfaceParams,
    mechanistic_surface_arrival_speed_mm_min,
    run_mechanistic_surface_simulation,
)
import csd_sulcus.surface_mechanistic as sm
from csd_sulcus.surface_model import median_arrival
from run_surface_representative import choose_auto_vertices


SEED = 20260714
BASELINE_STATUS_BEFORE_ANALYSIS = """ M manuscript/figures/fig2_rep_quantitative.pdf
 M manuscript/figures/fig2_rep_quantitative.png
 M scripts/run_fig2_rep_quantitative.py
 M src/csd_sulcus_model.egg-info/PKG-INFO
 M src/csd_sulcus_model.egg-info/SOURCES.txt
 M src/csd_sulcus_model.egg-info/dependency_links.txt
 M src/csd_sulcus_model.egg-info/requires.txt
 M src/csd_sulcus_model.egg-info/top_level.txt
?? outputs/sindy_physics_export/flat_no_dipole/
?? outputs/sindy_physics_export/folded_dipole_aligned/
?? outputs/sindy_physics_export/folded_distance_only_null/
?? outputs/sindy_physics_export/folded_no_dipole/
?? outputs/sindy_physics_export/folded_scrambled_normal_null/
?? uv.lock"""


def run_cmd(args: list[str], *, check: bool = True) -> str:
    proc = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)
    if check and proc.returncode:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(args)}\n{proc.stderr}")
    return (proc.stdout + proc.stderr).strip()


def ensure_output_path(path: Path) -> Path:
    path = path.resolve()
    if OUT.resolve() not in path.parents and path != OUT.resolve():
        raise RuntimeError(f"Refusing to write outside {OUT}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_text(path: Path, text: str) -> None:
    ensure_output_path(path).write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, data) -> None:
    ensure_output_path(path).write_text(json.dumps(json_ready(data), indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path = ensure_output_path(path)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_figure(fig, prefix: Path) -> None:
    for suffix, kwargs in ((".pdf", {}), (".svg", {}), (".png", {"dpi": 300})):
        path = ensure_output_path(prefix.with_suffix(suffix))
        fig.savefig(path, bbox_inches="tight", **kwargs)
    plt.close(fig)


def shortest_path(graph: sparse.csr_matrix, start: int, end: int) -> np.ndarray:
    _, pred = csgraph.dijkstra(graph, directed=False, indices=int(start), return_predecessors=True)
    result = [int(end)]
    cursor = int(end)
    while cursor != int(start):
        cursor = int(pred[cursor])
        if cursor < 0:
            raise RuntimeError("Could not reconstruct geodesic path")
        result.append(cursor)
    return np.asarray(result[::-1], dtype=int)


def cumulative_distance(vertices: np.ndarray) -> np.ndarray:
    out = np.zeros(len(vertices), dtype=float)
    if len(vertices) > 1:
        out[1:] = np.cumsum(np.linalg.norm(np.diff(vertices, axis=0), axis=1))
    return out


def full_cross_fold_path(mesh, x_target: float, *, descending_y: bool = True) -> np.ndarray:
    x_values = np.unique(mesh.vertices[:, 0])
    x_value = float(x_values[np.argmin(np.abs(x_values - x_target))])
    idx = np.where(np.isclose(mesh.vertices[:, 0], x_value, rtol=0.0, atol=1e-10))[0]
    order = np.argsort(mesh.vertices[idx, 1])
    if descending_y:
        order = order[::-1]
    return idx[order]


def roi_metrics(output, e1: int, e2: int, radius_mm: float = 1.0) -> dict:
    graph = output.operators.graph
    d1 = np.asarray(csgraph.dijkstra(graph, directed=False, indices=int(e1)), dtype=float)
    d2 = np.asarray(csgraph.dijkstra(graph, directed=False, indices=int(e2)), dtype=float)
    roi1 = d1 <= radius_mm
    roi2 = d2 <= radius_mm
    t1 = median_arrival(output.arrival_times, roi1)
    t2 = median_arrival(output.arrival_times, roi2)
    distance = float(d1[e2])
    return {
        "roi_e1": roi1,
        "roi_e2": roi2,
        "roi_e1_arrival_s": float(t1),
        "roi_e2_arrival_s": float(t2),
        "roi_delay_s": float(t2 - t1),
        "geodesic_distance_mm": distance,
        "speed_mm_min_recomputed": float(60.0 * distance / (t2 - t1)),
        "speed_mm_min_function": float(mechanistic_surface_arrival_speed_mm_min(output, e1, e2, radius_mm=radius_mm)),
        "vertex_e1_arrival_s": float(output.arrival_times[e1]),
        "vertex_e2_arrival_s": float(output.arrival_times[e2]),
        "vertex_delay_s": float(output.arrival_times[e2] - output.arrival_times[e1]),
    }


def collect_environment() -> dict:
    packages = {}
    for name in ("numpy", "scipy", "matplotlib", "pytest", "nibabel", "nilearn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    gpu = run_cmd(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], check=False) if shutil.which("nvidia-smi") else "nvidia-smi unavailable; no CUDA GPU inventory exposed"
    memory = ""
    if platform.system() == "Windows":
        memory = run_cmd(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"], check=False)
    env = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": run_cmd(["git", "branch", "--show-current"]),
        "git_commit": run_cmd(["git", "rev-parse", "HEAD"]),
        "git_status_at_driver_start": run_cmd(["git", "status", "--short"], check=False),
        "git_status_before_analysis_from_initial_reconnaissance": BASELINE_STATUS_BEFORE_ANALYSIS,
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "packages": packages,
        "os": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "physical_memory_bytes": int(memory.strip()) if memory.strip().isdigit() else "unavailable",
        "gpu": gpu or "nvidia-smi returned no devices",
        "random_seed": SEED,
    }
    lines = [
        f"Captured UTC: {env['captured_utc']}",
        f"Git branch: {env['git_branch']}",
        f"Git commit: {env['git_commit']}",
        "Git status before analysis (captured during initial reconnaissance):",
        env["git_status_before_analysis_from_initial_reconnaissance"],
        "Git status at driver start:",
        env["git_status_at_driver_start"],
        f"Python: {env['python']}",
        f"Executable: {env['python_executable']}",
        f"Packages: {json.dumps(packages, sort_keys=True)}",
        f"Operating system: {env['os']}",
        f"Processor: {env['processor']}",
        f"Logical CPUs: {env['logical_cpu_count']}",
        f"Physical memory bytes: {env['physical_memory_bytes']}",
        f"GPU inventory: {env['gpu']}",
        f"Fixed seed: {SEED}",
    ]
    write_text(OUT / "00_environment.txt", "\n".join(lines))
    return env


def repository_inventory(env: dict) -> list[dict]:
    inventory = [
        ("Representative folded simulation", "scripts/run_surface_mechanistic_study.py", "_representative_rows; REPRESENTATIVE_CASES"),
        ("Representative solver", "src/csd_sulcus/surface_mechanistic.py", "MechanisticSurfaceParams; run_mechanistic_surface_simulation"),
        ("Synthetic mesh and Gaussian fold", "src/csd_sulcus/surface_io.py", "generate_folded_strip_mesh"),
        ("Readout selection", "scripts/run_surface_representative.py", "choose_auto_vertices"),
        ("Speed/readout functions", "src/csd_sulcus/surface_mechanistic.py", "mechanistic_surface_arrival_speed_mm_min"),
        ("Current representative CSV", "outputs/surface_mechanistic_study/mechanistic_representative_summary.csv", "2.609/2.537 current saved outputs"),
        ("Current mechanistic summary", "outputs/surface_mechanistic_study/mechanistic_study_summary.json", "representative, sweep, null, atlas, convergence summaries"),
        ("Stale generated Table S2", "outputs/surface_mechanistic_study/table_s2_exact_representative_run.tex", "2.558/2.476 pre-swelling-revision outputs"),
        ("Current manuscript Table S2 source", "manuscript/table_s2_exact_representative_run.tex", "2.609/2.537 current outputs; read only"),
        ("Figure 2 generator", "scripts/run_fig2_rep_quantitative.py", "working-tree hard-coded display values; read only"),
        ("Figure 2 production assets", "manuscript/figures/fig2_rep_quantitative.pdf; manuscript/figures/fig2_rep_quantitative.png", "pre-existing working-tree modifications; read only"),
        ("Figure 3 propagation source", "scripts/run_surface_mechanistic_study.py", "_save_propagation_figure"),
        ("Figure 3 archived data/figure", "outputs/surface_mechanistic_study/mechanistic_wave_propagation.png", "generated from current mechanistic pipeline"),
        ("Synthetic geometry sweep", "outputs/surface_mechanistic_study/mechanistic_geometry_sweep.csv", "52x24 meshes, 210 s"),
        ("Atlas patches", "src/csd_sulcus/atlas_patch.py; outputs/surface_mechanistic_study/mechanistic_atlas_patch_check.csv", "multi-patch pipeline and outputs"),
        ("Null kernels", "outputs/surface_mechanistic_study/mechanistic_null_models.csv", "aligned, distance-only, scrambled-normal"),
        ("Convergence", "outputs/surface_mechanistic_study/mechanistic_convergence.csv", "neighboring mesh and time-step checks"),
        ("Model equations/parameters", "src/csd_sulcus/surface_mechanistic.py; src/csd_sulcus/surface_model.py; src/csd_sulcus/surface_ops.py", "implemented model and discretization"),
        ("Manuscript declarations", "manuscript/reframed_submission.tex", "primary specification and reported values; read only"),
        ("SINDy/null raw snapshots", "outputs/sindy_physics_export/*", "untracked pre-existing physics exports; not used as representative authority"),
    ]
    rows = [{"category": a, "path": b, "role": c, "exists": all((ROOT / p.strip()).exists() for p in b.split(";") if "*" not in p)} for a, b, c in inventory]
    md = [
        "# Repository inventory and provenance reconnaissance",
        "",
        f"- Branch: `{env['git_branch']}`",
        f"- Commit: `{env['git_commit']}`",
        "- The working tree was already dirty before this analysis. Those pre-existing files were treated as user-owned and were not modified.",
        "",
        "## Relevant files",
        "",
        "| Category | Path(s) | Role |",
        "|---|---|---|",
    ]
    md.extend(f"| {a} | `{b}` | {c} |" for a, b, c in inventory)
    md += [
        "",
        "## Provenance resolution",
        "",
        "The 2.558/2.476 mm/min result is an archived generated table from the implementation before commit `7d054b8` changed the swelling law from an unbounded/clipped 1.5 state to a bounded saturating target with separate recovery. The representative CSV and JSON were rerun after that model change and contain 2.609/2.537 mm/min. The archived output-side Table S2 was not regenerated and is stale.",
        "",
        "The current tracked Figure 2 generator at HEAD uses 2.609/2.537, but the working tree contains a pre-existing uncommitted edit that replaces those labels with the stale 2.558/2.476 values. The current mechanistic propagation figure and manuscript-side Table S2 use the post-revision values. Neither result comes from the distance-only or scrambled-normal null pipeline; the null pipeline uses the same vertices but distinct kernel modes and is stored separately.",
        "",
        "Both representative result sets used the 64x28 (1,792 vertex, 3,402 face) mesh; stimulus vertex 471; E1 vertex 636; E2 vertex 624; 1.2 mm initial perturbation; -28 mV arrival threshold; 220 s duration; and automatic 0.049969957 s step. The material difference is the swelling implementation and its defaults, not geometry, electrodes, seed, stimulus, threshold, duration, or time step.",
        "",
        "## Readout-definition finding",
        "",
        "The saved `cross_fold_delay_s` is E2 vertex arrival minus E1 vertex arrival. The saved speed instead uses the same E1/E2 centers but median arrival over separate 1 mm geodesic regions, divided into the center-to-center geodesic distance. The two time differences are close but not definitionally identical. New canonical outputs preserve both.",
    ]
    write_text(OUT / "00_repository_inventory.md", "\n".join(md))
    return rows


def provenance_map() -> list[dict]:
    common = {
        "mesh": "synthetic 64x28; 1792 vertices; 3402 faces; 22x10 mm; depth 2.4 mm; sigma 1.5 mm",
        "stimulus_definition": "initial-only Euclidean 3D radius <=1.2 mm about vertex 471; Ke>=22 mM; Nae-=10 mM (floor 5); theta>=0.92",
        "electrode_definition": "E1 center vertex 636; E2 center vertex 624; speed uses 1 mm geodesic ROIs; saved delay uses center vertices",
        "arrival_definition": "first explicit-Euler sample t>=0.5 s with Vm>=-28 mV; no temporal interpolation",
    }
    entries = []
    old_values = [
        ("baseline speed", "2.5582349901822656 mm/min"), ("dipole speed", "2.475787705816325 mm/min"),
        ("baseline delay", "130.52152862104504 s"), ("dipole delay", "134.66903508182097 s"),
        ("delay increase", "4.14750646077593 s"), ("baseline max |Ve|", "18.987306193669685 mV"),
        ("dipole max |Ve|", "19.50089527429308 mV"),
    ]
    for item, value in old_values:
        entries.append({"reported_item": f"stale/pre-revision {item}", "reported_value": value,
            "source_file": "outputs/surface_mechanistic_study/table_s2_exact_representative_run.tex",
            "source_script": "scripts/run_surface_mechanistic_study.py at commit 4614df6",
            "configuration": "pre-7d054b8 swelling: linear osmotic target +0.20 theta, clip s to 1.5",
            **common, "run_timestamp_if_available": "file committed 2026-04-12 19:44:19 -0300; run timestamp absent", "status_current_or_stale": "stale after model implementation change"})
    current_values = [
        ("baseline speed", "2.6091699831548576 mm/min"), ("dipole speed", "2.5368718587402426 mm/min"),
        ("baseline vertex delay", "127.97306079574898 s"), ("dipole vertex delay", "131.42098785350245 s"),
        ("delay increase", "3.44792705775347 s"), ("baseline max |Ve|", "15.433485430844861 mV"),
        ("dipole max |Ve|", "16.384820032184784 mV"),
    ]
    for item, value in current_values:
        entries.append({"reported_item": f"current/post-revision {item}", "reported_value": value,
            "source_file": "outputs/surface_mechanistic_study/mechanistic_representative_summary.csv; outputs/surface_mechanistic_study/mechanistic_study_summary.json",
            "source_script": "scripts/run_surface_mechanistic_study.py::_representative_rows",
            "configuration": "current bounded saturating swelling target; cap 1.10; recovery tau 28 s; no vascular feedback",
            **common, "run_timestamp_if_available": "outputs committed 2026-05-03 20:18:02 -0300; JSON runtime 566.877 s; run timestamp absent", "status_current_or_stale": "current authoritative implementation output"})
    entries.append({"reported_item": "working-tree Figure 2 labels", "reported_value": "2.558/2.476 mm/min; 130.5/134.7 s; 18.99/19.50 mV",
        "source_file": "scripts/run_fig2_rep_quantitative.py; manuscript/figures/fig2_rep_quantitative.*",
        "source_script": "scripts/run_fig2_rep_quantitative.py (hard-coded DISPLAY_VALUES and panel-C rows)",
        "configuration": "pre-existing uncommitted label substitution; trace pipeline still imports current REPRESENTATIVE_CASES",
        **common, "run_timestamp_if_available": "working-tree file mtime 2026-07-03; exact run timestamp unavailable", "status_current_or_stale": "internally mixed/stale labels in uncommitted working tree"})
    write_csv(OUT / "00_provenance_map.csv", entries)
    return entries


def canonical_run(log: list[str]) -> dict:
    out_dir = OUT / "01_canonical_representative"
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh = generate_folded_strip_mesh(nx=64, ny=28, length_mm=22.0, width_mm=10.0, fold_depth_mm=2.4, fold_sigma_mm=1.5)
    stim, e1, e2 = choose_auto_vertices(mesh)
    params = MechanisticSurfaceParams(final_t_end=220.0, enable_dipole_alignment=False, enable_vascular_feedback=False)
    dip_params = dc.replace(params, enable_dipole_alignment=True, dipole_kernel_mode="aligned")
    requested_times = np.arange(0.0, params.final_t_end + 0.25, 0.5)
    np.random.seed(SEED)
    start = time.perf_counter()
    baseline = run_mechanistic_surface_simulation(mesh, params, stimulus_vertex=stim, snapshot_times=requested_times)
    t_baseline = time.perf_counter() - start
    start = time.perf_counter()
    dipole = run_mechanistic_surface_simulation(mesh, dip_params, stimulus_vertex=stim, snapshot_times=requested_times)
    t_dipole = time.perf_counter() - start
    bm = roi_metrics(baseline, e1, e2)
    dm = roi_metrics(dipole, e1, e2)
    stim_mask = np.linalg.norm(mesh.vertices - mesh.vertices[stim], axis=1) <= params.stim_radius_mm
    readout_path = shortest_path(baseline.operators.graph, e1, e2)
    readout_s = cumulative_distance(mesh.vertices[readout_path])
    full_path = full_cross_fold_path(mesh, mesh.vertices[e1, 0], descending_y=True)
    full_s = cumulative_distance(mesh.vertices[full_path])
    config = {
        "authority": "current declared primary mechanistic specification and current post-swelling-revision implementation",
        "seed": SEED,
        "mesh_arguments": {"nx": 64, "ny": 28, "length_mm": 22.0, "width_mm": 10.0, "fold_depth_mm": 2.4, "fold_sigma_mm": 1.5},
        "mesh_realized": {"vertices": mesh.n_vertices, "faces": mesh.n_faces, "max_sampled_depth_mm": float(-np.min(mesh.vertices[:, 2]))},
        "params_no_dipole": dc.asdict(params),
        "params_dipole_aligned": dc.asdict(dip_params),
        "stimulus_vertex": stim,
        "stimulus_coordinate_mm": mesh.vertices[stim],
        "stimulus_vertex_count": int(stim_mask.sum()),
        "e1_vertex": e1,
        "e1_coordinate_mm": mesh.vertices[e1],
        "e2_vertex": e2,
        "e2_coordinate_mm": mesh.vertices[e2],
        "electrode_roi_radius_mm": 1.0,
        "arrival_definition": "first discrete solver sample at t>=0.5 s with Vm>=-28 mV; no temporal interpolation",
        "speed_definition": "60 * E1-center-to-E2-center geodesic distance (mm) / (E2 ROI median arrival - E1 ROI median arrival) (s)",
        "saved_delay_definition_in_original_pipeline": "E2 center-vertex arrival minus E1 center-vertex arrival",
        "snapshot_request_interval_s": 0.5,
        "software_commit": run_cmd(["git", "rev-parse", "HEAD"]),
    }
    write_json(out_dir / "canonical_config.json", config)

    def save_timeseries(path: Path, output) -> None:
        ensure_output_path(path)
        np.savez_compressed(
            path,
            snapshot_times_s=output.snapshot_times,
            membrane_voltage_mV=output.snapshot_voltage_mv,
            extracellular_potassium_mM=output.snapshot_potassium_e,
            extracellular_potential_hat=output.snapshot_potential,
            extracellular_potential_mV=output.params.field_reference_mV * output.snapshot_potential,
            arrival_times_s=output.arrival_times,
            final_membrane_voltage_mV=output.membrane_voltage_mv,
            final_extracellular_potassium_mM=output.potassium_e,
            final_extracellular_sodium_mM=output.sodium_e,
            final_extracellular_chloride_mM=output.chloride_e,
            final_intracellular_potassium_mM=output.potassium_i,
            final_intracellular_sodium_mM=output.sodium_i,
            final_intracellular_chloride_mM=output.chloride_i,
            final_activation=output.activation,
            final_swelling=output.swelling,
            final_ecs_volume_fraction=output.ecs_volume_fraction,
            final_ecs_tortuosity=output.ecs_tortuosity,
            e1_voltage_trace_mV=output.snapshot_voltage_mv[:, e1],
            e2_voltage_trace_mV=output.snapshot_voltage_mv[:, e2],
            e1_potassium_trace_mM=output.snapshot_potassium_e[:, e1],
            e2_potassium_trace_mM=output.snapshot_potassium_e[:, e2],
            e1_potential_trace_mV=output.params.field_reference_mV * output.snapshot_potential[:, e1],
            e2_potential_trace_mV=output.params.field_reference_mV * output.snapshot_potential[:, e2],
            dt_used_s=output.dt_used,
        )

    save_timeseries(out_dir / "no_dipole_timeseries.npz", baseline)
    save_timeseries(out_dir / "dipole_aligned_timeseries.npz", dipole)
    np.savez_compressed(
        ensure_output_path(out_dir / "mesh_and_readouts.npz"),
        vertices_mm=mesh.vertices,
        faces=mesh.faces,
        stimulus_vertex=stim,
        stimulus_vertices=np.where(stim_mask)[0],
        e1_vertex=e1,
        e2_vertex=e2,
        e1_roi_vertices=np.where(bm["roi_e1"])[0],
        e2_roi_vertices=np.where(bm["roi_e2"])[0],
        readout_geodesic_path_vertices=readout_path,
        readout_geodesic_path_distance_mm=readout_s,
        full_cross_fold_path_vertices=full_path,
        full_cross_fold_path_distance_mm=full_s,
    )
    metrics = []
    for label, output, m, runtime in (("no_dipole", baseline, bm, t_baseline), ("dipole_aligned", dipole, dm, t_dipole)):
        metrics.append({
            "condition": label,
            "dt_used_s": output.dt_used,
            "geodesic_distance_mm": m["geodesic_distance_mm"],
            "roi_e1_arrival_s": m["roi_e1_arrival_s"],
            "roi_e2_arrival_s": m["roi_e2_arrival_s"],
            "roi_within_condition_delay_s": m["roi_delay_s"],
            "vertex_e1_arrival_s": m["vertex_e1_arrival_s"],
            "vertex_e2_arrival_s": m["vertex_e2_arrival_s"],
            "vertex_within_condition_delay_s": m["vertex_delay_s"],
            "speed_mm_min": m["speed_mm_min_recomputed"],
            "speed_function_mm_min": m["speed_mm_min_function"],
            "max_abs_Ve_mV": float(output.params.field_reference_mV * np.max(np.abs(output.electric_potential))),
            "max_Ke_mM": float(np.max(output.potassium_e)),
            "max_swelling": float(np.max(output.swelling)),
            "runtime_s": runtime,
        })
    comparison = {
        "speed_change_dipole_minus_no_mm_min": metrics[1]["speed_mm_min"] - metrics[0]["speed_mm_min"],
        "speed_slowdown_no_minus_dipole_mm_min": metrics[0]["speed_mm_min"] - metrics[1]["speed_mm_min"],
        "speed_percent_change_vs_no_dipole": 100.0 * (metrics[1]["speed_mm_min"] - metrics[0]["speed_mm_min"]) / metrics[0]["speed_mm_min"],
        "vertex_delay_increase_s": metrics[1]["vertex_within_condition_delay_s"] - metrics[0]["vertex_within_condition_delay_s"],
        "roi_delay_increase_s": metrics[1]["roi_within_condition_delay_s"] - metrics[0]["roi_within_condition_delay_s"],
        "between_condition_E1_arrival_shift_s": metrics[1]["vertex_e1_arrival_s"] - metrics[0]["vertex_e1_arrival_s"],
        "between_condition_downstream_E2_arrival_shift_s": metrics[1]["vertex_e2_arrival_s"] - metrics[0]["vertex_e2_arrival_s"],
        "max_abs_Ve_change_mV": metrics[1]["max_abs_Ve_mV"] - metrics[0]["max_abs_Ve_mV"],
    }
    write_csv(out_dir / "canonical_metrics.csv", metrics)
    write_json(out_dir / "canonical_metrics.json", {"conditions": metrics, "paired_comparison": comparison})
    decision = f"""# Canonical representative provenance decision

The authoritative run is the current post-`7d054b8` implementation on the declared 64x28 synthetic folded strip. It exactly verifies the declared 22x10 mm domain, requested depth 2.4 mm and sigma 1.5 mm, 1,792 vertices, 3,402 faces, 1.2 mm initial-only stimulus, -28 mV arrival threshold, 220 s duration, automatically selected vertices 471/636/624, and {baseline.dt_used:.12f} s time step.

This choice is based on implementation chronology and the manuscript's declared bounded saturating swelling specification, not effect size. The competing 2.558/2.476 result was generated before the swelling implementation was changed and survives only in an output-side Table S2 that was not regenerated. The current representative CSV/JSON, manuscript-side Table S2, and tracked Figure 2 source all identify the post-revision run. The working-tree Figure 2 labels were manually changed back to the stale values without changing the underlying trace simulation, creating a mixed artifact.

The null-control pipeline is not the source of either representative pair. Its distance-only and scrambled-normal kernel modes are separate rows and outputs.

## Readout definitions

- Original saved `cross_fold_delay_s`: E2 center-vertex arrival minus E1 center-vertex arrival.
- Speed denominator: E2 median arrival minus E1 median arrival over 1 mm geodesic ROIs centered on the same vertices.
- Speed formula: `60 * {bm['geodesic_distance_mm']:.12f} mm / ROI delay (s)`.
- Between-condition downstream shift: dipole E2 center arrival minus no-dipole E2 center arrival.

Both center-vertex and ROI-median delays are reported in the canonical metrics; they are not averaged or substituted.
"""
    write_text(out_dir / "provenance_decision.md", decision)
    log += [
        f"Canonical baseline runtime_s={t_baseline:.6f}",
        f"Canonical dipole runtime_s={t_dipole:.6f}",
        f"Canonical vertices stim/e1/e2={stim}/{e1}/{e2}",
        f"Canonical speed no/dip={metrics[0]['speed_mm_min']:.12f}/{metrics[1]['speed_mm_min']:.12f}",
    ]
    write_text(out_dir / "run_log.txt", "\n".join(log))
    return {"mesh": mesh, "stim": stim, "e1": e1, "e2": e2, "baseline": baseline, "dipole": dipole, "bm": bm, "dm": dm, "metrics": metrics, "comparison": comparison, "readout_path": readout_path, "readout_s": readout_s, "full_path": full_path, "full_s": full_s}


def gaussian_profile(y: np.ndarray, depth: float, sigma: float) -> np.ndarray:
    return -float(depth) * np.exp(-0.5 * (np.asarray(y, dtype=float) / float(sigma)) ** 2)


def gaussian_curvature(y: np.ndarray, depth: float, sigma: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    exp_term = np.exp(-0.5 * (y / sigma) ** 2)
    dz = depth * y / sigma**2 * exp_term
    d2z = depth / sigma**2 * (1.0 - (y / sigma) ** 2) * exp_term
    return d2z / (1.0 + dz**2) ** 1.5


def curvature_regions(y: np.ndarray, depth: float, sigma: float) -> tuple[np.ndarray, dict]:
    """Seven geometry-only regions, fixed before condition differences are inspected."""
    y = np.asarray(y, dtype=float)
    # The Gaussian's analytic inflections are at |y|=sigma.  We predeclare
    # |y|=2*sigma as the shoulder-to-outer/flatter boundary (profile amplitude
    # exp(-2)=0.1353 of depth), independent of every propagation result.
    flat_boundary = min(float(2.0 * sigma), float(np.max(np.abs(y))))
    labels = np.empty(y.size, dtype=object)
    # Path orientation is positive-y to negative-y (stimulus bank toward opposite bank).
    for i, val in enumerate(y):
        if val > flat_boundary:
            labels[i] = "pre-fold flatter region"
        elif val > sigma:
            labels[i] = "first shoulder/curvature-transition region"
        elif val > 0.5 * sigma:
            labels[i] = "first sulcal bank"
        elif val >= -0.5 * sigma:
            labels[i] = "fundus"
        elif val >= -sigma:
            labels[i] = "opposite bank"
        elif val >= -flat_boundary:
            labels[i] = "second shoulder/curvature-transition region"
        else:
            labels[i] = "post-fold flatter region"
    return labels.astype(str), {
        "inflection_points_y_mm": [-sigma, sigma],
        "fundus_bank_boundaries_y_mm": [-0.5 * sigma, 0.5 * sigma],
        "flat_boundary_abs_y_mm": flat_boundary,
        "flat_boundary_rule": "analytic Gaussian scale |y|=2*sigma, where profile amplitude is exp(-2) of requested depth",
    }


def geometry_outputs(canonical: dict) -> list[dict]:
    out_dir = OUT / "02_geometry_profiles"
    out_dir.mkdir(parents=True, exist_ok=True)
    y = np.linspace(-5.0, 5.0, 28)
    geometries = [(0.0, 1.5)] + [(d, s) for d in (1.2, 1.8, 2.4, 3.0) for s in (1.1, 1.5, 1.9)]
    coordinates, realized = [], []
    for depth, sigma in geometries:
        z = np.zeros_like(y) if depth == 0 else gaussian_profile(y, depth, sigma)
        label = "flat_control" if depth == 0 else f"depth_{depth:.1f}_sigma_{sigma:.1f}"
        for index, (yi, zi) in enumerate(zip(y, z)):
            coordinates.append({"geometry": label, "requested_depth_mm": depth, "requested_sigma_mm": sigma, "profile_index": index, "y_mm": yi, "z_mm": zi})
        sampled_depth = float(-np.min(z))
        if depth > 0:
            valid = (z < 0) & (np.abs(y) > 0)
            sigma_fit = float(np.sqrt(np.mean(-0.5 * y[valid] ** 2 / np.log((-z[valid]) / depth))))
            positive = y >= 0
            yp, zp = y[positive], -z[positive]
            target = depth / 2.0
            crossing = np.where(zp <= target)[0]
            if crossing.size:
                j = int(crossing[0])
                if j == 0:
                    half_y = float(yp[0])
                else:
                    half_y = float(np.interp(target, [zp[j], zp[j - 1]], [yp[j], yp[j - 1]]))
                fwhm_sampled = 2.0 * half_y
            else:
                fwhm_sampled = float("nan")
        else:
            sigma_fit = float("nan")
            fwhm_sampled = float("nan")
        realized.append({
            "geometry": label, "requested_depth_mm": depth, "sampled_max_depth_mm": sampled_depth,
            "depth_sampling_error_mm": sampled_depth - depth, "requested_sigma_mm": sigma,
            "fitted_sigma_from_sampled_coordinates_mm": sigma_fit,
            "analytic_FWHM_mm": 2.0 * math.sqrt(2.0 * math.log(2.0)) * sigma if depth > 0 else float("nan"),
            "sampled_interpolated_FWHM_mm": fwhm_sampled,
            "profile_numerical_points": len(y),
        })
    write_csv(out_dir / "profile_coordinates.csv", coordinates)
    write_csv(out_dir / "realized_geometry_parameters.csv", realized)

    fig, axes = plt.subplots(4, 3, figsize=(10.5, 10.5), sharex=True, sharey=True, constrained_layout=True)
    folded = geometries[1:]
    for ax, (depth, sigma) in zip(axes.flat, folded):
        z = gaussian_profile(y, depth, sigma)
        representative = math.isclose(depth, 2.4) and math.isclose(sigma, 1.5)
        ax.plot(y, np.zeros_like(y), color="0.70", ls="--", lw=1.0, label="flat control")
        ax.plot(y, z, color="#b2182b" if representative else "#2166ac", lw=2.3 if representative else 1.7)
        ax.fill_between(y, z, 0, color="#b2182b" if representative else "#2166ac", alpha=0.10)
        ax.set_title(f"depth={depth:.1f} mm, sigma={sigma:.1f} mm" + ("\nrepresentative" if representative else ""), color="#b2182b" if representative else "black")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel("Cross-fold coordinate y (mm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Surface height z (mm)")
    save_figure(fig, out_dir / "synthetic_fold_profiles")

    mesh = canonical["mesh"]
    path = canonical["full_path"]
    fig = plt.figure(figsize=(10.5, 7.5), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], triangles=mesh.faces, cmap="Blues_r", alpha=0.72, linewidth=0.05)
    ax.plot(mesh.vertices[path, 0], mesh.vertices[path, 1], mesh.vertices[path, 2] + 0.05, color="#542788", lw=3, label="full cross-fold path")
    for vertex, label, color, marker in ((canonical["stim"], "stimulus", "#1b9e77", "*"), (canonical["e1"], "E1", "#d95f02", "o"), (canonical["e2"], "E2", "#7570b3", "s")):
        p = mesh.vertices[vertex]
        ax.scatter(*p, s=110 if label == "stimulus" else 70, color=color, marker=marker, depthshade=False, label=label)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_zlabel("z (mm)")
    ax.set_box_aspect((22, 10, 5))
    ax.view_init(elev=27, azim=-55)
    ax.legend(frameon=False, loc="upper left")
    save_figure(fig, out_dir / "representative_surface_readouts")
    write_text(out_dir / "geometry_equation_and_implementation.md", f"""# Geometry equation and implementation

The exact code in `src/csd_sulcus/surface_io.py:160-205` evaluates

`z(y) = -d exp[-0.5 (y/sigma)^2]`

on a tensor grid with `x=linspace(0,22,nx)` and `y=linspace(-5,5,ny)`. The representative uses `nx=64`, `ny=28`; the family profiles in `profile_coordinates.csv` therefore use the exact 28 y samples of the representative mesh. The flat control is computed by setting `d=0`, not inserted as a reported result.

Because `ny=28` is even, y=0 is not a mesh vertex. The sampled maximum depth is consequently slightly smaller than the requested analytic depth; `realized_geometry_parameters.csv` reports this sampling difference. A fit of the exact sampled coordinates recovers sigma, and both analytic and sampled-interpolated full width at half maximum are reported.

All profile panels use equal physical axis scaling in millimeters. The representative depth 2.4 mm, sigma 1.5 mm panel is highlighted without vertical exaggeration.
""")
    return realized


def triangulation_xy(mesh):
    return mtri.Triangulation(mesh.vertices[:, 0], mesh.vertices[:, 1], triangles=mesh.faces)


def mark_points_2d(ax, canonical: dict) -> None:
    mesh = canonical["mesh"]
    for vertex, label, color, marker in ((canonical["stim"], "S", "#1b9e77", "*"), (canonical["e1"], "E1", "white", "o"), (canonical["e2"], "E2", "white", "s")):
        p = mesh.vertices[vertex]
        ax.scatter(p[0], p[1], s=60, facecolor=color, edgecolor="black", marker=marker, linewidth=0.7, zorder=5)
        ax.text(p[0] + 0.25, p[1] + 0.15, label, fontsize=7, zorder=6)


def add_region_lines(ax, canonical: dict, *, axis="x") -> None:
    path = canonical["full_path"]
    mesh = canonical["mesh"]
    labels, meta = curvature_regions(mesh.vertices[path, 1], 2.4, 1.5)
    s = canonical["full_s"]
    changes = np.where(labels[1:] != labels[:-1])[0] + 1
    for idx in changes:
        if axis == "x":
            ax.axvline(s[idx], color="white", lw=0.55, alpha=0.55)
        else:
            ax.axhline(s[idx], color="white", lw=0.55, alpha=0.55)
    for vertex, text in ((canonical["e1"], "E1"), (canonical["e2"], "E2")):
        j = int(np.argmin(np.linalg.norm(mesh.vertices[path] - mesh.vertices[vertex], axis=1)))
        if axis == "x": ax.axvline(s[j], color="#fdae61", lw=1.0, ls="--")
        else: ax.axhline(s[j], color="#fdae61", lw=1.0, ls="--")


def difference_visualizations(canonical: dict) -> dict:
    out_dir = OUT / "03_difference_visualization"
    out_dir.mkdir(parents=True, exist_ok=True)
    b, d = canonical["baseline"], canonical["dipole"]
    t1 = canonical["bm"]["vertex_e1_arrival_s"]
    t2 = canonical["bm"]["vertex_e2_arrival_s"]
    requested = t1 + np.asarray([0.25, 0.50, 0.75]) * (t2 - t1)
    idx = np.asarray([int(np.argmin(np.abs(b.snapshot_times - t))) for t in requested])
    actual = b.snapshot_times[idx]
    tri = triangulation_xy(canonical["mesh"])
    abs_min = float(min(np.min(b.snapshot_potassium_e[idx]), np.min(d.snapshot_potassium_e[idx])))
    abs_max = float(max(np.max(b.snapshot_potassium_e[idx]), np.max(d.snapshot_potassium_e[idx])))
    delta_maps = d.snapshot_potassium_e[idx] - b.snapshot_potassium_e[idx]
    delta_limit = float(np.max(np.abs(delta_maps))) or 1e-12
    fig, axes = plt.subplots(3, 3, figsize=(13, 8.5), sharex=True, sharey=True, constrained_layout=True)
    for col, (ii, tt) in enumerate(zip(idx, actual)):
        a0 = axes[0, col].tripcolor(tri, b.snapshot_potassium_e[ii], shading="gouraud", cmap="viridis", vmin=abs_min, vmax=abs_max)
        axes[1, col].tripcolor(tri, d.snapshot_potassium_e[ii], shading="gouraud", cmap="viridis", vmin=abs_min, vmax=abs_max)
        ad = axes[2, col].tripcolor(tri, d.snapshot_potassium_e[ii] - b.snapshot_potassium_e[ii], shading="gouraud", cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0.0, vmin=-delta_limit, vmax=delta_limit))
        for row in range(3): mark_points_2d(axes[row, col], canonical)
        axes[0, col].set_title(f"t={tt:.2f} s")
    axes[0, 0].set_ylabel("No dipole\ny (mm)"); axes[1, 0].set_ylabel("Dipole aligned\ny (mm)"); axes[2, 0].set_ylabel("Delta Ke\ny (mm)")
    for ax in axes[-1]: ax.set_xlabel("Projected native x (mm)")
    fig.colorbar(a0, ax=axes[:2, :], shrink=0.75, label="Extracellular K+ (mM)")
    fig.colorbar(ad, ax=axes[2, :], shrink=0.75, label="Dipole - no dipole (mM)")
    fig.suptitle("Matched maps at fixed fractions of the no-dipole E1-E2 traversal interval")
    save_figure(fig, out_dir / "matched_maps")

    delta_t = d.arrival_times - b.arrival_times
    finite = np.isfinite(delta_t)
    dt_limit = float(np.nanmax(np.abs(delta_t[finite]))) if finite.any() else 1.0
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True, sharey=True, constrained_layout=True)
    lo = float(np.nanmin([b.arrival_times, d.arrival_times])); hi = float(np.nanmax([b.arrival_times, d.arrival_times]))
    a = axes[0].tripcolor(tri, b.arrival_times, shading="flat", cmap="viridis", vmin=lo, vmax=hi)
    axes[1].tripcolor(tri, d.arrival_times, shading="flat", cmap="viridis", vmin=lo, vmax=hi)
    c = axes[2].tripcolor(tri, delta_t, shading="flat", cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0.0, vmin=-dt_limit, vmax=dt_limit))
    for ax, title in zip(axes, ("No dipole arrival", "Dipole-aligned arrival", "Delta T = dipole - no dipole")):
        ax.set_title(title); ax.set_xlabel("Projected native x (mm)"); mark_points_2d(ax, canonical)
    axes[0].set_ylabel("Projected native y (mm)")
    fig.colorbar(a, ax=axes[:2], shrink=0.8, label="Arrival time (s)")
    fig.colorbar(c, ax=axes[2], shrink=0.8, label="Positive = later with dipole (s)")
    save_figure(fig, out_dir / "arrival_time_difference_map")

    path, s = canonical["full_path"], canonical["full_s"]
    kb = b.snapshot_potassium_e[:, path].T
    kd = d.snapshot_potassium_e[:, path].T
    kdelta = kd - kb
    klimit = float(np.max(np.abs(kdelta))) or 1e-12
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharex=True, sharey=True, constrained_layout=True)
    extent = [b.snapshot_times[0], b.snapshot_times[-1], s[0], s[-1]]
    im0 = axes[0].imshow(kb, origin="lower", aspect="auto", extent=extent, cmap="viridis", vmin=min(kb.min(), kd.min()), vmax=max(kb.max(), kd.max()))
    axes[1].imshow(kd, origin="lower", aspect="auto", extent=extent, cmap="viridis", vmin=min(kb.min(), kd.min()), vmax=max(kb.max(), kd.max()))
    im2 = axes[2].imshow(kdelta, origin="lower", aspect="auto", extent=extent, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0.0, vmin=-klimit, vmax=klimit))
    threshold = b.params.arrival_voltage_threshold_mv
    for ax in axes:
        ax.contour(b.snapshot_times, s, b.snapshot_voltage_mv[:, path].T, levels=[threshold], colors=["white"], linewidths=1.1)
        ax.contour(d.snapshot_times, s, d.snapshot_voltage_mv[:, path].T, levels=[threshold], colors=["black"], linewidths=1.0, linestyles="--")
        add_region_lines(ax, canonical, axis="y")
        ax.set_xlabel("Time (s)")
    axes[0].set_title("No dipole"); axes[1].set_title("Dipole aligned"); axes[2].set_title("Signed difference")
    axes[0].set_ylabel("Unfolded cross-fold path distance (mm)")
    fig.colorbar(im0, ax=axes[:2], shrink=0.8, label="Extracellular K+ (mM)")
    fig.colorbar(im2, ax=axes[2], shrink=0.8, label="Dipole - no dipole (mM)")
    save_figure(fig, out_dir / "paired_kymographs")

    tb, td = b.arrival_times[path], d.arrival_times[path]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    axes[0].plot(s, tb, color="black", lw=1.8, marker="o", ms=3, label="no dipole")
    axes[0].plot(s, td, color="#b2182b", lw=1.8, marker="s", ms=3, label="dipole aligned")
    axes[0].set_ylabel("Arrival time (s)"); axes[0].legend(frameon=False); axes[0].grid(alpha=0.25)
    axes[1].plot(s, td - tb, color="#542788", lw=2, marker="o", ms=3)
    axes[1].axhline(0, color="black", lw=0.8); axes[1].set_ylabel("Delta T (s)\npositive = later"); axes[1].set_xlabel("Unfolded path distance (mm)"); axes[1].grid(alpha=0.25)
    for ax in axes: add_region_lines(ax, canonical, axis="x")
    save_figure(fig, out_dir / "path_arrival_and_deltaT")

    np.savez_compressed(ensure_output_path(out_dir / "plotted_arrays.npz"),
        selected_requested_times_s=requested, selected_actual_times_s=actual,
        no_dipole_selected_Ke_mM=b.snapshot_potassium_e[idx], dipole_selected_Ke_mM=d.snapshot_potassium_e[idx], delta_Ke_mM=delta_maps,
        no_dipole_arrival_s=b.arrival_times, dipole_arrival_s=d.arrival_times, delta_arrival_s=delta_t,
        path_vertices=path, path_distance_mm=s, kymograph_times_s=b.snapshot_times,
        no_dipole_kymograph_Ke_mM=kb, dipole_kymograph_Ke_mM=kd, delta_kymograph_Ke_mM=kdelta,
        no_dipole_path_arrival_s=tb, dipole_path_arrival_s=td, path_deltaT_s=td-tb)
    metadata = {
        "display_time_rule": "25%, 50%, and 75% of the no-dipole E1-center to E2-center traversal interval; nearest saved 0.5 s snapshot",
        "requested_times_s": requested, "actual_times_s": actual,
        "matched_map_coordinates": "native x-y coordinates projected to plan view; both axes in mm; z omitted",
        "arrival_map_coordinates": "native x-y plan-view projection in mm",
        "kymograph_space": "unfolded cumulative 3D edge length along a fixed-x full cross-fold mesh path, mm",
        "path_plot_space": "same unfolded full cross-fold path distance, mm",
        "difference_sign": "dipole aligned minus no dipole; positive arrival difference means later under dipole alignment",
        "paired_absolute_color_limits_mM": [abs_min, abs_max], "difference_color_limit_mM": [-delta_limit, delta_limit],
    }
    write_json(out_dir / "plotting_metadata.json", metadata)
    write_text(out_dir / "visualization_methods.md", """# Difference-visualization methods

Maps use native x-y surface coordinates as a physically labeled plan-view projection; z is not used as a plotting axis. Kymographs and path plots use cumulative three-dimensional edge length along the fixed-x cross-section that contains E1 and E2. The path spans both outer flatter regions and the complete fold.

Display times were fixed before inspecting differences: 25%, 50%, and 75% of the no-dipole center-vertex E1-E2 traversal interval, rounded to the nearest saved 0.5 s snapshot. Absolute paired maps share color limits. Signed difference maps use zero-centered diverging limits. White solid and black dashed kymograph contours are the no-dipole and dipole-aligned Vm=-28 mV fronts, respectively.

Stimulus, E1, and E2 are marked on maps. Region boundaries on path plots derive only from the analytic Gaussian geometry: fundus/bank boundaries at |y|=0.5 sigma, inflections at |y|=sigma, and shoulder-to-flatter boundaries at |y|=2 sigma (profile amplitude exp(-2) of requested depth).
""")
    return metadata


def safe_savgol(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < 5:
        return np.full_like(values, np.nan)
    interp = np.interp(np.arange(values.size), np.where(finite)[0], values[finite])
    window = min(window, values.size if values.size % 2 else values.size - 1)
    window = max(5, window)
    if window % 2 == 0:
        window -= 1
    return savgol_filter(interp, window_length=window, polyorder=2, mode="interp")


def pathwise_arrays(mesh, baseline, dipole, e1: int, e2: int, depth: float, sigma: float) -> dict:
    path = full_cross_fold_path(mesh, 0.5 * (mesh.vertices[e1, 0] + mesh.vertices[e2, 0]), descending_y=True)
    coords = mesh.vertices[path]
    s = cumulative_distance(coords)
    y = coords[:, 1]
    curvature = gaussian_curvature(y, depth, sigma) if depth > 0 else np.zeros_like(y)
    regions, region_meta = curvature_regions(y, max(depth, 1e-12), sigma) if depth > 0 else (np.full(y.size, "flat control", dtype=str), {})
    tb, td = baseline.arrival_times[path], dipole.arrival_times[path]
    result = {"path": path, "s": s, "y": y, "z": coords[:, 2], "curvature": curvature, "regions": regions, "tb": tb, "td": td, "delta": td - tb, "region_meta": region_meta}
    for window in (5, 7, 9):
        sb = safe_savgol(tb, window); sd = safe_savgol(td, window)
        db = np.gradient(sb, s); dd = np.gradient(sd, s)
        vb = np.where(db > 1e-6, 60.0 / db, np.nan)
        vd = np.where(dd > 1e-6, 60.0 / dd, np.nan)
        result[f"smooth_b_{window}"] = sb; result[f"smooth_d_{window}"] = sd
        result[f"speed_b_{window}"] = vb; result[f"speed_d_{window}"] = vd; result[f"speed_delta_{window}"] = vd - vb
    # Unsmooth adjacent-segment definition assigned to segment midpoints/right vertex.
    dtb = np.diff(tb); dtd = np.diff(td); ds = np.diff(s)
    seg_b = np.full(tb.size, np.nan); seg_d = np.full(td.size, np.nan)
    seg_b[1:] = np.divide(60.0 * ds, dtb, out=np.full_like(dtb, np.nan), where=dtb > 1e-9)
    seg_d[1:] = np.divide(60.0 * ds, dtd, out=np.full_like(dtd, np.nan), where=dtd > 1e-9)
    result["segment_speed_b"] = seg_b; result["segment_speed_d"] = seg_d; result["segment_speed_delta"] = seg_d - seg_b
    return result


def summarize_regions(arr: dict, depth: float, sigma: float, geometry: str, dt: float) -> list[dict]:
    rows = []
    order = list(dict.fromkeys(arr["regions"]))
    for region in order:
        mask = (arr["regions"] == region) & np.isfinite(arr["delta"])
        if not mask.any():
            continue
        delta = arr["delta"][mask]
        speed_delta = arr["speed_delta_7"][mask]
        segment_delta = arr["segment_speed_delta"][mask]
        mean_delta = float(np.mean(delta))
        tolerance = float(dt)  # one time step, declared before interpretation
        sign = "slowing" if mean_delta > tolerance else "acceleration" if mean_delta < -tolerance else "no material difference"
        rows.append({
            "geometry": geometry, "depth": depth, "sigma": sigma, "region": region,
            "mean_DeltaT": mean_delta, "median_DeltaT": float(np.median(delta)), "minimum_DeltaT": float(np.min(delta)), "maximum_DeltaT": float(np.max(delta)),
            "local_speed_difference": float(np.nanmean(speed_delta)) if np.isfinite(speed_delta).any() else float("nan"),
            "segment_speed_difference": float(np.nanmean(segment_delta)) if np.isfinite(segment_delta).any() else float("nan"),
            "curvature_summary": f"mean={np.mean(arr['curvature'][mask]):.8g}; min={np.min(arr['curvature'][mask]):.8g}; max={np.max(arr['curvature'][mask]):.8g} mm^-1",
            "sign_of_effect": sign, "materiality_tolerance_s": tolerance, "n_path_vertices": int(mask.sum()),
        })
    return rows


def run_geometry_pair(nx: int, ny: int, depth: float, sigma: float, *, final_t: float, fixed_coords=None, dt=None):
    mesh = generate_folded_strip_mesh(nx=nx, ny=ny, length_mm=22.0, width_mm=10.0, fold_depth_mm=depth, fold_sigma_mm=sigma)
    if fixed_coords is None:
        stim, e1, e2 = choose_auto_vertices(mesh)
    else:
        stim, e1, e2 = [int(np.argmin(np.linalg.norm(mesh.vertices - np.asarray(c)[None, :], axis=1))) for c in fixed_coords]
    base_params = MechanisticSurfaceParams(final_t_end=final_t, dt=dt, enable_dipole_alignment=False, enable_vascular_feedback=False)
    dip_params = dc.replace(base_params, enable_dipole_alignment=True, dipole_kernel_mode="aligned")
    b = run_mechanistic_surface_simulation(mesh, base_params, stimulus_vertex=stim)
    d = run_mechanistic_surface_simulation(mesh, dip_params, stimulus_vertex=stim)
    return mesh, stim, e1, e2, b, d


def shoulder_analysis(canonical: dict, log: list[str]) -> dict:
    out_dir = OUT / "04_shoulder_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = pathwise_arrays(canonical["mesh"], canonical["baseline"], canonical["dipole"], canonical["e1"], canonical["e2"], 2.4, 1.5)
    rep_rows = []
    for i, vertex in enumerate(rep["path"]):
        rep_rows.append({
            "path_index": i, "vertex": int(vertex), "path_distance_mm": rep["s"][i], "y_mm": rep["y"][i], "z_mm": rep["z"][i],
            "region": rep["regions"][i], "signed_curvature_mm_inv": rep["curvature"][i], "absolute_curvature_mm_inv": abs(rep["curvature"][i]),
            "no_dipole_arrival_s_unsmoothed": rep["tb"][i], "dipole_arrival_s_unsmoothed": rep["td"][i], "DeltaT_s": rep["delta"][i],
            "no_dipole_local_speed_window5_mm_min": rep["speed_b_5"][i], "dipole_local_speed_window5_mm_min": rep["speed_d_5"][i], "local_speed_difference_window5_mm_min": rep["speed_delta_5"][i],
            "no_dipole_local_speed_window7_mm_min": rep["speed_b_7"][i], "dipole_local_speed_window7_mm_min": rep["speed_d_7"][i], "local_speed_difference_window7_mm_min": rep["speed_delta_7"][i],
            "no_dipole_local_speed_window9_mm_min": rep["speed_b_9"][i], "dipole_local_speed_window9_mm_min": rep["speed_d_9"][i], "local_speed_difference_window9_mm_min": rep["speed_delta_9"][i],
            "no_dipole_segment_speed_mm_min": rep["segment_speed_b"][i], "dipole_segment_speed_mm_min": rep["segment_speed_d"][i], "segment_speed_difference_mm_min": rep["segment_speed_delta"][i],
            "surface_normal_opposition_index": "not a local path field in implementation; pairwise kernel weights only",
        })
    write_csv(out_dir / "representative_pathwise_metrics.csv", rep_rows)

    all_rows = []
    sweep_start = time.perf_counter()
    for depth in (1.2, 1.8, 2.4, 3.0):
        for sigma in (1.1, 1.5, 1.9):
            mesh, stim, e1, e2, b, d = run_geometry_pair(52, 24, depth, sigma, final_t=210.0)
            arr = pathwise_arrays(mesh, b, d, e1, e2, depth, sigma)
            all_rows.extend(summarize_regions(arr, depth, sigma, f"depth_{depth:.1f}_sigma_{sigma:.1f}", b.dt_used))
    write_csv(out_dir / "all_geometries_region_metrics.csv", all_rows)
    log.append(f"12-geometry pathwise sweep runtime_s={time.perf_counter()-sweep_start:.6f}")

    colors = {
        "pre-fold flatter region": "#bdbdbd", "first shoulder/curvature-transition region": "#fdae61", "first sulcal bank": "#abd9e9",
        "fundus": "#2c7bb6", "opposite bank": "#abd9e9", "second shoulder/curvature-transition region": "#fdae61", "post-fold flatter region": "#bdbdbd",
    }
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
    for region in dict.fromkeys(rep["regions"]):
        mask = rep["regions"] == region
        for ax in axes:
            ax.axvspan(rep["s"][mask].min(), rep["s"][mask].max(), color=colors.get(region, "0.9"), alpha=0.16)
    axes[0].plot(rep["s"], rep["tb"], "o-", color="black", ms=3, label="no dipole, unsmoothed")
    axes[0].plot(rep["s"], rep["td"], "s-", color="#b2182b", ms=3, label="dipole aligned, unsmoothed")
    axes[0].legend(frameon=False); axes[0].set_ylabel("Arrival time (s)"); axes[0].grid(alpha=0.2)
    axes[1].plot(rep["s"], rep["delta"], "o-", color="#542788", ms=3); axes[1].axhline(0, color="black", lw=0.8); axes[1].set_ylabel("Delta T (s)"); axes[1].grid(alpha=0.2)
    axes[2].plot(rep["s"], rep["curvature"], color="#1b9e77", lw=2, label="signed curvature")
    ax2 = axes[2].twinx(); ax2.plot(rep["s"], rep["speed_delta_7"], color="#d95f02", lw=1.6, label="local speed difference")
    axes[2].axhline(0, color="black", lw=0.8); axes[2].set_ylabel("Curvature (mm^-1)"); ax2.set_ylabel("Dipole - no speed (mm/min)"); axes[2].set_xlabel("Full cross-fold path distance (mm)")
    save_figure(fig, out_dir / "curvature_and_deltaT")

    region_order = list(dict.fromkeys(rep["regions"]))
    counts = Counter(row["sign_of_effect"] for row in all_rows)
    grouped = {region: [r for r in all_rows if r["region"] == region] for region in region_order}
    x = np.arange(len(region_order))
    means = [np.nanmean([r["mean_DeltaT"] for r in grouped[region]]) if grouped[region] else np.nan for region in region_order]
    mins = [np.nanmin([r["minimum_DeltaT"] for r in grouped[region]]) if grouped[region] else np.nan for region in region_order]
    maxs = [np.nanmax([r["maximum_DeltaT"] for r in grouped[region]]) if grouped[region] else np.nan for region in region_order]
    fig, ax = plt.subplots(figsize=(12, 5.4), constrained_layout=True)
    ax.bar(x, means, color=[colors.get(r, "0.5") for r in region_order], edgecolor="black", linewidth=0.6)
    ax.errorbar(x, means, yerr=[np.asarray(means)-np.asarray(mins), np.asarray(maxs)-np.asarray(means)], fmt="none", color="black", capsize=3)
    ax.axhline(0, color="black", lw=0.8); ax.set_ylabel("Delta T across 12 geometries (s)"); ax.set_xticks(x, [r.replace("/curvature-transition region", "") for r in region_order], rotation=25, ha="right"); ax.grid(axis="y", alpha=0.2)
    save_figure(fig, out_dir / "regional_effect_summary")

    fixed = [canonical["mesh"].vertices[i] for i in (canonical["stim"], canonical["e1"], canonical["e2"])]
    numerical_rows = []
    configs = [
        ("reference", 64, 28, None),
        ("neighboring_mesh_72x32", 72, 32, None),
        ("half_time_step", 64, 28, canonical["baseline"].dt_used / 2.0),
    ]
    for label, nx, ny, dt in configs:
        if label == "reference":
            mesh, e1, e2, b, d = canonical["mesh"], canonical["e1"], canonical["e2"], canonical["baseline"], canonical["dipole"]
        else:
            mesh, _, e1, e2, b, d = run_geometry_pair(nx, ny, 2.4, 1.5, final_t=220.0, fixed_coords=fixed, dt=dt)
        arr = pathwise_arrays(mesh, b, d, e1, e2, 2.4, 1.5)
        for region in ("first shoulder/curvature-transition region", "second shoulder/curvature-transition region"):
            mask = (arr["regions"] == region) & np.isfinite(arr["delta"])
            numerical_rows.append({
                "configuration": label, "nx": nx, "ny": ny, "dt_s": b.dt_used, "region": region,
                "min_DeltaT_s": float(np.min(arr["delta"][mask])) if mask.any() else float("nan"),
                "mean_DeltaT_s": float(np.mean(arr["delta"][mask])) if mask.any() else float("nan"),
                "mean_local_speed_difference_window5_mm_min": float(np.nanmean(arr["speed_delta_5"][mask])) if np.isfinite(arr["speed_delta_5"][mask]).any() else float("nan"),
                "mean_local_speed_difference_window7_mm_min": float(np.nanmean(arr["speed_delta_7"][mask])) if np.isfinite(arr["speed_delta_7"][mask]).any() else float("nan"),
                "mean_local_speed_difference_window9_mm_min": float(np.nanmean(arr["speed_delta_9"][mask])) if np.isfinite(arr["speed_delta_9"][mask]).any() else float("nan"),
                "mean_segment_speed_difference_mm_min": float(np.nanmean(arr["segment_speed_delta"][mask])) if np.isfinite(arr["segment_speed_delta"][mask]).any() else float("nan"),
                "any_negative_DeltaT": bool(np.any(arr["delta"][mask] < -b.dt_used)) if mask.any() else False,
            })
    write_csv(out_dir / "numerical_check.csv", numerical_rows)

    rep_region = summarize_regions(rep, 2.4, 1.5, "representative", canonical["baseline"].dt_used)
    min_delta = float(np.nanmin(rep["delta"]))
    shoulder_checks = [r for r in numerical_rows if "shoulder" in r["region"]]
    robust_negative = all(r["any_negative_DeltaT"] for r in shoulder_checks)
    robust_speed_accel = all((r["mean_local_speed_difference_window5_mm_min"] > 0 and r["mean_local_speed_difference_window7_mm_min"] > 0 and r["mean_local_speed_difference_window9_mm_min"] > 0 and r["mean_segment_speed_difference_mm_min"] > 0) for r in shoulder_checks)
    reference_shoulder_checks = [r for r in shoulder_checks if r["configuration"] == "reference"]
    representative_shoulder_acceleration_evidence = any(
        r["any_negative_DeltaT"]
        or (
            r["mean_local_speed_difference_window5_mm_min"] > 0
            and r["mean_local_speed_difference_window7_mm_min"] > 0
            and r["mean_local_speed_difference_window9_mm_min"] > 0
            and r["mean_segment_speed_difference_mm_min"] > 0
        )
        for r in reference_shoulder_checks
    )
    if robust_negative and robust_speed_accel:
        conclusion = "clear shoulder acceleration"
    elif representative_shoulder_acceleration_evidence:
        conclusion = "localized but numerically fragile acceleration"
    elif np.isfinite(rep["delta"]).any():
        conclusion = "no shoulder-specific acceleration"
    else:
        conclusion = "analysis inconclusive"
    negative_regions = [str(r["region"]) for r in rep_region if r["minimum_DeltaT"] < -canonical["baseline"].dt_used]
    positive_curvature = rep["delta"][(rep["curvature"] > 0) & np.isfinite(rep["delta"])]
    negative_curvature = rep["delta"][(rep["curvature"] < 0) & np.isfinite(rep["delta"])]
    report = [
        "# Shoulder/gyral-effect analysis", "", f"**Required classification: {conclusion}.**", "",
        "Regions were fixed from the implemented Gaussian geometry before examining paired differences. Inflection points are y=+/-sigma. Fundus/bank boundaries are y=+/-0.5 sigma. Shoulder-to-flatter boundaries are y=+/-2 sigma, where the Gaussian profile amplitude is exp(-2) of requested depth.", "",
        "Arrival arrays are shown unsmoothed. Local speed uses a quadratic Savitzky-Golay derivative with nominal 7-vertex window; 5- and 9-vertex windows and unsmoothed adjacent-segment speeds are included as sensitivity definitions.", "",
        f"The representative minimum Delta T over the full path was {min_delta:.6f} s (positive means later with dipole).",
        f"Across the 12 deterministic folded geometries, region classifications were: {dict(counts)}.", "",
        f"Delta T was not positive at every path vertex. Regions containing values more negative than one reference time step were: {negative_regions}. Neither representative shoulder region met that criterion.",
        f"Mean Delta T was {np.mean(positive_curvature):.6f} s where analytic signed curvature was positive and {np.mean(negative_curvature):.6f} s where it was negative. The effect therefore did not follow a simple curvature-sign reversal.",
        "Delay accumulated across the fundus, opposite bank, and second shoulder. It then decreased in the outer post-fold flatter region. Arrival time versus fixed cross-sectional path distance was non-monotonic there, so a directional local speed is not numerically defined and the apparent outer-region acceleration is not shoulder evidence.", "",
        "No population-level biological inference is made. The sweep summary describes deterministic paired simulations only.", "",
        "The implementation has no precomputed per-vertex surface-normal opposition index. Opposition is pairwise inside the nonlocal kernel, so inventing a local path field would change the model; the path CSV records this as unavailable.", "",
        "## Representative regional summaries", "",
        "| Region | Mean Delta T (s) | Minimum Delta T (s) | Mean local speed difference (mm/min) |", "|---|---:|---:|---:|",
    ]
    report.extend(f"| {r['region']} | {r['mean_DeltaT']:.6f} | {r['minimum_DeltaT']:.6f} | {r['local_speed_difference']:.6f} |" for r in rep_region)
    report += ["", "The numerical check repeats both shoulders on a neighboring 72x32 mesh and with the reference time step halved. The reference and half-step shoulders show no negative Delta T; the neighboring mesh has a one-step negative fluctuation at the first shoulder with mutually inconsistent local-speed definitions. This does not support shoulder-specific acceleration."]
    write_text(out_dir / "shoulder_analysis_report.md", "\n".join(report))
    return {"conclusion": conclusion, "representative_region_rows": rep_region, "all_rows": all_rows, "counts": dict(counts), "numerical_rows": numerical_rows, "minimum_representative_deltaT_s": min_delta}


def source_range(obj) -> str:
    lines, start = inspect.getsourcelines(obj)
    return f"{start}-{start + len(lines) - 1}"


def model_audit() -> list[dict]:
    out_dir = OUT / "05_model_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = MechanisticSurfaceParams()
    ranges = {
        "params": source_range(MechanisticSurfaceParams), "dynamic": source_range(sm._dynamic_extracellular_fields),
        "nernst": source_range(sm._nernst_potential), "ghk": source_range(sm._ghk_voltage), "pump": source_range(sm._pump_rate),
        "theta": source_range(sm._activation_target), "currents": source_range(sm._membrane_currents), "transport": source_range(sm._ion_transport_rhs),
        "kernel": source_range(sm._build_dipole_interaction_matrix), "potential_solver": source_range(sm._build_potential_solver),
        "potential": source_range(sm._electric_potential), "simulation": source_range(sm.run_mechanistic_surface_simulation),
    }
    def row(item, expr, definition, units, default, function, documented, discrepancy="None identified"):
        return {"audit_item": item, "mathematical_expression_matching_implementation": expr, "variable_definition": definition, "units": units,
            "default_value": default, "code_file": "src/csd_sulcus/surface_mechanistic.py", "function_or_class": function,
            "line_number_or_range": ranges.get(function, function), "currently_documented_in_manuscript": documented, "code_manuscript_discrepancy": discrepancy}
    rows = [
        row("1 beta_ed", "beta_ed = m_ed * V0 / 26.64 = 0.0938438438", "dimensionless coefficient multiplying z*c*Delta(phi_hat) in edge drift", "dimensionless", p.electrodiffusion_mobility_fraction*p.field_reference_mV/26.64, "transport", "Yes, formula and defaults", "Code uses the numerical thermal voltage directly rather than explicit F, R, T constants."),
        row("2 phi_hat and Ve", "phi_hat = solve(A, M[gamma_rho*rho - gamma_d*K(Im)]); phi_hat -= mass-weighted mean; Ve[mV]=V0*phi_hat", "electric_potential stores dimensionless phi_hat", "phi_hat dimensionless; Ve mV", f"V0={p.field_reference_mV} mV", "potential", "Partly", "Mass-weighted mean-zero gauge/fallback is not stated."),
        row("3 gamma_rho", "charge source = gamma_rho * [(Ke-Ke0)+(Nae-Nae0)-(Cle-Cle0)]", "charge_field_gain_per_mM", "mM^-1 in manuscript; implementation source scale", p.charge_field_gain_per_mM, "potential", "Yes", "Phenomenological gain; no derivation or physical charge conversion in code."),
        row("4 gamma_d", "dipole source = -gamma_d * K_d[Im]", "dipole_field_gain", "implementation scale; dimension not enforced", p.dipole_field_gain, "potential", "Yes", "Units are not defined by code."),
        row("5 sigma_phi", "electrical_parallel=1.8*d_parallel_base; electrical_perp=1.8*d_perp_base", "conductivity-like tensor used to build potential stiffness", "inherits model diffusivity scale; not physical S/m", "1.8 times transport fields", "potential_solver", "Yes", "It is not a calibrated electrical conductivity."),
        row("6 kappa", "A_phi = stiffness_phi + kappa*M", "potential_screening", "implementation screening scale; dimensional unit not enforced", p.potential_screening, "potential_solver", "Yes", "No physical screening-length derivation."),
        row("7 complete Na/K pump", "P=Pmax*[Nai^3/(Nai^3+KNa^3)]*[Ke^2/(Ke^2+KK^2)]*clip(O,minO,1)", "pump rate", "phenomenological rate scale", p.pump_max_rate, "pump", "Only product form", "Manuscript omits explicit factors, exponents, clipping, and constants."),
        row("8 f_Na", "f_Na=Nai^3/(Nai^3+14^3)", "intracellular sodium pump activation", "dimensionless", "K_Na=14 mM", "pump", "Half constant only", "Hill exponent 3 not documented."),
        row("9 f_K", "f_K=Ke^2/(Ke^2+4.2^2)", "extracellular potassium pump activation", "dimensionless", "K_K=4.2 mM", "pump", "Half constant only", "Hill exponent 2 not documented."),
        row("10 f_O2", "f_O2=clip(O,0.20,1.0)", "oxygen pump support", "dimensionless", "O in [0.20,1.0] for pump", "pump", "Only qualitative", "Exact clipping is undocumented."),
        row("11 Hill exponents", "n_Na=3; n_K=2; oxygen linear after clipping", "pump activation exponents", "dimensionless", "3,2,1", "pump", "No", "Missing from manuscript."),
        row("12 half-activation constants", "K_Na=14; K_K=4.2", "pump half activation", "mM", "14.0, 4.2", "params", "Yes", "None."),
        row("13 all transmembrane fluxes J_s^m", "J_K=0.038*(I_K-I_K0)-2*0.011*(P-P0); J_Na=0.038*(I_Na-I_Na0)+3*0.011*(P-P0); J_Cl=-0.038*(I_Cl-I_Cl0)", "code variables k_membrane, na_membrane, cl_membrane", "mM/s effective source after compartment division; physical conversion not defined", "0.038 current factor; 0.011 pump factor", "simulation", "No", "Manuscript equation uses -J/alpha extracellular but code adds these J/alpha; J sign convention is not explicitly defined."),
        row("14 current-to-molar conversions", "membrane_flux_scale=0.038; pump_flux_scale=0.011", "phenomenological conversion factors", "implementation scale", "0.038, 0.011", "params", "No", "No Faraday constant, membrane capacitance, area, or derivation is implemented."),
        row("15 membrane area / surface-to-volume", "no membrane area factor; extracellular divide by alpha, intracellular divide by beta=1-alpha", "local compartment fraction conversion", "dimensionless fractions", "alpha dynamic; beta=1-alpha", "simulation", "No", "A physical surface-to-volume ratio is absent."),
        row("16 pump stoichiometry", "K term -2*P; Na term +3*P; no Cl pump term", "3 Na / 2 K coefficients in effective flux variables", "stoichiometric coefficients", "3 Na, 2 K", "simulation", "No", "Signs depend on undocumented J convention."),
        row("17 GHK voltage", "Vm=26.64 ln[(Pk Ke + PNa Nae + PCl Cli)/(Pk Ki + PNa Nai + PCl Cle)]", "instantaneous membrane voltage", "mV", "thermal voltage 26.64 mV", "ghk", "Qualitative only", "Exact equation and 1e-6 floors are omitted."),
        row("18 Nernst potentials", "E_s=(26.64/z_s) ln(max(c_out,1e-6)/max(c_in,1e-6))", "ion reversal potentials", "mV", "zK=zNa=+1,zCl=-1", "nernst", "Qualitative only", "Exact formula/floor omitted."),
        row("19 leak conductances", "gK=0.14; gNa=0.035; gCl=0.06", "Ohmic current coefficients I_s=g_s(Vm-E_s)", "implementation conductance scale", "0.14,0.035,0.06", "params", "No", "Units and values omitted."),
        row("20 active permeabilities/conductances", "P=P_leak+P_active*theta; g=g_leak+g_active*theta", "GHK permeabilities and Ohmic conductances", "relative permeability / implementation conductance", "P active K/Na/Cl=1.25/1.55/0.18; g active=0.34/0.46/0.14", "simulation", "Qualitative only", "Exact values and distinction between GHK P and current g omitted."),
        row("21 theta ODE", "dtheta/dt=(theta_target-theta)/4.5", "activation relaxation", "s^-1; theta dimensionless", "tau=4.5 s", "simulation", "Yes, generic", "Exact update order is after concentration update but uses pre-update Vm."),
        row("22 theta sigmoid/slopes/bounds", "target=logistic[(Ke-8.5)/1.35+(Vm+58)/6.5]; if Ke>=threshold target=max(target,0.55); clip theta to [0,1.5]", "activation target and clipping", "dimensionless", "8.5 mM,-58 mV,1.35 mM,6.5 mV; floor .55; cap 1.5", "theta", "Midpoints/slopes documented", "0.55 target floor, logistic clipping [-60,60], and theta cap 1.5 omitted."),
        row("23 swelling target s_infinity", "q=5*r/(r+0.06)+0.20*clip(theta,0,1.5); s_inf=1.10*q/(1+q)", "bounded swelling target; r=max((Ki+Nai+Cli-rest_osm)/rest_osm,0)", "dimensionless", "gain=5, half=.06, activity=.20, cap=1.10", "simulation", "Only qualitative", "Exact saturation law and constants omitted."),
        row("24 alpha(s)", "alpha=clip(alpha0*(1-0.34*clip(s,0,1.10)),0.08,0.20)", "dynamic ECS volume fraction", "dimensionless", "base .20,min .08,gain .34", "dynamic", "Only bounds/qualitative", "Exact gain and geometric alpha0 construction not fully documented."),
        row("25 lambda(s)", "lambda=clip(lambda0*(1+0.28*clip(s,0,1.10)),1.0,4.8)", "dynamic tortuosity", "dimensionless", "base 1.60; gain .28; cap 4.8", "dynamic", "Bounds documented", "Exact multiplier omitted."),
        row("26 swelling recovery", "tau=12 s if s_inf>=s else 28 s", "piecewise relaxation branch", "s", "12,28", "simulation", "Yes, times only", "Branch condition omitted."),
        row("27 initial concentrations", "Ke=3.5,Ki=140,Nae=145,Nai=18,Cle=112,Cli=7", "uniform resting ion fields", "mM", "listed values", "simulation", "No", "Initial concentrations absent from manuscript table."),
        row("28 initial voltage/activation", "Vm=rest GHK; theta=0; s=0; Ve=0; O=1; constriction=0; perfusion=baseline_reserve", "initial states before focal perturbation", "mixed", "derived/rest values", "simulation", "No", "Exact initial voltage/state not reported."),
        row("29 exact focal perturbation", "3D Euclidean distance <=1.2 mm: Ke=max(Ke,22), Nae=max(Nae-10,5), theta=max(theta,.92)", "initial-only perturbation", "mm,mM,dimensionless", "radius 1.2; Ke22; Na drop10; theta.92", "simulation", "Yes", "Stimulus region is Euclidean 3D, not geodesic; manuscript does not state this."),
        row("30 ionic boundary conditions", "edge flux only over mesh edges; no exterior-edge flux term", "implicit natural zero-flux boundary", "not applicable", "no-flux by discretization", "transport", "No", "Boundary condition omitted."),
        row("31 potential boundary conditions", "screened FEM system with no explicit boundary term; natural no-flux; mass-weighted mean removed", "potential boundary/gauge handling", "not applicable", "natural no-flux + mean zero", "potential", "No", "Boundary/gauge omitted."),
        row("32 finite mesh boundaries", "open strip boundary remains; cotangent graph has no neighbors outside domain", "finite-domain treatment", "not applicable", "natural reflecting/no-flux", "simulation", "No", "No absorbing layer or infinite-domain approximation."),
        row("33 concentration field interpretation", "one value per surface vertex for each intra/extracellular ion", "reduced two-compartment surface fields", "mM", "surface nodal fields", "simulation", "Calls them overlapping surface compartments", "Not explicitly identified as cortical-column averages or thickness-integrated fields."),
        row("34 unresolved-thickness mixing", "no through-thickness coordinate; each local intra/extracellular compartment is a scalar nodal state", "well-mixed unresolved local compartments", "not applicable", "implicit", "simulation", "No", "Well-mixed-through-thickness assumption is implicit, not stated."),
        row("35 V0,m_ed,ell_d,d_c,gamma terms", "V0 and m_ed enter beta_ed; ell_d/cutoff define normalized Kd; gamma_rho/gamma_d enter phi RHS", "parameter entry points", "mixed", "5,.5,4,12,.084,1.5", "transport", "Yes", "Code normalization means gamma_d multiplies a row-normalized average, not an unnormalized integral."),
        row("36 caps/floors/clipping/fallbacks", "concentration clips: Ke[1,80],Nae/Cle[20,170],Ki[20,160],Nai/Cli[2,80]; Vm[-95,35]; theta[0,1.5]; s[0,1.10]; logistic input[-60,60]; numerous 1e-6 floors", "numerical safeguards", "state units", "as expression", "simulation", "No", "Extensive safeguards and explicit-Euler arrival quantization omitted."),
        row("37 null kernels", "distance-only w=exp(-dg/ell); scrambled-normal uses fixed RNG seed 13 permutation before opposition; both cutoff, threshold, row normalize", "orientation null definitions", "dimensionless weights", "ell=4 mm, cutoff=12 mm, seed=13", "kernel", "Qualitative definitions", "Kernel >1e-8 pruning and exact seed omitted."),
    ]
    write_csv(out_dir / "model_equation_audit.csv", rows)
    param_rows = []
    for field in dc.fields(MechanisticSurfaceParams):
        value = getattr(p, field.name)
        param_rows.append({"parameter": field.name, "default_value": value, "python_type": str(field.type), "code_file": "src/csd_sulcus/surface_mechanistic.py", "line_range": ranges["params"], "units_or_status": "See model_equation_audit.csv; code does not attach unit metadata"})
    write_csv(out_dir / "parameter_audit.csv", param_rows)
    md = ["# Exact model-equation audit", "", "This is a code-level transcription, not a proposed Methods rewrite. Quantities without enforced physical units are labeled as implementation/phenomenological scales.", "", "| Item | Implemented expression | Default | Source | Manuscript status/discrepancy |", "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['audit_item']} | {r['mathematical_expression_matching_implementation']} | {r['default_value']} | `{r['code_file']}:{r['line_number_or_range']}` | {r['currently_documented_in_manuscript']}; {r['code_manuscript_discrepancy']} |")
    write_text(out_dir / "model_equation_audit.md", "\n".join(md))
    write_text(out_dir / "initial_and_boundary_conditions.md", f"""# Initial and boundary conditions

## Initial state

Uniform concentrations are Ke={p.k_e_rest}, Ki={p.k_i_rest}, Nae={p.na_e_rest}, Nai={p.na_i_rest}, Cle={p.cl_e_rest}, and Cli={p.cl_i_rest} mM. Vm is the GHK value from those concentrations and leak permeabilities. Theta, swelling, constriction, and phi_hat start at zero; oxygen starts at one; perfusion starts at the geometry-dependent baseline reserve.

The focal perturbation is applied once at t=0 to vertices within 1.2 mm **three-dimensional Euclidean** distance of the frozen stimulus vertex: Ke is raised to at least 22 mM, Nae is reduced by 10 mM with a 5 mM floor, and theta is raised to at least 0.92. There is no sustained source.

## Boundaries

Ion transport and potential use mesh-edge cotangent operators with no exterior-edge source or flux. On the open rectangular strip this is an implicit natural zero-normal-flux boundary. The screened potential solve has no explicit Dirichlet boundary; after solving, the mass-weighted mean is subtracted. There is no absorbing layer, surrounding conductor, or continuation beyond the finite mesh.

## Reduced compartments

All concentrations are scalar nodal surface fields. The code has no thickness coordinate, membrane area-to-volume ratio, or through-thickness gradients. Intracellular and extracellular compartments are therefore locally well mixed in the unresolved direction; alpha and 1-alpha rescale effective compartment source terms.
""")
    discrepancies = [r for r in rows if r["code_manuscript_discrepancy"] != "None identified"]
    disc = ["# Code-manuscript discrepancies and omissions", "", "These findings report the current implementation; they do not propose replacement prose.", ""]
    for r in discrepancies:
        disc.append(f"- **{r['audit_item']}**: {r['code_manuscript_discrepancy']}")
    disc += ["", "The most consequential reproducibility omissions are the phenomenological current-to-concentration factors, absence of a membrane area/volume conversion, exact pump factors and Hill exponents, exact theta and swelling laws, all clipping rules, implicit boundary conditions, and unresolved-thickness mixing assumption. The manuscript's extracellular J sign cannot be reconciled unambiguously because J is not explicitly assigned the code's sign convention."]
    write_text(out_dir / "code_manuscript_discrepancies.md", "\n".join(disc))
    return rows


def skull_feasibility() -> None:
    write_text(OUT / "06_skull_feasibility.md", """# Skull / surrounding-conductor feasibility

1. **Represented layers:** The repository's mechanistic surface solver represents neither CSF, meninges, skull, scalp, nor any surrounding volume-conductor layer. It evolves two overlapping ionic surface compartments on the cortical mesh only.
2. **External conductivity boundary:** The screened surface potential equation cannot assign an exterior conductivity or interface condition. Its finite-strip boundary is an implicit natural no-flux boundary.
3. **Existing physical skull parameter:** None. `potential_screening` (manuscript kappa), the 1.8 electrical surface-operator multiplier (sigma_phi), `field_reference_mV`, and dipole gains are internal phenomenological surface-equation parameters, not skull conductivity.
4. **Interpretability of changing an internal parameter:** Relabeling or varying one of those terms as skull conductivity would be arbitrary because the implementation contains no derivation connecting it to a layered volume conductor or conductivity contrast.
5. **Answerability:** The reviewer's skull question is not answerable with the current model. A three-dimensional volume-conductor or defensibly derived boundary-layer extension with CSF/skull interfaces would be required.

No sensitivity simulation was run because the required external-conductivity/boundary-layer option does not exist. This is a feasibility limit, not a null result about skull effects.
""")


def consistency_checks(canonical: dict) -> list[dict]:
    checks = []
    def check(name: str, passed: bool, observed, expected, tolerance=0.0, note=""):
        checks.append({"check": name, "passed": bool(passed), "observed": observed, "expected": expected, "absolute_tolerance": tolerance, "note": note})
    m0, m1 = canonical["metrics"]
    c = canonical["comparison"]
    check("speed paired subtraction", math.isclose(c["speed_slowdown_no_minus_dipole_mm_min"], m0["speed_mm_min"]-m1["speed_mm_min"], abs_tol=1e-12), c["speed_slowdown_no_minus_dipole_mm_min"], m0["speed_mm_min"]-m1["speed_mm_min"], 1e-12)
    check("vertex no-dipole delay underlying arrivals", math.isclose(m0["vertex_within_condition_delay_s"], m0["vertex_e2_arrival_s"]-m0["vertex_e1_arrival_s"], abs_tol=1e-12), m0["vertex_within_condition_delay_s"], m0["vertex_e2_arrival_s"]-m0["vertex_e1_arrival_s"], 1e-12)
    check("vertex dipole delay underlying arrivals", math.isclose(m1["vertex_within_condition_delay_s"], m1["vertex_e2_arrival_s"]-m1["vertex_e1_arrival_s"], abs_tol=1e-12), m1["vertex_within_condition_delay_s"], m1["vertex_e2_arrival_s"]-m1["vertex_e1_arrival_s"], 1e-12)
    pct = 100*(m1["speed_mm_min"]-m0["speed_mm_min"])/m0["speed_mm_min"]
    check("percentage denominator is no-dipole speed", math.isclose(c["speed_percent_change_vs_no_dipole"], pct, abs_tol=1e-12), c["speed_percent_change_vs_no_dipole"], pct, 1e-12)
    speed_formula = 60*m0["geodesic_distance_mm"]/(m0["roi_e2_arrival_s"]-m0["roi_e1_arrival_s"])
    check("mm/s to mm/min conversion", math.isclose(m0["speed_mm_min"], speed_formula, abs_tol=1e-12), m0["speed_mm_min"], speed_formula, 1e-12, "factor 60 explicitly applied")
    check("paired physical readouts", canonical["baseline"].mesh is canonical["dipole"].mesh, "same mesh object and frozen indices", "same mesh object and frozen indices")
    with np.load(OUT / "01_canonical_representative" / "mesh_and_readouts.npz") as meshdata:
        check("NPZ readout indices", int(meshdata["e1_vertex"]) == canonical["e1"] and int(meshdata["e2_vertex"]) == canonical["e2"], f"{int(meshdata['e1_vertex'])},{int(meshdata['e2_vertex'])}", f"{canonical['e1']},{canonical['e2']}")
    saved_json = json.loads((OUT / "01_canonical_representative" / "canonical_metrics.json").read_text(encoding="utf-8"))
    check("CSV/JSON canonical speed agreement", math.isclose(saved_json["conditions"][0]["speed_mm_min"], m0["speed_mm_min"], abs_tol=1e-12), saved_json["conditions"][0]["speed_mm_min"], m0["speed_mm_min"], 1e-12)
    with np.load(OUT / "03_difference_visualization" / "plotted_arrays.npz") as plotted:
        check("difference array sign/data", np.allclose(plotted["delta_arrival_s"], canonical["dipole"].arrival_times-canonical["baseline"].arrival_times, equal_nan=True), "saved delta array", "dipole minus no dipole")
    for stem in ("matched_maps", "arrival_time_difference_map", "paired_kymographs", "path_arrival_and_deltaT"):
        for suffix in (".pdf", ".svg", ".png"):
            check(f"candidate figure exists: {stem}{suffix}", (OUT/"03_difference_visualization"/(stem+suffix)).exists(), "exists" if (OUT/"03_difference_visualization"/(stem+suffix)).exists() else "missing", "exists")
    archived_rows = list(csv.DictReader((ROOT / "outputs" / "surface_mechanistic_study" / "mechanistic_representative_summary.csv").open(encoding="utf-8")))
    archive_base = next(r for r in archived_rows if r["case_label"] == "mechanistic_multion_baseline")
    archive_dip = next(r for r in archived_rows if r["case_label"] == "mechanistic_multion_dipole")
    check("canonical baseline equals current archived pipeline", math.isclose(float(archive_base["arrival_speed_mm_min"]), m0["speed_mm_min"], abs_tol=1e-12), m0["speed_mm_min"], archive_base["arrival_speed_mm_min"], 1e-12)
    check("canonical dipole equals current archived pipeline", math.isclose(float(archive_dip["arrival_speed_mm_min"]), m1["speed_mm_min"], abs_tol=1e-12), m1["speed_mm_min"], archive_dip["arrival_speed_mm_min"], 1e-12)
    check("representative not mixed with stale table", not math.isclose(m0["speed_mm_min"], 2.5582349901822656, abs_tol=1e-9), m0["speed_mm_min"], "must differ from stale pre-revision value", 1e-9)
    mesh, stim, e1, e2, fb, fd = run_geometry_pair(52, 24, 0.0, 1.5, final_t=210.0)
    fmb, fmd = roi_metrics(fb, e1, e2), roi_metrics(fd, e1, e2)
    check("flat control computed in driver", True, f"computed speeds {fmb['speed_mm_min_recomputed']},{fmd['speed_mm_min_recomputed']}", "computed, not inserted")
    check("flat aligned kernel has zero effect", math.isclose(fmb["speed_mm_min_recomputed"], fmd["speed_mm_min_recomputed"], abs_tol=1e-12), fmd["speed_mm_min_recomputed"]-fmb["speed_mm_min_recomputed"], 0.0, 1e-12)
    check("single-command driver exists", Path(__file__).resolve().exists(), str(Path(__file__).resolve()), "exists")
    write_csv(OUT / "07_consistency_checks.csv", checks)
    failed = [r for r in checks if not r["passed"]]
    if failed:
        raise RuntimeError(f"Consistency checks failed: {[r['check'] for r in failed]}")
    return checks


def reproduction_commands(env: dict) -> None:
    pytest_command = (
        f"{sys.executable} -m pytest tests/test_surface_mechanistic.py tests/test_surface_representative.py"
        if env["packages"].get("pytest") != "not installed"
        else "python -m pytest tests/test_surface_mechanistic.py tests/test_surface_representative.py  # pytest is absent from .venv; uses an existing system installation"
    )
    text = f"""# Run from repository root: {ROOT}
# The driver fails on missing core dependencies/inputs and writes only below results/reviewer_revision_analysis.
{sys.executable} scripts/reviewer_revision_analysis/run_all_revision_analyses.py

# Optional verification
{pytest_command}
git status --short
git diff --name-only

Git commit analyzed: {env['git_commit']}
Fixed random seed: {SEED}
"""
    write_text(OUT / "07_reproduction_commands.txt", text)


def final_git_safety() -> dict:
    status = run_cmd(["git", "status", "--short"], check=False)
    diff_names = run_cmd(["git", "diff", "--name-only"], check=False)
    baseline_tracked = {
        "manuscript/figures/fig2_rep_quantitative.pdf", "manuscript/figures/fig2_rep_quantitative.png", "scripts/run_fig2_rep_quantitative.py",
        "src/csd_sulcus_model.egg-info/PKG-INFO", "src/csd_sulcus_model.egg-info/SOURCES.txt", "src/csd_sulcus_model.egg-info/dependency_links.txt",
        "src/csd_sulcus_model.egg-info/requires.txt", "src/csd_sulcus_model.egg-info/top_level.txt",
    }
    current_tracked = {
        line.strip().replace("\\", "/")
        for line in diff_names.splitlines()
        if line.strip() and not line.lower().startswith("warning:")
    }
    unexpected = sorted(current_tracked - baseline_tracked)
    if unexpected:
        raise RuntimeError(f"Unexpected tracked modifications outside analysis directories: {unexpected}")
    return {"git_status": status, "git_diff_name_only": diff_names, "preexisting_tracked_modifications": sorted(baseline_tracked), "unexpected_new_tracked_modifications": unexpected, "safety_passed": not unexpected}


def final_report(env: dict, canonical: dict, shoulder: dict, checks: list[dict], safety: dict) -> None:
    m0, m1 = canonical["metrics"]
    c = canonical["comparison"]
    generated = sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in OUT.rglob("*") if p.is_file())
    report = f"""# Final computational revision analysis report

## 1. Executive factual summary

The authoritative current representative run was reproducibly identified and rerun. Cross-fold speed was {m0['speed_mm_min']:.12f} mm/min without the dipole kernel and {m1['speed_mm_min']:.12f} mm/min with aligned dipole coupling (slowdown {c['speed_slowdown_no_minus_dipole_mm_min']:.12f} mm/min; {abs(c['speed_percent_change_vs_no_dipole']):.6f}% relative to no dipole). Center-vertex E1-E2 delays were {m0['vertex_within_condition_delay_s']:.12f} and {m1['vertex_within_condition_delay_s']:.12f} s; the increase was {c['vertex_delay_increase_s']:.12f} s. The between-condition downstream E2 arrival shift was {c['between_condition_downstream_E2_arrival_shift_s']:.12f} s. Maximum absolute Ve was {m0['max_abs_Ve_mV']:.12f} and {m1['max_abs_Ve_mV']:.12f} mV.

## 2. Repository and environment

Branch `{env['git_branch']}`, commit `{env['git_commit']}`, Python `{env['python']}` on `{env['os']}`. Full package, CPU, memory, GPU, initial status, and pre-analysis dirty-tree state are in `00_environment.txt`.

## 3. Representative-run provenance

The current implementation and declared bounded swelling specification identify the post-`7d054b8` run as authoritative. The mesh, physical stimulus/readout locations, threshold, duration, and time step were verified in code. The pipeline uses no randomness for aligned/no-dipole runs; seed {SEED} is fixed for all potentially random operations.

## 4. Authoritative representative metrics

Speed uses 60 times the {m0['geodesic_distance_mm']:.12f} mm center-to-center geodesic distance divided by the 1 mm ROI median-arrival difference. The original saved delay is instead the E2 center-vertex arrival minus E1 center-vertex arrival. ROI delays were {m0['roi_within_condition_delay_s']:.12f} and {m1['roi_within_condition_delay_s']:.12f} s. Both definitions are preserved in the canonical CSV/JSON.

## 5. Competing Figure 3 / Table 2 values

The 2.558/2.476 mm/min, 130.5/134.7 s, and 18.99/19.50 mV set was generated before the swelling law was changed to the current bounded saturating target with a separate recovery branch. Its output-side Table S2 was never regenerated and is stale. The 2.609/2.537 set comes from the current representative CSV/JSON and current manuscript-side Table S2. Neither set is a null-control result. A pre-existing uncommitted edit to `scripts/run_fig2_rep_quantitative.py` hard-codes the stale labels while still importing the current simulation configuration, producing a mixed working-tree Figure 2 artifact. No production file was changed here.

## 6. Geometry-profile outputs

All 12 implemented Gaussian folds plus a computed flat control are provided with equal physical scaling. Because the 28-point cross-section has no y=0 vertex, sampled maximum depths are slightly below analytic requested depths; exact sampled coordinates and fitted sigma/FWHM checks are tabulated.

## 7. Difference-map findings

Signed Ke and arrival-time differences were computed as dipole minus no dipole. Display times were fixed at 25%, 50%, and 75% of the no-dipole E1-E2 traversal interval. Paired absolute maps share limits; signed maps are zero-centered. Arrays, coordinate conventions, and selected times are saved separately.

## 8. Shoulder/gyral-effect findings

Required classification: **{shoulder['conclusion']}**. The minimum representative path Delta T was {shoulder['minimum_representative_deltaT_s']:.12f} s. Region definitions depend only on analytic Gaussian landmarks, with a predeclared shoulder-to-flatter boundary at |y|=2 sigma. Across the 12 deterministic geometries, classifications were {shoulder['counts']}.

## 9. Numerical robustness of localized acceleration

The two shoulders were checked on the 64x28 reference mesh, a 72x32 neighboring mesh, and with the reference time step halved. Local speed was evaluated with 5-, 7-, and 9-vertex Savitzky-Golay derivatives and unsmoothed adjacent segments. Detailed outcomes are in `04_shoulder_analysis/numerical_check.csv`; the classification above requires agreement across these definitions and resolutions.

## 10. Exact model-equation audit

All 37 requested implementation items are transcribed with expressions, defaults, units/status, code locations, manuscript coverage, and discrepancies. Several quantities are phenomenological implementation scales rather than dimensionally derived biophysical conversions. In particular, the code has no Faraday/current-to-molar derivation or membrane area-to-volume factor; it uses fixed factors 0.038 and 0.011.

## 11. Initial and boundary conditions

Initial ion states, GHK voltage, activation, focal perturbation, clipping, and implicit natural no-flux boundaries are documented in `05_model_audit/initial_and_boundary_conditions.md`. The model is a two-compartment surface reduction with no through-thickness coordinate.

## 12. Skull/surrounding-conductor feasibility

The current solver has no CSF, meninges, skull, scalp, external conductivity, or boundary-layer option. Internal kappa/sigma/gain parameters cannot defensibly be relabeled as skull conductivity. The skull question therefore cannot be answered without a three-dimensional volume-conductor or derived boundary-layer extension; no arbitrary sensitivity was run.

## 13. Files generated

{chr(10).join('- `'+p+'`' for p in generated)}

## 14. Reproduction

Run `{sys.executable} scripts/reviewer_revision_analysis/run_all_revision_analyses.py` from `{ROOT}`. Exact commands are in `07_reproduction_commands.txt`. All {len(checks)} automated consistency checks passed.

## 15. Incomplete analyses or unresolved ambiguities

- A local scalar surface-normal opposition index is not implemented; opposition exists only pairwise inside the nonlocal kernel. It was not invented.
- The manuscript symbol J_s^m has no explicit sign convention mapping to the code's `*_membrane` variables; the apparent extracellular sign mismatch is therefore reported as unresolved.
- Skull effects are outside the current solver.
- No atlas rerun was needed for the reviewer-requested direct synthetic shoulder test; existing atlas provenance was inventoried only.

## 16. Git safety confirmation

No manuscript, supplement, bibliography, response-letter, cover-letter, or production figure was edited by this analysis. New writes are confined to `scripts/reviewer_revision_analysis/` and `results/reviewer_revision_analysis/`. The repository already contained tracked and untracked changes before this work; they were preserved. No unexpected new tracked modification was detected.

Final `git diff --name-only` (all entries pre-existing):

```text
{safety['git_diff_name_only']}
```

Final `git status --short`:

```text
{safety['git_status']}
```
"""
    write_text(OUT / "FINAL_ANALYSIS_REPORT.md", report)


def write_manifest() -> None:
    rows = []
    manifest = OUT / "manifest_sha256.txt"
    for path in sorted(p for p in OUT.rglob("*") if p.is_file() and p != manifest):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")
    write_text(manifest, "\n".join(rows))


def load_completed_state_for_finalize():
    metrics_json = json.loads((OUT / "01_canonical_representative" / "canonical_metrics.json").read_text(encoding="utf-8"))
    canonical = {"metrics": metrics_json["conditions"], "comparison": metrics_json["paired_comparison"]}
    shoulder_text = (OUT / "04_shoulder_analysis" / "shoulder_analysis_report.md").read_text(encoding="utf-8")
    classification_line = next(line for line in shoulder_text.splitlines() if line.startswith("**Required classification:"))
    conclusion = classification_line.split(":", 1)[1].strip().strip("*.")
    rep_rows = list(csv.DictReader((OUT / "04_shoulder_analysis" / "representative_pathwise_metrics.csv").open(encoding="utf-8")))
    all_rows = list(csv.DictReader((OUT / "04_shoulder_analysis" / "all_geometries_region_metrics.csv").open(encoding="utf-8")))
    shoulder = {
        "conclusion": conclusion,
        "minimum_representative_deltaT_s": min(float(r["DeltaT_s"]) for r in rep_rows if r["DeltaT_s"] not in ("", "nan")),
        "counts": dict(Counter(r["sign_of_effect"] for r in all_rows)),
    }
    checks = list(csv.DictReader((OUT / "07_consistency_checks.csv").open(encoding="utf-8")))
    return canonical, shoulder, checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalize-only", action="store_true", help="Finalize reports/hashes from already completed analysis outputs.")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    log = [
        f"command={' '.join(sys.argv)}", f"cwd={Path.cwd()}", f"start_utc={datetime.now(timezone.utc).isoformat()}",
        f"python={sys.version.replace(chr(10),' ')}", f"seed={SEED}",
    ]
    env = collect_environment()
    if args.finalize_only:
        canonical, shoulder, checks = load_completed_state_for_finalize()
        reproduction_commands(env)
        safety = final_git_safety()
        final_report(env, canonical, shoulder, checks, safety)
        write_manifest()
        print(f"Revision analyses finalized: {OUT}")
        return
    repository_inventory(env)
    provenance_map()
    canonical = canonical_run(log)
    geometry_outputs(canonical)
    difference_visualizations(canonical)
    shoulder = shoulder_analysis(canonical, log)
    model_audit()
    skull_feasibility()
    checks = consistency_checks(canonical)
    reproduction_commands(env)
    safety = final_git_safety()
    final_report(env, canonical, shoulder, checks, safety)
    write_manifest()
    # Update the canonical log after all timed analyses without rerunning simulations.
    log += [f"end_utc={datetime.now(timezone.utc).isoformat()}", f"consistency_checks_passed={len(checks)}", f"git_safety_passed={safety['safety_passed']}"]
    write_text(OUT / "01_canonical_representative" / "run_log.txt", "\n".join(log))
    write_manifest()
    print(f"Revision analyses complete: {OUT}")


if __name__ == "__main__":
    main()
