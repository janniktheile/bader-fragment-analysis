"""
Title: Fragment-Resolved MO Energies (H-1, H, L, L+1) with Broken Y-Axis
Python: 3.9+
Dependencies: numpy, matplotlib, brokenaxes

Purpose
-------
This script reads (i) fragment-resolved orbital compositions from Multiwfn
(`orbcomp.txt`) and (ii) orbital eigenvalues from a Gaussian log or text export
containing lines like:
    "Alpha  occ. eigenvalues --  -0.50000  -0.45000 ..."
    "Alpha virt. eigenvalues --   0.02000   0.05000 ..."
Energies are converted from Hartree to eV. For each system, the script selects
four orbitals (H-1, H, L, L+1), and draws horizontal, stacked bars whose
segment widths encode per-fragment contributions (in %). The y-axis is broken
to show occupied and low-lying virtual levels clearly in one panel.

This layout is suitable for inclusion in a scientific paper (e.g., as a
vector PDF). The code is heavily commented for readability and reproducibility.

Notes & Assumptions
-------------------
1) Multiwfn `orbcomp.txt` is expected to be grouped by orbitals with headers
   like "Orbital   123" followed by lines "atom_index  contribution%".
2) Gaussian output must include the "Alpha  occ. eigenvalues" and
   "Alpha virt. eigenvalues" lines (typical for SCF or TD-DFT printouts).
3) Fragment definitions use atom indices that correspond to positions in the
   geometry file used to run the calculation (1-based indexing, as in Multiwfn).
4) If any path is missing, the system is skipped (a warning is printed).
"""

# ============================== Imports =====================================

import os
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from brokenaxes import brokenaxes


# ============================ Configuration =================================

# Fragment labels, colors, and atom-index sets (1-based indices as in Multiwfn)
fragment_labels = ['A1', 'D', 'A2']
fragment_colors = ['#FF0000', '#0072BD', '#FFD700']  # red, blue, gold
fragment_indices: List[List[int]] = [
    [1, 2, 3, 4, 5, 6],          # A1
    list(range(7, 21)),          # D
    list(range(21, 33))          # A2
]

# Systems to plot (keep your absolute paths if you wish; relative paths also work)
# Tip: If distributing, consider replacing absolute paths with relative project paths.
systems = [
    {
        "name": "A1-D-A2",
        "orbcomp_path": "path_to/orbcomp.txt",
        "log_path": "path_to/go_ada.txt",
        # Element groups (S, N, H, C, O, F) as 1-based indices; kept from your original
        "fragments_by_element": [
            [5, 11, 18, 29],                          # S
            [2, 3, 14, 27, 28],                      # N
            [6, 15, 19, 20, 30, 31, 32],             # H
            [1, 4, 7, 8, 9, 10, 12, 13, 16, 17,
             21, 22, 23, 24, 25, 26],                # C
            [],                                      # O
            []                                       # F
        ]
    },
    {
        "name": "A2(1F)",
        "orbcomp_path": "path_to/orbcomp.txt",
        "log_path": "path_to/go_ada.txt",
        "fragments_by_element": [
            [5, 11, 18, 29],                         # S
            [2, 3, 14, 27, 28],                      # N
            [6, 15, 19, 20, 30, 31],                 # H
            [1, 4, 7, 8, 9, 10, 12, 13, 16, 17,
             21, 22, 23, 24, 25, 26],                # C
            [],                                      # O
            [32]                                     # F
        ]
    },
    {
        "name": "A2(2F)",
        "orbcomp_path": "path_to/orbcomp.txt",
        "log_path": "path_to/go_ada.txt",
        "fragments_by_element": [
            [5, 11, 18, 29],                         # S
            [2, 3, 14, 27, 28],                      # N
            [6, 15, 19, 20, 30],                     # H
            [1, 4, 7, 8, 9, 10, 12, 13, 16, 17,
             21, 22, 23, 24, 25, 26],                # C
            [],                                      # O
            [31, 32]                                 # F
        ]
    },
    {
        "name": "A2(3F)",
        "orbcomp_path": "path_to/orbcomp.txt",
        "log_path": "path_to/go_ada.txt",
        "fragments_by_element": [
            [5, 11, 18, 29],                         # S
            [2, 3, 14, 27, 28],                      # N
            [6, 15, 19, 20],                         # H
            [1, 4, 7, 8, 9, 10, 12, 13, 16, 17,
             21, 22, 23, 24, 25, 26],                # C
            [],                                      # O
            [30, 31, 32]                             # F
        ]
    },
    {
        "name": "D(1F)",
        "orbcomp_path": "path_to/orbcomp.txt",
        "log_path": "path_to/go_ada.txt",
        "fragments_by_element": [
            [5, 11, 18, 29],                         # S
            [2, 3, 14, 27, 28],                      # N
            [6, 15, 19, 30, 31, 32],                 # H
            [1, 4, 7, 8, 9, 10, 12, 13, 16, 17,
             21, 22, 23, 24, 25, 26],                # C
            [],                                      # O
            [20]                                     # F
        ]
    },
    {
        "name": "D(2F)",
        "orbcomp_path": "path_to/orbcomp.txt",
        "log_path": "path_to/go_ada.txt",
        "fragments_by_element": [
            [5, 11, 18, 29],                         # S
            [2, 3, 14, 27, 28],                      # N
            [6, 19, 30, 31, 32],                     # H
            [1, 4, 7, 8, 9, 10, 12, 13, 16, 17,
             21, 22, 23, 24, 25, 26],                # C
            [],                                      # O
            [15, 20]                                  # F
        ]
    },
    {
        "name": "D(3F)",
        "orbcomp_path": "path_to/orbcomp.txt",
        "log_path": "path_to/go_ada.txt",
        "fragments_by_element": [
            [5, 11, 18, 29],                         # S
            [2, 3, 14, 27, 28],                      # N
            [6, 19, 30, 31, 32],                     # H
            [1, 4, 7, 8, 9, 10, 12, 13, 16, 17,
             21, 22, 23, 24, 25, 26],                # C
            [],                                      # O
            [15, 19, 20]                              # F
        ]
    }
]

# Plot appearance
HARTREE_TO_EV = 27.2114
BAR_HEIGHT = 0.18
BAR_WIDTH = 0.70

# Visible y-ranges for the broken axis (adjust if your levels differ)
VISIBLE_LOWER = (-9.25, -5.75)
VISIBLE_UPPER = (-2.75, 0.15)


DPI = 300  # ignored for vector backends like .pdf/.svg


# ============================== Functions ===================================

def read_orbcomp(file_path: Path) -> Dict[int, List[Tuple[int, float]]]:
    """
    Parse Multiwfn 'orbcomp.txt'.

    Parameters
    ----------
    file_path : Path
        Path to orbcomp.txt.

    Returns
    -------
    data : dict
        Mapping: MO_number (int) -> List[(atom_index (int), contribution_percent (float))]

    Robustness
    ----------
    * Skips empty/malformed lines gracefully.
    * Accepts '%' presence/absence in the contribution field.
    """
    data: Dict[int, List[Tuple[int, float]]] = {}
    current_orbital = None

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            # Detect orbital header lines (e.g., "Orbital   123")
            if parts[0].lower().startswith("orbital") and len(parts) >= 2:
                try:
                    current_orbital = int(parts[1])
                    data[current_orbital] = []
                except ValueError:
                    current_orbital = None
                continue
            # Read atom contribution lines: "<atom_index>  <percent>"
            if current_orbital is not None and len(parts) >= 2:
                try:
                    atom_index = int(parts[0])
                    contrib_str = parts[1].replace("%", "")
                    contribution = float(contrib_str)
                    data[current_orbital].append((atom_index, contribution))
                except ValueError:
                    # Silently skip malformed rows to keep parsing resilient
                    continue

    if not data:
        warnings.warn(f"No orbital data parsed from {file_path}. "
                      f"Check format and headers like 'Orbital  <n>'.")
    return data


def calculate_fragment_contributions(
    data: Dict[int, List[Tuple[int, float]]],
    fragments: List[List[int]]
) -> Dict[int, List[float]]:
    """
    Sum per-atom contributions into per-fragment contributions (percent).

    Parameters
    ----------
    data : dict
        MO -> [(atom_index, contribution_percent)]
    fragments : list of lists
        Each inner list contains atom indices belonging to that fragment.

    Returns
    -------
    frag_contribs : dict
        MO -> [contrib_fragment_0, contrib_fragment_1, ...] in percent
    """
    frag_contribs: Dict[int, List[float]] = {}
    for mo, atom_contribs in data.items():
        per_frag = []
        for frag in fragments:
            # Sum contributions for atoms that belong to this fragment
            s = sum(val for atom, val in atom_contribs if atom in frag)
            per_frag.append(s)
        frag_contribs[mo] = per_frag
    return frag_contribs


def extract_alpha_energies_eV(file_path: Path) -> Tuple[List[float], List[float]]:
    """
    Extract Alpha occupied and virtual orbital eigenvalues from a Gaussian log/text.

    Parameters
    ----------
    file_path : Path
        Path to Gaussian output or a stripped text containing eigenvalue lines.

    Returns
    -------
    occupied_eV : list of float
    virtual_eV  : list of float

    Notes
    -----
    * Energies are parsed in Hartree and converted to eV.
    * Only Alpha eigenvalues are considered (typical for closed-shell systems).
    """
    occupied_h: List[float] = []
    virtual_h: List[float] = []

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # Match canonical Gaussian formatting
            if line.startswith("Alpha  occ. eigenvalues"):
                # Values start at token index 4 for lines like:
                # "Alpha  occ. eigenvalues --  -0.50000  -0.45000 ..."
                parts = line.split()
                # find numeric tokens (robust to varying dashes)
                nums = [p for p in parts if _is_number(p)]
                occupied_h.extend(map(float, nums))
            elif line.startswith("Alpha virt. eigenvalues"):
                parts = line.split()
                nums = [p for p in parts if _is_number(p)]
                virtual_h.extend(map(float, nums))

    if not occupied_h or not virtual_h:
        warnings.warn(
            f"Could not find both occupied and virtual alpha eigenvalues in {file_path}. "
            f"Check for the expected Gaussian lines."
        )

    occupied_eV = [e * HARTREE_TO_EV for e in occupied_h]
    virtual_eV = [e * HARTREE_TO_EV for e in virtual_h]
    return occupied_eV, virtual_eV


def _is_number(token: str) -> bool:
    """Return True if token can be parsed as float (handles leading dashes)."""
    try:
        float(token)
        return True
    except ValueError:
        return False


def prepare_plot_data(
    systems_cfg: List[dict],
    frag_definition: List[List[int]]
) -> List[Tuple[str, List[float], Dict[int, List[float]], List[int]]]:
    """
    For each system, read orbcomp + energies and select H-1, H, L, L+1.

    Returns a list of tuples:
        (system_name, energies_eV, frag_contribs_by_MO, MO_numbers)

    If any required file is missing or parsing fails, that system is skipped.
    """
    plot_bundle = []

    for sys_cfg in systems_cfg:
        name = sys_cfg.get("name", "Unnamed")
        orb_path = Path(sys_cfg["orbcomp_path"])
        log_path = Path(sys_cfg["log_path"])

        if not orb_path.exists():
            warnings.warn(f"[{name}] Missing orbcomp: {orb_path}. Skipping.")
            continue
        if not log_path.exists():
            warnings.warn(f"[{name}] Missing log: {log_path}. Skipping.")
            continue

        orb_data = read_orbcomp(orb_path)
        if not orb_data:
            warnings.warn(f"[{name}] No orbital composition parsed. Skipping.")
            continue

        frag_contribs = calculate_fragment_contributions(orb_data, frag_definition)
        occupied, virtual = extract_alpha_energies_eV(log_path)

        # Determine MO indices present in orbcomp (sorted, 1-based from Multiwfn)
        mo_numbers_sorted = sorted(orb_data.keys())

        # The HOMO index in energy arrays = len(occupied) - 1 (0-based)
        num_occ = len(occupied)
        if num_occ < 2 or len(virtual) < 2:
            warnings.warn(f"[{name}] Not enough orbitals to select H-1/H/L/L+1. Skipping.")
            continue

        homo_idx_energy = num_occ - 1  # 0-based in energy list
        # Select indices into energy arrays: H-1, H, L, L+1
        selected_energy_values = [
            occupied[homo_idx_energy - 1],
            occupied[homo_idx_energy],
            virtual[0],
            virtual[1],
        ]

        # Map these selections to MO numbers from orbcomp.
        # Assumption: orbcomp includes all MOs in energy order. We map by position.
        # (If not strictly aligned, consider a dedicated MO-number/energy mapping.)
        selection_positions = [homo_idx_energy - 1, homo_idx_energy, num_occ, num_occ + 1]
        # Guard against short 'mo_numbers_sorted'
        if max(selection_positions) >= len(mo_numbers_sorted):
            warnings.warn(f"[{name}] orbcomp lacks some MO entries matching energies. Skipping.")
            continue
        selected_mo_numbers = [mo_numbers_sorted[i] for i in selection_positions]

        plot_bundle.append((name, selected_energy_values, frag_contribs, selected_mo_numbers))

    if not plot_bundle:
        warnings.warn("No systems available for plotting after preprocessing.")
    return plot_bundle


def configure_matplotlib():
    """Apply manuscript-friendly matplotlib settings."""
    plt.rcParams.update({
        'font.size': 16,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'axes.linewidth': 2.5,
        'xtick.major.width': 2.0,
        'ytick.major.width': 2.0,
        # Avoid clipping large labels
        'figure.constrained_layout.use': False,
    })


def plot_fragment_bars_with_broken_axis(
    plot_data: List[Tuple[str, List[float], Dict[int, List[float]], List[int]]],
    frag_labels: List[str],
    frag_colors: List[str],
    save: bool = False,
    path: Path = FIG_PATH,
    dpi: int = DPI,
):
    """
    Draw the stacked, fragment-colored bars for H-1, H, L, L+1 across systems.

    Parameters
    ----------
    plot_data : list of tuples
        Each tuple: (system_name, [E(H-1), E(H), E(L), E(L+1)],
                     frag_contribs_by_MO, [MO_numbers_for_these_levels])
    frag_labels : list of str
    frag_colors : list of str
    save : bool
        If True, figure is written to 'path'.
    path : Path
        Output file path.
    dpi : int
        DPI for raster formats (ignored for vector formats like PDF/SVG).
    """
    if not plot_data:
        print("No data to plot.")
        return

    # Figure width scales with number of systems to keep labels readable
    fig_w = max(6.0, len(plot_data) * 1.75)
    fig_h = 4.0
    fig = plt.figure(figsize=(fig_w, fig_h))

    bax = brokenaxes(
        ylims=[VISIBLE_LOWER, VISIBLE_UPPER],
        hspace=0.02,
        despine=True,
        diag_color="none"  # draw custom break marks below
    )

    # Remove automatic diagonal break markers that sometimes appear as tiny lines
    for axis in bax.axs:
        for line in list(axis.lines):
            if (line.get_linestyle() == 'None'
                and line.get_marker() == 'None'
                and len(getattr(line, 'get_xydata', lambda: [])()) == 2):
                line.remove()

    # Label mapping for the first system only (HOMO/LUMO)
    orbital_labels = ['H-1', 'H', 'L', 'L+1']
    label_by_mo_first = {}
    _, _, _, mo_numbers0 = plot_data[0]
    want_map = {'H': 'HOMO', 'L': 'LUMO'}
    for idx, lab in enumerate(orbital_labels):
        if lab in want_map and idx < len(mo_numbers0):
            label_by_mo_first[mo_numbers0[idx]] = want_map[lab]

    # Draw bars
    for i, (name, energies, frag_contribs, mo_numbers) in enumerate(plot_data):
        left = i - BAR_WIDTH / 2.0
        for j, energy in enumerate(energies):
            mo_num = mo_numbers[j]
            # Fragment contributions for this MO (percent)
            contribs = frag_contribs.get(mo_num, [0.0] * len(frag_labels))
            # Normalize tiny numerical drift so total ≈ 100
            total = sum(contribs) or 1.0
            contribs = [max(0.0, v) for v in contribs]  # avoid negatives
            # Draw horizontal stacked segments by varying x-extent (width)
            cum_width = 0.0
            for k, value in enumerate(contribs):
                width = (value / 100.0) * BAR_WIDTH
                bax.broken_barh(
                    [(left + cum_width, width)],
                    (energy - BAR_HEIGHT / 2.0, BAR_HEIGHT),
                    facecolors=frag_colors[k],
                    edgecolors='black',
                    linewidth=0.3
                )
                cum_width += width

            # Annotate HOMO/LUMO on the first system only
            if i == 0 and mo_num in label_by_mo_first:
                bax.annotate(
                    label_by_mo_first[mo_num],
                    xy=(i, energy - 0.11),
                    ha='center', va='top',
                    fontsize=34, fontweight='bold'
                )

    # System names below the lower window (tweak the y for your data if needed)
    for i, (name, _, _, _) in enumerate(plot_data):
        bax.annotate(
            name,
            xy=(i, VISIBLE_LOWER[0] + 0.75),
            ha='center', va='top',
            fontsize=30, rotation=0, weight='bold'
        )

    # Custom break marks on the main (lower) axis, purely cosmetic
    main_axis = bax.axs[0]
    main_axis.plot([-0.015, 0.015], [-0.02, 0.01], transform=main_axis.transAxes,
                   color='k', linewidth=1.3, clip_on=False)
    main_axis.plot([-0.015, 0.015], [0.01, 0.04], transform=main_axis.transAxes,
                   color='k', linewidth=1.3, clip_on=False)
    main_axis.plot([0.985, 1.015], [-0.02, 0.01], transform=main_axis.transAxes,
                   color='k', linewidth=1.3, clip_on=False)
    main_axis.plot([0.985, 1.015], [0.01, 0.04], transform=main_axis.transAxes,
                   color='k', linewidth=1.3, clip_on=False)

    # Axes cosmetics
    bax.set_ylabel("Energy (eV)", fontsize=38, labelpad=43)
    for axis in bax.axs:
        axis.tick_params(labelsize=26, width=2, length=6)
        axis.spines['left'].set_linewidth(2.5)
        axis.spines['right'].set_visible(True)
        ylims = axis.get_ylim()
        axis.spines['top'].set_visible(ylims[1] > VISIBLE_UPPER[0] + 1e-3)
        axis.spines['bottom'].set_visible(ylims[0] < VISIBLE_LOWER[1] - 1e-3)
        axis.set_xticks([])

    # Legend with colored patches (centered near the top-left; adjust bbox_to_anchor as needed)
    handles = [Patch(facecolor=frag_colors[i], edgecolor='black', label=frag_labels[i])
               for i in range(len(frag_labels))]
    fig.legend(
        handles, frag_labels,
        loc='upper left',
        bbox_to_anchor=(0.13, 0.9),
        ncol=len(frag_labels),
        fontsize=34,
        frameon=False
    )

    # Avoid tight_layout with brokenaxes; a small top margin helps legend breathing room
    fig.subplots_adjust(top=0.92, bottom=0.12, left=0.10, right=0.98)

    if save:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        print(f"Figure written to: {path.resolve()}")
    else:
        plt.show()


# ================================ Main ======================================

def main():
    """End-to-end execution: parse inputs, prepare data, and plot/optionally save."""
    configure_matplotlib()

    # By default we use the three-fragment partition you defined at the top:
    frag_def = fragment_indices

    # Build the plotting bundle across all systems
    plot_data = prepare_plot_data(systems, frag_def)

    # For labeling HOMO/LUMO on the first system, define which four orbitals we selected:
    # ['H-1', 'H', 'L', 'L+1'] (already used in the plotting function)

    # Finally, draw the figure
    plot_fragment_bars_with_broken_axis(
        plot_data,
        frag_labels=fragment_labels,
        frag_colors=fragment_colors,
        save=SAVE_FIG,
        path=FIG_PATH,
        dpi=DPI
    )


if __name__ == "__main__":
    main()
