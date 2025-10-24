# -*- coding: utf-8 -*-
"""
Fragment-wise heat map (A1–D–A2, 8 systems)
--------------------------------------------
Reads Multiwfn outputs (multipole.txt), basin-to-atom mapping, and XYZ coordinates
for each system. Computes per-atom net charges, applies optional correction,
sums charges over user-defined fragments (A1, D, A2), and plots a heatmap
(fragments × systems).
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cms = 1 / 2.54
plt.rcParams.update({
    'font.size': 13,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica'],
    'axes.linewidth': 2.0,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
})

atomic_numbers = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
    "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17,
    "Ar": 18, "K": 19, "Ca": 20, "Fe": 26, "Ni": 28, "Cu": 29, "Zn": 30, "Br": 35, "I": 53
}

# === Helper parsers ===
def parse_basin_zuordnung(filepath):
    mapping = {}
    pattern = r"Attractor\s+(\d+)\s+corresponds to atom\s+(\d+)\s+\((\w+)\s*\)"
    with open(filepath, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                mapping[int(match.group(1))] = (match.group(3), int(match.group(2)))
    return mapping


def extract_bader_electrons(file_path):
    electrons = []
    with open(file_path, "r") as f:
        for line in f:
            if "Basin monopole moment:" in line:
                val = float(line.strip().split()[-1])
                electrons.append(abs(val))
    return electrons


def read_xyz_coords(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()[2:]
    coords = []
    for line in lines:
        if line.strip():
            parts = line.split()
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(coords)


# === Analysis ===
def bader_charge_analysis(multipole_file, mapping_file, xyz_file, correction="none"):
    electrons = extract_bader_electrons(multipole_file)
    mapping = parse_basin_zuordnung(mapping_file)
    coords = read_xyz_coords(xyz_file)

    data = []
    for i, e in enumerate(electrons, start=1):
        element, atom_index = mapping[i]
        Z = atomic_numbers.get(element)
        x, y, z = coords[atom_index - 1]
        net_charge = Z - e
        data.append({
            "Attractor": i,
            "Atom_index": atom_index,
            "Element": element,
            "Z": Z,
            "X": x, "Y": y, "Z_coord": z,
            "Bader_electrons": e,
            "Net_charge": net_charge
        })

    df = pd.DataFrame(data).sort_values("Atom_index").reset_index(drop=True)

    if correction == "shift":
        delta = df["Net_charge"].sum() / len(df)
        df["Net_charge"] -= delta
        print(f"🔧 Correction 'shift' applied (offset removed: {delta:+.4f} e)")
    elif correction == "scale":
        df["Bader_electrons"] *= df["Z"].sum() / df["Bader_electrons"].sum()
        df["Net_charge"] = df["Z"] - df["Bader_electrons"]
        print("🔧 Correction 'scale' applied.")
    elif correction != "none":
        raise ValueError("Correction must be 'none', 'shift', or 'scale'")

    total = df["Net_charge"].sum()
    print(f"👉 Total molecular charge: {total:+.4f} e")
    return df


# === Plot (fixed: resolves fragments per system) ===
def plot_fragment_sums_multiple(multipole_files, mapping_files, xyz_files,
                                fragments, fragment_labels, system_labels,
                                correction="shift"):

    def resolve_fragments_for(label):
        """Return fragment lists for given system label."""
        if isinstance(fragments, dict):
            return fragments[label]
        elif isinstance(fragments, list):
            return fragments
        else:
            raise TypeError("fragments must be dict or list of lists")

    all_sums = []
    for mfile, zfile, xfile, label in zip(multipole_files, mapping_files, xyz_files, system_labels):
        df = bader_charge_analysis(mfile, zfile, xfile, correction=correction)
        frag_lists = resolve_fragments_for(label)
        frag_sums = []
        for frag in frag_lists:
            frag_sums.append(df.loc[df["Atom_index"].isin(frag), "Net_charge"].sum())
        all_sums.append(frag_sums)
        print(f"{label}: total net charge = {df['Net_charge'].sum():.4f} e")

    fragment_sums = np.array(all_sums).T

    plt.figure(figsize=(18 * cms, 12 * cms))
    ax = sns.heatmap(
        fragment_sums,
        annot=True, fmt=".3f",
        cmap='jet', center=0,
        vmin=-0.3, vmax=0.4,
        xticklabels=system_labels,
        yticklabels=fragment_labels,
        annot_kws={"size": 22},
        cbar_kws={'label': 'Sum. Net Charge per Fragment (e)'}
    )
    plt.title("Fragment-Wise Net Charges Across Systems", fontsize=24, pad=20)
    plt.xlabel("")
    plt.ylabel("Fragment", fontsize=22)
    ax.tick_params(labelsize=20)
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel('Sum. Net Charge per Fragment (e)', fontsize=20, labelpad=20)
    cbar.ax.tick_params(labelsize=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


# === Paths === Example:
multipole_files = [
    "path_to/A1_D_A2/multipole.txt",
    "path_to/A1-π-D-π-A2/multipole.txt",
    "path_to/A1_D_A2(1F)/multipole.txt",
    "path_to/A1_D_A2(2F)/multipole.txt",
    "path_to/A1_D_A2(3F)/multipole.txt",
    "path_to/A1_D(1F)_A2/multipole.txt",
    "path_to/A1_D(2F)_A2/multipole.txt",
    "path_to/A1_D(3F)_A2/multipole.txt",
]

# mapping=Zuordnung
mapping_files = [
    "path_to/A1_D_A2/mapping_basin.txt",
    "path_to/A1-π-D-π-A2/mapping_basin.txt",
    "path_to/A1_D_A2(1F)/mapping_basin.txt",
    "path_to/A1_D_A2(2F)/mapping_basin.txt",
    "path_to/A1_D_A2(3F)/mapping_basin.txt",
    "path_to/A1_D(1F)_A2/mapping_basin.txtt",
    "path_to/A1_D(2F)_A2/mapping_basin.txt",
    "path_to/A1_D(3F)_A2/mapping_basin.txt",
]

xyz_files = [
    "path_to/A1_D_A2/geo.xyz",
    "path_to/A1-π-D-π-A2/geo.xyz",
    "path_to/A1_D_A2(1F)/geo.xyz",
    "path_to/A1_D_A2(2F)/geo.xyz",
    "path_to/A1_D_A2(3F)/geo.xyz",
    "path_to/A1_D(1F)_A2/geo.xyz",
    "path_to/A1_D(2F)_A2/geo.xyz",
    "path_to/A1_D(3F)_A2/geo.xyz",
]

charge_fragments = [
    [1, 2, 3, 4, 5, 6],                                 # A1
    [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],  # D
    [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]    # A2
]
fragment_labels = ['A1', 'D', 'A2']

system_labels = [
    "A1-D-A2",
    "A1-π-D-π-A2",
    "A1-D-A2(1F)",
    "A1-D-A2(2F)",
    "A1-D-A2(3F)",
    "A1-D(1F)-A2",
    "A1-D(2F)-A2",
    "A1-D(3F)-A2",
]

fragments_per_system = {label: [frag[:] for frag in charge_fragments] for label in system_labels}
fragments_per_system["A1-π-D-π-A2"] = [
    [23, 24, 25, 26, 27, 28],  # A1
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
    [29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
]

# === Run ===
if __name__ == "__main__":
    plot_fragment_sums_multiple(
        multipole_files,
        mapping_files,
        xyz_files,
        fragments_per_system,
        fragment_labels,
        system_labels,
        correction="shift"
    )




