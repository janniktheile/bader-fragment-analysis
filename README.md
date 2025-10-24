# Bader and Fragment Analysis Scripts

[![DOI](https://doi.org/10.5281/zenodo.17427520)

This repository contains two Python scripts developed for analyzing output data from **Gaussian 16** and **Multiwfn**.  
They facilitate the extraction and processing of **Bader partial charges** and **fragment orbital energy contributions** for a set of conjugated oligomer systems.

---

## 🧩 Overview

- `fragment_contri_energy_levels.py`:  
  Processes Gaussian 16 output files to extract fragment orbital contributions and energy levels.

- `fragments_net_charge_bader.py`:  
  Processes Bader charge analysis results to determine net charge distribution per molecular fragment.

Both scripts were used to generate the data presented in the associated publication and Zenodo dataset.

---

## ⚙️ Requirements

- Python ≥ 3.8  
- NumPy  
- Pandas  
- Matplotlib  
