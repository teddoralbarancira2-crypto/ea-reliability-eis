# Reliability of process activation energies from temperature-resolved impedance

Analysis code for the paper *"Structured model misfit limits the reliability of process activation energies from temperature-resolved impedance of aging lithium-ion cells"* (Barancira Ted Doral and Jonghoon Kim, submitted).

The code fits an Arrhenius-coupled equivalent circuit jointly to impedance spectra measured at several temperatures, tests how stable each process activation energy is when the temperature window changes, and computes the misfit-to-arc ratio used as a one-fit screening test. It also contains the synthetic tests, the temperature-compensation benchmark and the scripts that draw every figure and table of the paper.

## Data

The measured spectra are **not** included.

| | Cells | Availability |
|---|---|---|
| Dataset A | two commercial 21700 cells, 11 aging checkpoints, 0 to 55 °C, three voltages | on reasonable request, with the permission of the data owner |
| Dataset B | two fresh commercial 21700 cells, 0 to 45 °C, six states of charge | on reasonable request |

The differential voltage analysis uses the open-circuit potentials of Chen et al. (J. Electrochem. Soc. 167 (2020) 080534) as proxies. They are distributed with PyBaMM (parameter set `Chen2020`); place the two files as `arrhenius_labels/dataset_a_fits/dvdq/ref/graphite_LGM50_ocp_Chen2020.csv` and `nmc_LGM50_ocp_Chen2020.csv`.

## Setup

Python 3.11 with the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

Point the code to the data with environment variables (or edit `config.py`):

| Variable | Meaning | Default |
|---|---|---|
| `EA_DATA_A_RAW` | dataset A raw exports (`OCV-EIS test/` and `Capacity test/`) | `data/dataset_a` |
| `EA_DATA_B_DIR` | dataset B `.mat` files | `data/dataset_b` |
| `EA_DATA_A_PREFIX` | file-name prefix of the dataset A raw exports (before `Cell1_...`) | empty |
| `EA_WORK` | intermediate files (dataset A impedance cube) | `work/` |

Every script writes its results (`.json`) next to itself, and the scripts in `paper/` read them from there.

## Run order

**Dataset A (aging cells)**, `S` = voltage step (3 = 4.06 V, 10 = 3.73 V, 17 = 3.41 V)

1. `dataset_a/extract_cube.py`: raw exports to the impedance cube `work/cube.npz`
2. `arrhenius_labels/dataset_a_fits/joint_hf.py B S`: joint fit at every checkpoint (main model, variant B). Variants `A`, `C`, `D`, `E` are the robustness checks of Section 4.4 (free R0, 2 kHz limit, temperature-dependent and asymmetric slow-arc shape)
3. `arrhenius_labels/dataset_a_fits/window_scan.py S`: refits over the temperature windows (window span)
4. `arrhenius_labels/dataset_a_fits/joint_age.py`, `jk_age.py`, `checks.py`: the five checks of the slow label (earlier 2 kHz configuration)
5. `arrhenius_labels/dataset_a_fits/synth_leak.py`, `fastband.py`, `drt_mid.py`: fast-band artifact tests and distribution of relaxation times
6. `arrhenius_labels/dataset_a_fits/dvdq/dvdq_load.py`, `dva_fit.py`: differential voltage analysis

**Dataset B (fresh cells) and synthetic tests**

1. `arrhenius_labels/drt_all.py`: loader and distribution of relaxation times
2. `arrhenius_labels/joint_ms.py 2 N`: multistart joint fit with K = 2 processes and N random starts
3. `arrhenius_labels/robust.py`, `arrhenius_labels/ident/curvature.py`, `cold_split.py`: jackknife, sliding windows, cold-end tests
4. `arrhenius_labels/ident/synth_ident.py`, `synth_transplant.py`: synthetic tests of Section 4.2

**Temperature-compensation benchmark**

1. `arrhenius_labels/ident/consequence2.py`: fixed aging state
2. `arrhenius_labels/ident/aging_transfer.py`: calibrated once and carried across aging

**Paper numbers, tables and figures**

| Script | Produces |
|---|---|
| `paper/fig_check.py` | the 44 cases of Section 4.3 (`fig3_rows.json`) and the numbers behind the figures |
| `paper/ci_dataset_b.py` | Jacobian 95 % intervals of dataset B |
| `paper/boot_ci_v13.py` | bootstrap intervals of dataset B, 1000 draws (Section 4.1) |
| `paper/analysis_v13.py` | Kramers–Kronig checks, residual map (Fig. 3c), calibrate-once breakdown, fixed-threshold folds |
| `paper/holdout_v10.py` | out-of-sample test of the misfit-to-arc threshold (Table 1) |
| `paper/tables_2_3.py` | Tables 2 and 3 |
| `paper/make_fig1_v11.py` | Fig. 1 |
| `paper/make_figs_v11.py` | Figs. 2 to 5 |

Figures are written to `figures/` at the repository root.

## Notes

- The random starts use fixed seeds, so reruns give the same fits on the same machine; small differences across platforms are possible because the problems are nonconvex.
- The misfit-to-arc threshold of 0.015 depends on the residual definition, the frequency range and the model structure. Re-establish it with a window scan before using it on another cell type.

## License

MIT, see `LICENSE`.
