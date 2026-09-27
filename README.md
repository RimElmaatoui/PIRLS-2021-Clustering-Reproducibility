# PIRLS 2021 clustering reproducibility code

This repository contains the experimental code accompanying the manuscript:

> **Exploring Model-Dependent Learning Profiles in PIRLS 2021: A Comparative Study of Classical and Deep Clustering Methods**

Authors: Rim Elmaataoui, Soukaina Elharouri, Soufiane Lyaqini, and Imad Hafidi.

The study compares classical clustering, PCA followed by k-means, and deep clustering on PIRLS 2021 data. It also evaluates common-input-space performance, external educational associations, initialization stability, missing-data sensitivity, hyperparameter sensitivity, cross-country robustness, and computational efficiency.

## Repository structure

```text
PIRLS-2021-Clustering-Reproducibility/
|-- Data/
|   |-- Main_Revision_Imputation_Robuste.ipynb
|   |-- robust_primary_outputs.py
|   |-- robust_seed_stability.py
|   |-- robust_multiseed_external.py
|   |-- second_comment_external_validation.py
|   |-- Comparison_PCA_UMAP_KMeans.py
|   |-- hyperparameter_ablation_robust.py
|   |-- cross_country_validation_core.py
|   `-- computational_efficiency_benchmark.py
|-- figures/                 # Generated figures
|-- results/                 # Aggregate results reported with the manuscript
|-- .gitignore
|-- CITATION.cff
|-- DATA_AVAILABILITY.md
|-- README.md
`-- requirements.txt
```

## Data availability

The PIRLS 2021 source files are not redistributed in this repository. They must be obtained from the official PIRLS/IEA data portal and used in accordance with its conditions. See [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md) for the official source and dataset DOI, [`Data/README.md`](Data/README.md) for placement instructions, and [`Data/data_manifest.csv`](Data/data_manifest.csv) for the exact filenames, sizes, and SHA-256 checksums used in the analyses.

Do not commit raw PIRLS files, student-level labels, identifiers, latent arrays, or completed individual-level matrices. The supplied `.gitignore` excludes these materials.

## Environment

The reported analyses used Python 3.11.6. Install the frozen core dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

The primary neural experiments can use CUDA when it is available. The computational-efficiency benchmark deliberately runs every method on the CPU to provide a common-device comparison.

## Reproduction workflow

Place the required PIRLS files directly in `Data/`, then run the following steps from the repository root.

1. Open `Data/Main_Revision_Imputation_Robuste.ipynb` and execute all cells. This produces the type-aware primary preprocessing, iterative-imputation comparisons, and the primary method labels.
2. Generate the primary common-space metrics, educational profiles, statistical tests, and figures:

   ```bash
   python Data/robust_primary_outputs.py
   ```

3. Evaluate DEC and IDEC over seeds 0, 7, 42, 123, and 2024:

   ```bash
   python Data/robust_seed_stability.py
   ```

4. Pool the reading-achievement evidence across the five plausible values:

   ```bash
   python Data/robust_multiseed_external.py
   ```

5. Run the matched PCA+k-means baseline and optional UMAP+k-means analysis:

   ```bash
   python Data/Comparison_PCA_UMAP_KMeans.py
   ```

6. Run the hyperparameter screening ablation:

   ```bash
   python Data/hyperparameter_ablation_robust.py
   ```

7. Run the harmonized Morocco-France-Singapore-South Africa analysis:

   ```bash
   python Data/cross_country_validation_core.py
   ```

8. Run the common-CPU timing, peak-memory, and scalability benchmark:

   ```bash
   python Data/computational_efficiency_benchmark.py
   ```

The script `second_comment_external_validation.py` reproduces the focused external-validation plots and tables developed in response to the reviewer. Its substantive results are also incorporated into the primary-output workflow.

## Randomness management

- Primary reported seed: `42`.
- Stability seeds: `0`, `7`, `42`, `123`, and `2024`.
- Python, NumPy, PyTorch, k-means/GMM, data-loader, metric-subsampling, PCA, and t-SNE randomness is explicitly controlled where applicable.
- K-means uses `n_init=20`.

Exact runtimes and small floating-point differences can vary across operating systems, BLAS implementations, CPUs, and GPUs.

## Interpretation note

A high latent-space internal index does not, by itself, establish naturally separated learner types. DEC and IDEC optimize the representation jointly with the partition, so the full evaluation must also consider common-input-space metrics, assignment stability, imputation sensitivity, held-out educational variables, and conventional dimensionality-reduction baselines.

## Contact

Questions about the code may be addressed to the corresponding author, Rim Elmaataoui: `rim.elmaataoui@usms.ac.ma`.
