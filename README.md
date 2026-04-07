# AI-Augmented Transdiagnostic Pleiotropy & Discordance Explorer for PGC Psychiatric GWAS

Code license: MIT. Data license: CC-BY 4.0 via the OpenMed_AI PGC collection on Hugging Face, derived from Psychiatric Genomics Consortium (PGC) summary statistics. Every downstream use of this repository must cite OpenMed_AI, the Psychiatric Genomics Consortium, and the original PGC publication/config used for each trait.

## Abstract

This repository provides a fast, reproducible, open-science workflow for discovering cross-disorder psychiatric GWAS signals in the new OpenMed_AI PGC collection on Hugging Face. The analysis focuses on six major psychiatric disorders, harmonizes heterogeneous summary-stat schemas automatically, ranks pleiotropic loci shared across disorders, flags discordant loci with opposite effect directions after allele alignment, and produces publication-ready figures, tables, and an AI-ready interpretation prompt for biological hypothesis generation and drug-repurposing discussion.

The pipeline is designed to run on a free Google Colab session or a standard laptop in under one hour by avoiding full-table materialization, using `datasets` for one-line remote loading, and reducing each trait to genome-wide-significant loci plus a tractable top-hit panel.

## Why This Repo Is Novel

To the best of our knowledge, this is the first reproducible open-source cross-disorder analysis workflow built specifically for the OpenMed_AI PGC Hugging Face collection released on April 7, 2026. The collection exposes approximately one billion cleaned Parquet GWAS summary-stat rows across 52 PGC studies, but no comparable public transdiagnostic pleiotropy and discordance explorer was available at release time.

This repository is intentionally opinionated:

- It uses the new one-line Hugging Face access pattern instead of bespoke file wrangling.
- It validates live configs before analysis rather than hard-coding stale dataset names.
- It keeps attribution front and center in every file.
- It ends with an explicit LLM interpretation layer so the biological follow-up is reproducible, reviewable, and easy to extend.

## Core Questions

1. Which SNPs are genome-wide significant in multiple psychiatric disorders?
2. Which loci show effect-direction discordance after allele harmonization?
3. Which disorder pairs share the densest burden of significant loci?
4. Do signed Z-score patterns recover interpretable latent transdiagnostic factors?
5. Which genes and pathways should be prioritized for mechanistic follow-up or drug repurposing?

## Verified Default Data Inputs

The notebook inspects available Hugging Face configs at runtime with `datasets.get_dataset_config_names(...)` and then uses one representative config per disorder. During live validation on April 7, 2026, these were the latest usable defaults:

| Disorder | Dataset ID | Preferred config | Runtime default used here | Reason |
|---|---|---:|---:|---|
| Schizophrenia | `OpenMed/pgc-schizophrenia` | `scz2022` | `scz2022` | Latest flagship schizophrenia release exposed in the collection |
| Bipolar disorder | `OpenMed/pgc-bipolar` | `bip2024` | `bip2024` | Latest bipolar release exposed in the collection |
| Major depressive disorder | `OpenMed/pgc-mdd` | `mdd2025` | `mdd2025` | Latest MDD release exposed in the collection |
| ADHD | `OpenMed/pgc-adhd` | `adhd2022` | `adhd2022` | Latest ADHD release exposed in the collection |
| Autism spectrum disorder | `OpenMed/pgc-autism` | `asd2023` or latest | `asd2019` | `asd2019` was the latest available Autism config exposed on April 7, 2026 |
| PTSD | `OpenMed/pgc-ptsd` | `ptsd2023` or latest | `ptsd2019` | `ptsd2024` exists in the collection, but its current parquet conversion resolves to VCF header metadata rather than analyzable GWAS rows; this repo falls back to the newest usable PTSD table |

Important note: the PTSD fallback is deliberate. If OpenMed_AI later fixes the `ptsd2024` parquet schema, the notebook will show that during config inspection and you can switch back with a one-line edit.

## What The Pipeline Produces

- Harmonized per-trait hit tables with unified SNP, locus, allele, beta, SE, P, Z, and sample-size fields
- A pleiotropy table ranking loci by the number of disorders with `P < 5e-8`
- A discordance table ranking loci with opposite aligned effects across disorder pairs
- A signed Z-score correlation matrix across disorders
- Optional latent-factor summaries using PCA and non-negative matrix factorization on `abs(Z)`
- An UpSet plot for overlap structure
- A shared-loci heatmap
- A multi-trait Manhattan-style plot across sentinel loci
- A disorder network graph weighted by shared significant loci
- CSV exports for the top 50 pleiotropic loci and top 20 discordant loci
- A markdown report and an LLM-ready prompt scaffold for pathway, mechanism, and drug-repurposing interpretation

## Repository Layout

```text
.
├── README.md
├── requirements.txt
├── LICENSE
├── CITATION.cff
├── .gitignore
├── notebooks/
│   └── PGC_Pleiotropy_Discordance_Explorer.ipynb
└── src/
    └── harmonize.py
```

## Installation

Python 3.10+ is recommended.

```bash
git clone https://github.com/YOUR-USER/pgc-pleiotropy-discordance-explorer.git
cd pgc-pleiotropy-discordance-explorer

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Quick Start

### Option 1: Jupyter or Colab

```bash
jupyter lab
```

Open [notebooks/PGC_Pleiotropy_Discordance_Explorer.ipynb](notebooks/PGC_Pleiotropy_Discordance_Explorer.ipynb) and run all cells.

### Option 2: Headless execution

```bash
jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  notebooks/PGC_Pleiotropy_Discordance_Explorer.ipynb
```

## Runtime Expectations

- Config inspection: 1 to 3 minutes
- Remote hit extraction across six disorders: 10 to 40 minutes depending on network speed and backend
- Gene annotation and figure generation: 5 to 15 minutes
- Total end-to-end runtime target: under 60 minutes

Internet access is required because the notebook reads OpenMed_AI datasets directly from Hugging Face and downloads a compact public gene-coordinate reference for nearest-gene annotation.

## Analysis Strategy

1. Inspect available configs per trait with `datasets`.
2. Resolve the chosen config, applying a PTSD fallback if the newest config is malformed.
3. Harmonize schema differences such as `SNP` vs `rsid`, `BP` vs `base_pair_location`, and `OR` vs `beta`.
4. Convert `OR` to `log(OR)` when needed so effect directions can be compared across traits.
5. Retain genome-wide-significant variants plus a top-hit panel per disorder.
6. Align alleles across traits before computing discordance.
7. Count cross-trait significance to define a pleiotropy score.
8. Pivot to a signed Z-score matrix and compute pairwise correlations.
9. Annotate loci with nearest genes.
10. Export tables, figures, a markdown report, and an LLM-ready interpretation prompt.

## Example Outputs And Expected Biological Insight

This repository does not ship precomputed claims because upstream dataset cards can change and this analysis should remain reproducible. Instead, the notebook generates the following interpretable outputs on demand:

- A prioritized pleiotropy table showing loci shared across psychosis, mood, neurodevelopmental, and trauma-related traits
- A discordance panel highlighting loci whose risk-increasing allele differs across disorders after harmonization
- A correlation heatmap that often separates psychosis-related and internalizing-related axes while still exposing shared burden
- A network view that makes high-overlap disorder pairs visually obvious
- An AI interpretation scaffold that turns the exported loci into a structured hypothesis-generation and repurposing workflow

Illustrative biological themes that the final LLM interpretation step should test, rather than assume, include synaptic signaling, calcium-channel biology, neuronal development, stress-axis regulation, immune signaling, and transcriptional regulation in brain-relevant tissues.

## Output Files

After a successful run, the notebook writes:

- `results/tables/trait_hit_counts.csv`
- `results/tables/pleiotropic_loci_top50.csv`
- `results/tables/discordant_loci_top20.csv`
- `results/tables/cross_trait_variant_summary.csv`
- `results/tables/zscore_matrix.csv`
- `results/figures/upset_overlap.png`
- `results/figures/shared_loci_heatmap.png`
- `results/figures/multi_trait_manhattan.png`
- `results/figures/disorder_network.png`
- `results/figures/zscore_correlation_heatmap.png`
- `results/analysis_report.md`
- `results/llm_interpretation_prompt.md`

## Reproducibility Notes

- The notebook prints the selected configs at runtime.
- The code caches only reduced trait-level hit tables, not the full billion-row collection.
- Every random process uses a fixed seed where applicable.
- The notebook preserves the original `_source_file` column when available.
- Trait-level extraction is intentionally conservative when upstream schema is malformed.

## Attribution And Required Citations

This repository uses summary statistics from the OpenMed_AI PGC collection on Hugging Face. Those data are derived from Psychiatric Genomics Consortium studies and remain governed by the original study terms and the collection license.

You must cite all of the following when using this repository:

1. This code repository
2. OpenMed_AI Hugging Face dataset cards for each dataset used
3. The Psychiatric Genomics Consortium
4. The original PGC publication for each config included in the analysis

### OpenMed_AI dataset entry points used here

- `OpenMed/pgc-schizophrenia`
- `OpenMed/pgc-bipolar`
- `OpenMed/pgc-mdd`
- `OpenMed/pgc-adhd`
- `OpenMed/pgc-autism`
- `OpenMed/pgc-ptsd`

### Original PGC-linked configs used by default

- `scz2022` for schizophrenia
- `bip2024` for bipolar disorder
- `mdd2025` for major depressive disorder
- `adhd2022` for ADHD
- `asd2019` for autism spectrum disorder
- `ptsd2019` for PTSD, with `ptsd2024` monitored as a future upgrade path

### Suggested acknowledgment

> This work uses the OpenMed_AI PGC collection on Hugging Face, derived from Psychiatric Genomics Consortium summary statistics. We cite OpenMed_AI, the PGC, and the original PGC publication for each dataset configuration analyzed.

## Citation Format

Plain text:

> AI-Augmented Transdiagnostic Pleiotropy & Discordance Explorer for PGC Psychiatric GWAS. 2026. MIT-licensed analysis code for OpenMed_AI / PGC psychiatric GWAS summary statistics. Data accessed from the OpenMed_AI PGC collection on Hugging Face under CC-BY 4.0; original PGC studies cited separately.

BibTeX:

```bibtex
@software{pgc_pleiotropy_discordance_explorer_2026,
  title = {AI-Augmented Transdiagnostic Pleiotropy \& Discordance Explorer for PGC Psychiatric GWAS},
  year = {2026},
  license = {MIT},
  note = {Analysis code for the OpenMed\_AI PGC collection on Hugging Face. Data license: CC-BY 4.0. Cite OpenMed\_AI, PGC, and the underlying PGC studies.}
}
```

The machine-readable citation metadata lives in [CITATION.cff](CITATION.cff).

## Data And Code Licensing

- Code in this repository: MIT
- Data accessed by this repository: CC-BY 4.0 via OpenMed_AI dataset cards and underlying PGC terms

This repository does not relicense the underlying data. The code is MIT; the data are not.

## Future Work

- Switch PTSD back to `ptsd2024` after upstream parquet harmonization is fixed
- Add ancestry-stratified analyses and explicit European-only filters where metadata permit
- Add LD clumping and locus collapsing to reduce redundant sentinel hits
- Add colocalization against brain eQTL and pQTL resources
- Add pathway enrichment against MSigDB, Reactome, and drug-target knowledge bases
- Add Mendelian randomization and cross-trait fine-mapping follow-up
- Add interactive dashboards for locus browsing and disorder-pair comparison

## Ethics And Interpretation

These analyses operate on summary statistics, not individual-level data, but psychiatric genetics still warrants cautious interpretation. Pleiotropy does not imply a shared disease mechanism by itself, and AI-generated biological narratives should be treated as hypothesis scaffolds, not conclusions.

## License

See [LICENSE](LICENSE) for the MIT code license. See the OpenMed_AI dataset cards and PGC source publications for the governing data terms.
