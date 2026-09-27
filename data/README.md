# PIRLS 2021 data

The source data are not included. Download PIRLS 2021 Edition 2 directly from the official IEA repository:

<https://www.iea.nl/data-tools/repository-direct/1970?cycle=1036>

Dataset DOI: <https://doi.org/10.58150/PIRLS_2021_edition_2_including_Log-file_data>

After accepting the IEA conditions, extract and place the following files in this directory.

## Moroccan primary analysis

- `ASPMARR5.csv`: student questionnaire variables used for clustering.
- `ATGMARR5.csv`: teacher questionnaire variables used for clustering.
- `ACGMARR5.csv`: school questionnaire variables used for clustering.
- `ASGMARR5.sav`: student achievement/context file used for survey weights, five overall-reading plausible values, and held-out educational validation.

The five overall-reading plausible values (`ASRREA01` to `ASRREA05`), Home Resources for Learning, and home-language frequency are excluded from cluster construction and linked to cluster labels only during external-variable validation.

## Cross-country analysis

For each country code `MAR`, `FRA`, `SGP`, and `ZAF`, provide:

- `ASP<CODE>R5.sav`
- `ATG<CODE>R5.sav`
- `ACG<CODE>R5.sav`
- `ASG<CODE>R5.sav`

For example, the French files are `ASPFRAR5.sav`, `ATGFRAR5.sav`, `ACGFRAR5.sav`, and `ASGFRAR5.sav`.

The scripts preserve the original PIRLS variable names. Do not rename columns.

## Privacy and licensing

Do not upload the source data or generated student-level files to a public repository. In particular, keep files containing `IDSTUD`, `IDSCHOOL`, individual cluster assignments, completed feature matrices, or latent coordinates outside version control.
