# NF1_organoid_profile_analysis
This repo contains analysis code of profiles generated in multiple image-based profiling repos.

- 3D image-based profiles of NF1 organoids were generated from: [NF1 3D Organoid profiling pipeline](https://github.com/WayScience/NF1_3D_organoid_profiling_pipeline)
- 2D image-based profiles of NF1 organoids were generated from: [NF1 2D organoid profiling pipeline](https://github.com/WayScience/NF1_2D_organoid_profiling_pipeline)
- Cell types will be predicted using: [Cell type prediction in NF1 organoids](https://github.com/WayScience/NF1_organoid_cell_segmentation)

## Repo modules
- `0.download_data`: Download image-based profiles from the internet (not yet available)
- `1.EDA`: Exploratory data analysis of image-based profiles

## Computational environment
Notebooks in this repo are split between Python and R, each managed by a separate environment.

### Python (uv)
Python notebooks use a `uv`-managed virtual environment defined in `pyproject.toml`/`uv.lock`.

```bash
source uv_setup.sh
```

This creates `.venv` and registers a `python3` Jupyter kernel.
Select this kernel when running the Python notebooks.

### R (mamba)
R notebooks use a mamba/conda environment defined in `environments/r_env.yml`.

```bash
mamba env create -f environments/r_env.yml
# or, if the environment already exists:
mamba env update -f environments/r_env.yml
```

This creates the `gff_figure_env` environment but does not register a Jupyter kernel for it.
Register the kernel once, from inside the activated environment:

```bash
mamba activate gff_figure_env
Rscript -e 'IRkernel::installspec(name = "gff_figure_env", displayname = "R (gff_figure_env)")'
```

Select the `R (gff_figure_env)` kernel when running the R notebooks for visualization.
