list_of_packages <- c("dplyr", "ggplot2", "patchwork", "arrow", "RColorBrewer", "ggrastr")
for (package in list_of_packages) {
    suppressPackageStartupMessages(
        suppressWarnings(
            library(
                package,
                character.only = TRUE,
                quietly = TRUE,
                warn.conflicts = FALSE
            )
        )
    )
}

# Get the current working directory and find Git root
find_git_root <- function() {
    # Get current working directory
    cwd <- getwd()

    # Check if current directory has .git
    if (dir.exists(file.path(cwd, ".git"))) {
        return(cwd)
    }

    # If not, search parent directories
    current_path <- cwd
    while (dirname(current_path) != current_path) {  # While not at root
        parent_path <- dirname(current_path)
        if (dir.exists(file.path(parent_path, ".git"))) {
            return(parent_path)
        }
        current_path <- parent_path
    }

    # If no Git root found, stop with error
    stop("No Git root directory found.")
}

# Find the Git root directory
root_dir <- find_git_root()
source(file.path(root_dir, "utils", "r_plot_funcs.r"))
source(file.path(root_dir, "utils", "r_plot_themes.r"))

figures_path <- file.path(root_dir, "6.technical_platemap_analysis/figures/pca_vs_z_dim")
if (!dir.exists(figures_path)) {
  dir.create(figures_path, recursive = TRUE)
}

pca_results_dir <- file.path(root_dir, "1.EDA/results/pca")

# Single cell's centroid Z position (depth within the organoid/image stack).
# Only present for ZEDProfiler-derived features (Metadata_Location_*); the
# SAM-Med3D and MorphEm/CHAMMI feature sets carry no centroid location at
# all, so PC-vs-Z is not computable for those normalization variants -- the
# missing-column check below skips them and logs why.
z_col <- "Metadata_Location_Cell_CenterZ"

# Scatter one PC against the Z dimension, annotated with the Pearson
# correlation between the two.
plot_pc_vs_z <- function(df, pc_col, title) {
    r <- cor(df[[pc_col]], df[[z_col]], use = "pairwise.complete.obs")

    (
        ggplot(df, aes(x = .data[[z_col]], y = .data[[pc_col]]))
        + rasterise(geom_point(alpha = 0.15, size = 0.4), dpi = 150)
        + labs(
            title = title,
            subtitle = sprintf("Pearson r = %.3f", r),
            x = "Cell center Z",
            y = pc_col
        )
        + theme_manuscript(base_size = 10)
    )
}

# One plate per patient: build a combined PC0 + PC1 vs Z page for each
# patient separately, keyed by patient.
build_patient_z_pages <- function(df, patient_col, title_prefix) {
    patient_plots <- list()
    for (patient in sort(unique(df[[patient_col]]))) {
        patient_df <- df[df[[patient_col]] == patient, ]
        patient_title_prefix <- paste0(title_prefix, " (", patient, ")")
        patient_plots[[patient]] <- (
            plot_pc_vs_z(patient_df, "PC0", paste0(patient_title_prefix, ": PC0 vs Z"))
            + plot_pc_vs_z(patient_df, "PC1", paste0(patient_title_prefix, ": PC1 vs Z"))
        )
    }
    patient_plots
}

# Restricted to the single-cell-granularity normalization variants (one row
# per cell): excludes organoid_norm and sammed_organoid_norm, whose
# "feature_selected_profiles" are organoid-level (one row per organoid, far
# fewer rows) rather than single-cell.
normalized_profiles <- c(
    "sc_norm",
    "sammed_sc_norm",
    "nucleocentric_morphem_norm",
    "sammed_nucleocentric_norm"
)

z_plots <- list()
z_plots_per_patient <- list()

for (norm_profile in normalized_profiles) {
    file_name <- paste0("3D_1.feature_selected_profiles_", norm_profile, "_fs_profiles_embeddings.parquet")
    file_path <- file.path(pca_results_dir, file_name)

    if (!file.exists(file_path)) {
        cat("Missing file, skipping:", file_path, "\n")
        next
    }

    df <- arrow::read_parquet(file_path)
    if (!all(c("PC0", "PC1", z_col, "Metadata_Biology_PatientTumor") %in% colnames(df))) {
        cat("Required columns not found, skipping:", file_path, "\n")
        next
    }

    title_prefix <- paste0("3D ", normalization_variant_labels[[norm_profile]], " (Single Cell, FS)")
    combined_plot <- (
        plot_pc_vs_z(df, "PC0", paste0(title_prefix, ": PC0 vs Z"))
        + plot_pc_vs_z(df, "PC1", paste0(title_prefix, ": PC1 vs Z"))
    )

    page_key <- paste0("3D_1.feature_selected_profiles_", norm_profile, "_fs_profiles")
    z_plots[[page_key]] <- combined_plot

    patient_plots <- build_patient_z_pages(df, "Metadata_Biology_PatientTumor", title_prefix)
    for (patient in names(patient_plots)) {
        z_plots_per_patient[[paste0(page_key, "_", patient)]] <- patient_plots[[patient]]
    }
}

save_umap_pca_plots_pdf(
    z_plots, file.path(figures_path, "PCs_vs_z_dim.pdf"), width = 12, height = 5
)
save_umap_pca_plots_pdf(
    z_plots_per_patient, file.path(figures_path, "PCs_vs_z_dim_per_patient.pdf"), width = 12, height = 5
)
