list_of_packages <- c("dplyr", "stringr", "ggplot2", "platetools", "patchwork", "arrow", "RColorBrewer")
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

figures_path <- file.path(root_dir, "6.technical_platemap_analysis/figures/pca_per_well")
if (!dir.exists(figures_path)) {
  dir.create(figures_path, recursive = TRUE)
}

pca_results_dir <- file.path(root_dir, "1.EDA/results/pca")

# platetools::raw_map expects well IDs like "A01" (zero-padded column number);
# the PCA embeddings store wells as "A1" (matches platemap_figure.r).
add_leading_zero <- function(well) {
    str_replace(well, "([A-Za-z])([1-9])$", "\\10\\2")
}

# Mean a single PC's value per well and draw it as a 96-well platemap.
plot_pc_platemap <- function(df, well_col, pc_col, title) {
    well_means <- df %>%
        mutate(.well = add_leading_zero(.data[[well_col]])) %>%
        group_by(.well) %>%
        summarize(mean_value = mean(.data[[pc_col]], na.rm = TRUE), .groups = "drop")

    (
        raw_map(data = well_means$mean_value, well = well_means$.well, plate = 96)
        + ggtitle(title)
        + labs(fill = paste0("Mean ", pc_col))
        + theme_manuscript(base_size = 10)
        + theme(axis.text = element_text(size = 8))
    )
}

# One plate (and one patient_col value) per patient: build a combined
# PC0 + PC1 platemap for each patient separately, keyed by patient.
build_patient_platemap_pages <- function(df, well_col, patient_col, title_prefix) {
    patient_plots <- list()
    for (patient in sort(unique(df[[patient_col]]))) {
        patient_df <- df[df[[patient_col]] == patient, ]
        patient_title_prefix <- paste0(title_prefix, " (", patient, ")")
        patient_plots[[patient]] <- (
            plot_pc_platemap(patient_df, well_col, "PC0", paste0(patient_title_prefix, ": PC0"))
            + plot_pc_platemap(patient_df, well_col, "PC1", paste0(patient_title_prefix, ": PC1"))
        )
    }
    patient_plots
}

slice_specs <- list(
    list(file_prefix = "2D_max_projection", title_label = "2D MIP"),
    list(file_prefix = "2D_middle_slice", title_label = "2D Middle Slice"),
    list(file_prefix = "2D_middle_n_slice", title_label = "2D Middle N Slice")
)
profile_specs_2d <- list(
    list(suffix = "fs_profiles", label = "FS"),
    list(suffix = "agg_profiles", label = "Aggregate"),
    list(suffix = "consensus_profiles", label = "Consensus")
)
entity_specs <- list(
    list(entity = "sc", entity_label = "Single Cell"),
    list(entity = "organoid", entity_label = "Organoid")
)

platemap_plots <- list()
platemap_plots_per_patient <- list()

for (slice in slice_specs) {
    for (entity in entity_specs) {
        for (profile in profile_specs_2d) {
            file_name <- paste0(slice$file_prefix, "_", entity$entity, "_", profile$suffix, "_embeddings.parquet")
            file_path <- file.path(pca_results_dir, file_name)

            if (!file.exists(file_path)) {
                cat("Missing file, skipping:", file_path, "\n")
                next
            }

            df <- arrow::read_parquet(file_path)
            if (!all(c("PC0", "PC1", "Metadata_Well") %in% colnames(df))) {
                cat("Required columns not found, skipping:", file_path, "\n")
                next
            }

            title_prefix <- paste0(slice$title_label, " ", entity$entity_label, " ", profile$label)
            combined_plot <- (
                plot_pc_platemap(df, "Metadata_Well", "PC0", paste0(title_prefix, ": PC0"))
                + plot_pc_platemap(df, "Metadata_Well", "PC1", paste0(title_prefix, ": PC1"))
            )

            page_key <- paste0(slice$file_prefix, "_", entity$entity, "_", profile$suffix)
            platemap_plots[[page_key]] <- combined_plot

            patient_plots <- build_patient_platemap_pages(df, "Metadata_Well", "Metadata_patient_tumor", title_prefix)
            for (patient in names(patient_plots)) {
                platemap_plots_per_patient[[paste0(page_key, "_", patient)]] <- patient_plots[[patient]]
            }
        }
    }
}

normalized_profiles <- c(
    "organoid_norm",
    "sammed_organoid_norm",
    "sc_norm",
    "sammed_sc_norm",
    "nucleocentric_morphem_norm",
    "sammed_nucleocentric_norm"
)

for (norm_profile in normalized_profiles) {
    file_name <- paste0("3D_2.aggregated_profiles_", norm_profile, "_sc_agg_profiles_embeddings.parquet")
    file_path <- file.path(pca_results_dir, file_name)

    if (!file.exists(file_path)) {
        cat("Missing file, skipping:", file_path, "\n")
        next
    }

    df <- arrow::read_parquet(file_path)
    if (!all(c("PC0", "PC1", "Metadata_Experiment_Well") %in% colnames(df))) {
        cat("Required columns not found, skipping:", file_path, "\n")
        next
    }

    title_prefix <- paste0("3D ", normalization_variant_labels[[norm_profile]], " Aggregated")
    combined_plot <- (
        plot_pc_platemap(df, "Metadata_Experiment_Well", "PC0", paste0(title_prefix, ": PC0"))
        + plot_pc_platemap(df, "Metadata_Experiment_Well", "PC1", paste0(title_prefix, ": PC1"))
    )

    page_key <- paste0("3D_2.aggregated_profiles_", norm_profile, "_sc_agg_profiles")
    platemap_plots[[page_key]] <- combined_plot

    patient_plots <- build_patient_platemap_pages(df, "Metadata_Experiment_Well", "Metadata_Biology_PatientTumor", title_prefix)
    for (patient in names(patient_plots)) {
        platemap_plots_per_patient[[paste0(page_key, "_", patient)]] <- patient_plots[[patient]]
    }
}

save_umap_pca_plots_pdf(
    platemap_plots, file.path(figures_path, "PCs_per_well.pdf"), width = 12, height = 5
)
save_umap_pca_plots_pdf(
    platemap_plots_per_patient, file.path(figures_path, "PCs_per_well_per_patient.pdf"), width = 12, height = 5
)
