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

figures_path <- file.path(root_dir, "6.technical_platemap_analysis/figures/counts_per_well")
if (!dir.exists(figures_path)) {
  dir.create(figures_path, recursive = TRUE)
}

profiles_2D_dir <- file.path(root_dir, "data/profiles_2D/all_patients")
profiles_3D_dir <- file.path(root_dir, "data/profiles_3D/all_patients/0.normalized_profiles")

# NF0037_T1_CQ1 is a separate analysis and is excluded everywhere in this EDA
# (matches 1.EDA/scripts/2.generate_pca.py).
exclude_patient <- "NF0037_T1_CQ1"

# platetools::raw_map expects well IDs like "A01" (zero-padded column number);
# the profiles store wells as "A1" (matches platemap_figure.r).
add_leading_zero <- function(well) {
    str_replace(well, "([A-Za-z])([1-9])$", "\\10\\2")
}

# Count rows (= objects: single cells or organoids) per well and draw it as
# a 96-well platemap.
plot_count_platemap <- function(df, well_col, title) {
    well_counts <- df %>%
        mutate(.well = add_leading_zero(.data[[well_col]])) %>%
        count(.well, name = "n")

    (
        raw_map(data = well_counts$n, well = well_counts$.well, plate = 96)
        + ggtitle(title)
        + labs(fill = "Count")
        + theme_manuscript(base_size = 10)
        + theme(axis.text = element_text(size = 8))
    )
}

# Single cell count + organoid count platemaps, side by side.
build_combined_count_plot <- function(sc_df, organoid_df, well_col, title_prefix) {
    (
        plot_count_platemap(sc_df, well_col, paste0(title_prefix, ": Single Cell Count"))
        + plot_count_platemap(organoid_df, well_col, paste0(title_prefix, ": Organoid Count"))
    )
}

# One plate per patient: build a combined single-cell + organoid count
# platemap for each patient separately, keyed by patient.
build_patient_count_pages <- function(sc_df, organoid_df, well_col, patient_col, title_prefix) {
    patients <- sort(unique(c(sc_df[[patient_col]], organoid_df[[patient_col]])))
    patient_plots <- list()
    for (patient in patients) {
        sc_patient_df <- sc_df[sc_df[[patient_col]] == patient, ]
        organoid_patient_df <- organoid_df[organoid_df[[patient_col]] == patient, ]
        patient_title_prefix <- paste0(title_prefix, " (", patient, ")")
        patient_plots[[patient]] <- build_combined_count_plot(
            sc_patient_df, organoid_patient_df, well_col, patient_title_prefix
        )
    }
    patient_plots
}

count_plots <- list()
count_plots_per_patient <- list()

slice_specs <- list(
    list(dir_name = "max_projection", title_label = "2D MIP"),
    list(dir_name = "middle_slice", title_label = "2D Middle Slice"),
    list(dir_name = "middle_n_slice", title_label = "2D Middle N Slice")
)

for (slice in slice_specs) {
    sc_path <- file.path(profiles_2D_dir, slice$dir_name, "sc_profiles.parquet")
    organoid_path <- file.path(profiles_2D_dir, slice$dir_name, "organoid_profiles.parquet")

    if (!file.exists(sc_path) || !file.exists(organoid_path)) {
        cat("Missing file, skipping:", slice$dir_name, "\n")
        next
    }

    sc_df <- arrow::read_parquet(sc_path, col_select = c("Metadata_Well", "Metadata_patient_tumor"))
    sc_df <- sc_df[sc_df$Metadata_patient_tumor != exclude_patient, ]
    organoid_df <- arrow::read_parquet(organoid_path, col_select = c("Metadata_Well", "Metadata_patient_tumor"))
    organoid_df <- organoid_df[organoid_df$Metadata_patient_tumor != exclude_patient, ]

    page_key <- slice$dir_name
    count_plots[[page_key]] <- build_combined_count_plot(sc_df, organoid_df, "Metadata_Well", slice$title_label)

    patient_plots <- build_patient_count_pages(
        sc_df, organoid_df, "Metadata_Well", "Metadata_patient_tumor", slice$title_label
    )
    for (patient in names(patient_plots)) {
        count_plots_per_patient[[paste0(page_key, "_", patient)]] <- patient_plots[[patient]]
    }
}

sc_path_3d <- file.path(profiles_3D_dir, "sc_norm_norm_profile.parquet")
organoid_path_3d <- file.path(profiles_3D_dir, "organoid_norm_norm_profile.parquet")

if (file.exists(sc_path_3d) && file.exists(organoid_path_3d)) {
    sc_df_3d <- arrow::read_parquet(
        sc_path_3d, col_select = c("Metadata_Experiment_Well", "Metadata_Biology_PatientTumor")
    )
    sc_df_3d <- sc_df_3d[sc_df_3d$Metadata_Biology_PatientTumor != exclude_patient, ]
    organoid_df_3d <- arrow::read_parquet(
        organoid_path_3d, col_select = c("Metadata_Experiment_Well", "Metadata_Biology_PatientTumor")
    )
    organoid_df_3d <- organoid_df_3d[organoid_df_3d$Metadata_Biology_PatientTumor != exclude_patient, ]

    page_key <- "3D"
    count_plots[[page_key]] <- build_combined_count_plot(
        sc_df_3d, organoid_df_3d, "Metadata_Experiment_Well", "3D"
    )

    patient_plots <- build_patient_count_pages(
        sc_df_3d, organoid_df_3d, "Metadata_Experiment_Well", "Metadata_Biology_PatientTumor", "3D"
    )
    for (patient in names(patient_plots)) {
        count_plots_per_patient[[paste0(page_key, "_", patient)]] <- patient_plots[[patient]]
    }
} else {
    cat("Missing 3D source file(s), skipping 3D counts\n")
}

save_umap_pca_plots_pdf(
    count_plots, file.path(figures_path, "Counts_per_well.pdf"), width = 12, height = 5
)
save_umap_pca_plots_pdf(
    count_plots_per_patient, file.path(figures_path, "Counts_per_well_per_patient.pdf"), width = 12, height = 5
)
