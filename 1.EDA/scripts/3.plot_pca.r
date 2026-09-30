list_of_packages <- c("ggplot2", "dplyr", "tidyr", "RColorBrewer")
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

figures_path <- file.path(root_dir,"1.EDA/figures/pca")
if (!dir.exists(figures_path)) {
  dir.create(figures_path, recursive = TRUE)
}

pca_results_dir <- file.path(root_dir, "1.EDA/results/pca")

slice_specs <- list(
    list(dir_name = "max_projection", file_prefix = "2D_max_projection", title_label = "2D MIP"),
    list(dir_name = "middle_slice", file_prefix = "2D_middle_slice", title_label = "2D Middle Slice"),
    list(dir_name = "middle_n_slice", file_prefix = "2D_middle_n_slice", title_label = "2D Middle N Slice")
)

profile_specs_2d <- list(
    list(suffix = "fs_profiles", label = "FS", facet = TRUE),
    list(suffix = "agg_profiles", label = "Aggregate", facet = FALSE),
    list(suffix = "consensus_profiles", label = "Consensus", facet = FALSE)
)

entity_specs <- list(
    list(entity = "sc", entity_label = "Single Cell"),
    list(entity = "organoid", entity_label = "Organoid")
)

pca_plots_2D <- list()

for (slice in slice_specs) {
    for (entity in entity_specs) {
        for (profile in profile_specs_2d) {
            file_name <- paste0(slice$file_prefix, "_", entity$entity, "_", profile$suffix, "_embeddings.parquet")
            file_path <- file.path(pca_results_dir, file_name)
            explained_variance_file_name <- paste0(slice$file_prefix, "_", entity$entity, "_", profile$suffix, "_explained_variance.parquet")
            explained_variance_file_path <- file.path(pca_results_dir, explained_variance_file_name)

            if (!file.exists(file_path) || !file.exists(explained_variance_file_path)) {
                cat("Missing file, skipping:", file_path, "\n")
                next
            }

            df <- arrow::read_parquet(file_path)
            if (!"PC1" %in% colnames(df)) {
                cat("PC1 not found, skipping:", file_path, "\n")
                next
            }
            # Harmonize 2D column names to the 3D metadata convention, as
            # done in 1.plot_umap.ipynb
            df <- df %>%
                rename(
                    Metadata_Experiment_Treatment = Metadata_treatment,
                    Metadata_Biology_PatientTumor = Metadata_patient_tumor
                )
            df$Metadata_Biology_TumorType <- tumor_type_lookup[df$Metadata_Biology_PatientTumor]

            explained_variance_df <- arrow::read_parquet(explained_variance_file_path)
            title <- paste0("All patients: ", slice$title_label, " ", entity$entity_label, " ", profile$label, " Profiles")

            page_key <- paste0(slice$dir_name, "_", entity$entity, "_", profile$suffix)

            pca_plots_2D[[paste0(page_key, "_by_treatment")]] <- plot_pca(
                data = df, explained_variance_df = explained_variance_df, title = title,
                color_by = "Metadata_Experiment_Treatment", palette = custom_treatment_palette,
                legend_title = "Treatment", alpha = 0.3, rasterize_dpi = 150
            )
            pca_plots_2D[[paste0(page_key, "_by_tumor_type")]] <- plot_pca(
                data = df, explained_variance_df = explained_variance_df, title = title,
                color_by = "Metadata_Biology_TumorType", palette = tumor_type_palette,
                legend_title = "Tumor type", alpha = 0.18, rasterize_dpi = 150
            )

            if (profile$facet) {
                pca_plots_2D[[paste0(page_key, "_facet_by_patient")]] <- plot_pca(
                    data = df, explained_variance_df = explained_variance_df, title = title,
                    color_by = "Metadata_Experiment_Treatment", palette = custom_treatment_palette,
                    legend_title = "Treatment", facet_by = "Metadata_Biology_PatientTumor", facet_nrow = 3,
                    alpha = 0.5, point_size = 0.3, rasterize_dpi = 150
                )
                pca_plots_2D[[paste0(page_key, "_facet_by_tumor_type")]] <- plot_pca(
                    data = df, explained_variance_df = explained_variance_df, title = title,
                    color_by = "Metadata_Experiment_Treatment", palette = custom_treatment_palette,
                    legend_title = "Treatment", facet_by = "Metadata_Biology_TumorType", facet_nrow = 2,
                    alpha = 0.3, point_size = 0.15, rasterize_dpi = 150
                )
            }
        }
    }
}

save_umap_pca_plots_pdf(pca_plots_2D, file.path(figures_path, "2D_all_patients_pca.pdf"), width = 7, height = 7)

# normalized profiles
normalized_profiles <- c(
    "organoid_norm",
    "sammed_organoid_norm",
    "sc_norm",
    "sammed_sc_norm",
    "nucleocentric_morphem_norm",
    "sammed_nucleocentric_norm"
)

profile_specs_3d <- list(
    list(file_prefix = "3D_1.feature_selected_profiles", suffix = "fs_profiles", label = "Feature Selected", facet = TRUE),
    list(file_prefix = "3D_2.aggregated_profiles", suffix = "sc_agg_profiles", label = "Aggregated", facet = FALSE),
    list(file_prefix = "3D_3.consensus_profiles", suffix = "sc_consensus_profiles", label = "Consensus", facet = FALSE)
)

pca_plots_3D <- list()

for (norm_profile in normalized_profiles) {
    for (profile in profile_specs_3d) {
        file_name <- paste0(profile$file_prefix, "_", norm_profile, "_", profile$suffix, "_embeddings.parquet")
        file_path <- file.path(pca_results_dir, file_name)
        explained_variance_file_name <- paste0(profile$file_prefix, "_", norm_profile, "_", profile$suffix, "_explained_variance.parquet")
        explained_variance_file_path <- file.path(pca_results_dir, explained_variance_file_name)

        if (!file.exists(file_path) || !file.exists(explained_variance_file_path)) {
            cat("Missing file, skipping:", file_path, "\n")
            next
        }

        df <- arrow::read_parquet(file_path)
        if (!"PC1" %in% colnames(df)) {
            cat("PC1 not found, skipping:", file_path, "\n")
            next
        }
        df$Metadata_Biology_TumorType <- tumor_type_lookup[df$Metadata_Biology_PatientTumor]

        explained_variance_df <- arrow::read_parquet(explained_variance_file_path)
        title <- paste0("All patients: 3D ", normalization_variant_labels[[norm_profile]], " ", profile$label, " Profiles")

        page_key <- paste0(norm_profile, "_", profile$suffix)

        pca_plots_3D[[paste0(page_key, "_by_treatment")]] <- plot_pca(
            data = df, explained_variance_df = explained_variance_df, title = title,
            color_by = "Metadata_Experiment_Treatment", palette = custom_treatment_palette,
            legend_title = "Treatment", alpha = 0.3, rasterize_dpi = 150
        )
        pca_plots_3D[[paste0(page_key, "_by_tumor_type")]] <- plot_pca(
            data = df, explained_variance_df = explained_variance_df, title = title,
            color_by = "Metadata_Biology_TumorType", palette = tumor_type_palette,
            legend_title = "Tumor type", alpha = 0.18, rasterize_dpi = 150
        )

        if (profile$facet) {
            pca_plots_3D[[paste0(page_key, "_facet_by_patient")]] <- plot_pca(
                data = df, explained_variance_df = explained_variance_df, title = title,
                color_by = "Metadata_Experiment_Treatment", palette = custom_treatment_palette,
                legend_title = "Treatment", facet_by = "Metadata_Biology_PatientTumor", facet_nrow = 3,
                alpha = 0.5, point_size = 0.3, rasterize_dpi = 150
            )
            pca_plots_3D[[paste0(page_key, "_facet_by_tumor_type")]] <- plot_pca(
                data = df, explained_variance_df = explained_variance_df, title = title,
                color_by = "Metadata_Experiment_Treatment", palette = custom_treatment_palette,
                legend_title = "Treatment", facet_by = "Metadata_Biology_TumorType", facet_nrow = 2,
                alpha = 0.3, point_size = 0.15, rasterize_dpi = 150
            )
        }
    }
}

save_umap_pca_plots_pdf(pca_plots_3D, file.path(figures_path, "3D_all_patients_pca.pdf"), width = 7, height = 7)

# Reuses normalized_profiles from the 3D PCA section above.
scree_df_list <- list()
for (norm_profile in normalized_profiles) {
    explained_variance_file_name <- paste0("3D_1.feature_selected_profiles_", norm_profile, "_fs_profiles_explained_variance.parquet")
    explained_variance_file_path <- file.path(pca_results_dir, explained_variance_file_name)

    if (!file.exists(explained_variance_file_path)) {
        cat("Missing file, skipping:", explained_variance_file_path, "\n")
        next
    }

    ev_df <- arrow::read_parquet(explained_variance_file_path)
    n_components <- ncol(ev_df)
    scree_df_list[[norm_profile]] <- data.frame(
        component = seq_len(n_components),
        variance_explained = as.numeric(ev_df[1, ]) * 100,
        norm_profile = normalization_variant_labels[[norm_profile]]
    )
}
scree_df <- do.call(rbind, scree_df_list)

scree_plot <- ggplot(scree_df, aes(x = component, y = variance_explained, color = norm_profile)) +
    geom_line() +
    geom_point(size = 1) +
    labs(
        title = "Scree Plot: 3D Single Cell Feature Selected Profiles",
        x = "Principal Component",
        y = "Variance Explained (%)",
        color = "Normalization"
    ) +
    theme_manuscript(base_size = 10)

ggsave(scree_plot, filename = file.path(figures_path, "3D_scfs_scree_plot.png"), width = 8, height = 6, dpi = 300)
scree_plot

# 2D scree plot: feature-selected profiles, one panel per slice type,
# colored by entity (single cell vs organoid).
scree_df_list_2d <- list()
for (slice in slice_specs) {
    for (entity in entity_specs) {
        explained_variance_file_name <- paste0(slice$file_prefix, "_", entity$entity, "_fs_profiles_explained_variance.parquet")
        explained_variance_file_path <- file.path(pca_results_dir, explained_variance_file_name)

        if (!file.exists(explained_variance_file_path)) {
            cat("Missing file, skipping:", explained_variance_file_path, "\n")
            next
        }

        ev_df <- arrow::read_parquet(explained_variance_file_path)
        scree_df_list_2d[[paste0(slice$dir_name, "_", entity$entity)]] <- data.frame(
            component = seq_len(ncol(ev_df)),
            variance_explained = as.numeric(ev_df[1, ]) * 100,
            slice = slice$title_label,
            entity = entity$entity_label
        )
    }
}
scree_df_2d <- do.call(rbind, scree_df_list_2d)

scree_plot_2d <- (
    ggplot(scree_df_2d, aes(x = component, y = variance_explained, color = entity))
    + geom_line()
    + geom_point(size = 1)
    + facet_wrap(~slice, nrow = 1)
    + labs(
        title = "Scree Plot: 2D Feature Selected Profiles",
        x = "Principal Component",
        y = "Variance Explained (%)",
        color = "Entity"
    )
    + theme_manuscript(base_size = 10)
)

ggsave(scree_plot_2d, filename = file.path(figures_path, "2D_fs_scree_plot.png"), width = 12, height = 5, dpi = 600)
scree_plot_2d
