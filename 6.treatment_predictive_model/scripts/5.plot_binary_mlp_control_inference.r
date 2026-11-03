list_of_packages <- c("ggplot2", "dplyr", "tidyr", "arrow", "ggrastr", "RColorBrewer")
for (package in list_of_packages) {
    suppressPackageStartupMessages(
        suppressWarnings(
            library(package, character.only = TRUE, quietly = TRUE, warn.conflicts = FALSE)
        )
    )
}

find_git_root <- function() {
    cwd <- getwd()
    if (dir.exists(file.path(cwd, ".git"))) {
        return(cwd)
    }
    current_path <- cwd
    while (dirname(current_path) != current_path) {
        parent_path <- dirname(current_path)
        if (dir.exists(file.path(parent_path, ".git"))) {
            return(parent_path)
        }
        current_path <- parent_path
    }
    stop("No Git root directory found.")
}

root_dir <- find_git_root()
source(file.path(root_dir, "utils", "r_plot_themes.r"))

results_dir <- file.path(root_dir, "6.treatment_predictive_model", "model_results")
figures_dir <- file.path(root_dir, "6.treatment_predictive_model", "figures")
dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)

inference_path <- file.path(results_dir, "binary_mlp_control_all_treatments_inference.parquet")
per_treatment_summary_path <- file.path(results_dir, "binary_mlp_control_per_treatment_summary.parquet")
per_patient_treatment_summary_path <- file.path(
    results_dir, "binary_mlp_control_per_patient_treatment_summary.parquet"
)

distribution_figure_path <- file.path(figures_dir, "binary_mlp_control_probability_by_treatment.png")
scatter_figure_path <- file.path(figures_dir, "binary_mlp_control_probability_vs_dmso_distance.png")
heatmap_figure_path <- file.path(figures_dir, "binary_mlp_control_heatmap.png")

inference_df <- read_parquet(inference_path)
per_treatment_summary_df <- read_parquet(per_treatment_summary_path)
per_patient_treatment_summary_df <- read_parquet(per_patient_treatment_summary_path)

treatment_levels <- intersect(
    custom_treatment_order, unique(inference_df$Metadata_Experiment_Treatment)
)
inference_df$Metadata_Experiment_Treatment <- factor(
    inference_df$Metadata_Experiment_Treatment, levels = treatment_levels
)
per_treatment_summary_df$Metadata_Experiment_Treatment <- factor(
    per_treatment_summary_df$Metadata_Experiment_Treatment, levels = treatment_levels
)
per_patient_treatment_summary_df$Metadata_Experiment_Treatment <- factor(
    per_patient_treatment_summary_df$Metadata_Experiment_Treatment, levels = treatment_levels
)

head(inference_df)

options(repr.plot.width = 16, repr.plot.height = 8)

probability_by_treatment_plot <- (
    ggplot(
        inference_df,
        aes(x = Metadata_Experiment_Treatment, y = Metadata_predicted_cytotoxic_probability)
    )
    + geom_violin(fill = "grey70", scale = "width", alpha = 0.7)
    + geom_boxplot(width = 0.12, outlier.shape = NA, fill = "white", alpha = 0.6)
    + labs(
        x = "Treatment",
        y = "Predicted cytotoxic probability",
        title = "Predicted cytotoxic-likeness by treatment (all patients, all cells)"
    )
    + theme_manuscript(base_size = 16, x_text = "angled")
)

ggsave(
    distribution_figure_path, probability_by_treatment_plot,
    width = 16, height = 8, dpi = 600, bg = "white"
)
probability_by_treatment_plot

options(repr.plot.width = 20, repr.plot.height = 16)

probability_vs_distance_plot <- (
    ggplot(
        inference_df,
        aes(
            x = Metadata_distance_from_dmso_centroid,
            y = Metadata_predicted_cytotoxic_probability,
            colour = Metadata_Experiment_Treatment
        )
    )
    + rasterise(geom_point(size = 0.4, alpha = 0.3), dpi = 600)
    + geom_hline(yintercept = 0.5, linetype = "dashed", colour = "grey40")
    + scale_colour_manual(values = custom_treatment_palette, guide = "none")
    + facet_wrap(~Metadata_Experiment_Treatment, scales = "free_x")
    + labs(
        x = "Distance from DMSO centroid (standardized feature space)",
        y = "Predicted cytotoxic probability",
        title = "Predicted cytotoxic probability vs. distance from DMSO, by treatment"
    )
    + theme_manuscript(base_size = 14, x_text = "default")
)

ggsave(
    scatter_figure_path, probability_vs_distance_plot,
    width = 20, height = 16, dpi = 600, bg = "white"
)
probability_vs_distance_plot

options(repr.plot.width = 14, repr.plot.height = 8)

per_patient_treatment_summary_df$Metadata_Biology_TumorType <- tumor_type_lookup[
    as.character(per_patient_treatment_summary_df$Metadata_Biology_PatientTumor)
]
per_patient_treatment_summary_df$patient_label <- paste0(
    per_patient_treatment_summary_df$Metadata_Biology_PatientTumor,
    " (", per_patient_treatment_summary_df$Metadata_Biology_TumorType, ")"
)

patient_order_df <- per_patient_treatment_summary_df %>%
    distinct(Metadata_Biology_PatientTumor, Metadata_Biology_TumorType, patient_label) %>%
    arrange(
        match(Metadata_Biology_TumorType, names(tumor_type_palette)),
        Metadata_Biology_PatientTumor
    )
per_patient_treatment_summary_df$patient_label <- factor(
    per_patient_treatment_summary_df$patient_label, levels = patient_order_df$patient_label
)

per_patient_treatment_summary_df <- per_patient_treatment_summary_df %>%
    complete(patient_label, Metadata_Experiment_Treatment)

heatmap_plot <- (
    ggplot(
        per_patient_treatment_summary_df,
        aes(
            x = Metadata_Experiment_Treatment,
            y = patient_label,
            fill = Metadata_predicted_cytotoxic_probability_mean
        )
    )
    + geom_tile(colour = "white")
    + scale_fill_distiller(
        palette = "RdBu", direction = -1, limits = c(0, 1), name = "Mean P(cytotoxic)",
        na.value = "grey80"
    )
    + labs(
        x = "Treatment",
        y = "Patient (tumor type)",
        title = "Mean predicted cytotoxic probability per patient x treatment"
    )
    + theme_manuscript(base_size = 14, x_text = "angled")
    + theme(
        axis.text.y = element_text(
            colour = tumor_type_palette[patient_order_df$Metadata_Biology_TumorType]
        ),
        panel.grid = element_blank()
    )
)

ggsave(
    heatmap_figure_path, heatmap_plot,
    width = 14, height = 8, dpi = 600, bg = "white"
)
heatmap_plot
