list_of_packages <- c("ggplot2", "dplyr", "arrow", "RColorBrewer")
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
source(file.path(root_dir, "utils", "r_plot_funcs.r"))

results_dir <- file.path(root_dir, "5.differential_analysis", "results")
figures_dir <- file.path(root_dir, "5.differential_analysis", "figures")
dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)

patient_viability_path <- file.path(results_dir, "viability_log2fc_by_patient_tumor_type.parquet")
tumor_type_viability_path <- file.path(results_dir, "viability_log2fc_by_tumor_type.parquet")
figure_path <- file.path(figures_dir, "viability_percent_by_tumor_type.pdf")

tumor_type_levels <- c("cNF", "pNF", "MPNST", "Other")

add_treatment_label <- function(df) {
    df %>%
        mutate(
            treatment_label = paste0(
                Metadata_Experiment_Treatment, " ",
                as.character(Metadata_Experiment_Dose), " ",
                Metadata_Experiment_Unit
            ),
            Metadata_Biology_TumorType = factor(
                Metadata_Biology_TumorType,
                levels = intersect(tumor_type_levels, unique(Metadata_Biology_TumorType))
            )
        )
}

patient_viability_df <- add_treatment_label(read_parquet(patient_viability_path))
tumor_type_viability_df <- add_treatment_label(read_parquet(tumor_type_viability_path))

treatment_label_levels <- patient_viability_df %>%
    distinct(Metadata_Experiment_Treatment, Metadata_Experiment_Dose, treatment_label) %>%
    arrange(match(Metadata_Experiment_Treatment, custom_treatment_order), Metadata_Experiment_Dose) %>%
    pull(treatment_label)
patient_viability_df$treatment_label <- factor(patient_viability_df$treatment_label, levels = treatment_label_levels)
tumor_type_viability_df$treatment_label <- factor(tumor_type_viability_df$treatment_label, levels = treatment_label_levels)
head(tumor_type_viability_df)

heatmap_df_from_tables <- function(patient_df, tumor_type_df) {
    bind_rows(
        patient_df %>%
            select(
                Metadata_Biology_TumorType,
                row_label = Metadata_Biology_PatientTumor,
                treatment_label,
                viability_percent = Metadata_Viability_Percentage
            ),
        tumor_type_df %>%
            transmute(
                Metadata_Biology_TumorType,
                row_label = paste0(Metadata_Biology_TumorType, " mean"),
                treatment_label,
                viability_percent = mean_viability_percent
            )
    )
}

make_viability_heatmap <- function(heatmap_df, title) {
    heatmap_df <- heatmap_df %>%
        mutate(Metadata_Biology_TumorType = droplevels(Metadata_Biology_TumorType))
    row_label_levels <- heatmap_df %>%
        distinct(Metadata_Biology_TumorType, row_label) %>%
        arrange(Metadata_Biology_TumorType, grepl(" mean$", row_label), row_label) %>%
        pull(row_label)
    heatmap_df$row_label <- factor(heatmap_df$row_label, levels = rev(row_label_levels))

    (
        ggplot(heatmap_df, aes(x = treatment_label, y = row_label, fill = viability_percent))
        + geom_tile(color = "white", linewidth = 0.3)
        + geom_text(
            data = filter(heatmap_df, !is.na(viability_percent)),
            aes(label = round(viability_percent)),
            size = 3
        )
        + scale_fill_gradient2(
            low = "#b2182b",
            mid = "white",
            high = "#2166ac",
            midpoint = 100,
            name = "Viability\n(% of DMSO)"
        )
        + facet_grid(Metadata_Biology_TumorType ~ ., scales = "free_y", space = "free_y")
        + labs(title = title, x = "Treatment", y = "Patient")
        + theme_manuscript(base_size = 14)
        + theme(
            axis.text.x = element_text(angle = 45, hjust = 1),
            panel.grid = element_blank(),
            panel.background = element_rect(fill = "grey85"),
            strip.text.y = element_text(angle = 0)
        )
    )
}

# page 1: every patient, including those without viability data (grey tiles)
all_heatmap_df <- heatmap_df_from_tables(patient_viability_df, tumor_type_viability_df)

# page 2: only patients with viability data, and only tumor types that have some
tested_heatmap_df <- heatmap_df_from_tables(
    patient_viability_df %>% filter(!is.na(Metadata_Viability_Percentage)),
    tumor_type_viability_df %>% filter(n_patients > 0)
)

options(repr.plot.width = 18, repr.plot.height = 10)
viability_all_plot <- make_viability_heatmap(all_heatmap_df, "All patients")
viability_tested_plot <- make_viability_heatmap(tested_heatmap_df, "Patients with viability data")
save_plots_pdf(
    list(viability_all_plot, viability_tested_plot),
    figure_path,
    width = 18,
    height = 10
)
viability_all_plot
viability_tested_plot
