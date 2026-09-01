list_of_packages <- c("ggplot2", "dplyr", "arrow", "RColorBrewer", "ggrastr")
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

results_dir <- file.path(root_dir, "1.EDA", "results")
figures_dir <- file.path(root_dir, "1.EDA", "figures", "count_viability")
dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)

plot_theme <- (
    theme_bw()
    + theme(
        plot.title = element_text(hjust = 0.5, size = 14),
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14),
        axis.text.x = element_text(size = 9),
        axis.text.y = element_text(size = 10),
        legend.title = element_text(size = 12),
        legend.text = element_text(size = 9),
        strip.text = element_text(size = 10)
    )
)

# Biological question: does viability track with how many cells/organoids
# survive treatment? Restricted to 3D (the ground-truth modality) -- 2D
# projection-method comparisons don't add to that story and were dropped, as
# were the patient x treatment facets, most of which had only 1-2 points per
# panel and were not interpretable.

# --- 3D: mean cells per organoid vs. viability ---
df <- read_parquet(file.path(results_dir, "count_viability_joined.parquet"))
df$Treatment <- factor(df$Treatment, levels = intersect(custom_treatment_order, unique(df$Treatment)))

p_pooled <- (
    ggplot(df, aes(x = min_max_viability, y = mean_cell_count, color = Treatment, shape = factor(Metadata_dose)))
    + rasterise(geom_point(size = 2, alpha = 0.8), dpi = 300)
    + geom_smooth(method = "lm", se = TRUE, color = "black", linewidth = 0.5)
    + scale_color_manual(values = custom_treatment_palette, na.value = "grey70")
    + labs(
        title = "3D pooled (all patients): mean cells per organoid vs. viability",
        x = "Viability (min-max normalized)", y = "Mean cells per organoid", shape = "Dose"
    )
    + plot_theme
)

p_by_patient <- (
    ggplot(df, aes(x = min_max_viability, y = mean_cell_count, color = Treatment, shape = factor(Metadata_dose)))
    + rasterise(geom_point(size = 2, alpha = 0.8), dpi = 300)
    + scale_color_manual(values = custom_treatment_palette, na.value = "grey70")
    + facet_wrap(~Metadata_patient_tumor, scales = "free_y")
    + labs(
        title = "3D: mean cells per organoid vs. viability, by patient",
        x = "Viability (min-max normalized)", y = "Mean cells per organoid", shape = "Dose"
    )
    + plot_theme
)

# --- 3D: FOV-normalized total cell count vs. viability ---
df_norm <- read_parquet(file.path(results_dir, "count_norm_viability_joined.parquet"))
df_norm$Treatment <- factor(df_norm$Treatment, levels = intersect(custom_treatment_order, unique(df_norm$Treatment)))
df_norm_3d <- df_norm %>% filter(modality == "3D")

p_norm_pooled <- (
    ggplot(df_norm_3d, aes(x = min_max_viability, y = total_cell_count_norm, color = Treatment, shape = factor(Metadata_dose)))
    + rasterise(geom_point(size = 2, alpha = 0.8), dpi = 300)
    + geom_smooth(method = "lm", se = TRUE, color = "black", linewidth = 0.5)
    + scale_color_manual(values = custom_treatment_palette, na.value = "grey70")
    + labs(
        title = "3D pooled (all patients): total cells per treatment (FOV-normalized) vs. viability",
        x = "Viability (min-max normalized)", y = "Total cells per treatment (FOV-normalized)", shape = "Dose"
    )
    + plot_theme
)

p_norm_by_patient <- (
    ggplot(df_norm_3d, aes(x = min_max_viability, y = total_cell_count_norm, color = Treatment, shape = factor(Metadata_dose)))
    + rasterise(geom_point(size = 2, alpha = 0.8), dpi = 300)
    + scale_color_manual(values = custom_treatment_palette, na.value = "grey70")
    + facet_wrap(~Metadata_patient_tumor, scales = "free_y")
    + labs(
        title = "3D: total cells per treatment (FOV-normalized) vs. viability, by patient",
        x = "Viability (min-max normalized)", y = "Total cells per treatment (FOV-normalized)", shape = "Dose"
    )
    + plot_theme
)

pdf(file.path(figures_dir, "3D_count_vs_viability.pdf"), width = 11, height = 8.5, onefile = TRUE)
print(p_pooled)
print(p_by_patient)
print(p_norm_pooled)
print(p_norm_by_patient)
dev.off()

cat("Wrote 1 PDF (4 pages) to", figures_dir, "\n")
