list_of_packages <- c("ggplot2", "dplyr", "arrow", "tidyr", "RColorBrewer")
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

results_dir <- file.path(root_dir, "1.EDA", "results", "neighbors")
figures_dir <- file.path(root_dir, "1.EDA", "figures", "neighbors")
dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)

plot_theme <- theme_bw() + theme(
    plot.title = element_text(hjust = 0.5, size = 13),
    axis.title.x = element_text(size = 13),
    axis.title.y = element_text(size = 13),
    axis.text.x = element_text(size = 9, angle = 45, hjust = 1),
    axis.text.y = element_text(size = 10),
    strip.text = element_text(size = 9),
    legend.position = "none"
)

# --- 2D nuclei-level: distance to first closest neighbor, by patient (faceted by projection) ---
nuc_2d <- read_parquet(file.path(results_dir, "nuclei_neighbors_2D.parquet"))
p_nuc_2d_patient <- (
    ggplot(nuc_2d, aes(x = Metadata_patient, y = Nuclei_Neighbors_FirstClosestDistance_Adjacent, fill = Metadata_patient))
    + geom_violin(alpha = 0.6, trim = TRUE)
    + geom_boxplot(width = 0.15, alpha = 0.85, outlier.size = 0.2, outlier.alpha = 0.2)
    + facet_wrap(~projection, ncol = 1)
    + labs(title = "2D: nuclei first-closest-neighbor distance, by patient and projection method",
           x = "Patient", y = "First closest neighbor distance (z-scored)")
    + plot_theme
)
ggsave(filename = file.path(figures_dir, "2D_nuclei_first_closest_distance_per_patient.png"),
       plot = p_nuc_2d_patient, width = 11, height = 12, dpi = 600, units = "in")

p_nuc_2d_pooled <- nuc_2d %>%
    mutate(Metadata_treatment = factor(Metadata_treatment, levels = intersect(custom_treatment_order, unique(Metadata_treatment)))) %>%
    ggplot(aes(x = Metadata_treatment, y = Nuclei_Neighbors_FirstClosestDistance_Adjacent, fill = Metadata_treatment))
p_nuc_2d_pooled <- (
    p_nuc_2d_pooled
    + geom_violin(alpha = 0.6, trim = TRUE)
    + geom_boxplot(width = 0.15, alpha = 0.85, outlier.size = 0.2, outlier.alpha = 0.2)
    + scale_fill_manual(values = custom_treatment_palette, na.value = "grey70")
    + facet_wrap(~projection, ncol = 1)
    + labs(title = "2D pooled (all patients): nuclei first-closest-neighbor distance, by treatment",
           x = "Treatment", y = "First closest neighbor distance (z-scored)")
    + plot_theme
)
ggsave(filename = file.path(figures_dir, "2D_nuclei_first_closest_distance_pooled.png"),
       plot = p_nuc_2d_pooled, width = 11, height = 12, dpi = 600, units = "in")

# --- 2D nuclei-level: number of neighbors + percent touching (long format), pooled ---
nuc_long <- nuc_2d %>%
    select(Metadata_patient, projection, Nuclei_Neighbors_NumberOfNeighbors_Adjacent, Nuclei_Neighbors_PercentTouching_Adjacent) %>%
    pivot_longer(cols = starts_with("Nuclei_Neighbors"), names_to = "metric", values_to = "value")
p_nuc_metrics <- (
    ggplot(nuc_long, aes(x = Metadata_patient, y = value, fill = Metadata_patient))
    + geom_boxplot(outlier.size = 0.2, outlier.alpha = 0.2)
    + facet_grid(metric ~ projection, scales = "free_y")
    + labs(title = "2D: nuclei neighbor count and percent touching, by patient and projection method",
           x = "Patient", y = "Value (z-scored)")
    + plot_theme + theme(strip.text.y = element_text(size = 7))
)
ggsave(filename = file.path(figures_dir, "2D_nuclei_neighbor_metrics_per_patient.png"),
       plot = p_nuc_metrics, width = 14, height = 8, dpi = 600, units = "in")

# --- 2D organoid-level: number of neighboring organoids, by patient ---
org_2d <- read_parquet(file.path(results_dir, "organoid_neighbors_2D.parquet"))
p_org_2d <- (
    ggplot(org_2d, aes(x = Metadata_patient, y = Organoid_Neighbors_NumberOfNeighbors_Adjacent, fill = Metadata_patient))
    + geom_boxplot(outlier.size = 0.2, outlier.alpha = 0.2)
    + facet_wrap(~projection, ncol = 1)
    + labs(title = "2D: number of neighboring organoids, by patient and projection method",
           x = "Patient", y = "Number of neighboring organoids (z-scored)")
    + plot_theme
)
ggsave(filename = file.path(figures_dir, "2D_organoid_neighbor_count_per_patient.png"),
       plot = p_org_2d, width = 11, height = 12, dpi = 600, units = "in")

# --- 3D: shell/distance-based neighbor metrics, by patient (one plot per metric) ---
nuc_3d <- read_parquet(file.path(results_dir, "nuclei_neighbors_3D.parquet"))

p_3d_neighbor_count <- (
    ggplot(nuc_3d, aes(x = Metadata_patient, y = Metadata_Neighbors_NeighborsCountAdjacent, fill = Metadata_patient))
    + geom_boxplot(outlier.size = 0.2, outlier.alpha = 0.2)
    + labs(title = "3D: adjacent neighbor count, by patient",
           x = "Patient", y = "Adjacent neighbor count")
    + plot_theme
)
ggsave(filename = file.path(figures_dir, "3D_neighbor_count_per_patient.png"),
       plot = p_3d_neighbor_count, width = 9, height = 6, dpi = 600, units = "in")

p_3d_distance_from_center <- (
    ggplot(nuc_3d, aes(x = Metadata_patient, y = Metadata_Neighbors_DistancesFromCenter, fill = Metadata_patient))
    + geom_boxplot(outlier.size = 0.2, outlier.alpha = 0.2)
    + labs(title = "3D: nucleus distance from organoid center, by patient",
           x = "Patient", y = "Distance from organoid center")
    + plot_theme
)
ggsave(filename = file.path(figures_dir, "3D_distance_from_center_per_patient.png"),
       plot = p_3d_distance_from_center, width = 9, height = 6, dpi = 600, units = "in")

p_3d_distance_from_exterior <- (
    ggplot(nuc_3d, aes(x = Metadata_patient, y = Metadata_Neighbors_DistancesFromExterior, fill = Metadata_patient))
    + geom_boxplot(outlier.size = 0.2, outlier.alpha = 0.2)
    + labs(title = "3D: nucleus distance from organoid exterior, by patient",
           x = "Patient", y = "Distance from organoid exterior")
    + plot_theme
)
ggsave(filename = file.path(figures_dir, "3D_distance_from_exterior_per_patient.png"),
       plot = p_3d_distance_from_exterior, width = 9, height = 6, dpi = 600, units = "in")

p_3d_pooled_data <- nuc_3d %>%
    mutate(Metadata_treatment = factor(Metadata_treatment, levels = intersect(custom_treatment_order, unique(Metadata_treatment))))
p_3d_pooled <- (
    ggplot(p_3d_pooled_data, aes(x = Metadata_treatment, y = Metadata_Neighbors_NeighborsCountAdjacent, fill = Metadata_treatment))
    + geom_boxplot(outlier.size = 0.2, outlier.alpha = 0.2)
    + scale_fill_manual(values = custom_treatment_palette, na.value = "grey70")
    + labs(title = "3D pooled (all patients): adjacent neighbor count, by treatment",
           x = "Treatment", y = "Adjacent neighbor count")
    + plot_theme
)
ggsave(filename = file.path(figures_dir, "3D_neighbor_count_pooled_by_treatment.png"),
       plot = p_3d_pooled, width = 9, height = 6, dpi = 600, units = "in")

cat("Wrote 8 figures to", figures_dir, "\n")
