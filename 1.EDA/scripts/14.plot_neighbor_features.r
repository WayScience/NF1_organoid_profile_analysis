list_of_packages <- c("ggplot2", "dplyr", "arrow", "tidyr", "RColorBrewer", "ggrastr", "patchwork")
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

# --- local crowding: 3D nuc (single-cell) and 3D org (organoid), plus 3D org
# density and organoid volume ---
# Patient x treatment is shown as a single grid per metric (one facet per
# patient, treatment on the x-axis) rather than two separately-pooled
# marginals: pooling over the other factor is exactly what hides whether an
# effect is patient-specific, treatment-general, or a combination of both.
nuc_3d <- read_parquet(file.path(results_dir, "nuclei_neighbors_3D.parquet"))
# 3D organoid neighbor counts are derived in notebook 13 (bounding-sphere
# touch test within each organoid's own image), since no Organoid_Neighbors_*
# column exists in the 3D profiles the way it does in 2D.
org_3d <- read_parquet(file.path(results_dir, "organoid_neighbors_3D.parquet"))
# Organoid cell density (cells / organoid volume in um^3), also derived in
# notebook 13 since it isn't a profile column either.
org_density_3d <- read_parquet(file.path(results_dir, "organoid_density_3D.parquet"))
# Organoid volume (the size story itself, not just crowding/density around it).
org_volume_3d <- read_parquet(file.path(results_dir, "organoid_volume_3D.parquet"))

# Facets are grouped by tumor type (not just patient ID) so that a pattern
# shared by several patients of the same subtype reads as a tumor-subtype
# pattern, not an opaque "some patients differ" observation.
add_patient_facet_label <- function(df) {
    df %>%
        mutate(
            Metadata_Experiment_Treatment = factor(Metadata_Experiment_Treatment, levels = intersect(custom_treatment_order, unique(Metadata_Experiment_Treatment))),
            Metadata_TumorType = tumor_type_lookup[as.character(Metadata_Biology_PatientTumor)],
            Metadata_PatientLabel = paste0(Metadata_Biology_PatientTumor, " (", Metadata_TumorType, ")")
        )
}

nuc_3d <- add_patient_facet_label(nuc_3d)
org_3d <- add_patient_facet_label(org_3d)
org_density_3d <- add_patient_facet_label(org_density_3d)
org_volume_3d <- add_patient_facet_label(org_volume_3d)

patient_by_treatment_panel <- function(df, y_col, y_label, title, y_trans = "identity") {
    # Patients ordered by ID (not by tumor type) so the facet sequence reads
    # straightforwardly top-to-bottom, left-to-right; tumor type stays in the
    # facet label for context.
    patient_order <- df %>%
        distinct(Metadata_Biology_PatientTumor, Metadata_PatientLabel) %>%
        arrange(Metadata_Biology_PatientTumor) %>%
        pull(Metadata_PatientLabel)
    df <- df %>% mutate(Metadata_PatientLabel = factor(Metadata_PatientLabel, levels = patient_order))

    (
        ggplot(df, aes(x = Metadata_Experiment_Treatment, y = .data[[y_col]], fill = Metadata_Experiment_Treatment))
        + rasterise(geom_boxplot(outlier.size = 0.2, outlier.alpha = 0.2), dpi = 300)
        + scale_fill_manual(values = custom_treatment_palette, na.value = "grey70")
        + scale_y_continuous(trans = y_trans)
        + facet_wrap(~Metadata_PatientLabel, ncol = 4)
        + labs(title = paste0(title, " - patient x treatment"),
               x = "Treatment", y = y_label)
        + plot_theme
        + theme(axis.text.x = element_text(size = 6, angle = 45, hjust = 1))
    )
}

p_volume_organoid <- patient_by_treatment_panel(
    org_volume_3d, "Organoid_NoChannel_VolumeSizeShape_Volume", "Organoid volume (z-scored, pseudo-log scale)",
    "Organoid size", y_trans = scales::pseudo_log_trans(sigma = 1)
)
p_crowding_nuc <- patient_by_treatment_panel(
    nuc_3d, "Metadata_Neighbors_NucleiNoChannelNeighborsNeighborsCountAdjacent", "Adjacent neighbor count",
    "Nuclei number of neighbors"
)
p_crowding_organoid <- patient_by_treatment_panel(
    org_3d, "Organoid_Neighbors_NumberOfNeighbors_Adjacent", "Adjacent neighbor count",
    "Organoids number of neighbors"
)
p_density_organoid <- patient_by_treatment_panel(
    org_density_3d, "Organoid_CellDensity_CellsPerUm3", "Cell density (cells / um^3, pseudo-log scale)",
    "Organoids cell density", y_trans = scales::pseudo_log_trans(sigma = 1e-4)
)

pdf_path <- file.path(figures_dir, "3D_neighbor_organoid_story.pdf")
pdf(pdf_path, width = 20, height = 14, onefile = TRUE)
print(p_volume_organoid)
print(p_crowding_nuc)
print(p_crowding_organoid)
print(p_density_organoid)
invisible(dev.off())
