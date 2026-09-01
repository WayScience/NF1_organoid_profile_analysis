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
figures_dir <- file.path(root_dir, "1.EDA", "figures", "volume_area_vs_count")
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
        legend.text = element_text(size = 9)
    )
)

# points are small and semi-transparent in-plot (for dense scatters); make
# the legend swatches larger and fully opaque so treatments stay legible
legend_guide <- guides(color = guide_legend(override.aes = list(size = 4, alpha = 1)))

# Biological question: as organoids are exposed to more cells (a proxy for
# treatment effect on growth), does organoid/cell volume change? Restricted
# to 3D: the 2D projection-method variants of this analysis compared
# representation choices, not biology, and were dropped.
#
# Organoid_NoChannel_AreaSizeShape_Volume and Cell_NoChannel_AreaSizeShape_Volume
# are z-scored PER PATIENT upstream (each patient's population is normalized
# against its own mean/SD), so patients are NOT on a common scale: e.g.
# mean/SD of organoid volume range from (0.49, 2.15) to (3.67, 12.0) across
# our 12 patients. We therefore never pool/overlay this column across
# patients on one shared axis -- only ever facet by patient, where each
# panel is valid on its own terms.

# --- FOV-normalized total cell count per patient x treatment x dose, from the
# canonical (non-sammed/non-nucleocentric) 3D single-cell profile type,
# matching sc_norm.parquet. There is no true per-organoid cell count in the
# current pipeline output, so this treatment-level count is broadcast onto
# every organoid below. ---
raw_counts <- read_parquet(file.path(results_dir, "cell_counts", "cell_counts.parquet"))

counts_3d <- raw_counts %>%
    filter(Metadata_profile_type == "sc_norm_norm_profile_3D") %>%
    group_by(Metadata_Biology_PatientTumor, Metadata_Experiment_Treatment, Metadata_Experiment_Dose) %>%
    summarise(
        total_cells = sum(Metadata_n_cells),
        total_fovs = sum(Metadata_n_fovs),
        .groups = "drop"
    ) %>%
    mutate(total_cell_count_norm = total_cells / total_fovs) %>%
    rename(
        Metadata_patient_tumor = Metadata_Biology_PatientTumor,
        Metadata_treatment = Metadata_Experiment_Treatment,
        Metadata_dose = Metadata_Experiment_Dose
    ) %>%
    select(Metadata_patient_tumor, Metadata_treatment, Metadata_dose, total_cell_count_norm)

# --- mean cells per organoid per patient x treatment x dose, broadcast onto
# every organoid below. Computed directly from organoid_cell_counts.parquet
# (not via count_viability_joined.parquet, which inner-joins against the
# viability platemap and would silently drop the 5 patients with no
# viability coverage even though viability isn't used here) ---
mean_counts <- read_parquet(file.path(results_dir, "cell_counts", "organoid_cell_counts.parquet")) %>%
    group_by(Metadata_Biology_PatientTumor, Metadata_Experiment_Treatment, Metadata_Experiment_Dose) %>%
    summarise(mean_cell_count = mean(mean_cells_per_organoid, na.rm = TRUE), .groups = "drop") %>%
    rename(
        Metadata_patient_tumor = Metadata_Biology_PatientTumor,
        Metadata_treatment = Metadata_Experiment_Treatment,
        Metadata_dose = Metadata_Experiment_Dose
    )

# --- per-organoid volume (z-scored within patient) joined to that count ---
patients_3d <- setdiff(list.dirs(file.path(root_dir, "data", "profiles_3D"), recursive = FALSE, full.names = FALSE), c("all_patients", "NF0037_T1_CQ1"))
vol_rows <- list()
for (patient in patients_3d) {
    f <- file.path(root_dir, "data", "profiles_3D", patient, "5.normalized_profiles", "organoid_norm.parquet")
    if (!file.exists(f)) next
    df <- read_parquet(f, col_select = c("Metadata_Experiment_Treatment", "Metadata_Experiment_Dose", "Organoid_NoChannel_AreaSizeShape_Volume"))
    df$Metadata_patient_tumor <- patient
    vol_rows[[patient]] <- df
}
vol_df <- bind_rows(vol_rows)
colnames(vol_df)[colnames(vol_df) == "Metadata_Experiment_Treatment"] <- "Metadata_treatment"
colnames(vol_df)[colnames(vol_df) == "Metadata_Experiment_Dose"] <- "Metadata_dose"

joined_3d <- vol_df %>%
    inner_join(mean_counts, by = c("Metadata_patient_tumor", "Metadata_treatment", "Metadata_dose"))
joined_3d$Metadata_treatment <- factor(joined_3d$Metadata_treatment,
                                          levels = intersect(custom_treatment_order, unique(joined_3d$Metadata_treatment)))

# --- per-organoid scatter, faceted by patient (the only axis this z-score is
# valid on), colored by treatment ---
p_facet_patient <- (
    ggplot(joined_3d, aes(x = mean_cell_count, y = Organoid_NoChannel_AreaSizeShape_Volume, color = Metadata_treatment))
    + rasterise(geom_point(size = 1, alpha = 0.5), dpi = 300)
    + scale_color_manual(values = custom_treatment_palette, na.value = "grey70")
    + facet_wrap(~Metadata_patient_tumor, scales = "free")
    + labs(
        title = "3D: organoid volume vs. mean cells per organoid, by patient",
        x = "Mean cells per organoid", y = "Organoid volume (z-scored within patient)", color = "Treatment"
    )
    + plot_theme
    + legend_guide
)

# --- single-cell volume vs. the same FOV-normalized count, faceted by patient ---
sc_vol_rows <- list()
for (patient in patients_3d) {
    f <- file.path(root_dir, "data", "profiles_3D", patient, "5.normalized_profiles", "sc_norm.parquet")
    if (!file.exists(f)) next
    df <- read_parquet(f, col_select = c("Metadata_Experiment_Treatment", "Metadata_Experiment_Dose", "Cell_NoChannel_AreaSizeShape_Volume"))
    df$Metadata_patient_tumor <- patient
    sc_vol_rows[[patient]] <- df
}
sc_vol_df <- bind_rows(sc_vol_rows)
colnames(sc_vol_df)[colnames(sc_vol_df) == "Metadata_Experiment_Treatment"] <- "Metadata_treatment"
colnames(sc_vol_df)[colnames(sc_vol_df) == "Metadata_Experiment_Dose"] <- "Metadata_dose"

joined_sc <- sc_vol_df %>%
    inner_join(counts_3d, by = c("Metadata_patient_tumor", "Metadata_treatment", "Metadata_dose"))
joined_sc$Metadata_treatment <- factor(joined_sc$Metadata_treatment,
                                          levels = intersect(custom_treatment_order, unique(joined_sc$Metadata_treatment)))

p_sc_facet_patient <- (
    ggplot(joined_sc, aes(x = total_cell_count_norm, y = Cell_NoChannel_AreaSizeShape_Volume, color = Metadata_treatment))
    + rasterise(geom_point(size = 0.5, alpha = 0.4), dpi = 300)
    + scale_color_manual(values = custom_treatment_palette, na.value = "grey70")
    + facet_wrap(~Metadata_patient_tumor, scales = "free")
    + labs(
        title = "3D: single-cell volume vs. total cell count (FOV-normalized), by patient",
        x = "Total cells per treatment (FOV-normalized)", y = "Cell volume (z-scored within patient)", color = "Treatment"
    )
    + plot_theme
    + legend_guide
)

pdf(file.path(figures_dir, "3D_volume_vs_count.pdf"), width = 11, height = 8.5, onefile = TRUE)
print(p_facet_patient)
print(p_sc_facet_patient)
dev.off()

cat("Wrote 1 PDF (2 pages) to", figures_dir, "\n")
