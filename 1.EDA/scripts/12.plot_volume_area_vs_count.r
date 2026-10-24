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

# Biological question: as organoids are exposed to more cells (a proxy for
# treatment effect on growth), does organoid/cell volume change? Restricted
# to 3D: the 2D projection-method variants of this analysis compared
# representation choices, not biology, and were dropped.
#
# Plotted in RAW (unnormalized) units, read directly from each modality's
# pre-normalization stage (4.qc_profiles) rather than the z-scored
# 5.normalized_profiles stage used previously -- volume spans several orders
# of magnitude, so the y axis is log10. Faceted by patient (never
# pooled/overlaid on one shared axis) since patients differ in absolute
# volume scale even in raw units; treatment is shown via point color within
# each patient facet instead.

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

# --- per-organoid volume and its own per-organoid cell count (both live on
# the same row in organoid_flagged_outliers.parquet -- no need to broadcast a
# patient x treatment x dose mean onto every organoid; the real, unaggregated
# per-organoid count gives each point its own x value instead of collapsing
# a whole treatment group onto one repeated number) ---
patients_3d <- setdiff(list.dirs(file.path(root_dir, "data", "profiles_3D"), recursive = FALSE, full.names = FALSE), c("all_patients", "NF0037_T1_CQ1"))
vol_rows <- list()
for (patient in patients_3d) {
    f <- file.path(root_dir, "data", "profiles_3D", patient, "4.qc_profiles", "organoid_flagged_outliers.parquet")
    if (!file.exists(f)) next
    df <- read_parquet(f, col_select = c(
        "Metadata_Experiment_Treatment", "Metadata_Experiment_Dose",
        "Metadata_Object_OrganoidSingleCellCount", "Organoid_NoChannel_VolumeSizeShape_Volume"
    ))
    df$Metadata_patient_tumor <- patient
    vol_rows[[patient]] <- df
}
vol_df <- bind_rows(vol_rows)
colnames(vol_df)[colnames(vol_df) == "Metadata_Experiment_Treatment"] <- "Metadata_treatment"
colnames(vol_df)[colnames(vol_df) == "Metadata_Experiment_Dose"] <- "Metadata_dose"

joined_3d <- vol_df
joined_3d$Metadata_treatment <- factor(joined_3d$Metadata_treatment,
                                          levels = intersect(custom_treatment_order, unique(joined_3d$Metadata_treatment)))

# --- per-organoid/single-cell scatter, one plot faceted by one of
# patient/treatment and colored by the other (never both faceted -- ~20
# treatments made for cramped, hard-to-read facets when combined with
# per-patient paging, and free_x scales per facet don't help when the facet
# itself is the problem). Y axis is log10-scaled (raw volume spans several
# orders of magnitude); x axis is free per facet. Title is split across two
# lines and x-axis tick labels angled so they don't overlap. Legend points
# are drawn larger and fully opaque (override.aes) so they stay legible even
# though the plotted points themselves are semi-transparent (alpha = 0.7) to
# show overplotting density. ---
make_faceted_scatter <- function(df, x_col, y_col, facet_col, color_col, color_values, legend_name,
                                  title, x_lab, y_lab, point_size, point_alpha = 0.7) {
    (
        ggplot(df, aes(x = .data[[x_col]], y = .data[[y_col]], color = .data[[color_col]]))
        + rasterise(geom_point(size = point_size, alpha = point_alpha), dpi = 300)
        + facet_wrap(as.formula(paste("~", facet_col)), scales = "free_x")
        + scale_y_log10()
        + scale_color_manual(values = color_values, name = legend_name)
        + guides(color = guide_legend(override.aes = list(size = 3, alpha = 1)))
        + labs(title = title, x = x_lab, y = y_lab)
        + theme_manuscript(x_text = "angled")
    )
}

p_vol_by_patient <- make_faceted_scatter(
    joined_3d, "Metadata_Object_OrganoidSingleCellCount", "Organoid_NoChannel_VolumeSizeShape_Volume",
    facet_col = "Metadata_patient_tumor", color_col = "Metadata_treatment",
    color_values = custom_treatment_palette, legend_name = "Treatment",
    title = "3D: organoid volume vs. cells per organoid\nfaceted by patient, colored by treatment",
    x_lab = "Cells per organoid", y_lab = "Organoid volume (log10 scale)",
    point_size = 1
)

p_vol_by_treatment <- make_faceted_scatter(
    joined_3d, "Metadata_Object_OrganoidSingleCellCount", "Organoid_NoChannel_VolumeSizeShape_Volume",
    facet_col = "Metadata_treatment", color_col = "Metadata_patient_tumor",
    color_values = tab20_palette_for_patients, legend_name = "Patient",
    title = "3D: organoid volume vs. cells per organoid\nfaceted by treatment, colored by patient",
    x_lab = "Cells per organoid", y_lab = "Organoid volume (log10 scale)",
    point_size = 1
)

# --- single-cell volume vs. the same FOV-normalized count ---
sc_vol_rows <- list()
for (patient in patients_3d) {
    f <- file.path(root_dir, "data", "profiles_3D", patient, "4.qc_profiles", "sc_flagged_outliers.parquet")
    if (!file.exists(f)) next
    df <- read_parquet(f, col_select = c("Metadata_Experiment_Treatment", "Metadata_Experiment_Dose", "Cell_NoChannel_VolumeSizeShape_Volume"))
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

p_sc_vol_by_patient <- make_faceted_scatter(
    joined_sc, "total_cell_count_norm", "Cell_NoChannel_VolumeSizeShape_Volume",
    facet_col = "Metadata_patient_tumor", color_col = "Metadata_treatment",
    color_values = custom_treatment_palette, legend_name = "Treatment",
    title = "3D: single-cell volume vs. total cell count (FOV-normalized)\nfaceted by patient, colored by treatment",
    x_lab = "Total cells per treatment (FOV-normalized)", y_lab = "Cell volume (log10 scale)",
    point_size = 0.5
)

p_sc_vol_by_treatment <- make_faceted_scatter(
    joined_sc, "total_cell_count_norm", "Cell_NoChannel_VolumeSizeShape_Volume",
    facet_col = "Metadata_treatment", color_col = "Metadata_patient_tumor",
    color_values = tab20_palette_for_patients, legend_name = "Patient",
    title = "3D: single-cell volume vs. total cell count (FOV-normalized)\nfaceted by treatment, colored by patient",
    x_lab = "Total cells per treatment (FOV-normalized)", y_lab = "Cell volume (log10 scale)",
    point_size = 0.5
)

pdf(file.path(figures_dir, "3D_volume_vs_count.pdf"), width = 14, height = 10, onefile = TRUE)
print(p_vol_by_patient)
print(p_vol_by_treatment)
print(p_sc_vol_by_patient)
print(p_sc_vol_by_treatment)
dev.off()
