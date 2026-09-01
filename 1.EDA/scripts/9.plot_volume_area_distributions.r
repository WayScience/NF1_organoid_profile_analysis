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

figures_dir <- file.path(root_dir, "1.EDA", "figures", "volume_area")
dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)

plot_theme <- (
    theme_bw()
    + theme(
        plot.title = element_text(hjust = 0.5, size = 14),
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14),
        axis.text.x = element_text(size = 9, angle = 45, hjust = 1),
        axis.text.y = element_text(size = 10),
        legend.position = "none"
    )
)

excluded_patients <- c("all_patients", "NF0037_T1_CQ1")
patients_2d <- setdiff(list.dirs(file.path(root_dir, "data", "profiles_2D"), recursive = FALSE, full.names = FALSE), excluded_patients)
patients_3d <- setdiff(list.dirs(file.path(root_dir, "data", "profiles_3D"), recursive = FALSE, full.names = FALSE), excluded_patients)

projection_prefix <- c(max_projection = "max_projected", middle_slice = "middle_slice", middle_n_slice = "middle_n_slice")

# --- 2D: organoid Area, by patient x projection ---
area_rows <- list()
for (proj in names(projection_prefix)) {
    prefix <- projection_prefix[[proj]]
    for (patient in patients_2d) {
        f <- file.path(root_dir, "data", "profiles_2D", patient, "5.normalized", paste0(prefix, "_organoid.parquet"))
        if (!file.exists(f)) next
        df <- read_parquet(f, col_select = c("Metadata_treatment", "Organoid_AreaShape_Area"))
        df$patient <- patient
        df$projection <- proj
        area_rows[[paste(proj, patient)]] <- df
    }
}
area_df <- bind_rows(area_rows)
area_df$Metadata_treatment <- factor(area_df$Metadata_treatment,
                                       levels = intersect(custom_treatment_order, unique(area_df$Metadata_treatment)))

p_area_patient <- (
    ggplot(area_df, aes(x = patient, y = Organoid_AreaShape_Area, fill = patient))
    + geom_violin(alpha = 0.6, trim = TRUE)
    + geom_boxplot(width = 0.15, alpha = 0.85, outlier.size = 0.2, outlier.alpha = 0.2)
    + facet_wrap(~projection, ncol = 1, scales = "free_y")
    + labs(
        title = "2D: organoid area by patient, by projection method",
        x = "Patient", y = "Organoid area (z-scored)"
    )
    + plot_theme
)
ggsave(
    filename = file.path(figures_dir, "2D_area_per_patient_by_projection.png"),
    plot = p_area_patient, width = 11, height = 14, dpi = 600, units = "in"
)

p_area_pooled <- (
    ggplot(area_df, aes(x = Metadata_treatment, y = Organoid_AreaShape_Area, fill = Metadata_treatment))
    + geom_violin(alpha = 0.6, trim = TRUE)
    + geom_boxplot(width = 0.15, alpha = 0.85, outlier.size = 0.2, outlier.alpha = 0.2)
    + scale_fill_manual(values = custom_treatment_palette, na.value = "grey70")
    + facet_wrap(~projection, ncol = 1, scales = "free_y")
    + labs(
        title = "2D pooled (all patients): organoid area by treatment, by projection method",
        x = "Treatment", y = "Organoid area (z-scored)"
    )
    + plot_theme
)
ggsave(
    filename = file.path(figures_dir, "2D_area_pooled_by_treatment.png"),
    plot = p_area_pooled, width = 11, height = 14, dpi = 600, units = "in"
)

# --- 2D: single-cell Area, by patient x projection ---
sc_area_rows <- list()
for (proj in names(projection_prefix)) {
    prefix <- projection_prefix[[proj]]
    for (patient in patients_2d) {
        f <- file.path(root_dir, "data", "profiles_2D", patient, "5.normalized", paste0(prefix, "_sc.parquet"))
        if (!file.exists(f)) next
        df <- read_parquet(f, col_select = c("Metadata_treatment", "Cells_AreaShape_Area"))
        df$patient <- patient
        df$projection <- proj
        sc_area_rows[[paste(proj, patient)]] <- df
    }
}
sc_area_df <- bind_rows(sc_area_rows)
sc_area_df$Metadata_treatment <- factor(sc_area_df$Metadata_treatment,
                                          levels = intersect(custom_treatment_order, unique(sc_area_df$Metadata_treatment)))

p_sc_area_patient <- (
    ggplot(sc_area_df, aes(x = patient, y = Cells_AreaShape_Area, fill = patient))
    + geom_violin(alpha = 0.6, trim = TRUE)
    + geom_boxplot(width = 0.15, alpha = 0.85, outlier.size = 0.2, outlier.alpha = 0.2)
    + facet_wrap(~projection, ncol = 1, scales = "free_y")
    + labs(
        title = "2D: single-cell area by patient, by projection method",
        x = "Patient", y = "Cell area (z-scored)"
    )
    + plot_theme
)
ggsave(
    filename = file.path(figures_dir, "2D_sc_area_per_patient_by_projection.png"),
    plot = p_sc_area_patient, width = 11, height = 14, dpi = 600, units = "in"
)

p_sc_area_pooled <- (
    ggplot(sc_area_df, aes(x = Metadata_treatment, y = Cells_AreaShape_Area, fill = Metadata_treatment))
    + geom_violin(alpha = 0.6, trim = TRUE)
    + geom_boxplot(width = 0.15, alpha = 0.85, outlier.size = 0.2, outlier.alpha = 0.2)
    + scale_fill_manual(values = custom_treatment_palette, na.value = "grey70")
    + facet_wrap(~projection, ncol = 1, scales = "free_y")
    + labs(
        title = "2D pooled (all patients): single-cell area by treatment, by projection method",
        x = "Treatment", y = "Cell area (z-scored)"
    )
    + plot_theme
)
ggsave(
    filename = file.path(figures_dir, "2D_sc_area_pooled_by_treatment.png"),
    plot = p_sc_area_pooled, width = 11, height = 14, dpi = 600, units = "in"
)


# --- 3D: organoid Volume, by patient ---
vol_rows <- list()
for (patient in patients_3d) {
    f <- file.path(root_dir, "data", "profiles_3D", patient, "5.normalized_profiles", "organoid_norm.parquet")
    if (!file.exists(f)) next
    df <- read_parquet(f, col_select = c("Metadata_Experiment_Treatment", "Organoid_NoChannel_AreaSizeShape_Volume"))
    df$patient <- patient
    vol_rows[[patient]] <- df
}
vol_df <- bind_rows(vol_rows)
colnames(vol_df)[colnames(vol_df) == "Metadata_Experiment_Treatment"] <- "Metadata_treatment"
vol_df$Metadata_treatment <- factor(vol_df$Metadata_treatment,
                                      levels = intersect(custom_treatment_order, unique(vol_df$Metadata_treatment)))

p_vol_patient <- (
    ggplot(vol_df, aes(x = patient, y = Organoid_NoChannel_AreaSizeShape_Volume, fill = patient))
    + geom_violin(alpha = 0.6, trim = TRUE)
    + geom_boxplot(width = 0.15, alpha = 0.85, outlier.size = 0.2, outlier.alpha = 0.2)
    + labs(title = "3D: organoid volume by patient", x = "Patient", y = "Organoid volume (z-scored)")
    + plot_theme
)
ggsave(
    filename = file.path(figures_dir, "3D_volume_per_patient.png"),
    plot = p_vol_patient, width = 10, height = 6, dpi = 600, units = "in"
)

p_vol_pooled <- (
    ggplot(vol_df, aes(x = Metadata_treatment, y = Organoid_NoChannel_AreaSizeShape_Volume, fill = Metadata_treatment))
    + geom_violin(alpha = 0.6, trim = TRUE)
    + geom_boxplot(width = 0.15, alpha = 0.85, outlier.size = 0.2, outlier.alpha = 0.2)
    + scale_fill_manual(values = custom_treatment_palette, na.value = "grey70")
    + labs(
        title = "3D pooled (all patients): organoid volume by treatment",
        x = "Treatment", y = "Organoid volume (z-scored)"
    )
    + plot_theme
)
ggsave(
    filename = file.path(figures_dir, "3D_volume_pooled_by_treatment.png"),
    plot = p_vol_pooled, width = 10, height = 6, dpi = 600, units = "in"
)

p_vol_patient_treatment <- (
    ggplot(vol_df, aes(x = Metadata_treatment, y = Organoid_NoChannel_AreaSizeShape_Volume, fill = Metadata_treatment))
    + geom_violin(alpha = 0.6, trim = TRUE)
    + geom_boxplot(width = 0.15, alpha = 0.85, outlier.size = 0.2, outlier.alpha = 0.2)
    + scale_fill_manual(values = custom_treatment_palette, na.value = "grey70")
    + facet_wrap(~patient, scales = "free_y")
    + labs(
        title = "3D: organoid volume by treatment, faceted by patient",
        x = "Treatment", y = "Organoid volume (z-scored)"
    )
    + plot_theme
)
ggsave(
    filename = file.path(figures_dir, "3D_volume_by_patient_and_treatment.png"),
    plot = p_vol_patient_treatment, width = 16, height = 14, dpi = 600, units = "in"
)

cat("Wrote 5 figures to", figures_dir, "\n")

# --- 3D: single-cell Volume, by patient ---
sc_vol_rows <- list()
for (patient in patients_3d) {
    f <- file.path(root_dir, "data", "profiles_3D", patient, "5.normalized_profiles", "sc_norm.parquet")
    if (!file.exists(f)) next
    df <- read_parquet(f, col_select = c("Metadata_Experiment_Treatment", "Cell_NoChannel_AreaSizeShape_Volume"))
    df$patient <- patient
    sc_vol_rows[[patient]] <- df
}
sc_vol_df <- bind_rows(sc_vol_rows)
colnames(sc_vol_df)[colnames(sc_vol_df) == "Metadata_Experiment_Treatment"] <- "Metadata_treatment"
sc_vol_df$Metadata_treatment <- factor(sc_vol_df$Metadata_treatment,
                                         levels = intersect(custom_treatment_order, unique(sc_vol_df$Metadata_treatment)))

p_sc_vol_patient <- (
    ggplot(sc_vol_df, aes(x = patient, y = Cell_NoChannel_AreaSizeShape_Volume, fill = patient))
    + geom_violin(alpha = 0.6, trim = TRUE)
    + geom_boxplot(width = 0.15, alpha = 0.85, outlier.size = 0.2, outlier.alpha = 0.2)
    + labs(title = "3D: single-cell volume by patient", x = "Patient", y = "Cell volume (z-scored)")
    + plot_theme
)
ggsave(
    filename = file.path(figures_dir, "3D_sc_volume_per_patient.png"),
    plot = p_sc_vol_patient, width = 10, height = 6, dpi = 600, units = "in"
)

p_sc_vol_pooled <- (
    ggplot(sc_vol_df, aes(x = Metadata_treatment, y = Cell_NoChannel_AreaSizeShape_Volume, fill = Metadata_treatment))
    + geom_violin(alpha = 0.6, trim = TRUE)
    + geom_boxplot(width = 0.15, alpha = 0.85, outlier.size = 0.2, outlier.alpha = 0.2)
    + scale_fill_manual(values = custom_treatment_palette, na.value = "grey70")
    + labs(
        title = "3D pooled (all patients): single-cell volume by treatment",
        x = "Treatment", y = "Cell volume (z-scored)"
    )
    + plot_theme
)
ggsave(
    filename = file.path(figures_dir, "3D_sc_volume_pooled_by_treatment.png"),
    plot = p_sc_vol_pooled, width = 10, height = 6, dpi = 600, units = "in"
)

p_sc_vol_patient_treatment <- (
    ggplot(sc_vol_df, aes(x = Metadata_treatment, y = Cell_NoChannel_AreaSizeShape_Volume, fill = Metadata_treatment))
    + geom_violin(alpha = 0.6, trim = TRUE)
    + geom_boxplot(width = 0.15, alpha = 0.85, outlier.size = 0.2, outlier.alpha = 0.2)
    + scale_fill_manual(values = custom_treatment_palette, na.value = "grey70")
    + facet_wrap(~patient, scales = "free_y")
    + labs(
        title = "3D: single-cell volume by treatment, faceted by patient",
        x = "Treatment", y = "Cell volume (z-scored)"
    )
    + plot_theme
)
ggsave(
    filename = file.path(figures_dir, "3D_sc_volume_by_patient_and_treatment.png"),
    plot = p_sc_vol_patient_treatment, width = 16, height = 14, dpi = 600, units = "in"
)

cat("Wrote 5 single-cell figures to", figures_dir, "\n")

