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

# --- helpers shared by every distribution plot below ---

# Read one parquet per (projection, patient), tagging each with its source.
load_2d_profiles <- function(patients, projections, path_fn, col_select) {
    rows <- list()
    for (proj in names(projections)) {
        prefix <- projections[[proj]]
        for (patient in patients) {
            f <- path_fn(patient, prefix)
            if (!file.exists(f)) next
            df <- read_parquet(f, col_select = all_of(col_select))
            df$patient <- patient
            df$projection <- proj
            rows[[paste(proj, patient)]] <- df
        }
    }
    bind_rows(rows)
}

# Read one parquet per patient, tagging each with its source.
load_3d_profiles <- function(patients, path_fn, col_select) {
    rows <- list()
    for (patient in patients) {
        f <- path_fn(patient)
        if (!file.exists(f)) next
        df <- read_parquet(f, col_select = all_of(col_select))
        df$patient <- patient
        rows[[patient]] <- df
    }
    bind_rows(rows)
}

# Order a treatment column by custom_treatment_order, unseen levels appended.
set_treatment_factor <- function(df, col = "Metadata_treatment") {
    df[[col]] <- factor(
        df[[col]],
        levels = c(
            intersect(custom_treatment_order, unique(df[[col]])),
            setdiff(unique(df[[col]]), custom_treatment_order)
        )
    )
    df
}

# The violin + boxplot layout every panel in this notebook uses.
violin_box_plot <- function(df, x, y, fill, title, ylab, xlab = NULL,
                             facet = NULL, facet_ncol = NULL, facet_scales = "free_y",
                             palette = NULL, log_scale = FALSE) {
    p <- (
        ggplot(df, aes(x = .data[[x]], y = .data[[y]], fill = .data[[fill]]))
        + geom_violin(alpha = 0.6, trim = TRUE)
        + geom_boxplot(width = 0.15, alpha = 0.85, outlier.size = 0.2, outlier.alpha = 0.2)
        + labs(title = title, x = if (is.null(xlab)) x else xlab, y = ylab)
        + plot_theme
    )
    if (!is.null(palette)) {
        p <- p + scale_fill_manual(values = palette, na.value = "grey70")
    }
    if (!is.null(facet)) {
        p <- p + facet_wrap(as.formula(paste("~", facet)), ncol = facet_ncol, scales = facet_scales)
    }
    if (log_scale) {
        p <- p + scale_y_log10()
    }
    p
}

save_fig <- function(plot, filename, width, height) {
    ggsave(
        filename = file.path(figures_dir, filename),
        plot = plot, width = width, height = height, dpi = 300, units = "in"
    )
}

# --- 2D: organoid Area, by patient x projection ---
area_df <- load_2d_profiles(
    patients_2d, projection_prefix,
    path_fn = function(patient, prefix) {
        file.path(root_dir, "data", "profiles_2D", patient, "4.annotated", paste0(prefix, "_organoid.parquet"))
    },
    col_select = c("Metadata_treatment", "Organoid_AreaShape_Area")
)
area_df <- set_treatment_factor(area_df)

p_area_patient <- violin_box_plot(
    area_df, x = "patient", y = "Organoid_AreaShape_Area", fill = "patient",
    title = "2D: organoid area by patient, by projection method",
    ylab = "Organoid area (log10 scale)", xlab = "Patient",
    facet = "projection", facet_ncol = 1, log_scale = TRUE
)
save_fig(p_area_patient, "2D_area_per_patient_by_projection.png", width = 11, height = 14)

p_area_pooled <- violin_box_plot(
    area_df, x = "Metadata_treatment", y = "Organoid_AreaShape_Area", fill = "Metadata_treatment",
    title = "2D pooled (all patients): organoid area by treatment, by projection method",
    ylab = "Organoid area (log10 scale)", xlab = "Treatment",
    facet = "projection", facet_ncol = 1, palette = custom_treatment_palette, log_scale = TRUE
)
save_fig(p_area_pooled, "2D_area_pooled_by_treatment.png", width = 11, height = 14)

# --- 2D: single-cell Area, by patient x projection ---
sc_area_df <- load_2d_profiles(
    patients_2d, projection_prefix,
    path_fn = function(patient, prefix) {
        file.path(root_dir, "data", "profiles_2D", patient, "4.annotated", paste0(prefix, "_sc.parquet"))
    },
    col_select = c("Metadata_treatment", "Cells_AreaShape_Area")
)
sc_area_df <- set_treatment_factor(sc_area_df)

p_sc_area_patient <- violin_box_plot(
    sc_area_df, x = "patient", y = "Cells_AreaShape_Area", fill = "patient",
    title = "2D: single-cell area by patient, by projection method",
    ylab = "Cell area (log10 scale)", xlab = "Patient",
    facet = "projection", facet_ncol = 1, log_scale = TRUE
)
save_fig(p_sc_area_patient, "2D_sc_area_per_patient_by_projection.png", width = 11, height = 14)

p_sc_area_pooled <- violin_box_plot(
    sc_area_df, x = "Metadata_treatment", y = "Cells_AreaShape_Area", fill = "Metadata_treatment",
    title = "2D pooled (all patients): single-cell area by treatment, by projection method",
    ylab = "Cell area (log10 scale)", xlab = "Treatment",
    facet = "projection", facet_ncol = 1, palette = custom_treatment_palette, log_scale = TRUE
)
save_fig(p_sc_area_pooled, "2D_sc_area_pooled_by_treatment.png", width = 11, height = 14)

# --- 3D: organoid Volume, by patient ---
vol_df <- load_3d_profiles(
    patients_3d,
    path_fn = function(patient) {
        file.path(root_dir, "data", "profiles_3D", patient, "4.qc_profiles", "organoid_flagged_outliers.parquet")
    },
    col_select = c("Metadata_Experiment_Treatment", "Organoid_NoChannel_VolumeSizeShape_Volume")
)
colnames(vol_df)[colnames(vol_df) == "Metadata_Experiment_Treatment"] <- "Metadata_treatment"
vol_df <- set_treatment_factor(vol_df)

p_vol_patient <- violin_box_plot(
    vol_df, x = "patient", y = "Organoid_NoChannel_VolumeSizeShape_Volume", fill = "patient",
    title = "3D: organoid volume by patient", ylab = "Organoid volume (log10 scale)", xlab = "Patient",
    log_scale = TRUE
)
save_fig(p_vol_patient, "3D_volume_per_patient.png", width = 10, height = 6)

p_vol_pooled <- violin_box_plot(
    vol_df, x = "Metadata_treatment", y = "Organoid_NoChannel_VolumeSizeShape_Volume", fill = "Metadata_treatment",
    title = "3D pooled (all patients): organoid volume by treatment",
    ylab = "Organoid volume (log10 scale)", xlab = "Treatment", palette = custom_treatment_palette,
    log_scale = TRUE
)
save_fig(p_vol_pooled, "3D_volume_pooled_by_treatment.png", width = 10, height = 6)

p_vol_patient_treatment <- violin_box_plot(
    vol_df, x = "Metadata_treatment", y = "Organoid_NoChannel_VolumeSizeShape_Volume", fill = "Metadata_treatment",
    title = "3D: organoid volume by treatment, faceted by patient",
    ylab = "Organoid volume (log10 scale)", xlab = "Treatment",
    facet = "patient", palette = custom_treatment_palette, log_scale = TRUE
)
save_fig(p_vol_patient_treatment, "3D_volume_by_patient_and_treatment.png", width = 16, height = 14)

# --- 3D: single-cell Volume, by patient ---
sc_vol_df <- load_3d_profiles(
    patients_3d,
    path_fn = function(patient) {
        file.path(root_dir, "data", "profiles_3D", patient, "4.qc_profiles", "sc_flagged_outliers.parquet")
    },
    col_select = c("Metadata_Experiment_Treatment", "Cell_NoChannel_VolumeSizeShape_Volume")
)
colnames(sc_vol_df)[colnames(sc_vol_df) == "Metadata_Experiment_Treatment"] <- "Metadata_treatment"
sc_vol_df <- set_treatment_factor(sc_vol_df)

p_sc_vol_patient <- violin_box_plot(
    sc_vol_df, x = "patient", y = "Cell_NoChannel_VolumeSizeShape_Volume", fill = "patient",
    title = "3D: single-cell volume by patient", ylab = "Cell volume (log10 scale)", xlab = "Patient",
    log_scale = TRUE
)
save_fig(p_sc_vol_patient, "3D_sc_volume_per_patient.png", width = 10, height = 6)

p_sc_vol_pooled <- violin_box_plot(
    sc_vol_df, x = "Metadata_treatment", y = "Cell_NoChannel_VolumeSizeShape_Volume", fill = "Metadata_treatment",
    title = "3D pooled (all patients): single-cell volume by treatment",
    ylab = "Cell volume (log10 scale)", xlab = "Treatment", palette = custom_treatment_palette,
    log_scale = TRUE
)
save_fig(p_sc_vol_pooled, "3D_sc_volume_pooled_by_treatment.png", width = 10, height = 6)

p_sc_vol_patient_treatment <- violin_box_plot(
    sc_vol_df, x = "Metadata_treatment", y = "Cell_NoChannel_VolumeSizeShape_Volume", fill = "Metadata_treatment",
    title = "3D: single-cell volume by treatment, faceted by patient",
    ylab = "Cell volume (log10 scale)", xlab = "Treatment",
    facet = "patient", palette = custom_treatment_palette, log_scale = TRUE
)
save_fig(p_sc_vol_patient_treatment, "3D_sc_volume_by_patient_and_treatment.png", width = 16, height = 14)
