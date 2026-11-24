packages <- c("dplyr", "arrow", "ComplexHeatmap", "circlize", "grid")
for (pkg in packages) {
    suppressPackageStartupMessages(
        suppressWarnings(
            library(pkg, character.only = TRUE)
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

# Panel B of Figure 3: the clustered all-patients heatmap of 3D single-cell consensus
# profiles (sc_norm, consensus from feature-selected cells), drawn at the panel's own
# size so its text stays legible in the assembled figure. Annotations, palettes,
# cleaning and the colour cap follow 1.EDA/scripts/19.plot_profile_heatmaps.r.
consensus_path <- file.path(
    root_dir, "data", "profiles_3D", "all_patients", "3.consensus_profiles",
    "sc_norm_consensus_from_fs.parquet"
)
# (patient, tumor type) pairs; the consensus profiles do not carry tumor type
umap_path <- file.path(root_dir, "1.EDA", "results", "umap", "3D_scfs_umap.parquet")
figures_dir <- file.path(root_dir, "figures", "assembled_figure_3", "figures")
heatmap_panel_path <- file.path(figures_dir, "heatmap_panel.pdf")
dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)

# Text sizes (pt) at the panel's printed size; page size in inches
FONT_TITLE <- 22
FONT_ANNOTATION <- 18
FONT_LEGEND_TITLE <- 17
FONT_LEGEND <- 15
PAGE_WIDTH <- 20
PAGE_HEIGHT <- 10.5

tumor_pairs <- unique(arrow::read_parquet(
    umap_path, col_select = c("Metadata_Biology_PatientTumor", "Metadata_Biology_TumorType")
))
tumor_type_map <- setNames(tumor_pairs$Metadata_Biology_TumorType, tumor_pairs$Metadata_Biology_PatientTumor)

df <- arrow::read_parquet(consensus_path)
# NF0037_T1_CQ1 is a separate analysis; samples without a treatment are an incomplete
# metadata join, not a condition (both as in 19.plot_profile_heatmaps.r)
df <- df[df$Metadata_Biology_PatientTumor != "NF0037_T1_CQ1" & !is.na(df$Metadata_Experiment_Treatment), ]
metadata_columns <- grep("^Metadata_", colnames(df), value = TRUE)
feature_columns <- setdiff(colnames(df), metadata_columns)
feature_columns <- feature_columns[!vapply(df[feature_columns], function(x) all(is.na(x)), logical(1))]
df <- df[stats::complete.cases(df[, feature_columns]), ]

metadata <- df[, metadata_columns]
metadata$Metadata_Biology_TumorType <- tumor_type_map[metadata$Metadata_Biology_PatientTumor]
missing_dose <- is.na(metadata$Metadata_Experiment_Dose) | is.na(metadata$Metadata_Experiment_Unit)
metadata$dose_label <- ifelse(
    missing_dose, "None", paste0(metadata$Metadata_Experiment_Dose, metadata$Metadata_Experiment_Unit)
)
metadata$dose_label <- ifelse(metadata$dose_label == "1%", "1% (DMSO only)", metadata$dose_label)

# 3D feature names: <Compartment>_<Channel>_<FeatureType>_<Measurement...>; a
# colocalization pair ("AGP-Mito") is coloured by its first channel
parts <- strsplit(feature_columns, "_")
feature_annotations <- data.frame(
    compartment = vapply(parts, `[`, character(1), 1),
    channel = vapply(strsplit(vapply(parts, `[`, character(1), 2), "-"), `[`, character(1), 1),
    feature_type = vapply(parts, `[`, character(1), 3)
)
feature_annotations$feature_type[feature_annotations$feature_type == "VolumeSizeShape"] <- "AreaSizeShape"

lookup_colors <- function(labels, palette, fallback = "#999999") {
    labels <- sort(unique(labels))
    colors <- setNames(rep(fallback, length(labels)), labels)
    known <- intersect(labels, names(palette))
    colors[known] <- palette[known]
    colors
}

legend_param <- function(title, ncol = 1) {
    list(
        title = title, ncol = ncol,
        title_gp = gpar(fontsize = FONT_LEGEND_TITLE, fontface = "bold"),
        labels_gp = gpar(fontsize = FONT_LEGEND),
        grid_height = unit(5.5, "mm"), grid_width = unit(5.5, "mm")
    )
}

# legend of a discrete annotation: only the levels present, sorted, as ComplexHeatmap
# would draw it
annotation_legend <- function(title, values, colors, ncol = 1) {
    levels_present <- sort(unique(values))
    do.call(Legend, c(
        list(labels = levels_present, legend_gp = gpar(fill = unname(colors[levels_present]))),
        legend_param(title, ncol)
    ))
}

# patients named with their tumour type, e.g. "SARCO219_T2 (MPNST)"
patients <- sort(unique(metadata$Metadata_Biology_PatientTumor))
patient_labels <- setNames(paste0(patients, " (", tumor_type_map[patients], ")"), patients)
metadata$patient_label <- unname(patient_labels[metadata$Metadata_Biology_PatientTumor])
top_colors <- list(
    Treatment = custom_treatment_palette,
    Dose = lookup_colors(metadata$dose_label, dose_label_palette),
    Patient = setNames(tab20_palette_for_patients[seq_along(patients)], patient_labels),
    `Tumor type` = tumor_type_palette
)
right_colors <- list(
    Compartment = lookup_colors(
        feature_annotations$compartment, c(sc_compartment_palette, organoid_compartment_palette)
    ),
    `Feature type` = lookup_colors(feature_annotations$feature_type, feature_type_palette),
    Channel = lookup_colors(feature_annotations$channel, channel_palette)
)

# legends are drawn separately (see Save), packed into two balanced columns
top_annotation <- HeatmapAnnotation(
    Treatment = metadata$Metadata_Experiment_Treatment,
    Dose = metadata$dose_label,
    Patient = metadata$patient_label,
    `Tumor type` = metadata$Metadata_Biology_TumorType,
    col = top_colors,
    show_legend = FALSE,
    annotation_name_side = "left",
    annotation_name_gp = gpar(fontsize = FONT_ANNOTATION),
    simple_anno_size = unit(6, "mm")
)

right_annotation <- rowAnnotation(
    Compartment = feature_annotations$compartment,
    `Feature type` = feature_annotations$feature_type,
    Channel = feature_annotations$channel,
    col = right_colors,
    show_legend = FALSE,
    annotation_name_side = "bottom",
    annotation_name_rot = 90,
    annotation_name_gp = gpar(fontsize = FONT_ANNOTATION),
    simple_anno_size = unit(6, "mm")
)


# features are rows, samples columns; colour scale capped at the 99th percentile
# absolute value so a few outliers do not wash out the rest
mat <- t(as.matrix(df[, feature_columns]))
cap <- quantile(abs(mat), 0.99, na.rm = TRUE)
value_col_fun <- colorRamp2(c(-cap, 0, cap), c("#2166AC", "white", "#B2182B"))
na_safe_dist <- function(m) {
    m[is.na(m)] <- 0
    stats::dist(m)
}

heatmap <- Heatmap(
    mat,
    name = "Value",
    col = value_col_fun,
    show_heatmap_legend = FALSE,
    na_col = "grey80",
    show_row_names = FALSE,
    show_column_names = FALSE,
    clustering_distance_rows = na_safe_dist,
    clustering_distance_columns = na_safe_dist,
    row_dend_width = unit(18, "mm"),
    column_dend_height = unit(18, "mm"),
    use_raster = TRUE,
    raster_quality = 1,
    top_annotation = top_annotation,
    right_annotation = right_annotation,
    column_title = "3D single-cell consensus profiles (all patients)",
    column_title_gp = gpar(fontsize = FONT_TITLE, fontface = "bold")
)

# Legends in two columns of similar height, so neither leaves a block of white space:
# the sample annotations (left), then the colour scale and feature annotations (right)
legend_gap <- unit(5, "mm")
legends <- packLegend(
    packLegend(
        annotation_legend("Treatment", metadata$Metadata_Experiment_Treatment, top_colors$Treatment, ncol = 2),
        annotation_legend("Dose", metadata$dose_label, top_colors$Dose),
        annotation_legend("Patient", metadata$patient_label, top_colors$Patient, ncol = 2),
        annotation_legend("Tumor type", metadata$Metadata_Biology_TumorType, top_colors$`Tumor type`),
        direction = "vertical", gap = legend_gap
    ),
    packLegend(
        do.call(Legend, c(
            list(col_fun = value_col_fun, legend_height = unit(35, "mm")),
            legend_param("Normalized\nfeature value")
        )),
        annotation_legend("Compartment", feature_annotations$compartment, right_colors$Compartment),
        annotation_legend("Feature type", feature_annotations$feature_type, right_colors$`Feature type`),
        annotation_legend("Channel", feature_annotations$channel, right_colors$Channel),
        direction = "vertical", gap = legend_gap
    ),
    direction = "horizontal", column_gap = unit(8, "mm")
)

pdf(heatmap_panel_path, width = PAGE_WIDTH, height = PAGE_HEIGHT)
# extra bottom padding holds the row-annotation names
draw(
    heatmap,
    heatmap_legend_list = list(legends),
    padding = unit(c(2, 2, 2, 2), "mm")
)
invisible(dev.off())
