packages <- c("ggplot2", "dplyr", "arrow", "ComplexHeatmap", "circlize", "scales", "RColorBrewer")
for (pkg in packages) {
    suppressPackageStartupMessages(
        suppressWarnings(
            library(pkg, character.only = TRUE)
        )
    )
}

# Get the current working directory and find Git root
find_git_root <- function() {
    # Get current working directory
    cwd <- getwd()

    # Check if current directory has .git
    if (dir.exists(file.path(cwd, ".git"))) {
        return(cwd)
    }

    # If not, search parent directories
    current_path <- cwd
    while (dirname(current_path) != current_path) {  # While not at root
        parent_path <- dirname(current_path)
        if (dir.exists(file.path(parent_path, ".git"))) {
            return(parent_path)
        }
        current_path <- parent_path
    }

    # If no Git root found, stop with error
    stop("No Git root directory found.")
}

# Find the Git root directory
root_dir <- find_git_root()
cat("Git root directory:", root_dir, "\n")
source(file.path(root_dir, "utils", "r_plot_themes.r"))

correlation_dir <- file.path(root_dir, "1.EDA", "results", "correlation")
figures_dir <- file.path(root_dir, "1.EDA", "figures", "correlation_heatmaps")
dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)

correlation_files <- sort(list.files(correlation_dir, full.names = TRUE))
length(correlation_files)

build_correlation_heatmap <- function(file_path) {
    df <- arrow::read_parquet(file_path)

    metadata_cols <- grep("^Metadata_", colnames(df), value = TRUE)
    sample_cols <- grep("^Sample_", colnames(df), value = TRUE)

    mat <- as.matrix(df[, sample_cols])
    dimnames(mat) <- NULL

    treatment_col <- metadata_cols[grepl("treatment", metadata_cols, ignore.case = TRUE)]
    dose_col <- metadata_cols[grepl("dose", metadata_cols, ignore.case = TRUE) &
        !grepl("unit", metadata_cols, ignore.case = TRUE)]
    patient_col <- metadata_cols[grepl("patient|tumor", metadata_cols, ignore.case = TRUE)]

    annotations <- list()

    if (length(treatment_col) >= 1) {
        annotations$Treatment <- df[[treatment_col[1]]]
    }

    if (length(patient_col) >= 1) {
        patient_values <- as.character(df[[patient_col[1]]])
        unique_patients <- sort(unique(patient_values))
        patient_colors <- setNames(
            tab20_palette_for_patients[seq_along(unique_patients)],
            unique_patients
        )
        annotations$Patient <- patient_values
    }

    top_annotation <- NULL
    left_annotation <- NULL
    if (length(annotations) > 0) {
        anno_cols <- list()
        if (!is.null(annotations$Treatment)) {
            anno_cols$Treatment <- custom_treatment_palette
        }
        if (!is.null(annotations$Patient)) {
            anno_cols$Patient <- patient_colors
        }
        top_annotation <- do.call(HeatmapAnnotation, c(
            annotations,
            list(
                col = anno_cols,
                show_legend = c(Treatment = TRUE, Patient = TRUE)[names(annotations)],
                annotation_legend_param = list(
                    Treatment = list(title = "Treatment", ncol = 1, labels_gp = gpar(fontsize = 8))
                ),
                annotation_name_side = "left",
                annotation_name_gp = gpar(fontsize = 10)
            )
        ))

        left_annotation <- do.call(rowAnnotation, c(
            annotations,
            list(
                col = anno_cols,
                show_legend = FALSE,
                show_annotation_name = FALSE
            )
        ))
    }

    col_fun <- colorRamp2(c(-1, 0, 1), c("#2166AC", "white", "#B2182B"))

    # A handful of samples have zero-variance feature vectors, giving NA
    # Pearson correlations for those rows/columns. Rather than disabling
    # clustering entirely, treat NA as "no correlation" (0) when computing
    # distances for hclust -- the plotted cells still show the true NA
    # (grey80) since we only touch a local copy for distance calculation.
    na_safe_dist <- function(m) {
        m[is.na(m)] <- 0
        stats::as.dist(1 - m)
    }

    heatmap_plot <- Heatmap(
        mat,
        name = "Corr",
        col = col_fun,
        na_col = "grey80",
        show_row_names = FALSE,
        show_column_names = FALSE,
        show_row_dend = FALSE,
        show_column_dend = TRUE,
        cluster_rows = TRUE,
        cluster_columns = TRUE,
        clustering_distance_rows = na_safe_dist,
        clustering_distance_columns = na_safe_dist,
        use_raster = TRUE,
        raster_quality = 2,
        top_annotation = top_annotation,
        left_annotation = left_annotation,
        column_title = tools::file_path_sans_ext(basename(file_path)),
        column_title_gp = gpar(fontsize = 10, fontface = "bold"),
        heatmap_legend_param = list(title = "Pearson\ncorrelation")
    )

    list(heatmap = heatmap_plot, n_samples = nrow(mat))
}

save_heatmaps_pdf <- function(files, output_path) {
    # Build every heatmap in the group up front so a single shared page
    # size (sized to the largest matrix) can be picked before opening the
    # PDF device -- a single pdf() device can't vary its page size per page.
    built <- lapply(files, build_correlation_heatmap)
    max_n <- max(vapply(built, function(b) b$n_samples, numeric(1)))
    width <- max(8, min(24, max_n / 40 + 3))
    height <- max(6, min(20, max_n / 40))

    pdf(output_path, width = width, height = height)
    on.exit(dev.off(), add = TRUE)
    for (b in built) {
        draw(b$heatmap, merge_legend = TRUE)
    }
}

dimensions <- c("2D", "3D")
for (dimension in dimensions) {
    dimension_files <- correlation_files[startsWith(basename(correlation_files), paste0(dimension, "_"))]
    if (length(dimension_files) == 0) {
        next
    }
    pdf_path <- file.path(figures_dir, paste0(dimension, "_all_patients_correlation_heatmaps.pdf"))
    save_heatmaps_pdf(dimension_files, pdf_path)
    cat("Saved:", pdf_path, "(", length(dimension_files), "pages)\n")
}
