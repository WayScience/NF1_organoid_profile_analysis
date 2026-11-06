# Load libraries
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

if (!requireNamespace("ComplexHeatmap", quietly = TRUE)) {
    BiocManager::install("ComplexHeatmap")
}
suppressPackageStartupMessages({
    library(arrow)
    library(ComplexHeatmap)
    library(circlize)
    library(grid)
})

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
    stop("No Git root directory found.")
}

# Find the Git root directory
root_dir <- find_git_root()

results_dir <- file.path(root_dir, "2.2d_vs_3d_analysis", "results", "correlation")
manifest_path <- file.path(results_dir, "correlation_manifest.parquet")
figures_dir <- file.path(root_dir, "2.2d_vs_3d_analysis", "figures", "correlation")
pdf_path <- file.path(figures_dir, "max_projection_pearson_correlations.pdf")

dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)

# Which comparisons from the manifest to plot. NULL keeps every value.
projections_to_plot <- "max_projection"
profile_types_to_plot <- NULL
cohorts_to_plot <- NULL

# Number of features per block when clustering; see correlation_distances()
distance_block_size <- 256L

projection_labels <- c(
    max_projection = "Max projection",
    middle_slice = "Middle slice",
    middle_n_slice = "Middle N slices"
)

build_correlation_matrix <- function(corr_long_df) {
    #' Pivot long-form correlations into a 2D x 3D matrix.
    #'
    #' @param corr_long_df data.frame - Long-form data with feature_2d, feature_3d, pearson_r columns.
    #'
    #' @return matrix - Pearson correlations with 2D features as rows and 3D features as columns.

    features_2d <- sort(unique(corr_long_df$feature_2d))
    features_3d <- sort(unique(corr_long_df$feature_3d))

    # These tables run to millions of rows, so the matrix is filled by cell
    # index in one shot instead of pivoting row by row
    corr_matrix <- matrix(
        NA_real_,
        nrow = length(features_2d),
        ncol = length(features_3d),
        dimnames = list(features_2d, features_3d)
    )
    corr_matrix[
        match(corr_long_df$feature_2d, features_2d) +
            (match(corr_long_df$feature_3d, features_3d) - 1L) * length(features_2d)
    ] <- corr_long_df$pearson_r

    # Drop features left with nothing to show; partial gaps stay blank
    corr_matrix[
        rowSums(is.finite(corr_matrix)) > 0L,
        colSums(is.finite(corr_matrix)) > 0L,
        drop = FALSE
    ]
}

correlation_distances <- function(corr_matrix, block_size = distance_block_size) {
    #' Calculate Euclidean distances between the rows of a correlation matrix.
    #'
    #' dist() would build the full pairwise matrix at once, which does not fit
    #' for the larger profile types (thousands of features), so the distances
    #' are accumulated from matrix products over blocks of rows instead. Every
    #' feature is kept.
    #'
    #' @param corr_matrix matrix - Correlation matrix to calculate row distances for.
    #' @param block_size integer - Number of rows per block.
    #'
    #' @return dist - Lower-triangle distances, as returned by dist().

    n_features <- nrow(corr_matrix)
    squared_norms <- rowSums(corr_matrix * corr_matrix)
    distances <- numeric(n_features * (n_features - 1) / 2)
    offset <- 0L

    for (start in seq.int(1L, n_features - 1L, by = block_size)) {
        block <- seq.int(start, min(start + block_size - 1L, n_features - 1L))
        products <- tcrossprod(corr_matrix, corr_matrix[block, , drop = FALSE])

        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2ab, clamped at 0 for rounding
        for (j in seq_along(block)) {
            column <- block[j]
            rows <- seq.int(column + 1L, n_features)
            distances[seq.int(offset + 1L, offset + length(rows))] <- sqrt(pmax(
                squared_norms[rows] + squared_norms[column] - 2 * products[rows, j], 0
            ))
            offset <- offset + length(rows)
        }
    }

    structure(
        distances,
        Size = n_features,
        Labels = rownames(corr_matrix),
        Diag = FALSE,
        Upper = FALSE,
        method = "euclidean",
        class = "dist"
    )
}

cluster_correlation_matrix <- function(corr_matrix) {
    #' Cluster the rows of a correlation matrix with Ward linkage.
    #'
    #' Passed to Heatmap() for both rows and columns, which hands this the
    #' matrix and its transpose in turn.
    #'
    #' @param corr_matrix matrix - Correlation matrix to cluster the rows of.
    #'
    #' @return hclust - Row clustering.

    # Blanked correlations are treated as 0 for ordering only; they stay blank
    # in the plot itself
    corr_matrix[!is.finite(corr_matrix)] <- 0
    hclust(correlation_distances(corr_matrix), method = "ward.D2")
}

plot_correlation_heatmap <- function(corr_long_df, comparison) {
    #' Draw a clustered heatmap of Pearson correlations between 2D and 3D
    #' features onto the open graphics device.
    #'
    #' @param corr_long_df data.frame - Long-form data with feature_2d, feature_3d, pearson_r columns.
    #' @param comparison data.frame - One manifest row describing this comparison.

    corr_matrix <- build_correlation_matrix(corr_long_df)

    if (nrow(corr_matrix) == 0L || ncol(corr_matrix) == 0L) {
        message(
            "Skipping comparison with no defined correlations: ",
            comparison$correlation_file
        )
        return(invisible(FALSE))
    }

    # Title: what was compared, then how it was imaged and who it covers
    cohort_label <- if (comparison$cohort == "all_patients") {
        "All patients"
    } else {
        comparison$cohort
    }

    projection_label <- unname(projection_labels[comparison$projection])
    if (is.na(projection_label)) projection_label <- comparison$projection

    # MorphEM is a 2D embedding, so that comparison is 2D vs. 2D
    is_morphem <- comparison$profile_type == "sc_nucleocentric_morphem"
    comparison_title <- if (is_morphem) {
        "2D handcrafted vs. 2D nucleocentric (MorphEM/CHAMMI-75)"
    } else {
        paste0("2D vs. 3D: ", comparison$profile_label)
    }

    plot_title <- paste0(
        comparison_title, "\n", projection_label, " | ", cohort_label
    )

    # Create the heatmap
    ht <- Heatmap(
        corr_matrix,
        name = "Pearson r",

        # Color scale from red (-1) to white (0) to blue (1)
        col = colorRamp2(
            c(-1, 0, 1),
            c("#B2182B", "white", "#2166AC")
        ),
        na_col = "white",
        border = TRUE,

        # Clustering, skipped where there is only one feature to order
        cluster_rows = if (nrow(corr_matrix) > 1L) {
            cluster_correlation_matrix
        } else {
            FALSE
        },
        cluster_columns = if (ncol(corr_matrix) > 1L) {
            cluster_correlation_matrix
        } else {
            FALSE
        },

        # Only label features when there are few enough to read
        show_row_names = nrow(corr_matrix) <= 30L,
        show_column_names = ncol(corr_matrix) <= 30L,
        row_names_gp = gpar(fontsize = 7),
        column_names_gp = gpar(fontsize = 7),

        # Axis labels
        row_title = if (is_morphem) "2D handcrafted features" else "2D features",
        column_title = if (is_morphem) {
            "2D nucleocentric features"
        } else {
            "3D features"
        },
        column_title_side = "bottom",
        row_title_gp = gpar(fontsize = 14),
        column_title_gp = gpar(fontsize = 14),

        # Legend
        heatmap_legend_param = list(
            at = c(-1, -0.5, 0, 0.5, 1),
            title_gp = gpar(fontsize = 12),
            labels_gp = gpar(fontsize = 9)
        ),
        use_raster = TRUE,
        raster_quality = 3,
        raster_by_magick = FALSE,
        raster_resize_mat = FALSE
    )

    # Draw with main title at top
    draw(
        ht,
        column_title = plot_title,
        column_title_gp = gpar(fontsize = 14),
        padding = unit(c(5, 5, 5, 5), "mm")
    )

    cat(sprintf("  Plotted: %s\n", comparison$correlation_file))
    invisible(TRUE)
}

# Plot comparisons from the correlation manifest
manifest <- as.data.frame(arrow::read_parquet(manifest_path))

if (!is.null(projections_to_plot)) {
    manifest <- manifest[manifest$projection %in% projections_to_plot, , drop = FALSE]
}
if (!is.null(profile_types_to_plot)) {
    manifest <- manifest[
        manifest$profile_type %in% profile_types_to_plot, ,
        drop = FALSE
    ]
}
if (!is.null(cohorts_to_plot)) {
    manifest <- manifest[manifest$cohort %in% cohorts_to_plot, , drop = FALSE]
}

cat(sprintf("Plotting %d comparisons\n", nrow(manifest)))

# Merge every page into one combined PDF, in manifest order
page_dir <- file.path(tempdir(), "correlation_heatmap_pages")
dir.create(page_dir, showWarnings = FALSE)
page_paths <- character(0)

for (i in seq_len(nrow(manifest))) {
    comparison <- manifest[i, , drop = FALSE]

    corr_data <- as.data.frame(arrow::read_parquet(
        file.path(results_dir, comparison$correlation_file)
    ))

    page_path <- file.path(page_dir, sprintf("page_%03d.pdf", i))
    pdf(page_path, width = 11, height = 8.5, onefile = FALSE)
    plotted <- plot_correlation_heatmap(corr_data, comparison)
    dev.off()

    if (isTRUE(plotted)) {
        page_paths <- c(page_paths, page_path)
    } else {
        unlink(page_path)
    }
}

# Each comparison is rendered to its own single-page PDF, then merged into one file with Ghostscript.
gs_bin <- Sys.which("gs")
if (gs_bin == "") {
    gs_candidate <- file.path(dirname(dirname(R.home())), "bin", "gs")
    if (file.exists(gs_candidate)) gs_bin <- gs_candidate
}
if (gs_bin == "" || length(page_paths) == 0L) {
    stop("Ghostscript (gs) not found, or no pages were plotted; cannot merge.")
}
status <- system2(
    gs_bin,
    c(
        "-dBATCH", "-dNOPAUSE", "-q", "-sDEVICE=pdfwrite",
        paste0("-sOutputFile=", pdf_path), page_paths
    )
)
if (status != 0L) stop("Ghostscript failed to merge the heatmap pages.")

unlink(page_dir, recursive = TRUE)

cat(sprintf("Saved: %s\n", pdf_path))
