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

# Tumor type classification (cNF = cutaneous/subcutaneous neurofibroma, pNF =
# plexiform neurofibroma, MPNST = malignant peripheral nerve sheath tumor).
# NF0030_T1 (myopericytoma), NF0040_T1 (schwannoma), and SARCO361_T1
# (sarcoma) are not NF1 nerve-sheath tumors and are grouped as "Other".
# Source: https://github.com/WayScience/NF1_3D_organoid_profiling_pipeline/blob/4072be16543851063df9bcd16500498f269f45fd/figures/table1_patients_and_counts/results/table1_patients_and_counts_results.tsv
tumor_type_lookup <- c(
  "NF0014_T1" = "cNF",
  "NF0014_T2" = "pNF",
  "NF0016_T1" = "pNF",
  "NF0018_T6" = "cNF",
  "NF0021_T1" = "cNF",
  "NF0030_T1" = "Other",
  "NF0035_T1" = "cNF",
  "NF0037_T1" = "cNF",
  "NF0040_T1" = "Other",
  "NF0055_T1" = "pNF",
  "SARCO219_T2" = "MPNST",
  "SARCO361_T1" = "Other"
)

tumor_type_palette <- c(
  "cNF" = "#1B9E77",
  "pNF" = "#D95F02",
  "MPNST" = "#7570B3",
  "Other" = "#999999"
)

reconstruct_from_pairs <- function(pairs_df, n) {
    # Reconstruct a full symmetric N x N matrix from an upper-triangle-only
    # long-format pairs table (sample_i, sample_j 0-indexed from Python).
    mat <- matrix(NA_real_, nrow = n, ncol = n)
    mat[cbind(pairs_df$sample_i + 1, pairs_df$sample_j + 1)] <- pairs_df$correlation
    mat[cbind(pairs_df$sample_j + 1, pairs_df$sample_i + 1)] <- pairs_df$correlation
    mat
}

reconstruct_from_flat <- function(flat, n) {
    # Reconstruct an N x N matrix from a row-major-flattened vector (matches
    # numpy's default .flatten() order used when this was written).
    matrix(unlist(flat), nrow = n, byrow = TRUE)
}

build_heatmap <- function(mat, treatment, patient = NULL, tumor_type = NULL, cluster = TRUE,
                           title, legend_fontsize = 8, column_title_fontsize = 10) {
    # Build a single correlation Heatmap. patient/tumor_type are optional
    # annotations -- omit them where they'd be constant across the whole
    # matrix (e.g. a single-patient matrix), where they'd add a solid color
    # bar with no information. When cluster = FALSE, rows/columns are
    # instead sorted by tumor type (if given), then patient (if given), then
    # treatment -- patient IDs aren't grouped by tumor type (e.g. NF0014_T1
    # is cNF but NF0014_T2 is pNF), so tumor type needs its own sort key.
    n <- nrow(mat)
    perm <- seq_len(n)
    if (!cluster) {
        sort_keys <- list()
        if (!is.null(tumor_type)) sort_keys$tumor_type <- tumor_type
        if (!is.null(patient)) sort_keys$patient <- patient
        sort_keys$treatment <- treatment
        perm <- do.call(order, sort_keys)
    }
    mat <- mat[perm, perm]
    treatment <- treatment[perm]
    if (!is.null(patient)) patient <- patient[perm]
    if (!is.null(tumor_type)) tumor_type <- tumor_type[perm]

    annotations <- list(Treatment = treatment)
    anno_cols <- list(Treatment = custom_treatment_palette)
    show_legend <- c(Treatment = TRUE)

    if (!is.null(patient)) {
        unique_patients <- sort(unique(patient))
        patient_colors <- setNames(
            tab20_palette_for_patients[seq_along(unique_patients)],
            unique_patients
        )
        annotations$Patient <- patient
        anno_cols$Patient <- patient_colors
        show_legend <- c(show_legend, Patient = TRUE)
    }

    if (!is.null(tumor_type)) {
        annotations$TumorType <- tumor_type
        anno_cols$TumorType <- tumor_type_palette
        show_legend <- c(show_legend, TumorType = TRUE)
    }

    # annotation_legend_param entries must exactly match the annotations
    # actually present -- ComplexHeatmap errors ("Amount of legend params is
    # larger than the number of simple annotations") if e.g. TumorType has a
    # legend param but wasn't passed in (as in Group B's calls, which only
    # pass treatment).
    legend_params <- list(
        Treatment = list(title = "Treatment", ncol = 1, labels_gp = gpar(fontsize = legend_fontsize))
    )
    if (!is.null(tumor_type)) {
        legend_params$TumorType <- list(title = "Tumor type", labels_gp = gpar(fontsize = legend_fontsize))
    }

    top_annotation <- do.call(HeatmapAnnotation, c(
        annotations,
        list(
            col = anno_cols,
            show_legend = show_legend,
            annotation_legend_param = legend_params,
            annotation_name_side = "left",
            annotation_name_gp = gpar(fontsize = column_title_fontsize)
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

    Heatmap(
        mat,
        name = "Corr",
        col = col_fun,
        na_col = "grey80",
        show_row_names = FALSE,
        show_column_names = FALSE,
        show_row_dend = FALSE,
        show_column_dend = cluster,
        cluster_rows = cluster,
        cluster_columns = cluster,
        clustering_distance_rows = na_safe_dist,
        clustering_distance_columns = na_safe_dist,
        use_raster = TRUE,
        # quality 1 (not the ComplexHeatmap default of 2): quality scales the
        # intermediate raster buffer quadratically, and for the largest
        # per-patient sc_fs matrices (up to ~12,600 samples) quality=2
        # exhausted R's vector memory building that buffer. quality=1 still
        # gives >=1 pixel per matrix cell (no visible loss at these sizes)
        # while cutting the buffer to a quarter.
        raster_quality = 1,
        top_annotation = top_annotation,
        left_annotation = left_annotation,
        column_title = title,
        column_title_gp = gpar(fontsize = column_title_fontsize, fontface = "bold"),
        heatmap_legend_param = list(
            title = "Pearson\ncorrelation",
            title_gp = gpar(fontsize = legend_fontsize, fontface = "bold"),
            labels_gp = gpar(fontsize = legend_fontsize)
        )
    )
}

save_heatmaps_pdf <- function(heatmaps, output_path, width, height) {
    pdf(output_path, width = width, height = height)
    on.exit(dev.off(), add = TRUE)
    for (ht in heatmaps) {
        # draw()'s own newpage=TRUE default silently fails to advance the
        # PDF page when run under the Jupyter/IRkernel evaluate() context
        # (verified: identical code in plain Rscript advances pages fine).
        # Calling grid.newpage() ourselves and telling draw() not to is a
        # reliable workaround in both contexts.
        grid::grid.newpage()
        draw(ht, newpage = FALSE, merge_legend = TRUE)
    }
}

page_size_for <- function(max_n) {
    list(
        width = max(8, min(24, max_n / 40 + 3)),
        height = max(6, min(20, max_n / 40))
    )
}

pairs_df <- arrow::read_parquet(file.path(correlation_dir, "2D_sc_correlation_pairs.parquet"))
samples_df <- arrow::read_parquet(file.path(correlation_dir, "2D_sc_correlation_samples.parquet"))
samples_df$Metadata_Biology_TumorType <- tumor_type_lookup[samples_df$Metadata_Biology_PatientTumor]

slice_display_names <- c(
    max_projection = "Max Projection",
    middle_slice = "Middle Slice",
    middle_n_slice = "Middle N Slice"
)

profile_type_labels <- c(agg = "Aggregate", consensus = "Consensus")

heatmaps_by_profile_type <- list(agg = list(), consensus = list())
max_n_by_profile_type <- list(agg = 0, consensus = 0)

for (slice_strategy in names(slice_display_names)) {
    slice_label <- slice_display_names[[slice_strategy]]

    for (profile_type in names(profile_type_labels)) {
        profile_label <- profile_type_labels[[profile_type]]

        group_samples <- samples_df %>%
            filter(slice_strategy == !!slice_strategy, profile_type == !!profile_type) %>%
            arrange(sample_index)
        n <- nrow(group_samples)

        group_pairs <- pairs_df %>%
            filter(slice_strategy == !!slice_strategy, profile_type == !!profile_type)
        mat <- reconstruct_from_pairs(group_pairs, n)

        treatment <- group_samples$Metadata_Experiment_Treatment
        patient <- group_samples$Metadata_Biology_PatientTumor
        tumor_type <- group_samples$Metadata_Biology_TumorType

        correlation_label <- if (profile_type == "agg") "Replicate Correlation" else "Correlation"
        title_clustered <- paste0("2D ", slice_label, " — Single-Cell ", profile_label, " Profiles — ", correlation_label, " (Clustered)")
        title_unclustered <- paste0("2D ", slice_label, " — Single-Cell ", profile_label, " Profiles — ", correlation_label, " (Unclustered)")

        heatmaps_by_profile_type[[profile_type]][[paste0(slice_strategy, "_clustered")]] <- build_heatmap(
            mat, treatment, patient, tumor_type, cluster = TRUE, title = title_clustered
        )
        heatmaps_by_profile_type[[profile_type]][[paste0(slice_strategy, "_unclustered")]] <- build_heatmap(
            mat, treatment, patient, tumor_type, cluster = FALSE, title = title_unclustered
        )
        max_n_by_profile_type[[profile_type]] <- max(max_n_by_profile_type[[profile_type]], n)
    }
}

for (profile_type in names(profile_type_labels)) {
    size <- page_size_for(max_n_by_profile_type[[profile_type]])
    output_path <- file.path(figures_dir, paste0("2D_sc_", profile_type, "_correlation_heatmaps.pdf"))
    save_heatmaps_pdf(heatmaps_by_profile_type[[profile_type]], output_path, size$width, size$height)
    cat("Saved:", output_path, "(", length(heatmaps_by_profile_type[[profile_type]]), "pages)\n")
}

pairs_df_3d <- arrow::read_parquet(file.path(correlation_dir, "3D_sc_correlation_pairs.parquet"))
samples_df_3d <- arrow::read_parquet(file.path(correlation_dir, "3D_sc_correlation_samples.parquet"))
samples_df_3d$Metadata_Biology_TumorType <- tumor_type_lookup[samples_df_3d$Metadata_Biology_PatientTumor]

variant_display_names_3d <- c(
    nucleocentric_morphem_norm = "Nucleocentric Morphem Norm",
    sammed_nucleocentric_norm = "Sammed Nucleocentric Norm",
    sammed_sc_norm = "Sammed Sc Norm",
    sc_norm = "Sc Norm"
)

heatmaps_by_profile_type_3d <- list(agg = list(), consensus = list())
max_n_by_profile_type_3d <- list(agg = 0, consensus = 0)

for (variant in names(variant_display_names_3d)) {
    variant_label <- variant_display_names_3d[[variant]]

    for (profile_type in names(profile_type_labels)) {
        profile_label <- profile_type_labels[[profile_type]]

        group_samples <- samples_df_3d %>%
            filter(normalization_variant == !!variant, profile_type == !!profile_type) %>%
            arrange(sample_index)
        n <- nrow(group_samples)

        group_pairs <- pairs_df_3d %>%
            filter(normalization_variant == !!variant, profile_type == !!profile_type)
        mat <- reconstruct_from_pairs(group_pairs, n)

        treatment <- group_samples$Metadata_Experiment_Treatment
        patient <- group_samples$Metadata_Biology_PatientTumor
        tumor_type <- group_samples$Metadata_Biology_TumorType

        correlation_label <- if (profile_type == "agg") "Replicate Correlation" else "Correlation"
        title_clustered <- paste0("3D ", variant_label, " — Single-Cell ", profile_label, " Profiles — ", correlation_label, " (Clustered)")
        title_unclustered <- paste0("3D ", variant_label, " — Single-Cell ", profile_label, " Profiles — ", correlation_label, " (Unclustered)")

        heatmaps_by_profile_type_3d[[profile_type]][[paste0(variant, "_clustered")]] <- build_heatmap(
            mat, treatment, patient, tumor_type, cluster = TRUE, title = title_clustered
        )
        heatmaps_by_profile_type_3d[[profile_type]][[paste0(variant, "_unclustered")]] <- build_heatmap(
            mat, treatment, patient, tumor_type, cluster = FALSE, title = title_unclustered
        )
        max_n_by_profile_type_3d[[profile_type]] <- max(max_n_by_profile_type_3d[[profile_type]], n)
    }
}

for (profile_type in names(profile_type_labels)) {
    size <- page_size_for(max_n_by_profile_type_3d[[profile_type]])
    output_path <- file.path(figures_dir, paste0("3D_sc_", profile_type, "_correlation_heatmaps.pdf"))
    save_heatmaps_pdf(heatmaps_by_profile_type_3d[[profile_type]], output_path, size$width, size$height)
    cat("Saved:", output_path, "(", length(heatmaps_by_profile_type_3d[[profile_type]]), "pages)\n")
}

matrices_dataset <- arrow::open_dataset(
    file.path(correlation_dir, "3D_sc_fs_per_patient_correlation_matrices.parquet"),
    format = "parquet"
)
# Filtered lazily per variant below rather than materialized here -- this
# file is ~9GB (some patients have >12,000 cells, so their flattened
# correlation matrices are large), and collect()-ing the whole thing into R
# memory before filtering risks the same kind of memory exhaustion the
# calc notebook hit before it was fixed to write incrementally.

variant_display_names <- c(
    sc_norm = "Sc Norm",
    nucleocentric_morphem_norm = "Nucleocentric Morphem Norm",
    sammed_nucleocentric_norm = "Sammed Nucleocentric Norm"
)

for (variant in names(variant_display_names)) {
    variant_label <- variant_display_names[[variant]]
    variant_rows <- matrices_dataset %>%
        filter(variant == !!variant) %>%
        arrange(patient) %>%
        collect()

    # n_samples alone (no matrix reconstruction) is enough to size the page
    # up front, so heatmaps can be drawn one at a time below instead of
    # building the full list first -- holding every patient's Heatmap
    # object (each with its own raster buffer) simultaneously for a variant
    # with several large patients (e.g. sammed_nucleocentric_norm, where
    # SARCO219_T2 alone is 15,144 cells) was enough to exhaust R's vector
    # memory even with a single patient's buffer already reduced via
    # raster_quality = 1.
    max_n <- max(variant_rows$n_samples)
    size <- page_size_for(max_n)
    output_path <- file.path(figures_dir, paste0("3D_sc_fs_", variant, "_per_patient_correlation_heatmaps.pdf"))

    pdf(output_path, width = size$width, height = size$height)
    n_pages <- 0
    for (i in seq_len(nrow(variant_rows))) {
        row <- variant_rows[i, ]
        n <- row$n_samples[[1]]
        mat <- reconstruct_from_flat(row$correlation[[1]], n)
        treatment <- unlist(row$treatment[[1]])
        patient_id <- row$patient[[1]]
        tumor_type <- tumor_type_lookup[[patient_id]]
        title <- paste0("3D sc_fs (", variant_label, ") — ", patient_id, " — ", tumor_type)
        ht <- build_heatmap(mat, treatment, cluster = TRUE, title = title)
        # See the comment in save_heatmaps_pdf() (c5_helpers) -- draw()'s
        # own newpage=TRUE default doesn't reliably advance the PDF page
        # under Jupyter/IRkernel, so we advance it ourselves.
        grid::grid.newpage()
        draw(ht, newpage = FALSE, merge_legend = TRUE)
        n_pages <- n_pages + 1

        rm(mat, ht)
        gc(full = TRUE)
    }
    dev.off()

    cat("Saved:", output_path, "(", n_pages, "pages)\n")

    rm(variant_rows)
    gc(full = TRUE)
}
