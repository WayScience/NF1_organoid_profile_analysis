list_of_packages <- c("ggplot2", "dplyr", "tidyr", "arrow", "RColorBrewer")
for (package in list_of_packages) {
    suppressPackageStartupMessages(
        suppressWarnings(
            library(
                package,
                character.only = TRUE,
                quietly = TRUE,
                warn.conflicts = FALSE
            )
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
source(file.path(root_dir, "utils/r_result_cache.r"))

# Caching: skip work whose outputs were built from unchanged inputs and code
CACHE_STEP <- "5.plot_metrics_and_mAP"
CODE_PATH <- file.path(root_dir, "2.2d_vs_3d_analysis", "scripts", "5.plot_metrics_and_mAP.r")
results_root <- file.path(root_dir, "2.2d_vs_3d_analysis", "results")
cat("Git root directory:", root_dir, "\n")

# custom_treatment_palette lives in utils/r_plot_themes.r so it isn't
# redefined (and potentially drift out of sync) in every plotting script.
source(file.path(root_dir, "utils/r_plot_themes.r"))

mAP_results_dir <- file.path(root_dir, "2.2d_vs_3d_analysis", "results", "mAP")
dist_results_dir <- file.path(root_dir, "2.2d_vs_3d_analysis", "results", "distance_metrics")
figures_dir <- file.path(root_dir, "2.2d_vs_3d_analysis", "figures", "mAP_and_distance_metrics")
dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)

# Every profile type from notebooks 3/4, and the 2D-vs-3D pairs for cross-modality plots.
# Each 3D profile is aggregated two ways -- from the feature-selected single cells
# (from_fs) and from the normalized single cells (from_norm) -- so notebooks 3/4 emit a
# _from_fs / _from_norm output per 3D profile type. 2D profiles have no such split.
modes <- tibble::tibble(
    mode = c("from_fs", "from_norm"),
    mode_label = c("from FS", "from norm")
)
base_3d <- tibble::tribble(
    ~base_label, ~base_slug,
    "3D Organoid - Handcrafted", "3D_organoid_handcrafted",
    "3D Organoid - DL (SAM-Med3D)", "3D_organoid_sammed",
    "3D Single-cell - Handcrafted", "3D_sc_handcrafted",
    "3D Single-cell - DL (SAM-Med3D)", "3D_sc_sammed",
    "3D Single-cell - Nucleocentric DL (SAM-Med3D)", "3D_sc_sammed_nucleocentric",
    "3D Single-cell - Nucleocentric DL (MorphEM)", "3D_sc_nucleocentric_morphem"
)
profile_types <- dplyr::bind_rows(
    tibble::tribble(
        ~label, ~slug,
        "2D Organoid", "2D_organoid",
        "2D Single-cell", "2D_sc"
    ),
    tidyr::expand_grid(base_3d, modes) %>%
        dplyr::transmute(
            label = paste0(base_label, " (", mode_label, ")"),
            slug = paste0(base_slug, "_", mode)
        )
)

base_pairs_2d_3d <- tibble::tribble(
    ~base_label, ~slug_2d, ~base_slug_3d,
    "Organoid - Handcrafted", "2D_organoid", "3D_organoid_handcrafted",
    "Organoid - DL (SAM-Med3D)", "2D_organoid", "3D_organoid_sammed",
    "Single-cell - Handcrafted", "2D_sc", "3D_sc_handcrafted",
    "Single-cell - DL (SAM-Med3D)", "2D_sc", "3D_sc_sammed",
    "Single-cell - Nucleocentric DL (SAM-Med3D)", "2D_sc", "3D_sc_sammed_nucleocentric",
    "Single-cell - Nucleocentric DL (MorphEM)", "2D_sc", "3D_sc_nucleocentric_morphem"
)
pairs_2d_3d <- tidyr::expand_grid(base_pairs_2d_3d, modes) %>%
    dplyr::transmute(
        label = paste0(base_label, " (", mode_label, ")"),
        slug_2d = slug_2d,
        slug_3d = paste0(base_slug_3d, "_", mode)
    )

# Dose is encoded as point shape (1 = circle, 10 = triangle); drug colors
# come from custom_treatment_palette in utils/r_plot_themes.r (sourced above).
dose_shape_palette <- c("1" = 16, "10" = 17)  # 16 = circle, 17 = triangle

# Results only carry the combined "Drug_Dose" string; split it into bare drug
# (for color) and bare dose (for shape) before plotting.
split_treatment_dose <- function(df) {
    # Exclude this sample from displays and plotted summaries only.
    # Pooled inter-patient scores retain their original calculation cohort.
    if ("Metadata_patient" %in% names(df)) {
        df <- df %>% filter(is.na(Metadata_patient) | Metadata_patient != "NF0037_T1_CQ1")
    }
    df$Metadata_treatment <- sub("_[0-9]+$", "", df$Metadata_treatment_dose)
    df$Metadata_dose <- sub(".*_([0-9]+)$", "\\1", df$Metadata_treatment_dose)
    df
}

plot_theme <- theme(
    aspect.ratio = 1,
    panel.spacing.x = grid::unit(1.2, "lines"),
    plot.title   = element_text(hjust = 0.5, size = 14),
    axis.title.x = element_text(size = 16),
    axis.title.y = element_text(size = 16),
    axis.text.x  = element_text(size = 14),
    axis.text.y  = element_text(size = 14),
    legend.position = "bottom",
    legend.title = element_text(size = 12, face = "bold", hjust = 0.5),
    legend.text  = element_text(size = 10)
)

# Rasterizes the point layer (via ggrastr) while everything else in the plot
# (axes, text, legend) stays vector, so combined multi-page PDFs with many
# points per page don't balloon in file size.
point_layer <- function(..., rasterize_dpi = 300) {
    layer <- geom_point(...)
    ggrastr::rasterise(layer, dpi = rasterize_dpi)
}

# Saves a list of ggplot objects as a single multi-page PDF, one page per plot.
save_plots_pdf <- function(plots, output_path, width = 10, height = 12) {
    pdf(output_path, width = width, height = height)
    on.exit(dev.off(), add = TRUE)
    for (p in plots) {
        print(p)
    }
}

intra_plots <- list()
inter_plots <- list()
cross_plots <- list()

# mAP volcano-style scatter (mAP vs -log10(p))
make_mAP_plot <- function(df, title, faceted) {
    p <- (
        ggplot(df, aes(x = mean_average_precision, y = `-log10(p-value)`, color = Metadata_treatment, shape = Metadata_dose))
        + point_layer(size = 3, alpha = 0.7)
        + scale_color_manual(name = "Treatment", values = custom_treatment_palette,
                              guide = guide_legend(ncol = 4, title.position = "top", title.hjust = 0.5, order = 1))
        + scale_shape_manual(name = "Dose", values = dose_shape_palette,
                              guide = guide_legend(title.position = "top", title.hjust = 0.5, order = 2))
        + geom_hline(yintercept = 1.3, linetype = "dashed", color = "red")
        + labs(x = "Mean Average Precision", y = "-log10(p-value)", title = title)
        + theme_bw()
        + plot_theme
        + xlim(0, 1)
    )
    if (faceted) {
        p <- p + facet_wrap(~Metadata_patient, ncol = 4)
    }
    p
}

# mAP vs cosine distance scatter
make_mAP_vs_distance_plot <- function(df, title, faceted) {
    p <- (
        ggplot(df, aes(x = mean_average_precision, y = cosine_distance_mean, color = Metadata_treatment, shape = Metadata_dose))
        + point_layer(size = 3, alpha = 0.7)
        + scale_color_manual(name = "Treatment", values = custom_treatment_palette,
                              guide = guide_legend(ncol = 4, title.position = "top", title.hjust = 0.5, order = 1))
        + scale_shape_manual(name = "Dose", values = dose_shape_palette,
                              guide = guide_legend(title.position = "top", title.hjust = 0.5, order = 2))
        + labs(x = "Mean Average Precision", y = "Cosine Distance", title = title)
        + theme_bw()
        + plot_theme
        + xlim(0, 1)
    )
    if (faceted) {
        p <- p + facet_wrap(~Metadata_patient, ncol = 4)
    }
    p
}

# mAP-vs-p and mAP-vs-distance, one intra + one inter pair of plots per profile type
for (i in seq_len(nrow(profile_types))) {
    label <- profile_types$label[i]
    slug  <- profile_types$slug[i]

    intra_mAP <- split_treatment_dose(arrow::read_parquet(file.path(mAP_results_dir, paste0(slug, "_intra_patient_mAP_by_dose.parquet"))))
    inter_mAP <- split_treatment_dose(arrow::read_parquet(file.path(mAP_results_dir, paste0(slug, "_inter_patient_mAP_by_dose.parquet"))))
    intra_dist <- split_treatment_dose(arrow::read_parquet(file.path(dist_results_dir, paste0(slug, "_intra_patient_cosine_distance.parquet"))))
    inter_dist <- split_treatment_dose(arrow::read_parquet(file.path(dist_results_dir, paste0(slug, "_inter_patient_cosine_distance.parquet"))))

    intra_plots[[length(intra_plots) + 1]] <- make_mAP_plot(intra_mAP, paste0("Intra-patient mAP - ", label), faceted = TRUE)
    inter_plots[[length(inter_plots) + 1]] <- make_mAP_plot(inter_mAP, paste0("Inter-patient mAP - ", label), faceted = FALSE)

    intra_mAP_dist <- inner_join(
        intra_mAP,
        intra_dist %>% select(Metadata_treatment_dose, Metadata_patient, cosine_distance_mean),
        by = c("Metadata_treatment_dose", "Metadata_patient")
    )
    inter_mAP_dist <- inner_join(
        inter_mAP,
        inter_dist %>% select(Metadata_treatment_dose, cosine_distance_mean),
        by = "Metadata_treatment_dose"
    )

    intra_plots[[length(intra_plots) + 1]] <- make_mAP_vs_distance_plot(intra_mAP_dist, paste0("Intra-patient mAP vs Distance - ", label), faceted = TRUE)
    inter_plots[[length(inter_plots) + 1]] <- make_mAP_vs_distance_plot(inter_mAP_dist, paste0("Inter-patient mAP vs Distance - ", label), faceted = FALSE)
}

# Ratio bar chart + inter-vs-intra mAP scatter, one pair of plots per profile type
for (i in seq_len(nrow(profile_types))) {
    label <- profile_types$label[i]
    slug  <- profile_types$slug[i]

    intra_df <- arrow::read_parquet(file.path(mAP_results_dir, paste0(slug, "_intra_patient_mAP_by_dose.parquet"))) %>%
        select(Metadata_treatment_dose, Metadata_patient, intra_patient_mAP = mean_average_precision)
    inter_df <- arrow::read_parquet(file.path(mAP_results_dir, paste0(slug, "_inter_patient_mAP_by_dose.parquet"))) %>%
        select(Metadata_treatment_dose, inter_patient_mAP = mean_average_precision)

    merged_df <- merge(intra_df, inter_df, by = "Metadata_treatment_dose", all = TRUE)
    merged_df$intra_to_inter_ratio <- merged_df$intra_patient_mAP / merged_df$inter_patient_mAP
    merged_df <- split_treatment_dose(merged_df)

    # merged_df has one row per (patient, treatment) since intra is per-patient and inter
    # isn't; summarize to one ratio per treatment (median across patients) before plotting
    # a bar, otherwise geom_bar sums the per-patient duplicates into an inflated bar height.
    ratio_by_treatment <- merged_df %>%
        group_by(Metadata_treatment_dose, Metadata_treatment, Metadata_dose) %>%
        summarize(median_ratio = median(intra_to_inter_ratio, na.rm = TRUE), .groups = "drop")

    ratio_plot <- (
        ggplot(ratio_by_treatment, aes(x = Metadata_treatment_dose, y = median_ratio, fill = Metadata_treatment))
        + geom_bar(stat = "identity", position = "dodge")
        + scale_fill_manual(name = "Treatment", values = custom_treatment_palette,
                             guide = guide_legend(title.position = "top", title.hjust = 0.5))
        + labs(x = "Treatment (dose)", y = "Median Intra:Inter Patient mAP Ratio",
               title = paste0("Intra:Inter mAP Ratio - ", label))
        + theme_bw()
        + plot_theme
        + theme(axis.text.x = element_text(angle = 90, hjust = 1, size = 10), legend.position = "none")
        + geom_hline(yintercept = 1, linetype = "dashed", color = "red")
    )
    cross_plots[[length(cross_plots) + 1]] <- ratio_plot

    scatter_plot <- (
        ggplot(merged_df, aes(x = inter_patient_mAP, y = intra_patient_mAP, color = Metadata_treatment, shape = Metadata_dose))
        + point_layer(size = 2, alpha = 0.7)
        + scale_color_manual(name = "Treatment", values = custom_treatment_palette,
                              guide = guide_legend(ncol = 4, title.position = "top", title.hjust = 0.5, order = 1))
        + scale_shape_manual(name = "Dose", values = dose_shape_palette,
                              guide = guide_legend(title.position = "top", title.hjust = 0.5, order = 2))
        + geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red")
        + labs(x = "Inter-patient mAP", y = "Intra-patient mAP", title = paste0("Inter vs Intra Patient mAP - ", label))
        + theme_bw()
        + plot_theme
        + xlim(0, 1)
    )
    cross_plots[[length(cross_plots) + 1]] <- scatter_plot
}

# 2D vs 3D mAP scatter, one intra (faceted by patient) + one inter plot per pair
for (i in seq_len(nrow(pairs_2d_3d))) {
    label   <- pairs_2d_3d$label[i]
    slug_2d <- pairs_2d_3d$slug_2d[i]
    slug_3d <- pairs_2d_3d$slug_3d[i]

    intra_2d <- arrow::read_parquet(file.path(mAP_results_dir, paste0(slug_2d, "_intra_patient_mAP_by_dose.parquet")))
    intra_3d <- arrow::read_parquet(file.path(mAP_results_dir, paste0(slug_3d, "_intra_patient_mAP_by_dose.parquet")))
    intra_merged <- inner_join(
        intra_2d %>% select(Metadata_treatment_dose, Metadata_patient, mAP_2D = mean_average_precision),
        intra_3d %>% select(Metadata_treatment_dose, Metadata_patient, mAP_3D = mean_average_precision),
        by = c("Metadata_treatment_dose", "Metadata_patient")
    )

    inter_2d <- arrow::read_parquet(file.path(mAP_results_dir, paste0(slug_2d, "_inter_patient_mAP_by_dose.parquet")))
    inter_3d <- arrow::read_parquet(file.path(mAP_results_dir, paste0(slug_3d, "_inter_patient_mAP_by_dose.parquet")))
    inter_merged <- inner_join(
        inter_2d %>% select(Metadata_treatment_dose, mAP_2D = mean_average_precision),
        inter_3d %>% select(Metadata_treatment_dose, mAP_3D = mean_average_precision),
        by = "Metadata_treatment_dose"
    )
    intra_merged <- split_treatment_dose(intra_merged)
    inter_merged <- split_treatment_dose(inter_merged)

    intra_plots[[length(intra_plots) + 1]] <- (
        ggplot(intra_merged, aes(x = mAP_2D, y = mAP_3D, color = Metadata_treatment, shape = Metadata_dose))
        + point_layer(size = 3, alpha = 0.7)
        + scale_color_manual(name = "Treatment", values = custom_treatment_palette,
                              guide = guide_legend(ncol = 4, title.position = "top", title.hjust = 0.5, order = 1))
        + scale_shape_manual(name = "Dose", values = dose_shape_palette,
                              guide = guide_legend(title.position = "top", title.hjust = 0.5, order = 2))
        + geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red")
        + labs(x = "2D mAP Score", y = "3D mAP Score", title = paste0("2D vs 3D Intra-patient mAP - ", label))
        + theme_bw()
        + plot_theme
        + xlim(0, 1)
        + ylim(0, 1)
        + coord_fixed(ratio = 1)
        + facet_wrap(~Metadata_patient, ncol = 4)
    )

    inter_plots[[length(inter_plots) + 1]] <- (
        ggplot(inter_merged, aes(x = mAP_2D, y = mAP_3D, color = Metadata_treatment, shape = Metadata_dose))
        + point_layer(size = 4, alpha = 0.7)
        + scale_color_manual(name = "Treatment", values = custom_treatment_palette,
                              guide = guide_legend(ncol = 4, title.position = "top", title.hjust = 0.5, order = 1))
        + scale_shape_manual(name = "Dose", values = dose_shape_palette,
                              guide = guide_legend(title.position = "top", title.hjust = 0.5, order = 2))
        + geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red")
        + labs(x = "2D mAP Score", y = "3D mAP Score", title = paste0("2D vs 3D Inter-patient mAP - ", label))
        + theme_bw()
        + plot_theme
        + xlim(0, 1)
        + ylim(0, 1)
        + coord_fixed(ratio = 1)
    )
}

# 2D vs 3D cosine distance scatter, one intra (faceted by patient) + one inter plot per pair
for (i in seq_len(nrow(pairs_2d_3d))) {
    label   <- pairs_2d_3d$label[i]
    slug_2d <- pairs_2d_3d$slug_2d[i]
    slug_3d <- pairs_2d_3d$slug_3d[i]

    intra_2d <- arrow::read_parquet(file.path(dist_results_dir, paste0(slug_2d, "_intra_patient_cosine_distance.parquet")))
    intra_3d <- arrow::read_parquet(file.path(dist_results_dir, paste0(slug_3d, "_intra_patient_cosine_distance.parquet")))
    intra_merged <- inner_join(
        intra_2d %>% select(Metadata_treatment_dose, Metadata_patient, cosine_2D = cosine_distance_mean),
        intra_3d %>% select(Metadata_treatment_dose, Metadata_patient, cosine_3D = cosine_distance_mean),
        by = c("Metadata_treatment_dose", "Metadata_patient")
    )

    inter_2d <- arrow::read_parquet(file.path(dist_results_dir, paste0(slug_2d, "_inter_patient_cosine_distance.parquet")))
    inter_3d <- arrow::read_parquet(file.path(dist_results_dir, paste0(slug_3d, "_inter_patient_cosine_distance.parquet")))
    inter_merged <- inner_join(
        inter_2d %>% select(Metadata_treatment_dose, cosine_2D = cosine_distance_mean),
        inter_3d %>% select(Metadata_treatment_dose, cosine_3D = cosine_distance_mean),
        by = "Metadata_treatment_dose"
    )
    intra_merged <- split_treatment_dose(intra_merged)
    inter_merged <- split_treatment_dose(inter_merged)

    intra_plots[[length(intra_plots) + 1]] <- (
        ggplot(intra_merged, aes(x = cosine_2D, y = cosine_3D, color = Metadata_treatment, shape = Metadata_dose))
        + point_layer(size = 3, alpha = 0.7)
        + scale_color_manual(name = "Treatment", values = custom_treatment_palette,
                              guide = guide_legend(ncol = 4, title.position = "top", title.hjust = 0.5, order = 1))
        + scale_shape_manual(name = "Dose", values = dose_shape_palette,
                              guide = guide_legend(title.position = "top", title.hjust = 0.5, order = 2))
        + geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red")
        + labs(x = "2D Cosine Distance", y = "3D Cosine Distance",
               title = paste0("2D vs 3D Intra-patient Cosine Distance - ", label))
        + coord_fixed(
            ratio = 1,
            xlim = range(c(intra_merged$cosine_2D, intra_merged$cosine_3D), finite = TRUE),
            ylim = range(c(intra_merged$cosine_2D, intra_merged$cosine_3D), finite = TRUE)
        )
        + theme_bw()
        + plot_theme
        + facet_wrap(~Metadata_patient, ncol = 4)
    )

    inter_plots[[length(inter_plots) + 1]] <- (
        ggplot(inter_merged, aes(x = cosine_2D, y = cosine_3D, color = Metadata_treatment, shape = Metadata_dose))
        + point_layer(size = 4, alpha = 0.7)
        + scale_color_manual(name = "Treatment", values = custom_treatment_palette,
                              guide = guide_legend(ncol = 4, title.position = "top", title.hjust = 0.5, order = 1))
        + scale_shape_manual(name = "Dose", values = dose_shape_palette,
                              guide = guide_legend(title.position = "top", title.hjust = 0.5, order = 2))
        + geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red")
        + labs(x = "2D Cosine Distance", y = "3D Cosine Distance",
               title = paste0("2D vs 3D Inter-patient Cosine Distance - ", label))
        + coord_fixed(
            ratio = 1,
            xlim = range(c(inter_merged$cosine_2D, inter_merged$cosine_3D), finite = TRUE),
            ylim = range(c(inter_merged$cosine_2D, inter_merged$cosine_3D), finite = TRUE)
        )
        + theme_bw()
        + plot_theme
    )
}

# One multi-page, rasterized-point PDF per category, instead of a PNG per plot.
# A PDF is redrawn only when the mAP or distance results behind it have changed.
cache_inputs <- c(
    list.files(mAP_results_dir, pattern = "_patient_mAP_by_dose\\.parquet$", full.names = TRUE),
    list.files(dist_results_dir, pattern = "_patient_cosine_distance\\.parquet$", full.names = TRUE)
)

save_plots_pdf_if_stale <- function(plots, output_path, width = 10, height = 12) {
    if (result_cache_is_fresh(CACHE_STEP, results_root, CODE_PATH, output_path, cache_inputs)) {
        cat("Using cached result:", basename(output_path), "\n")
        return(invisible(FALSE))
    }
    save_plots_pdf(plots, output_path, width = width, height = height)
    result_cache_mark_fresh(CACHE_STEP, results_root, CODE_PATH, output_path, cache_inputs)
    invisible(TRUE)
}

save_plots_pdf_if_stale(intra_plots, file.path(figures_dir, "intra_patient_metrics.pdf"), width = 14, height = 14)
save_plots_pdf_if_stale(inter_plots, file.path(figures_dir, "inter_patient_metrics.pdf"), width = 10, height = 12)
save_plots_pdf_if_stale(cross_plots, file.path(figures_dir, "intra_vs_inter_metrics.pdf"), width = 10, height = 12)
