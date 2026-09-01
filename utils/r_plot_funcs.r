# Shared plot builders so common chart shapes (density, boxplot, patterned
# bar/box, scatter comparison) aren't reimplemented per-notebook.
# All of them apply theme_manuscript() so styling stays consistent.

save_ggplot <- function(plot, path, width, height, dpi = 600) {
    #' Save a ggplot object and display the saved file inline.
    #'
    #' Displays the PNG that was just written to disk rather than
    #' auto-printing the ggplot object itself. Letting IRkernel auto-print
    #' the object re-renders it from scratch at a tiny default device size
    #' (to compute its text repr), which crashes for our heavily
    #' faceted/dodged ggpattern plots (gridpattern's crosshatch spacing math
    #' divides by zero on very narrow bars) and silently drops the image.
    #' Displaying the already-correctly-rendered file sidesteps that
    #' entirely.
    ggsave(filename = path, plot = plot, width = width, height = height, dpi = dpi)
    IRdisplay::display_png(file = path)
    invisible(plot)
}

set_plot_size <- function(width, height) {
    #' Set the inline preview size for the next plot and return
    #' list(width, height) for the matching save_ggplot() call, so the two
    #' don't drift out of sync and each plot doesn't need its own
    #' width/height/options() boilerplate.
    options(repr.plot.width = width, repr.plot.height = height)
    list(width = width, height = height)
}

save_plots_pdf <- function(plots, output_path, width, height) {
    #' Save a list of ggplot objects as a single multi-page PDF, one page
    #' per plot, in list order.
    #'
    #' Parameters
    #' ----------
    #' plots : list of ggplot
    #'     Plots to save, one per page.
    #' output_path : str
    #'     File path for the combined PDF.
    #' width, height : float or numeric vector
    #'     Page dimensions in inches. Either a single number (applied to
    #'     every page) or a vector the same length as plots giving each
    #'     page its own size. Each page is rendered to its own single-page
    #'     PDF at its own size, then combined with the `pdfunite` CLI
    #'     (poppler-utils), which preserves each page's own media box
    #'     rather than rescaling every page to a shared canvas -- so a
    #'     simple single-panel plot doesn't inherit the extra height/width
    #'     only a much busier faceted plot elsewhere in the same file
    #'     actually needs.
    n <- length(plots)
    width <- if (length(width) == 1) rep(width, n) else width
    height <- if (length(height) == 1) rep(height, n) else height
    stopifnot(length(width) == n, length(height) == n)

    tmp_dir <- tempfile("save_plots_pdf_")
    dir.create(tmp_dir)
    on.exit(unlink(tmp_dir, recursive = TRUE), add = TRUE)

    page_paths <- file.path(tmp_dir, sprintf("page_%03d.pdf", seq_len(n)))
    for (i in seq_len(n)) {
        pdf(page_paths[i], width = width[i], height = height[i])
        print(plots[[i]])
        dev.off()
    }
    status <- system2("pdfunite", c(page_paths, output_path))
    if (status != 0) stop("pdfunite failed with status ", status)
}

plot_density_by_facet <- function(
    data, facet_col, facet_val, x_col, fill_col, y_max,
    x_max = NULL, palette = "PRGn", base_size = 18,
    x_lab = NULL, y_lab = "Density", fill_lab = NULL
) {
    if (is.null(x_lab)) x_lab <- "Normalized object counts per well FOV"
    if (is.null(y_lab)) y_lab <- "Density"
    if (is.null(fill_lab)) fill_lab <- "Profile type"
    #' Density plot for one facet level, sharing a common y-axis max across facets.
    plot_data <- data[data[[facet_col]] == facet_val, ]
    p <- (
        ggplot(plot_data, aes(x = .data[[x_col]], fill = .data[[fill_col]]))
        + geom_density(alpha = 0.3)
        + ylim(0, y_max)
        + labs(
            x = x_lab,
            y = y_lab,
            fill = paste0("Profile type: ", facet_val)
        )
        + guides(fill = guide_legend(override.aes = list(alpha = 0.5)))
        + scale_fill_brewer(palette = palette)
        + theme_manuscript(base_size = base_size)
    )
    if (!is.null(x_max)) p <- p + xlim(0, x_max)
    p
}

plot_boxplot <- function(
    data, x_col, y_col, fill_col = NULL,
    x_lab, y_lab, fill_lab = NULL,
    palette = "PRGn", custom_palette = NULL,
    ylim_max = NULL, x_text = "blank", facet_formula = NULL, base_size = 18,
    guides_ncol = NULL, show_legend = TRUE, facet_nrow = NULL, facet_ncol = NULL,
    show_outliers = FALSE

) {
    #' Generic boxplot builder shared across the cell/organoid count summary
    #' plots. fill_col = NULL draws a single-color boxplot with no legend.
    #' show_outliers = FALSE (default) hides points beyond the whiskers, as
    #' before; set TRUE for boxes that pool a mix of subgroups with very
    #' different scales (e.g. a "Total" box alongside per-category boxes),
    #' where hiding outliers can hide the very spread that makes the pooled
    #' box differ from its components.
    outlier_shape <- if (show_outliers) 19 else NA
    if (is.null(fill_col)) {
        p <- (
            ggplot(data, aes(x = .data[[x_col]], y = .data[[y_col]]))
            + geom_boxplot(
                outlier.shape = outlier_shape, alpha = 0.5,
                fill = if (!is.null(custom_palette)) custom_palette[[1]] else "#3D7DCC"
            )
            + labs(x = x_lab, y = y_lab)
            + theme_manuscript(base_size = base_size, x_text = x_text)
        )
    } else {
        p <- (
            ggplot(data, aes(x = .data[[x_col]], y = .data[[y_col]], fill = .data[[fill_col]]))
            + geom_boxplot(outlier.shape = outlier_shape, alpha = 0.5)
            + labs(x = x_lab, y = y_lab, fill = fill_lab)
            + theme_manuscript(base_size = base_size, x_text = x_text)
        )

        p <- if (!is.null(custom_palette)) {
            p + scale_fill_manual(values = custom_palette)
        } else {
            p + scale_fill_brewer(palette = palette)
        }

        t <- if (!show_legend) {
            guides(fill = "none")
        } else if (!is.null(guides_ncol)) {
            guides(fill = guide_legend(ncol = guides_ncol))
        } else {
            guides(fill = guide_legend())
        }
        p <- p + t
    }

    if (!is.null(ylim_max)) p <- p + ylim(0, ylim_max)
    if (!is.null(facet_formula)) p <- p + facet_wrap(facet_formula, scales = "free_x", nrow = facet_nrow, ncol = facet_ncol)
    p
}

plot_bar_horizontal <- function(
    data, x_col, y_col, fill_col, fill_palette,
    x_lab, y_lab, fill_lab = "Dose",
    ylim_max = NULL, facet_formula = NULL, base_size = 18
) {
    #' Mean bar plot, treatments on y-axis, sorted by mean count, fill-only
    #' (e.g. by dose), shared across treatment bar plots. Axes are swapped
    #' directly (not via coord_flip). Each treatment's dose bars stay
    #' grouped on one row; the row order is by the dose-1 mean (falling
    #' back to the overall mean for treatments with no dose-1 data).

    order_df <- data %>%
        dplyr::group_by(.data[[x_col]]) %>%
        dplyr::summarise(
            order_val = {
                dose1_vals <- .data[[y_col]][.data[[fill_col]] == 1]
                if (length(dose1_vals) > 0 && !all(is.na(dose1_vals))) {
                    mean(dose1_vals, na.rm = TRUE)
                } else {
                    mean(.data[[y_col]], na.rm = TRUE)
                }
            },
            .groups = "drop"
        ) %>%
        dplyr::arrange(order_val)

    # reorder x_col levels by dose-1 mean (ascending -> highest at top);
    # dose 1 and dose 10 bars for a treatment remain on the same row
    data[[x_col]] <- factor(data[[x_col]], levels = order_df[[x_col]])

    p <- (
        ggplot(
            data,
            aes(
                y = .data[[x_col]],
                x = .data[[y_col]],
                fill = factor(.data[[fill_col]])
            )
        )
        + geom_bar(
            stat = "summary",
            fun = "mean",
            position = position_dodge(width = 0.9),
            color = "black"
        )
        + scale_fill_manual(values = fill_palette)
        + labs(x = y_lab, y = x_lab, fill = fill_lab)
        + theme_manuscript(base_size = base_size, x_text = "default")
    )

    if (!is.null(ylim_max)) p <- p + xlim(0, ylim_max)
    if (!is.null(facet_formula)) p <- p + facet_wrap(facet_formula, scales = "free_y")
    p
}
plot_boxplot_horizontal <- function(
    data, x_col, y_col, fill_col, fill_palette,
    x_lab, y_lab, fill_lab = "Dose",
    ylim_max = NULL, facet_formula = NULL, base_size = 18,
    facet_nrow = NULL, facet_ncol = NULL
) {
    #' Horizontal boxplot, fill-only (e.g. by dose), shared across treatment box plots.
    p <- (
        ggplot(
            data,
            aes(
                x = .data[[x_col]],
                y = .data[[y_col]],
                fill = factor(.data[[fill_col]])
            )
        )
        + geom_boxplot(outlier.shape = NA, alpha = 0.5)
        + scale_fill_manual(values = fill_palette)
        + labs(x = x_lab, y = y_lab, fill = fill_lab)
        + coord_flip()
        + theme_manuscript(base_size = base_size, x_text = "default")
    )

    if (!is.null(ylim_max)) p <- p + coord_flip(ylim = c(0, ylim_max))
    if (!is.null(facet_formula)) p <- p + facet_wrap(facet_formula, scales = "free_y", nrow = facet_nrow, ncol = facet_ncol)
    p
}

plot_group_density_with_stats <- function(
    data, x_col, x_lab, y_lab = "Density", fill_lab = "Group",
    stat_col = x_col, stat_fn = function(x) sum(x, na.rm = TRUE),
    stat_label = "Total", stat_fmt = function(v) format(v, big.mark = ","),
    tumor_type_col = "Metadata_Biology_TumorType", base_size = 14
) {
    #' Density plot of x_col overlaid by tumor type plus a pooled "All"
    #' curve, with a per-group summary statistic (stat_fn applied to
    #' stat_col -- e.g. sum of raw counts, or mean of an already-per-unit
    #' metric like cells-per-organoid) annotated top-right, one line per
    #' group. tumor_type_col must already be a factor leveled
    #' cNF/pNF/MPNST/Other (utils/r_plot_themes.r::tumor_type_palette) so
    #' "All" sorts last both in the legend and the annotation.
    group_order <- c(names(tumor_type_palette), "All")
    group_palette <- c("All" = "black", tumor_type_palette)
    df <- dplyr::bind_rows(
        data %>% mutate(Group = "All"),
        data %>% mutate(Group = .data[[tumor_type_col]])
    )
    df$Group <- factor(df$Group, levels = group_order)
    group_stats <- df %>%
        group_by(Group) %>%
        summarise(stat_val = stat_fn(.data[[stat_col]]), .groups = "drop") %>%
        arrange(match(Group, group_order))
    stat_text <- paste0(
        stat_label, " ", group_stats$Group, ": ", stat_fmt(group_stats$stat_val),
        collapse = "\n"
    )
    (
        ggplot(df, aes(x = .data[[x_col]], fill = Group))
        + geom_density(color = "black", alpha = 0.4)
        + scale_fill_manual(values = group_palette, breaks = group_order)
        + annotate(
            "text", x = Inf, y = Inf, label = stat_text,
            hjust = 1.05, vjust = 1.1, size = 5, lineheight = 1.1
        )
        + labs(x = x_lab, y = y_lab, fill = fill_lab)
        + theme_manuscript(base_size = base_size)
    )
}

build_tumor_type_count_plots <- function(
    data, y_col, y_lab, n_col = "Metadata_n_cells",
    treatment_col = "Metadata_Experiment_Treatment",
    patient_col = "Metadata_Biology_PatientTumor",
    dose_col = "Metadata_Experiment_Dose",
    tumor_type_col = "Metadata_Biology_TumorType",
    facet_ncol_patient = 4, base_size = 14, base_size_patient_facet = 8,
    base_size_overview = base_size
) {
    #' Standard 6-plot, tumor-type-centric count summary, shared between the
    #' single-cell and organoid biology sections (both call this on their
    #' own ZedProfiler count column): overall distribution overlaid by
    #' tumor type (+ pooled "All"), a tumor-type boxplot with a pooled
    #' "Total" box (outliers shown, since hiding them can hide a minority
    #' subgroup's pull on the pooled box's spread), by patient (colored by
    #' tumor type), by treatment split into one horizontal plot per dose
    #' (axes swapped so tumor-type fill colors are legible), and by
    #' treatment faceted by patient (axes swapped so the many treatment
    #' labels read horizontally, legend moved below to save width).
    #' tumor_type_col must already be a factor leveled cNF/pNF/MPNST/Other
    #' (see utils/r_plot_themes.r::tumor_type_palette) so "All"/"Total"
    #' sort last below. base_size_overview (defaults to base_size) sets the
    #' font size for just the first two (single-panel, low-information-
    #' density) plots, independent of the rest -- useful when the shared
    #' PDF page size is driven up by a later faceted plot, which would
    #' otherwise leave these two looking sparse.
    distribution_plot <- plot_group_density_with_stats(
        data, x_col = y_col, x_lab = y_lab, stat_col = n_col,
        tumor_type_col = tumor_type_col, base_size = base_size_overview
    )

    tumor_type_order <- c(names(tumor_type_palette), "Total")
    tumor_type_palette_with_total <- c("Total" = "black", tumor_type_palette)
    tumor_type_with_total_df <- dplyr::bind_rows(
        data %>% mutate(TumorTypeGroup = "Total"),
        data %>% mutate(TumorTypeGroup = .data[[tumor_type_col]])
    )
    tumor_type_with_total_df$TumorTypeGroup <- factor(
        tumor_type_with_total_df$TumorTypeGroup, levels = tumor_type_order
    )
    by_tumor_type_plot <- plot_boxplot(
        tumor_type_with_total_df,
        x_col = "TumorTypeGroup", y_col = y_col, fill_col = "TumorTypeGroup",
        x_lab = "Tumor type", y_lab = y_lab, fill_lab = "Tumor type",
        custom_palette = tumor_type_palette_with_total, x_text = "default",
        show_outliers = TRUE, base_size = base_size_overview
    )

    by_patient_plot <- plot_boxplot(
        data,
        x_col = patient_col, y_col = y_col, fill_col = tumor_type_col,
        x_lab = "Patient tumor", y_lab = y_lab, fill_lab = "Tumor type",
        custom_palette = tumor_type_palette, x_text = "angled", base_size = base_size
    )

    by_treatment_dose1_plot <- plot_boxplot_horizontal(
        data %>% filter(.data[[dose_col]] == 1),
        x_col = treatment_col, y_col = y_col,
        fill_col = tumor_type_col, fill_palette = tumor_type_palette,
        x_lab = "Treatment", y_lab = paste0(y_lab, " (Dose 1 uM)"), fill_lab = "Tumor type",
        base_size = base_size
    )
    by_treatment_dose10_plot <- plot_boxplot_horizontal(
        data %>% filter(.data[[dose_col]] == 10),
        x_col = treatment_col, y_col = y_col,
        fill_col = tumor_type_col, fill_palette = tumor_type_palette,
        x_lab = "Treatment", y_lab = paste0(y_lab, " (Dose 10 uM)"), fill_lab = "Tumor type",
        base_size = base_size
    )

    by_treatment_patient_plot <- plot_boxplot_horizontal(
        data,
        x_col = treatment_col, y_col = y_col,
        fill_col = tumor_type_col, fill_palette = tumor_type_palette,
        x_lab = "Treatment", y_lab = y_lab, fill_lab = "Tumor type",
        facet_formula = as.formula(paste0("~ ", patient_col)), facet_ncol = facet_ncol_patient,
        base_size = base_size_patient_facet
    ) + theme(legend.position = "bottom")

    list(
        distribution = distribution_plot,
        by_tumor_type = by_tumor_type_plot,
        by_patient = by_patient_plot,
        by_treatment_dose1 = by_treatment_dose1_plot,
        by_treatment_dose10 = by_treatment_dose10_plot,
        by_treatment_patient = by_treatment_patient_plot
    )
}

build_diagonal_distance_grid <- function(data, x_col, y_col, facet_col = NULL, res = 120) {
    #' Build a fine regular (x, y, signed_dist) grid -- one per facet group
    #' if facet_col is given, otherwise a single grid for the whole
    #' dataset -- spanning each group's own observed x/y range, with
    #' signed_dist = the signed perpendicular distance from (x, y) to the
    #' line y = x (i.e. (y - x) / sqrt(2): 0 exactly on the line, positive
    #' above it -- y > x -- negative below it -- y < x). Signed (not
    #' absolute) so the diverging blue/white/red color scale in
    #' plot_2d_vs_3d_scatter() reads which side of parity a region falls
    #' on, not just how far. Feeds geom_raster() in that function's
    #' diagonal-distance background shading.
    build_one <- function(df) {
        x_range <- range(df[[x_col]], na.rm = TRUE)
        y_range <- range(df[[y_col]], na.rm = TRUE)
        # guard against a degenerate (zero-width) range on either axis
        if (diff(x_range) == 0) x_range <- x_range + c(-0.5, 0.5)
        if (diff(y_range) == 0) y_range <- y_range + c(-0.5, 0.5)
        grid <- expand.grid(
            x = seq(x_range[1], x_range[2], length.out = res),
            y = seq(y_range[1], y_range[2], length.out = res)
        )
        grid$signed_dist <- (grid$y - grid$x) / sqrt(2)
        grid
    }
    if (is.null(facet_col)) {
        build_one(data)
    } else {
        dplyr::bind_rows(lapply(split(data, data[[facet_col]]), function(gdf) {
            out <- build_one(gdf)
            out[[facet_col]] <- gdf[[facet_col]][1]
            out
        }))
    }
}

plot_2d_vs_3d_scatter <- function(
    data, x_col, y_col, color_col, shape_col, color_palette,
    x_lab, y_lab, color_lab = "Treatment", shape_lab = "Dose",
    facet_formula = NULL, facet_ncol = 3, facet_col = NULL, base_size = 18,
    diagonal_shading = TRUE, diagonal_shading_max_alpha = 0.55, diagonal_shading_res = 120,
    diagonal_shading_lab = "3D vs. 2D\n(distance from y = x)"
) {
    #' Scatter comparison of a 2D vs 3D count metric with a reference y=x
    #' line. diagonal_shading = TRUE (default) additionally fills the
    #' panel background with a diverging blue-white-red gradient by the
    #' *signed* perpendicular distance to y = x -- white exactly on the
    #' line (y = x, the midpoint), reddening where y > x (3D over 2D),
    #' bluening where y < x (2D over 3D) -- with a colorbar legend
    #' (diagonal_shading_lab). The color scale's limits are the dataset-
    #' wide (not per-panel) max absolute signed distance, so panels stay
    #' comparable: a panel that's uniformly far from parity reads as
    #' uniformly saturated, and a panel that only partly strays reads with
    #' visible internal gradient -- both true to the data, unlike a
    #' per-panel-rescaled version where every panel would look equally
    #' "extreme" regardless of its actual distance. This keeps the
    #' diagonal reference legible even in free-scaled facets where a
    #' panel's visible range may not actually include any point where
    #' x == y, which leaves geom_abline's line outside the panel and never
    #' drawn. facet_col (the same grouping variable as facet_formula, as a
    #' string) computes the shading grid per facet (needed so each free-
    #' scaled panel's raster covers its own x/y range) while the color
    #' scale itself stays the one dataset-wide scale described above.
    p <- ggplot(
        data,
        aes(
            x = .data[[x_col]],
            y = .data[[y_col]],
            color = .data[[color_col]],
            shape = factor(.data[[shape_col]])
        )
    )

    if (diagonal_shading) {
        shading_df <- build_diagonal_distance_grid(
            data, x_col, y_col, facet_col = facet_col, res = diagonal_shading_res
        )
        max_abs_dist <- max(abs((data[[y_col]] - data[[x_col]]) / sqrt(2)), na.rm = TRUE)
        p <- (
            p
            + geom_raster(
                data = shading_df, aes(x = x, y = y, fill = signed_dist),
                inherit.aes = FALSE, interpolate = TRUE
            )
            + scale_fill_gradient2(
                low = scales::alpha("blue", diagonal_shading_max_alpha),
                mid = scales::alpha("white", 0),
                high = scales::alpha("red", diagonal_shading_max_alpha),
                midpoint = 0, limits = c(-max_abs_dist, max_abs_dist), oob = scales::squish,
                name = diagonal_shading_lab,
                guide = guide_colorbar(order = 3)
            )
        )
    }

    p <- (
        p
        + geom_point(size = 4, alpha = 0.7)
        + geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black")
        + scale_color_manual(values = color_palette, guide = guide_legend(order = 1))
        + guides(shape = guide_legend(order = 2))
        + labs(x = x_lab, y = y_lab, color = color_lab, shape = shape_lab)
        + theme_manuscript(base_size = base_size)
    )

    if (!is.null(facet_formula)) p <- p + facet_wrap(facet_formula, scales = "free", ncol = facet_ncol)
    p
}
