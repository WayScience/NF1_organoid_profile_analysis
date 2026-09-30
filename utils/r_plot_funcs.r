plot_umap <- function(data, output_path = NULL, title,
                       color_by = NULL, palette = NULL, legend_title = NULL,
                       facet_by = NULL, facet_nrow = 2, facet_legend = FALSE,
                       alpha = 0.5, point_size = 1, background_alpha = 0.15,
                       width = 10, height = 5, dpi = 300,
                       base_size = 8, fixed_coord = TRUE,
                       extra_theme = NULL, rasterize_dpi = NULL) {
  # Build, optionally save, and return a single UMAP scatterplot in a
  # consistent style.
  #
  # Two color modes:
  # - Unfaceted (facet_by is None): points colored by color_by using palette,
  #   with a legend.
  # - Faceted (facet_by is set): every panel shows the full point cloud dimmed
  #   grey as context, plus that panel's own group highlighted using color_by
  #   and palette. Usually color_by is the same variable as facet_by (matching
  #   the corresponding non-faceted plot's colors), in which case the facet
  #   strip label already identifies the group and no legend is shown. Set
  #   facet_legend = TRUE when color_by differs from facet_by (e.g. faceting
  #   by patient but coloring by tumor type) so the color mapping stays
  #   decodable.
  #
  # Parameters
  # ----------
  # data : data.frame
  #     Data frame containing UMAP1, UMAP2, and the columns referenced by
  #     color_by and (optionally) facet_by.
  # output_path : str or None, optional
  #     File path the plot is saved to via ggsave(). If None, the plot is
  #     built and returned but not saved (e.g. to collect pages for a
  #     combined multi-page PDF built by the caller).
  # title : str
  #     Plot title.
  # color_by : str
  #     Column name in `data` used to color points. In faceted mode this is
  #     normally the same column as facet_by.
  # palette : named vector
  #     Colors keyed by the values of `color_by`, passed to scale_color_manual().
  # legend_title : str or None, optional
  #     Legend title for the color scale (unfaceted mode, or faceted mode
  #     with facet_legend = TRUE). If None, no title override is applied.
  # facet_by : str or None, optional
  #     Column name in `data` to facet by. If None, the plot is not faceted.
  # facet_nrow : int, optional
  #     Number of rows to use when faceting (ignored if facet_by is None).
  # facet_legend : bool, optional
  #     If True, shows a color legend in faceted mode. Ignored (no legend)
  #     unless facet_by is set. Use this when color_by differs from
  #     facet_by; leave False when they match, since the facet strip label
  #     already identifies the group.
  # alpha : float, optional
  #     Point transparency, in [0, 1]. In faceted mode, this applies to the
  #     highlighted points only.
  # point_size : float, optional
  #     Point size.
  # background_alpha : float, optional
  #     Transparency of the grey context points in faceted mode (ignored if
  #     facet_by is None).
  # width, height : float, optional
  #     Plot dimensions in inches, used both for the inline render size and
  #     the saved file.
  # dpi : int, optional
  #     Resolution (dots per inch) for the saved file (ignored if
  #     output_path is None).
  # base_size : int, optional
  #     Base font size passed to theme_manuscript().
  # fixed_coord : bool, optional
  #     If True, applies coord_fixed() so UMAP1/UMAP2 share a 1:1 scale. If a
  #     particular plot looks wrong with this on, set it False and adjust
  #     width/height instead.
  # extra_theme : ggplot2 theme or None, optional
  #     Theme object added on top of theme_manuscript(), or None to skip.
  # rasterize_dpi : int or None, optional
  #     If set, point layers are rasterized (via ggrastr) at this resolution
  #     instead of staying vector, while axes/text/labels stay vector. Keeps
  #     file size down for plots with many points, e.g. combined PDFs.
  #
  # Returns
  # -------
  # ggplot
  #     The plot object, returned visibly. Auto-prints when called at the
  #     top level of a cell; wrap in print() when called inside a loop,
  #     since R does not auto-print loop-body expressions.
  options(repr.plot.width = width, repr.plot.height = height)

  point_layer <- function(...) {
    layer <- geom_point(...)
    if (!is.null(rasterize_dpi)) {
      layer <- ggrastr::rasterise(layer, dpi = rasterize_dpi)
    }
    layer
  }

  # Compact legend styling: smaller key swatches and tighter spacing so
  # many-row legends (e.g. 12 patients, 22 treatments) fit within the plot
  # instead of getting cut off, especially in the combined PDFs.
  compact_legend_theme <- theme(
    legend.key.size = unit(0.3, "cm"),
    legend.spacing.y = unit(0.05, "cm"),
    legend.text = element_text(size = base_size * 0.8),
    legend.title = element_text(size = base_size * 0.9)
  )

  guide_args <- list(override.aes = list(alpha = 1, size = 2), ncol = 1)
  if (!is.null(legend_title)) {
    guide_args$title <- legend_title
  }

  if (!is.null(facet_by)) {
    # Full point cloud, minus the facet column, so this layer repeats
    # unchanged in every panel instead of being split by facet.
    background_data <- data
    background_data[[facet_by]] <- NULL

    p <- ggplot(data, aes(x = UMAP1, y = UMAP2, color = .data[[color_by]])) +
      point_layer(data = background_data, color = "grey80", alpha = background_alpha, size = point_size) +
      point_layer(alpha = alpha, size = point_size) +
      scale_color_manual(values = palette) +
      labs(title = title, x = "UMAP 1", y = "UMAP 2") +
      theme_manuscript(base_size = base_size) +
      facet_wrap(as.formula(paste0("~", facet_by)), nrow = facet_nrow)

    if (facet_legend) {
      p <- p + compact_legend_theme + guides(color = do.call(guide_legend, guide_args))
    } else {
      p <- p + guides(color = "none")
    }
  } else {
    p <- ggplot(data, aes(x = UMAP1, y = UMAP2, color = .data[[color_by]])) +
      point_layer(alpha = alpha, size = point_size) +
      scale_color_manual(values = palette) +
      labs(title = title, x = "UMAP 1", y = "UMAP 2") +
      theme_manuscript(base_size = base_size) +
      compact_legend_theme +
      guides(color = do.call(guide_legend, guide_args))
  }

  if (fixed_coord) {
    p <- p + coord_fixed()
  }

  if (!is.null(extra_theme)) {
    p <- p + extra_theme
  }

  if (!is.null(output_path)) {
    ggsave(p, file = output_path, width = width, height = height, dpi = dpi)
  }
  p
}
save_umap_pca_plots_pdf <- function(plots, output_path, width = 10, height = 5) {
  # Save a list of ggplot objects as a single multi-page PDF, one page per
  # plot, all pages sharing the same size.
  #
  # Parameters
  # ----------
  # plots : list of ggplot
  #     Plots to save, one per page.
  # output_path : str
  #     File path for the combined PDF.
  # width, height : float, optional
  #     Page dimensions in inches.
  pdf(output_path, width = width, height = height)
  # Guarantees the device closes even if print(p) errors partway through,
  # so a failure stays local to this PDF instead of leaving the device open
  # for whatever gets plotted next.
  on.exit(dev.off(), add = TRUE)
  for (p in plots) {
    print(p)
  }
}
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
        # a function is a page drawn with grid (e.g. ComplexHeatmap), anything else is a ggplot
        if (is.function(plots[[i]])) plots[[i]]() else print(plots[[i]])
        dev.off()
    }
    status <- system2("pdfunite", c(page_paths, output_path))
    if (status != 0) stop("pdfunite failed with status ", status)
}
plot_pca <- function(data, explained_variance_df, title,
                      color_by, palette, legend_title = NULL,
                      facet_by = NULL, facet_nrow = 3,
                      alpha = 0.3, point_size = 0.5, background_alpha = 0.15,
                      width = 8, height = 8,
                      base_size = 8, rasterize_dpi = NULL) {
  # Build a single PC0 vs PC1 scatterplot in a consistent style.
  #
  # Faceting is a plain small-multiples split: facet_by only controls which
  # panel a point falls into, while color_by is independent and keeps its
  # own legend in every panel (unlike plot_umap()'s faceted mode, where
  # facet_by and color_by are the same variable and the legend is dropped).
  # Faceted panels additionally show the full point cloud dimmed grey as
  # context, matching plot_umap()'s background layer -- background_data has
  # the facet column removed so it repeats unchanged in every panel.
  #
  # Parameters
  # ----------
  # data : data.frame
  #     Data frame containing PC0, PC1, and the columns referenced by
  #     color_by and (optionally) facet_by.
  # explained_variance_df : data.frame
  #     Single-row data frame with PC0_explained_variance and
  #     PC1_explained_variance columns, used to label the axes.
  # title : str
  #     Plot title.
  # color_by : str
  #     Column name in `data` used to color points.
  # palette : named vector
  #     Colors keyed by the values of `color_by`, passed to scale_color_manual().
  # legend_title : str or None, optional
  #     Legend title for the color scale. If None, no title override is applied.
  # facet_by : str or None, optional
  #     Column name in `data` to facet by. If None, the plot is not faceted.
  # facet_nrow : int, optional
  #     Number of rows to use when faceting (ignored if facet_by is None).
  # alpha : float, optional
  #     Point transparency, in [0, 1]. In faceted mode, this applies to the
  #     highlighted (foreground) points only.
  # point_size : float, optional
  #     Point size.
  # background_alpha : float, optional
  #     Transparency of the grey context points in faceted mode (ignored if
  #     facet_by is None).
  # width, height : float, optional
  #     Plot dimensions in inches, used for the inline render size.
  # base_size : int, optional
  #     Base font size passed to theme_manuscript().
  # rasterize_dpi : int or None, optional
  #     If set, the point layer is rasterized (via ggrastr) at this
  #     resolution instead of staying vector. Keeps file size down for
  #     plots with many points, e.g. combined PDFs.
  #
  # Returns
  # -------
  # ggplot
  #     The plot object, returned visibly.
  options(repr.plot.width = width, repr.plot.height = height)

  var_pc0 <- explained_variance_df$PC0_explained_variance
  var_pc1 <- explained_variance_df$PC1_explained_variance
  x_label <- sprintf("PCA 1 (var explained %.0f%%)", var_pc0 * 100)
  y_label <- sprintf("PCA 2 (var explained %.0f%%)", var_pc1 * 100)

  point_layer <- function(...) {
    layer <- geom_point(...)
    if (!is.null(rasterize_dpi)) {
      layer <- ggrastr::rasterise(layer, dpi = rasterize_dpi)
    }
    layer
  }

  # Compact legend styling: smaller key swatches and tighter spacing so
  # many-row legends (e.g. 22 treatments) fit within the plot instead of
  # getting cut off, especially in the combined PDFs.
  compact_legend_theme <- theme(
    legend.key.size = unit(0.3, "cm"),
    legend.spacing.y = unit(0.05, "cm"),
    legend.text = element_text(size = base_size * 0.8),
    legend.title = element_text(size = base_size * 0.9)
  )

  guide_args <- list(override.aes = list(alpha = 1, size = 2), ncol = 1)
  if (!is.null(legend_title)) {
    guide_args$title <- legend_title
  }

  if (!is.null(facet_by)) {
    # Full point cloud, minus the facet column, so this layer repeats
    # unchanged in every panel instead of being split by facet.
    background_data <- data
    background_data[[facet_by]] <- NULL

    p <- ggplot(data, aes(x = PC0, y = PC1, color = .data[[color_by]])) +
      point_layer(data = background_data, color = "grey80", alpha = background_alpha, size = point_size) +
      point_layer(alpha = alpha, size = point_size) +
      scale_color_manual(values = palette) +
      labs(title = title, x = x_label, y = y_label) +
      theme_manuscript(base_size = base_size) +
      # PC0/PC1 carry different explained-variance scales, so coord_fixed()
      # (equal data-unit scaling) would often look badly non-square. Forcing
      # the panel aspect ratio to 1 instead keeps the rendered plot close to
      # square without distorting either axis's data range.
      theme(aspect.ratio = 1) +
      compact_legend_theme +
      guides(color = do.call(guide_legend, guide_args)) +
      facet_wrap(as.formula(paste0("~", facet_by)), nrow = facet_nrow)
  } else {
    p <- ggplot(data, aes(x = PC0, y = PC1, color = .data[[color_by]])) +
      point_layer(alpha = alpha, size = point_size) +
      scale_color_manual(values = palette) +
      labs(title = title, x = x_label, y = y_label) +
      theme_manuscript(base_size = base_size) +
      theme(aspect.ratio = 1) +
      compact_legend_theme +
      guides(color = do.call(guide_legend, guide_args))
  }

  p
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
    facet_nrow = NULL, facet_ncol = NULL, facet_scales = "free_y"
) {
    #' Horizontal boxplot, fill-only (e.g. by dose), shared across treatment
    #' box plots. facet_scales defaults to "free_y" (frees the count axis,
    #' which coord_flip() renders horizontally, per facet panel) -- pass
    #' "free" when a single outlier group (e.g. the lone MPNST patient)
    #' would otherwise force a shared value-axis range onto every panel.
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
    if (!is.null(facet_formula)) p <- p + facet_wrap(facet_formula, scales = facet_scales, nrow = facet_nrow, ncol = facet_ncol)
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
        facet_scales = "free", base_size = base_size_patient_facet
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

# ---- ComplexHeatmap helpers (packages ComplexHeatmap, circlize and grid must be loaded) ----

heat_col_fun <- function(values = NULL, palette = "Mako", rev = TRUE, limits = range(values, na.rm = TRUE)) {
    #' Continuous colour function for a Heatmap: an hcl.colors palette
    #' stretched over `limits` (default: the range of `values`). The default
    #' (Mako, light -> dark) matches viridis option "F" with direction = -1.
    #' Degenerate limits (non-finite or equal) become a unit range centred on
    #' the finite value, or on zero, so the breaks stay strictly increasing.
    if (!all(is.finite(limits)) || limits[1] == limits[2]) {
        centre <- limits[is.finite(limits)]
        centre <- if (length(centre) > 0) centre[1] else 0
        limits <- c(centre - 0.5, centre + 0.5)
    }
    circlize::colorRamp2(seq(limits[1], limits[2], length.out = 9), hcl.colors(9, palette, rev = rev))
}

long_to_matrix <- function(df, row, col, value, row_levels = NULL, col_levels = NULL) {
    #' Long table -> matrix (rows x columns), NA where a pair is missing.
    if (is.null(row_levels)) row_levels <- unique(as.character(df[[row]]))
    if (is.null(col_levels)) col_levels <- unique(as.character(df[[col]]))
    mat <- matrix(NA_real_, nrow = length(row_levels), ncol = length(col_levels), dimnames = list(row_levels, col_levels))
    row_idx <- match(as.character(df[[row]]), row_levels)
    col_idx <- match(as.character(df[[col]]), col_levels)
    keep <- !is.na(row_idx) & !is.na(col_idx)
    mat[cbind(row_idx[keep], col_idx[keep])] <- df[[value]][keep]
    mat
}

simple_heatmap <- function(mat, name, col, title = NULL, cell_fmt = NULL, cell_labels = NULL,
                           cell_size = 8, base_size = 11, cluster_rows = FALSE, cluster_columns = FALSE,
                           row_names_max_width = NULL, ...) {
    #' Heatmap with the manuscript text sizes. cell_fmt (a sprintf format)
    #' writes the value in every cell; cell_labels (a same-shaped matrix)
    #' writes those labels instead. Rows/columns keep the matrix order unless
    #' cluster_rows / cluster_columns are set (any Heatmap clustering argument can go in `...`).
    #' row_names_max_width defaults to just enough room for the longest row
    #' label at row_names_gp's fontsize (base_size) -- ComplexHeatmap's own
    #' default (a flat 6cm) doesn't grow with long feature names, so labels
    #' get silently clipped and run into the legend instead of wrapping or
    #' widening; pass a value explicitly to override.
    cell_fun <- NULL
    if (!is.null(cell_fmt) || !is.null(cell_labels)) {
        cell_fun <- function(j, i, x, y, width, height, fill) {
            label <- if (!is.null(cell_labels)) cell_labels[i, j] else if (is.na(mat[i, j])) NA else sprintf(cell_fmt, mat[i, j])
            if (!is.na(label)) grid::grid.text(label, x, y, gp = grid::gpar(fontsize = cell_size))
        }
    }
    if (is.null(row_names_max_width)) {
        row_names_max_width <- if (!is.null(rownames(mat))) {
            ComplexHeatmap::max_text_width(rownames(mat), gp = grid::gpar(fontsize = base_size)) + grid::unit(3, "mm")
        } else {
            grid::unit(6, "cm")
        }
    }
    ComplexHeatmap::Heatmap(
        mat, name = name, col = col, na_col = "grey90",
        cluster_rows = cluster_rows, cluster_columns = cluster_columns,
        column_title = title, column_title_gp = grid::gpar(fontsize = base_size + 3, fontface = "bold"),
        row_names_gp = grid::gpar(fontsize = base_size), column_names_gp = grid::gpar(fontsize = base_size),
        row_names_max_width = row_names_max_width,
        heatmap_legend_param = list(
            title = name,
            title_gp = grid::gpar(fontsize = base_size, fontface = "bold"),
            labels_gp = grid::gpar(fontsize = base_size)
        ),
        cell_fun = cell_fun, ...
    )
}

heatmap_grid_page <- function(heatmaps, ncol = 1, title = NULL) {
    #' A page (a function, for save_plots_pdf) that lays a list of Heatmaps
    #' out on an nrow x ncol grid, with an optional title on top.
    force(heatmaps); force(ncol); force(title)
    function() {
        grid::grid.newpage()
        n_row <- ceiling(length(heatmaps) / ncol)
        grid::pushViewport(grid::viewport(layout = grid::grid.layout(
            n_row + 1, ncol,
            heights = grid::unit.c(grid::unit(2, "lines"), grid::unit(rep(1, n_row), "null"))
        )))
        if (!is.null(title)) {
            grid::pushViewport(grid::viewport(layout.pos.row = 1, layout.pos.col = seq_len(ncol)))
            grid::grid.text(title, gp = grid::gpar(fontsize = 16, fontface = "bold"))
            grid::popViewport()
        }
        for (i in seq_along(heatmaps)) {
            grid::pushViewport(grid::viewport(layout.pos.row = (i - 1) %/% ncol + 2, layout.pos.col = (i - 1) %% ncol + 1))
            ComplexHeatmap::draw(heatmaps[[i]], newpage = FALSE, merge_legend = TRUE, padding = grid::unit(c(2, 2, 2, 2), "mm"))
            grid::popViewport()
        }
        grid::popViewport()
    }
}

# ---------------------------------------------------------------------------
# Linear modeling figures (4.linear_modeling)
# ---------------------------------------------------------------------------

# UpSet layout: at most UPSET_TOP_N combinations are drawn. Column widths are in units of 0.5 in:
# set-size bars, term labels (fits "cell_per_organoid_count"), and the bar-row height (fits the
# two-line y-axis title)
UPSET_TOP_N <- 25
UPSET_SIZE_COL <- 4.2
UPSET_LABEL_COL <- 6.4
UPSET_BAR_ROW <- 3.4

upset_dims <- function(n_combo, n_set) {
    #' Page width / height (in) of a plot_upset() page.
    c(
        width = 0.5 * (UPSET_SIZE_COL + UPSET_LABEL_COL + max(min(n_combo, UPSET_TOP_N), 4) + 1.5) + 0.3,
        height = 3.8 + 0.42 * n_set
    )
}

# four-set Venn layout (plot coordinates): ellipse x, y, width, height, angle
venn_ellipses <- data.frame(
    set = 1:4,
    x = c(0.350, 0.450, 0.544, 0.644),
    y = c(0.400, 0.500, 0.500, 0.400),
    w = 0.72, h = 0.45,
    angle = c(140, 140, 40, 40)
)
# where the count of each region (bits in term order) is written
venn_region_xy <- data.frame(
    region_key = c(
        "1000", "0100", "0010", "0001", "1100", "1010", "1001", "0110",
        "0101", "0011", "1110", "1101", "1011", "0111", "1111"
    ),
    x = c(0.14, 0.32, 0.68, 0.85, 0.23, 0.29, 0.50, 0.50, 0.71, 0.77, 0.35, 0.61, 0.39, 0.65, 0.50),
    y = c(0.42, 0.72, 0.72, 0.42, 0.59, 0.30, 0.17, 0.66, 0.30, 0.59, 0.50, 0.24, 0.24, 0.50, 0.38)
)
venn_set_label_xy <- data.frame(x = c(0.02, 0.22, 0.78, 0.98), y = c(0.86, 0.97, 0.97, 0.86), hjust = c(0, 0.5, 0.5, 1))

ellipse_polygon <- function(x, y, w, h, angle, set, n = 200) {
    t <- seq(0, 2 * pi, length.out = n)
    theta <- angle * pi / 180
    data.frame(
        set = set,
        px = x + (w / 2) * cos(t) * cos(theta) - (h / 2) * sin(t) * sin(theta),
        py = y + (w / 2) * cos(t) * sin(theta) + (h / 2) * sin(t) * cos(theta)
    )
}

plot_venn <- function(regions, sizes, title) {
    #' Four-set Venn of feature membership across the linear-model variates.
    terms <- sizes$term
    stopifnot(length(terms) == 4)
    polys <- do.call(rbind, lapply(seq_len(4), function(i) do.call(ellipse_polygon, as.list(venn_ellipses[i, ]))))
    polys$term <- factor(terms[polys$set], levels = terms)
    counts <- venn_region_xy |>
        left_join(regions[, c("region_key", "n_features")], by = "region_key") |>
        mutate(n_features = ifelse(is.na(n_features), 0, n_features))
    labels <- cbind(venn_set_label_xy, term = terms, n = sizes$set_size)
    labels$label <- paste0(labels$term, "\n(", labels$n, " features)")
    (
        ggplot()
        + geom_polygon(data = polys, aes(px, py, group = term, fill = term), alpha = 0.3)
        + geom_path(data = polys, aes(px, py, group = term, colour = term), linewidth = 0.6)
        + geom_text(data = counts, aes(x, y, label = n_features), size = 6.5)
        + geom_text(
            data = labels, aes(x, y, label = label, colour = term, hjust = hjust),
            fontface = "bold", size = 5, lineheight = 0.9
        )
        + scale_fill_manual(values = linear_modeling_term_palette[terms])
        + scale_colour_manual(values = linear_modeling_term_palette[terms])
        + coord_fixed(xlim = c(0, 1), ylim = c(0, 1))
        + labs(title = paste0(
            "Features that are significant for each variate (", title, ")\n",
            sizes$n_none[1], " of ", sizes$n_total[1],
            " features are significant for none of these ", length(terms), " variates"
        ))
        + theme_manuscript_void(base_size = 14)
        + theme(legend.position = "none")
    )
}

plot_upset <- function(combos, sizes, title, top_n = UPSET_TOP_N, show_title = TRUE) {
    #' UpSet plot: bars = features in exactly that term combination, dots = which terms.
    #' show_title = FALSE drops the overall plot_annotation title (used for the multiresult-figure
    #' panel, which embeds this plot without a title even though the standalone PDF keeps one).
    terms <- sizes$term
    n_set <- length(terms)
    shown <- head(combos, top_n)
    n_combo <- nrow(shown)
    shown$x <- seq_len(n_combo)
    note <- if (n_combo < nrow(combos)) paste0("top ", n_combo, " of ", nrow(combos), " combinations; ") else ""

    p_bar <- (
        ggplot(shown, aes(x = x))
        + geom_rect(
            aes(xmin = x - 0.45, xmax = x + 0.45, ymin = 0.8, ymax = n_features, fill = treatment_specific)
        )
        + geom_text(aes(y = n_features * 1.08, label = n_features), vjust = 0, size = 5)
        + scale_fill_manual(values = treatment_specific_palette)
        + scale_y_log10(expand = c(0, 0))
        + coord_cartesian(xlim = c(0.4, n_combo + 0.6), ylim = c(0.8, max(shown$n_features) * 1.6), expand = FALSE)
        + labs(x = NULL, y = "features in exactly\nthis combination (log)")
        + theme_manuscript(base_size = 16, x_text = "blank")
        + theme(
            legend.position = "none", panel.grid = element_blank(), panel.border = element_blank(),
            axis.line.y = element_line(), plot.margin = margin(2, 2, 0, 2)
        )
    )

    membership <- shown |>
        select(x, all_of(paste0("in_", terms))) |>
        pivot_longer(-x, names_to = "term", values_to = "on") |>
        mutate(term = sub("^in_", "", term), row = match(term, terms))
    stripes <- data.frame(row = seq_len(n_set), shade = seq_len(n_set) %% 2 == 1)
    lines_df <- membership |> filter(on) |> group_by(x) |> filter(n() > 1) |> ungroup()
    p_mat <- (
        ggplot()
        + geom_rect(
            data = filter(stripes, shade),
            aes(xmin = 0.4, xmax = n_combo + 0.6, ymin = row - 0.5, ymax = row + 0.5), fill = upset_stripe_colour
        )
        + geom_point(data = membership, aes(x, row), colour = upset_dot_off_colour, size = 5.5)
        + geom_line(data = lines_df, aes(x, row, group = x), linewidth = 0.9)
        + geom_point(data = filter(membership, on), aes(x, row, fill = term), colour = "black", shape = 21, size = 5.5)
        + scale_fill_manual(values = linear_modeling_term_palette[terms])
        + scale_y_reverse(limits = c(n_set + 0.5, 0.5), expand = c(0, 0))
        + scale_x_continuous(limits = c(0.4, n_combo + 0.6), expand = c(0, 0))
        + theme_manuscript_void()
        + theme(legend.position = "none")
    )

    sizes$row <- seq_len(n_set)
    p_size <- (
        ggplot(sizes, aes(y = row))
        + geom_rect(aes(xmin = 0, xmax = set_size, ymin = row - 0.45, ymax = row + 0.45), fill = upset_set_size_colour)
        + geom_text(aes(x = set_size, label = paste0(set_size, " ")), hjust = 1, size = 4.5)
        + scale_x_reverse(limits = c(max(sizes$set_size) * 1.35, 0))
        + scale_y_reverse(limits = c(n_set + 0.5, 0.5), expand = c(0, 0))
        + labs(x = "features in group", y = NULL)
        + theme_manuscript(base_size = 15)
        + theme(
            panel.grid = element_blank(), panel.border = element_blank(), axis.line.x = element_line(),
            axis.text.y = element_blank(), axis.ticks.y = element_blank(), plot.margin = margin(0, 0, 2, 2)
        )
    )
    p_lab <- (
        ggplot(sizes, aes(y = row))
        + geom_text(aes(x = 0.02, label = term, colour = term), hjust = 0, fontface = "bold", size = 5)
        + scale_colour_manual(values = linear_modeling_term_palette[terms])
        + scale_x_continuous(limits = c(0, 1), expand = c(0, 0))
        + scale_y_reverse(limits = c(n_set + 0.5, 0.5), expand = c(0, 0))
        + theme_manuscript_void()
        + theme(legend.position = "none", plot.margin = margin(0, 2, 0, 2))
    )

    combined <- (
        (plot_spacer() + plot_spacer() + p_bar + p_size + p_lab + p_mat)
        + plot_layout(ncol = 3, widths = c(UPSET_SIZE_COL, UPSET_LABEL_COL, max(n_combo, 4) + 1.5), heights = c(UPSET_BAR_ROW, n_set * 0.55))
    )
    if (!show_title) return(combined)
    combined + plot_annotation(
        title = paste0("Feature membership across variate groups, ", title, " (", note, "orange = treatment only)"),
        theme = theme(plot.title = element_text(size = 14, face = "bold"), plot.margin = margin(4, 4, 2, 4))
    )
}

plot_patient_upset <- function(combos, patients, class_label, treatment) {
    #' UpSet plot whose sets are patients: bars = features with exactly that patient combination,
    #' dot matrix = which patients. Titled with the variate class and treatment it shows.
    n <- nrow(combos)
    n_pat <- length(patients)
    combos$x <- seq_len(n)
    use_log <- max(combos$n_features) / max(min(combos$n_features), 1) > 30
    p_bar <- (
        ggplot(combos, aes(x = x))
        + geom_rect(
            aes(xmin = x - 0.35, xmax = x + 0.35, ymin = if (use_log) 0.8 else 0, ymax = n_features),
            fill = upset_class_bar_colour
        )
        + geom_text(aes(y = n_features, label = n_features), vjust = -0.3, size = 3)
        + labs(
            x = NULL,
            y = paste0("features with exactly this\npatient combination", if (use_log) " (log)" else "")
        )
        + theme_manuscript(base_size = 9, x_text = "blank")
        + theme(
            panel.grid = element_blank(), panel.border = element_blank(),
            axis.line.y = element_line()
        )
    )
    p_bar <- if (use_log) {
        p_bar + scale_y_log10(expand = c(0, 0)) + coord_cartesian(xlim = c(0.4, n + 0.6), ylim = c(0.8, max(combos$n_features) * 3))
    } else {
        p_bar + scale_y_continuous(expand = c(0, 0)) + coord_cartesian(xlim = c(0.4, n + 0.6), ylim = c(0, max(combos$n_features) * 1.18))
    }
    dots <- do.call(rbind, lapply(seq_len(n), function(i) {
        data.frame(
            x = i, row = seq_len(n_pat), patient = patients,
            on = patients %in% strsplit(combos$patients[i], ";", fixed = TRUE)[[1]]
        )
    }))
    stripes <- data.frame(row = seq_len(n_pat), shade = seq_len(n_pat) %% 2 == 1)
    lines_df <- dots |> filter(on) |> group_by(x) |> filter(n() > 1) |> ungroup()
    p_mat <- (
        ggplot()
        + geom_rect(
            data = filter(stripes, shade),
            aes(xmin = 0.4, xmax = n + 0.6, ymin = row - 0.5, ymax = row + 0.5), fill = upset_stripe_colour
        )
        + geom_point(data = dots, aes(x, row), colour = upset_dot_off_colour, size = 3)
        + geom_line(data = lines_df, aes(x, row, group = x), linewidth = 0.8)
        + geom_point(data = filter(dots, on), aes(x, row), colour = "black", size = 3)
        + scale_y_reverse(breaks = seq_len(n_pat), labels = patients, limits = c(n_pat + 0.5, 0.5), expand = c(0, 0))
        + scale_x_continuous(limits = c(0.4, n + 0.6), expand = c(0, 0))
        + labs(x = NULL, y = NULL)
        + theme_manuscript_void(base_size = 9)
        + theme(axis.text.y = element_text(size = 9, hjust = 1))
    )
    (
        (p_bar / p_mat)
        + plot_layout(heights = c(2.4, n_pat * 0.42))
        + plot_annotation(
            title = paste0(class_label, " | ", treatment),
            subtitle = paste0(
                "Features significant for exactly these variates, by the patients they occur in\n",
                "(bars: number of features shared by exactly that patient combination)"
            ),
            theme = theme(
                plot.title = element_text(size = 12, face = "bold"),
                plot.subtitle = element_text(size = 9)
            )
        )
    )
}

cooccurrence_heatmap <- function(sig, title, show_title = TRUE, legend_title_size = 14, legend_label_size = 12) {
    #' Clustered heatmap of treatment-only significant features (rows) across patient-treatments
    #' (columns), annotated with patient, tumor type, drug, dose and feature type.
    #' show_title = FALSE drops column_title (used for the multiresult-figure subpanel, which embeds
    #' this heatmap without a title even though the standalone PDF keeps one).
    #' legend_*_size set the legend font sizes; the multiresult-figure subpanel enlarges them because it is shrunk to fit.
    legend_title_gp <- gpar(fontsize = legend_title_size, fontface = "bold")
    legend_labels_gp <- gpar(fontsize = legend_label_size)
    sig <- sig |> mutate(patient_treatment = paste(patient, treatment, sep = " | "))
    columns <- sig |>
        distinct(patient, treatment, patient_treatment) |>
        mutate(
            tumor_type = tumor_type_lookup[patient],
            drug = sub("_.*$", "", treatment),
            dose = sub("^.*_", "", treatment)
        )
    features <- sig |> distinct(feature, Feature_type) |> mutate(Feature_type = ifelse(is.na(Feature_type), "Other", Feature_type))
    mat <- matrix(0, nrow = nrow(features), ncol = nrow(columns), dimnames = list(features$feature, columns$patient_treatment))
    mat[cbind(match(sig$feature, features$feature), match(sig$patient_treatment, columns$patient_treatment))] <- 1

    top_annotation <- HeatmapAnnotation(
        Patient = columns$patient, `Tumor type` = columns$tumor_type,
        Treatment = columns$drug, Dose = columns$dose,
        col = list(Patient = patient_palette, `Tumor type` = tumor_type_palette,
                   Treatment = custom_treatment_palette, Dose = dose_label_palette),
        annotation_name_side = "left", annotation_name_gp = gpar(fontsize = 14),
        annotation_legend_param = list(
            Treatment = list(ncol = 2, title_gp = legend_title_gp, labels_gp = legend_labels_gp),
            Patient = list(title_gp = legend_title_gp, labels_gp = legend_labels_gp),
            `Tumor type` = list(title_gp = legend_title_gp, labels_gp = legend_labels_gp),
            Dose = list(title_gp = legend_title_gp, labels_gp = legend_labels_gp)
        ),
        simple_anno_size = unit(5, "mm")
    )
    left_annotation <- rowAnnotation(
        `Feature type` = features$Feature_type,
        col = list(`Feature type` = feature_type_palette_with_other),
        annotation_name_gp = gpar(fontsize = 14), simple_anno_size = unit(5, "mm"),
        annotation_legend_param = list(`Feature type` = list(title_gp = legend_title_gp, labels_gp = legend_labels_gp))
    )
    # the matrix is a yes/no occurrence, so it gets a discrete two-colour legend (clustering uses the 0/1 matrix)
    row_hc <- if (nrow(mat) > 1) hclust(dist(mat, method = "binary"), method = "average") else FALSE
    col_hc <- if (ncol(mat) > 1) hclust(dist(t(mat), method = "binary"), method = "average") else FALSE
    sig_mat <- matrix(ifelse(mat == 1, "yes", "no"), nrow = nrow(mat), dimnames = dimnames(mat))
    Heatmap(
        sig_mat, name = "treatment-only significant",
        col = treatment_only_heatmap_palette,
        heatmap_legend_param = list(
            title = "treatment\nonly terms", at = c("yes", "no"), border = "black",
            title_gp = legend_title_gp, labels_gp = legend_labels_gp
        ),
        cluster_rows = row_hc, cluster_columns = col_hc,
        top_annotation = top_annotation, left_annotation = left_annotation,
        show_row_names = nrow(mat) <= 100, row_names_gp = gpar(fontsize = 8),
        show_column_names = FALSE, use_raster = TRUE, raster_quality = 2,
        column_title = if (show_title) paste0(title, "\n", nrow(mat), " features x ", ncol(mat), " patient-treatments") else NULL,
        column_title_gp = gpar(fontsize = 16, fontface = "bold"),
        row_title = "feature (clustered)", row_title_gp = gpar(fontsize = 14), column_title_side = "top"
    )
}

plot_clustermap <- function(read_result, matrix_name, order_prefix, label) {
    #' Clustermap of all patient x treatment class-count profiles: log1p, column min-max scaling,
    #' hierarchical clustering (Ward.D2 on the scaled values). Row annotations use the manuscript
    #' palettes; the bar at the bottom shows which variates are significant in each class (column).
    #' `read_result(name)` loads a table of the run (see 8.plot_variate_class_upsets_and_clustermap).
    mat <- read_result(matrix_name) |>
        pivot_longer(-c(patient, treatment), names_to = "class", values_to = "value") |>
        mutate(key = paste(patient, treatment, sep = "|"), value = log1p(value))
    col_order <- read_result(paste0(order_prefix, "_column_order.parquet"))$column_order
    row_order <- read_result(paste0(order_prefix, "_row_order.parquet"))$row_order
    mat <- mat |>
        filter(class %in% col_order, key %in% row_order) |>
        group_by(class) |>
        mutate(scaled = (value - min(value)) / (max(value) - min(value))) |>
        ungroup()
    scaled <- long_to_matrix(mat, "key", "class", "scaled", row_levels = row_order, col_levels = col_order)
    scaled[is.na(scaled)] <- 0  # constant columns have no range to scale

    patient_of_row <- sub("\\|.*$", "", row_order)
    treatment_of_row <- sub("^[^|]*\\|", "", row_order)
    left_annotation <- rowAnnotation(
        Patient = patient_of_row, `Tumor type` = tumor_type_lookup[patient_of_row],
        Treatment = sub("_.*$", "", treatment_of_row), Dose = sub("^.*_", "", treatment_of_row),
        col = list(Patient = patient_palette, `Tumor type` = tumor_type_palette,
                   Treatment = custom_treatment_palette, Dose = dose_label_palette),
        annotation_name_gp = gpar(fontsize = 9), simple_anno_size = unit(4, "mm"),
        annotation_legend_param = list(Treatment = list(ncol = 2))
    )
    # presence / absence of each variate in each class (column)
    in_class <- sapply(names(variate_letters), function(v) ifelse(sapply(strsplit(col_order, "+", fixed = TRUE), function(x) v %in% x), "yes", "no"))
    colnames(in_class) <- paste0(names(variate_letters), " ", variate_letters)
    bottom_annotation <- HeatmapAnnotation(
        df = as.data.frame(in_class),
        col = setNames(rep(list(variate_presence_palette), ncol(in_class)), colnames(in_class)),
        show_legend = c(TRUE, rep(FALSE, ncol(in_class) - 1)),
        annotation_legend_param = list(list(title = "variate significant\nin class", at = c("yes", "no"))),
        annotation_name_side = "right", annotation_name_gp = gpar(fontsize = 8),
        simple_anno_size = unit(3, "mm")
    )
    simple_heatmap(
        scaled, label, heat_col_fun(c(0, 1), palette = "Viridis", rev = FALSE),
        cluster_rows = TRUE, cluster_columns = TRUE,
        clustering_method_rows = "ward.D2", clustering_method_columns = "ward.D2",
        left_annotation = left_annotation, bottom_annotation = bottom_annotation,
        show_row_names = FALSE, show_column_names = FALSE, use_raster = TRUE
    )
}
