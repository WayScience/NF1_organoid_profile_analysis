list_of_packages <- c("ggplot2", "dplyr", "tidyr", "patchwork", "arrow", "ggrepel", "ggrastr", "scales", "ComplexHeatmap", "circlize", "grid", "RColorBrewer")
for (package in list_of_packages) {
    suppressPackageStartupMessages(
        suppressWarnings(
            library(package, character.only = TRUE, quietly = TRUE, warn.conflicts = FALSE)
        )
    )
}

find_git_root <- function() {
    current_path <- getwd()
    while (!dir.exists(file.path(current_path, ".git"))) {
        parent_path <- dirname(current_path)
        if (parent_path == current_path) stop("No Git root directory found.")
        current_path <- parent_path
    }
    current_path
}
root_dir <- find_git_root()
source(file.path(root_dir, "utils/r_plot_themes.r"))
source(file.path(root_dir, "utils/r_plot_funcs.r"))

# run tag of the tables made by 7.calculate_variate_class_upsets_and_clustermap (profile + variate groups)
tag <- "sc_T-O-C-N-M-X-Y-Z-D"
results_path <- file.path(root_dir, "4.linear_modeling/results/variate_class_plots", tag)

read_result <- function(name) arrow::read_parquet(file.path(results_path, name))

membership <- read_result("membership.parquet")
patients <- sort(unique(membership$patient))
treatments <- sort(unique(membership$treatment))
if (tag == "sc_T-O-C-N-M-X-Y-Z-D") {
    figure_name <- "sc_technical_model_results"
}
figures_path <- file.path(root_dir, "4.linear_modeling/figures/variate_class_plots", figure_name)
dir.create(file.path(figures_path, "upset"), recursive = TRUE, showWarnings = FALSE)
tag_pdf <- file.path(figures_path, paste0(figure_name, ".pdf"))

# Task A: UpSet plots, sets = patients. One pdf per class, one page per treatment.
# a class's pdf is only rebuilt when it is missing
upset_summary <- read_result("upset_top_combinations.parquet")
for (cls in sort(unique(upset_summary$class))) {
    letters_in_cls <- strsplit(cls, "+", fixed = TRUE)[[1]]
    cls_file_name <- paste(unname(variate_letters[letters_in_cls]), collapse = "__")
    class_label <- paste(unname(variate_letters[letters_in_cls]), collapse = " + ")

    pdf_path <- file.path(figures_path, "upset", paste0(cls_file_name, ".pdf"))
    if (file.exists(pdf_path)) next
    cls_df <- upset_summary |> filter(class == cls)
    plots <- list()
    widths <- c()
    heights <- c()
    for (trt in sort(unique(cls_df$treatment))) {
        combos <- cls_df |> filter(treatment == trt) |> arrange(rank)
        plots[[length(plots) + 1]] <- plot_patient_upset(combos, patients, class_label, trt)
        widths <- c(widths, max(7.5, 0.75 * nrow(combos) + 3.5))
        heights <- c(heights, 4.2 + 0.36 * length(patients))
    }
    save_plots_pdf(plots, pdf_path, width = widths, height = heights)
}

# every remaining figure is a page of the run's single pdf
pages <- list()
page_w <- c()
page_h <- c()
add_page <- function(plot, width, height) {
    pages[[length(pages) + 1]] <<- plot
    page_w <<- c(page_w, width)
    page_h <<- c(page_h, height)
}

# Task B: clustermap of all patient x treatment class-count profiles. log1p, column min-max scaling; rows and
# columns are hierarchically clustered here (Ward.D2 on the scaled values). Row annotations use the manuscript
# theme palettes; the bar at the bottom shows which variates are significant in each class (column).
add_page(
    heatmap_grid_page(list(plot_clustermap(read_result, "class_count_matrix.parquet", "clustermap_counts", "column-scaled\nlog1p(count)"))),
    10, 12
)
add_page(
    heatmap_grid_page(list(plot_clustermap(read_result, "class_fraction_matrix.parquet", "clustermap_fractions", "column-scaled\nlog1p(fraction)"))),
    10, 12
)

# Task C: which features are treatment-only (class T)?
# 7 writes the treatment-only tables only when some feature is in class T; without them Tasks C and D are skipped
has_treatment_only <- file.exists(file.path(results_path, "treatment_only_feature_recurrence.parquet"))
if (has_treatment_only) {
    recurrence <- read_result("treatment_only_feature_recurrence.parquet") |>
        mutate(channel = ifelse(is.na(channel), "none", channel))
    # the top features are the ones 7.calculate_variate_class_upsets_and_clustermap kept (its --top_features)
    per_treatment_wide <- read_result("treatment_only_feature_by_treatment.parquet")
    top <- recurrence |> filter(feature %in% per_treatment_wide$feature)
    top_features <- top$feature
    top$feature <- factor(top$feature, levels = rev(top$feature))
    p_bar <- (
        ggplot(top, aes(x = n_patient_treatments, y = feature, fill = feature_type))
        + geom_col()
        + labs(
            x = "# (patient, treatment) pairs where feature is treatment-only", y = NULL,
            fill = "feature type"
        )
        + theme_manuscript(base_size = 10)
        + theme(axis.text.y = element_text(size = 7))
    )
    add_page(p_bar, 12, max(6, 0.28 * nrow(top)))

    grid_df <- recurrence |> count(channel, feature_type, name = "n_features")
    grid_mat <- long_to_matrix(grid_df, "channel", "feature_type", "n_features")
    add_page(
        heatmap_grid_page(list(simple_heatmap(
            grid_mat, "# distinct\nfeatures", heat_col_fun(grid_mat), cell_fmt = "%d", cell_size = 10,
            column_names_rot = 45
        ))),
        9, 7
    )

    per_treatment <- per_treatment_wide |>
        pivot_longer(-feature, names_to = "treatment", values_to = "n_patients")
    per_treatment_mat <- long_to_matrix(per_treatment, "feature", "treatment", "n_patients", row_levels = top_features)
    add_page(
        heatmap_grid_page(list(simple_heatmap(
            per_treatment_mat, "# patients\n(feature is\ntreatment-only)", heat_col_fun(per_treatment_mat),
            base_size = 9
        ))),
        max(14, 0.4 * length(unique(per_treatment$treatment)) + 9), max(6, 0.28 * nrow(top))
    )
} else {
    cat("no treatment-only (class T) hits in this run; skipping Tasks C and D\n")
}


# Task D: distribution of the top shared treatment-only features
if (has_treatment_only) {
    shared <- read_result("top_shared_features_coefficients.parquet")
    top_shared <- intersect(recurrence$feature, unique(as.character(shared$feature)))
    shared <- shared |> mutate(feature = factor(feature, levels = rev(top_shared)))
    p_box <- (
        ggplot(shared, aes(x = coefficient, y = feature))
        + geom_boxplot(fill = "lightgrey", outlier.shape = NA)
        + rasterise(geom_jitter(aes(colour = treatment_only), height = 0.2, size = 0.6, alpha = 0.5), dpi = 600)
        + geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.4)
        + scale_colour_manual(values = treatment_only_palette, name = "treatment-only significant")
        + labs(x = "treatment coefficient (all patient x treatment models)", y = NULL)
        + theme_manuscript(base_size = 10)
    )
    frac <- shared |> group_by(feature) |> summarise(fraction = mean(treatment_only), .groups = "drop")
    p_frac <- (
        ggplot(frac, aes(x = fraction, y = feature))
        + geom_col(fill = treatment_only_fill)
        + labs(x = "fraction of models that are treatment-only significant", y = NULL)
        + theme_manuscript(base_size = 10)
        + theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())
    )
    add_page((p_box + p_frac) + plot_layout(widths = c(2.2, 1)), 16, 5.5)

    space <- read_result("top_shared_features_space_distribution.parquet") |>
        pivot_longer(-c(value, part), names_to = "set", values_to = "fraction") |>
        mutate(part = factor(part, levels = c("compartment", "channel", "feature_type")))
    p_space <- (
        ggplot(space, aes(x = value, y = fraction, fill = set))
        + geom_col(position = position_dodge(width = 0.8), width = 0.75)
        + facet_wrap(~part, scales = "free_x", nrow = 1)
        + scale_fill_manual(values = setNames(all_vs_top_palette, sort(unique(space$set))))
        + labs(x = NULL, y = "fraction of features", fill = NULL)
        + theme_manuscript(base_size = 10)
        + theme(axis.text.x = element_text(angle = 45, hjust = 1))
    )
    add_page(p_space, 18, 4.5)

    for (by in c("patient", "treatment")) {
        med <- shared |> group_by(feature, .data[[by]]) |> summarise(median_coef = median(coefficient), .groups = "drop")
        n_significant <- shared |> filter(treatment_only) |> count(feature, .data[[by]], name = "n_significant")
        med <- med |> left_join(n_significant, by = c("feature", by)) |> mutate(n_significant = ifelse(is.na(n_significant), 0L, n_significant))
        lim <- quantile(abs(med$median_coef), 0.98, na.rm = TRUE)
        med_mat <- long_to_matrix(med, "feature", by, "median_coef", row_levels = top_shared)
        significant_labels <- if (by == "patient") {
            matrix(as.character(long_to_matrix(med, "feature", by, "n_significant", row_levels = top_shared)), nrow = length(top_shared),
                   dimnames = dimnames(med_mat))
        } else NULL
        ht <- simple_heatmap(
            med_mat, "median treatment\ncoefficient",
            circlize::colorRamp2(c(-lim, 0, lim), coefficient_diverging_colours),
            cell_labels = significant_labels, cell_size = 9, column_names_rot = 90
        )
        # extra width to fit the long feature-name row labels (see simple_heatmap's row_names_max_width) next to the heatmap body and legend
        row_label_width <- max(nchar(top_shared)) * 0.09 + 0.5
        base_width <- if (by == "patient") 8 else max(9, 0.4 * length(treatments) + 5)
        add_page(heatmap_grid_page(list(ht)), base_width + row_label_width, 5)
    }
}


save_plots_pdf(pages, tag_pdf, width = page_w, height = page_h)

