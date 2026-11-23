list_of_packages <- c("ggplot2", "dplyr", "tidyr", "patchwork", "arrow", "ggrepel", "ggrastr", "scales", "RColorBrewer", "ComplexHeatmap", "circlize", "grid")
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

plot_data_path <- file.path(root_dir, "4.linear_modeling/results/explore_linear_models/plot_data")
figures_path <- file.path(root_dir, "4.linear_modeling/figures/explore_linear_models")
dir.create(figures_path, recursive = TRUE, showWarnings = FALSE)
pdf_path <- file.path(figures_path, "explore_linear_models.pdf")

# tables saved by 9.explore_linear_model_haystacks for each figure: plot_data/<figure>__<key>.parquet
read_plot <- function(figure, key) {
    arrow::read_parquet(file.path(plot_data_path, paste0(figure, "__", key, ".parquet")))
}

# significance threshold, identical to the calculation notebook
FDR_MAX <- 0.05

# every figure is a page of one pdf, in the order it is added
pages <- list()
page_w <- c()
page_h <- c()
add_page <- function(plot, width, height) {
    pages[[length(pages) + 1]] <<- plot
    page_w <<- c(page_w, width)
    page_h <<- c(page_h, height)
}

# the significance measure, named for what it counts
SIGNIFICANT_LABEL <- "statistically significant\nvariate per model"
SIGNIFICANT_LABEL_INLINE <- "statistically significant variate per model"

# treatments (e.g. "Trametinib_1uM") ordered as in the manuscript theme (drug order, then dose)
order_treatments <- function(x) {
    u <- unique(as.character(x))
    drug <- sub("_.*$", "", u)
    u <- u[order(match(drug, custom_treatment_order), u)]
    factor(x, levels = u)
}

# 4. volcano plots: significance on the y-axis, effect size on the x-axis, every point coloured by R2 (full 0-1 range)
volcano <- read_plot("4_volcano_by_profile", "volcano") |> mutate(profile = factor(profile, levels = profiles))
p_volcano <- (
    ggplot(volcano, aes(x = coefficient, y = neglog_fdr))
    + rasterise(geom_point(aes(colour = rsquared), size = 2, alpha = 0.25), dpi = 600)
    + geom_hline(yintercept = -log10(FDR_MAX), linetype = "dashed", linewidth = 0.4)
    + scale_colour_viridis_c(option = "A", name = "R2", limits = c(0, 1), breaks = seq(0, 1, 0.2), na.value = "lightgrey")
    + facet_wrap(~profile, ncol = 2, scales = "free")
    + labs(x = "treatment coefficient", y = "-log10 FDR")
    + theme_manuscript(base_size = 24)
    + theme(legend.key.height = grid::unit(2.2, "cm"))
    + guides(colour = guide_colourbar())
)
add_page(p_volcano, 18, 14)

# 5. variance partitioning: where does the variance go?
# variates in the order they are described in the model of 3.linear_modeling_technical_vars, then what is left
variate_levels <- c(
    "treatment", "cell_count", "organoid_count", "cell_per_organoid_count",
    "manhattan_distance_from_center", "cell_x_position", "cell_y_position",
    "cell_z_position", "cell_z_depth", "residual"
)
partition_long <- read_plot("5_variance_partition_by_profile", "partition_long") |>
    filter(profile %in% main_profiles) |>
    mutate(
        profile = factor(profile, levels = main_profiles),
        term = factor(term, levels = variate_levels)
    )

# number of linear models fit per profile
model_counts <- read_plot("5_variance_partition_by_profile", "model_counts")
cat("models fit per profile:\n")
print(model_counts)

p_partition <- (
    ggplot(partition_long, aes(x = term, y = pct, fill = profile))
    + geom_boxplot(position = position_dodge(width = 0.8), width = 0.7, outlier.size = 0.5)
    + scale_fill_manual(values = profile_palette, drop = FALSE, breaks = main_profiles)
    + labs(x = NULL, y = "% of total variance", fill = NULL)
    + theme_manuscript(base_size = 16)
    + theme(axis.text.x = element_text(angle = 45, hjust = 1))
)
add_page(p_partition, 16, 6)

share <- read_plot("5_treatment_variance_share_heatmap", "share") |> mutate(treatment = order_treatments(treatment))
share_heatmaps <- lapply(main_profiles, function(pf) {
    d <- filter(share, profile == pf)
    simple_heatmap(
        long_to_matrix(d, "patient", "treatment", "median_pct_var", col_levels = levels(share$treatment)),
        "median treatment\n% var", heat_col_fun(share$median_pct_var)
    )
})
add_page(heatmap_grid_page(share_heatmaps, ncol = 1), 16, 12)

# 6. significance landscape: which patients, drugs, MOAs, doses and tumor types produce the most significant variates?
landscape <- read_plot("6_hit_rate_by_group", "landscape")
levels_order <- c("patient", "treatment", "drug", "therapeutic_category", "tumor_type", "dose")
landscape <- landscape |> mutate(level = factor(level, levels = levels_order), profile = factor(profile, levels = profiles))
# groups ordered by the organoid significant-variate rate within each level (groups the organoid profile lacks go last)
group_order <- landscape |>
    filter(profile == "organoid") |>
    arrange(level, desc(hit_rate)) |>
    mutate(key = paste(level, group, sep = "___"))
landscape <- landscape |> mutate(key = paste(level, group, sep = "___"))
key_levels <- c(rev(group_order$key), setdiff(unique(landscape$key), group_order$key))
landscape$key <- factor(landscape$key, levels = key_levels)
p_landscape <- (
    ggplot(landscape, aes(x = hit_rate, y = key, fill = profile))
    + geom_col(position = position_dodge(width = 0.8), width = 0.75)
    + scale_y_discrete(labels = function(x) sub("^.*___", "", x))
    + scale_fill_manual(values = profile_palette)
    + facet_wrap(~level, scales = "free_y", ncol = 3)
    + labs(x = SIGNIFICANT_LABEL_INLINE, y = NULL, fill = "profile")
    + theme_manuscript(base_size = 14)
)
add_page(p_landscape, 22, 11)

hit_rate_pt <- read_plot("6_hit_rate_patient_by_treatment", "hit_rate") |> mutate(treatment = order_treatments(treatment))
hit_rate_heatmaps <- lapply(main_profiles, function(pf) {
    d <- filter(hit_rate_pt, profile == pf)
    simple_heatmap(
        long_to_matrix(d, "patient", "treatment", "hit_rate", col_levels = levels(hit_rate_pt$treatment)),
        SIGNIFICANT_LABEL, heat_col_fun(hit_rate_pt$hit_rate)
    )
})
add_page(heatmap_grid_page(hit_rate_heatmaps, ncol = 1), 16, 12)

forcats_rev <- function(f) factor(f, levels = rev(levels(f)))
direction <- read_plot("6_hit_direction_by_treatment", "direction") |>
    pivot_longer(c(down, up), names_to = "direction", values_to = "n_significant") |>
    mutate(profile = factor(profile, levels = main_profiles), treatment = order_treatments(treatment), direction = factor(direction, levels = c("down", "up")))
p_direction <- (
    ggplot(direction, aes(x = n_significant, y = forcats_rev(treatment), fill = direction))
    + geom_col()
    + geom_vline(xintercept = 0, linewidth = 0.3)
    + scale_fill_manual(values = direction_palette)
    + facet_wrap(~profile, ncol = 2, scales = "free")
    + labs(x = "number of significant variates (down < 0 < up)", y = NULL)
    + theme_manuscript(base_size = 14)
)
add_page(p_direction, 20, 7)

# 7. feature space enrichment: Fisher exact test of significant vs non-significant for every compartment / channel / feature type
enrichment <- read_plot("7_feature_family_enrichment", "enrichment") |>
    filter(family %in% c("compartment_label", "channel_label", "Feature_type")) |>
    mutate(
        profile = factor(profile, levels = main_profiles),
        family = factor(family, levels = c("compartment_label", "channel_label", "Feature_type")),
        colour = ifelse(fdr < 0.05, ifelse(log2_or > 0, "enriched", "depleted"), "n.s."),
        key = paste(profile, family, level, sep = "___")
    ) |>
    group_by(profile, family) |>
    arrange(log2_or, .by_group = TRUE) |>
    ungroup()
enrichment$key <- factor(enrichment$key, levels = enrichment$key)
p_enrichment <- (
    ggplot(enrichment, aes(x = log2_or, y = key, fill = colour))
    + geom_col()
    + geom_vline(xintercept = 0, linewidth = 0.3)
    + scale_y_discrete(labels = function(x) sub("^.*___", "", x))
    + scale_fill_manual(values = enrichment_palette, name = "FDR < 0.05")
    + facet_grid(profile ~ family, scales = "free")
    + labs(x = "log2 odds ratio of being significant", y = NULL)
    + theme_manuscript(base_size = 14)
)
add_page(p_enrichment, 22, 12)

channel_heat <- read_plot("7_channel_by_feature_type_heatmaps", "heatmap")
channel_heatmaps <- list()
for (pf in main_profiles) {
    for (k in unique(channel_heat$kind)) {
        d <- filter(channel_heat, profile == pf, kind == k)
        channel_heatmaps[[length(channel_heatmaps) + 1]] <- simple_heatmap(
            long_to_matrix(d, "channel", "column", "hit_rate"),
            SIGNIFICANT_LABEL, heat_col_fun(channel_heat$hit_rate),
            cell_fmt = "%.3f", cell_size = 9
        )
    }
}
add_page(
    heatmap_grid_page(channel_heatmaps, ncol = length(unique(channel_heat$kind))),
    20, 12
)

drug_channel <- read_plot("7_drug_by_channel_hit_rate", "heatmap")
drug_channel_heatmaps <- lapply(main_profiles, function(pf) {
    d <- filter(drug_channel, profile == pf)
    simple_heatmap(
        long_to_matrix(d, "drug", "channel_label", "hit_rate", row_levels = intersect(custom_treatment_order, unique(d$drug))),
        SIGNIFICANT_LABEL, heat_col_fun(drug_channel$hit_rate)
    )
})
add_page(heatmap_grid_page(drug_channel_heatmaps, ncol = 2), 20, 9)

# 8. MEK inhibitor concordance: do the MEK inhibitors (at each dose) agree on the treatment coefficient?
# Spearman correlation of the treatment coefficient across every (patient, feature), per profile
lm_results_path <- file.path(root_dir, "4.linear_modeling/results/linear_modeling")
lm_files <- c(
    organoid = "organoid_norm.parquet",
    sc = "sc_norm.parquet",
    organoid_agg = "organoid_agg.parquet",
    sc_agg = "single_cell_agg.parquet"
)
meki_drugs <- names(treatment_moa_map)[treatment_moa_map == "MEK1/2 inhibitor"]

meki_heatmaps <- lapply(profiles, function(profile) {
    coefs <- arrow::read_parquet(
        file.path(lm_results_path, lm_files[[profile]]),
        col_select = c("term", "patient", "treatment", "drug", "feature", "coefficient")
    ) |>
        filter(term == "treatment", drug %in% meki_drugs) |>
        select(patient, feature, treatment, coefficient) |>
        pivot_wider(names_from = treatment, values_from = coefficient)
    treatments <- levels(order_treatments(setdiff(names(coefs), c("patient", "feature"))))
    rho <- cor(coefs[treatments], method = "spearman", use = "pairwise.complete.obs")
    simple_heatmap(
        rho, "Spearman", circlize::colorRamp2(c(-1, 0, 1), correlation_diverging_colours),
        cell_fmt = "%.2f", cell_size = 9
    )
})
add_page(heatmap_grid_page(meki_heatmaps, ncol = 2), 20, 18)

save_plots_pdf(pages, pdf_path, width = page_w, height = page_h)
