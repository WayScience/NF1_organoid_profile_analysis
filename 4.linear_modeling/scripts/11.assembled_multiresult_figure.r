list_of_packages <- c("ggplot2", "dplyr", "tidyr", "patchwork", "arrow", "ggrastr", "RColorBrewer", "png", "grid")
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

module_dir <- file.path(root_dir, "4.linear_modeling")
results_path <- file.path(module_dir, "results")
figures_path <- file.path(module_dir, "figures", "headline_results_figure")
dir.create(figures_path, recursive = TRUE, showWarnings = FALSE)


# # Summary figure: the most load-bearing results across the linear-modeling pipeline
#
# One panel per question, all drawn from tables/figures already saved by the calculation
# notebooks (9.explore_linear_model_haystacks and 6.plot_variate_importance) -- nothing is recomputed here.
#
# A. Where does the variance go (treatment vs technical covariates)?
# B. How many/how strong are the treatment effects (volcano)?
# C. Tumor-type UpSet (title-free subpanel from 6.plot_variate_importance)
# D. Treatment-only variate co-occurrence (title-free subpanel from 6.plot_variate_importance)
#
# Panels C and D embed pre-rendered, title-free PNGs saved by 6.plot_variate_importance's
# "Multiresult-figure subpanels" section (the standalone PDFs in figures/variate_importance/ keep
# their titles; only these dedicated panel PNGs omit them).

# ---- A. variance partition -------------------------------------------------
# grouped boxplot of all variates (per-model % of total variance), matching
# 10.plot_explore_linear_model_haystacks.r's page 2
variate_levels <- c(
    "treatment", "cell_count", "organoid_count", "cell_per_organoid_count",
    "manhattan_distance_from_center", "cell_x_position", "cell_y_position",
    "cell_z_position", "cell_z_depth", "residual"
)
partition_long <- arrow::read_parquet(
    file.path(
        results_path, "explore_linear_models", "plot_data",
        "5_variance_partition_by_profile__partition_long.parquet"
    )
) |>
    filter(profile %in% main_profiles) |>
    mutate(profile = factor(profile, levels = main_profiles), term = factor(term, levels = variate_levels))
pA <- (
    ggplot(partition_long, aes(x = term, y = pct, fill = profile))
    + geom_boxplot(position = position_dodge(width = 0.8), width = 0.7, outlier.size = 0.3)
    + scale_fill_manual(values = profile_palette, drop = FALSE, breaks = main_profiles)
    + labs(x = NULL, y = "% of total variance", fill = NULL)
    + theme_manuscript(base_size = 14)
    + theme(
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "bottom", legend.text = element_text(size = 13), legend.key.size = unit(6, "mm")
    )
)

# ---- B. volcano (treatment effect size vs significance) -------------------
volcano <- arrow::read_parquet(
    file.path(results_path, "explore_linear_models", "plot_data", "4_volcano_by_profile__volcano.parquet")
) |>
    filter(profile %in% main_profiles) |>
    mutate(profile = factor(profile, levels = main_profiles))
pB <- (
    ggplot(volcano, aes(x = coefficient, y = neglog_fdr))
    + rasterise(geom_point(aes(colour = rsquared), size = 1, alpha = 0.3), dpi = 600)
    + geom_hline(yintercept = -log10(0.05), linetype = "dashed", linewidth = 0.3)
    + scale_colour_viridis_c(option = "A", name = "R2", limits = c(0, 1))
    + facet_wrap(~profile, ncol = 2, scales = "free")
    + labs(x = "treatment coefficient", y = "-log10 FDR")
    + theme_manuscript(base_size = 14)
)

# ---- treatment-only variate co-occurrence (assembled as panel D; title-free subpanel from 6.plot_variate_importance) --
cooccurrence_png <- file.path(
    module_dir, "figures", "variate_importance", "multiresult_figure_subpanels",
    "cooccurrence_organoid_original.png"
)
if (!file.exists(cooccurrence_png)) {
    stop(
        "missing ", cooccurrence_png, " -- run scripts/6.plot_variate_importance.r first ",
        "(it saves this PNG, which panel D renders)"
    )
}
img_cooccurrence <- png::readPNG(cooccurrence_png)
pC <- wrap_elements(full = grid::rasterGrob(img_cooccurrence, interpolate = TRUE))

# ---- tumor-type UpSet (assembled as panel C; title-free subpanel from 6.plot_variate_importance: original model, tumor_type=pNF) --
tumor_type_png <- file.path(
    module_dir, "figures", "variate_importance", "multiresult_figure_subpanels",
    "tumor_type_pNF_upset.png"
)
if (!file.exists(tumor_type_png)) {
    stop(
        "missing ", tumor_type_png, " -- run 6.plot_variate_importance first ",
        "(it saves this PNG, which panel C renders)"
    )
}
img_upset <- png::readPNG(tumor_type_png)
pD <- wrap_elements(full = grid::rasterGrob(img_upset, interpolate = TRUE))

# ---- assemble ---------------------------------------------------------------
# C and D are pre-rendered PNGs (landscape: wide bar/matrix charts), so giving them
# a portrait cell (as a shared, doubled-height row under A/B) let-boxed them with a
# lot of blank space top/bottom. Instead each gets its own full-width row sized to
# its own aspect ratio, so the embedded image fills its cell with no dead space.
fig_width <- 16
fig_height <- 16
layout <- "
AABB
CCDD
"
headline_results_figure <- (
    wrap_plots(A = pA, B = pB, C = pD, D = pC, design = layout)
    + plot_annotation(tag_levels = "A")
)
save_ggplot(
    headline_results_figure, file.path(figures_path, "multiresult_figure.png"),
    width = fig_width, height = fig_height, dpi = 600
)


