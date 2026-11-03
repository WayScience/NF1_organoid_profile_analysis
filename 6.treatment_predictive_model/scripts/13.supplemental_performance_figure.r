list_of_packages <- c("ggplot2", "patchwork", "magick", "grid", "RColorBrewer")
for (package in list_of_packages) {
    suppressPackageStartupMessages(
        suppressWarnings(
            library(package, character.only = TRUE, quietly = TRUE, warn.conflicts = FALSE)
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
source(file.path(root_dir, "utils", "r_plot_funcs.r"))

module_dir <- file.path(root_dir, "6.treatment_predictive_model")
figures_dir <- file.path(module_dir, "figures")
dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)

embed_png_panel <- function(path) {
    #' Reads a saved PNG and trims its (mostly white) border margin before
    #' wrapping it as a patchwork-compatible panel - each source figure was
    #' saved with its own surrounding whitespace, which otherwise stacks up
    #' when the panels are packed into a grid alongside each other.
    if (!file.exists(path)) {
        stop(
            "missing ", path, " - run the notebook that saves this figure first."
        )
    }
    img <- magick::image_trim(magick::image_read(path))
    wrap_elements(full = grid::rasterGrob(as.raster(img), interpolate = TRUE))
}

panel_a <- embed_png_panel(file.path(figures_dir, "binary_mlp_control_training_curves.png"))
panel_b <- embed_png_panel(file.path(figures_dir, "binary_mlp_control_pr_curves.png"))
panel_c <- embed_png_panel(file.path(figures_dir, "multiclass_mlp_treatment_dose_training_curve.png"))
panel_d <- embed_png_panel(file.path(figures_dir, "multiclass_mlp_treatment_dose_top_k_accuracy.png"))
panel_e <- embed_png_panel(file.path(figures_dir, "multiclass_mlp_treatment_dose_per_class_f1.png"))
panel_f <- embed_png_panel(file.path(figures_dir, "multiclass_mlp_target_training_curve.png"))
panel_g <- embed_png_panel(file.path(figures_dir, "multiclass_mlp_target_top_k_accuracy.png"))
panel_h <- embed_png_panel(file.path(figures_dir, "multiclass_mlp_target_per_class_f1.png"))

fig_width <- 22
fig_height <- 11
layout <- "
ABCD
EFGH
"

dims <- set_plot_size(fig_width, fig_height)

supplemental_performance_figure <- (
    wrap_plots(
        A = panel_a, B = panel_b, C = panel_c, D = panel_d,
        E = panel_e, F = panel_f, G = panel_g, H = panel_h,
        design = layout
    )
    + plot_annotation(tag_levels = "A")
    & theme(plot.margin = margin(2, 2, 2, 2))
)

save_ggplot(
    supplemental_performance_figure,
    file.path(figures_dir, "supplemental_performance_figure.png"),
    width = dims$width, height = dims$height, dpi = 600
)
