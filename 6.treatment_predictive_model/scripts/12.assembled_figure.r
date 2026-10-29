list_of_packages <- c("ggplot2", "dplyr", "arrow", "patchwork", "magick", "grid", "RColorBrewer")
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
results_dir <- file.path(module_dir, "model_results")
figures_dir <- file.path(module_dir, "figures")
dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)

embed_png_panel <- function(path) {
    #' Reads a saved PNG, trims its (mostly white) border margin, and wraps
    #' it as a patchwork-compatible panel that exactly fills its grid cell
    #' (unit(1, "npc") width/height). This is only distortion-free because
    #' the assembly below sizes each panel's cell to that panel's own
    #' trimmed aspect ratio - see the "Assemble" cell.
    if (!file.exists(path)) {
        stop(
            "missing ", path, " - run the notebook that saves this figure first."
        )
    }
    img <- magick::image_trim(magick::image_read(path))
    wrap_elements(
        full = grid::rasterGrob(
            as.raster(img), interpolate = TRUE,
            width = unit(1, "npc"), height = unit(1, "npc")
        )
    )
}

panel_a <- embed_png_panel(file.path(figures_dir, "binary_mlp_control_pr_curves.png"))
panel_b <- embed_png_panel(file.path(figures_dir, "binary_mlp_control_heatmap.png"))
panel_c <- embed_png_panel(
    file.path(figures_dir, "multiclass_mlp_treatment_dose_confusion_matrix.png")
)
panel_d <- embed_png_panel(file.path(figures_dir, "multiclass_mlp_target_confusion_matrix.png"))

# Trimmed aspect ratios (width/height) of each panel's saved PNG - see the
# "Assemble" markdown above for why these drive the layout.
aspect_a <- 0.993  # binary_mlp_control_pr_curves.png
aspect_b <- 1.761  # binary_mlp_control_heatmap.png
aspect_c <- 1.074  # multiclass_mlp_treatment_dose_confusion_matrix.png
aspect_d <- 1.107  # multiclass_mlp_target_confusion_matrix.png

row1 <- panel_a + panel_b + plot_layout(widths = c(aspect_a, aspect_b))
row2 <- panel_c + panel_d + plot_layout(widths = c(aspect_c, aspect_d))

# Row height is inversely proportional to its panels' combined aspect ratio,
# so that at a shared figure width, each row renders at the exact height its
# panels need (no stretch) - see the markdown above.
row1_height <- 1 / (aspect_a + aspect_b)
row2_height <- 1 / (aspect_c + aspect_d)

fig_width <- 20
fig_height <- fig_width * (row1_height + row2_height)
dims <- set_plot_size(fig_width, fig_height)

assembled_figure <- (
    (row1 / row2) + plot_layout(heights = c(row1_height, row2_height))
    + plot_annotation(tag_levels = "A")
    & theme(plot.margin = margin(2, 2, 2, 2))
)

save_ggplot(
    assembled_figure,
    file.path(figures_dir, "assembled_ml_summary_figure.png"),
    width = dims$width, height = dims$height, dpi = 600
)
