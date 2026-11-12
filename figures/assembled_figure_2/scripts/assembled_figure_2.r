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

figures_dir <- file.path(root_dir, "figures", "assembled_figure_2", "figures")
dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)

panel_a_path <- file.path(figures_dir, "featurization_figure.png")
panel_c_path <- file.path(root_dir, "2.2d_vs_3d_analysis", "figures", "entropy", "sc_fs_entropy.png")
panel_d_path <- file.path(root_dir, "2.2d_vs_3d_analysis", "figures", "entropy", "sc_fs_entropy_2D_vs_3D.png")

embed_png_panel <- function(path) {
    #' Reads a saved PNG, trims its (mostly white) border margin, and wraps
    #' it as a patchwork-compatible panel that exactly fills its grid cell
    #' (unit(1, "npc") width/height). This is only distortion-free because
    #' the assembly below sizes each panel's cell to that panel's own
    #' trimmed aspect ratio - see the "Assemble" section.
    if (!file.exists(path)) {
        stop("missing ", path, " - run the notebook that saves this figure first.")
    }
    img <- magick::image_trim(magick::image_read(path))
    wrap_elements(
        full = grid::rasterGrob(
            as.raster(img), interpolate = TRUE,
            width = unit(1, "npc"), height = unit(1, "npc")
        )
    )
}

placeholder_panel <- function(label) {
    #' A light-gray boxed panel with centred italic text, standing in for a
    #' figure that has not been produced yet (Panel B).
    (
        ggplot()
        + annotate("text", x = 0.5, y = 0.5, label = label, size = 6,
                   fontface = "italic", colour = "grey40", lineheight = 1.1)
        + xlim(0, 1) + ylim(0, 1)
        + theme_void()
        + theme(
            panel.background = element_rect(fill = "grey96", colour = NA),
            panel.border = element_rect(colour = "grey70", fill = NA, linewidth = 0.5)
        )
    )
}

# Panel A carries a larger top-left margin so the bold "A" tag sits in
# whitespace instead of overlapping the image's own italic "i" sublabel.
tight_margin <- margin(2, 2, 2, 2)
panel_a <- embed_png_panel(panel_a_path) & theme(
    plot.margin = margin(t = 34, l = 34, r = 2, b = 2)
)
panel_b <- placeholder_panel(
    "Panel B: Z-sampling influence on feature measurements\n(placeholder)"
) + theme(plot.margin = tight_margin)
panel_c <- embed_png_panel(panel_c_path) & theme(plot.margin = tight_margin)
panel_d <- embed_png_panel(panel_d_path) & theme(plot.margin = tight_margin)

# Trimmed aspect ratios (width / height) of each embedded PNG. Panel B is a
# placeholder; its aspect is matched to Panel A so the two top-row cells read
# as a balanced pair.
aspect_a <- 1.847  # featurization_figure.png
aspect_b <- 1.847  # placeholder (matched to Panel A)
aspect_c <- 1.171  # sc_fs_entropy.png
aspect_d <- 0.987  # sc_fs_entropy_2D_vs_3D.png

# 2x2 layout: A and B share row 1, C and D share row 2. Within each row the
# panel widths are set to their aspect ratios so neither is stretched, and each
# row's height is inversely proportional to its panels' combined aspect ratio,
# so at a shared figure width every panel renders at its native shape.
row1 <- panel_a + panel_b + plot_layout(widths = c(aspect_a, aspect_b))
row2 <- panel_c + panel_d + plot_layout(widths = c(aspect_c, aspect_d))

row1_height <- 1 / (aspect_a + aspect_b)
row2_height <- 1 / (aspect_c + aspect_d)

fig_width <- 20
fig_height <- fig_width * (row1_height + row2_height)
dims <- set_plot_size(fig_width, fig_height)

assembled_figure <- (
    (row1 / row2)
    + plot_layout(heights = c(row1_height, row2_height))
    + plot_annotation(tag_levels = "A")
    & theme(plot.tag = element_text(face = "bold", size = 18))
)

save_ggplot(
    assembled_figure,
    file.path(figures_dir, "assembled_figure_2.png"),
    width = dims$width, height = dims$height, dpi = 600
)
