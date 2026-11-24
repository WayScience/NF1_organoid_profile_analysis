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
analysis_figures_dir <- file.path(root_dir, "2.2d_vs_3d_analysis", "figures")
panel_c_path <- file.path(analysis_figures_dir, "mutual_information", "sc_mutual_information.pdf")
panel_c_page <- 3  # single-cell, feature-selected: MI per feature
panel_d_path <- panel_c_path
panel_d_page <- 4  # single-cell, feature-selected: fraction of features explained
PDF_DPI <- 300  # resolution PDF pages are rendered at

read_pdf_page <- function(path, page, dpi = PDF_DPI) {
    #' Renders one page of a PDF to an image (pdftoppm) and trims its white border.
    if (!file.exists(path)) {
        stop("missing ", path, " - run the notebook that saves this figure first.")
    }
    png_root <- tempfile()
    status <- system2(
        "pdftoppm", c("-f", page, "-l", page, "-r", dpi, "-png", "-singlefile", path, png_root)
    )
    if (status != 0) stop("pdftoppm failed with status ", status)
    on.exit(unlink(paste0(png_root, ".png")), add = TRUE)
    magick::image_trim(magick::image_read(paste0(png_root, ".png")))
}

embed_image_panel <- function(img) {
    #' Wraps a magick image as a patchwork-compatible panel that exactly fills its grid
    #' cell; distortion-free because each cell is sized to its image's aspect ratio.
    wrap_elements(
        # the panel area only, so the tag strip above it stays clear
        panel = grid::rasterGrob(
            as.raster(img), interpolate = TRUE  # fits its cell, keeping its aspect ratio
        )
    )
}

image_aspect <- function(img) {
    #' Width / height of a magick image.
    info <- magick::image_info(img)
    info$width / info$height
}

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
        # the panel area only, so the tag strip above it stays clear
        panel = grid::rasterGrob(
            as.raster(img), interpolate = TRUE  # fits its cell, keeping its aspect ratio
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

# Panel tags sit in their own margin strip above each panel (see Assemble), so
# they never overlap panel content
tight_margin <- margin(2, 2, 2, 2)
panel_a <- embed_png_panel(panel_a_path) & theme(plot.margin = tight_margin)
panel_b <- placeholder_panel(
    "Panel B: Z-sampling influence on feature measurements\n(placeholder)"
) + theme(plot.margin = tight_margin)
img_c <- read_pdf_page(panel_c_path, panel_c_page)
img_d <- read_pdf_page(panel_d_path, panel_d_page)
panel_c <- embed_image_panel(img_c) & theme(plot.margin = tight_margin)
panel_d <- embed_image_panel(img_d) & theme(plot.margin = tight_margin)

# Trimmed aspect ratios (width / height) of each embedded panel. Panel B is a
# placeholder; its aspect is matched to Panel A so the two top-row cells read
# as a balanced pair.
aspect_a <- 1.847  # featurization_figure.png
aspect_b <- 1.847  # placeholder (matched to Panel A)
aspect_c <- image_aspect(img_c)  # MI per feature (rendered PDF page)
aspect_d <- image_aspect(img_d)  # fraction of features explained (rendered PDF page)

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
    & theme(
        plot.tag = element_text(face = "bold", size = 18, margin = margin(b = 8)),
        # tags go in a strip above each panel's top-left corner, outside its content
        plot.tag.location = "margin",
        plot.tag.position = "topleft"
    )
)

save_ggplot(
    assembled_figure,
    file.path(figures_dir, "assembled_figure_2.png"),
    width = dims$width, height = dims$height, dpi = 600
)
