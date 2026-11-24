list_of_packages <- c("ggplot2", "dplyr", "patchwork", "magick", "grid", "RColorBrewer")
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

module_dir <- file.path(root_dir, "figures", "assembled_figure_3")
source_panels_dir <- file.path(module_dir, "data", "source_panels")
media_dir <- file.path(module_dir, "media")
figures_dir <- file.path(module_dir, "figures")
dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)

pad_image_to_aspect <- function(img, target_aspect, gravity = "center") {
    #' Pad a magick image with white to an exact width/height aspect ratio,
    #' keeping the original UNDISTORTED. Whitespace is added to whichever
    #' dimension is too short; `gravity` places the image within the padded
    #' box ("center", "west" = flush left, "east" = flush right, ...).
    #' Returns the image unchanged when `target_aspect` is NULL.
    if (is.null(target_aspect)) {
        return(img)
    }
    info <- magick::image_info(img)
    w <- info$width
    h <- info$height
    if (target_aspect > w / h) {
        new_w <- round(h * target_aspect)
        new_h <- h
    } else {
        new_w <- w
        new_h <- round(w / target_aspect)
    }
    magick::image_extent(
        img, geometry = paste0(new_w, "x", new_h),
        gravity = gravity, color = "white"
    )
}

embed_png_panel <- function(path, target_aspect = NULL, gravity = "center") {
    #' Reads a saved PNG, trims its (mostly white) border margin, and wraps
    #' it as a patchwork-compatible panel that exactly fills its grid cell
    #' (unit(1, "npc") width/height). Filling a cell only avoids distortion
    #' when the cell's aspect matches the image's, so pass `target_aspect`
    #' (the cell's width/height) to letterbox-pad the trimmed image to that
    #' aspect first: the plot then keeps its proportions and any leftover
    #' space in the cell becomes whitespace (placed per `gravity`) instead
    #' of stretch.
    if (!file.exists(path)) {
        stop(
            "missing ", path, " - run the notebook that saves this figure first."
        )
    }
    img <- pad_image_to_aspect(
        magick::image_trim(magick::image_read(path)), target_aspect, gravity
    )
    wrap_elements(
        full = grid::rasterGrob(
            as.raster(img), interpolate = TRUE,
            width = unit(1, "npc"), height = unit(1, "npc")
        )
    )
}

embed_pdf_page_panel <- function(path, page, target_aspect = NULL, gravity = "center") {
    #' Same as embed_png_panel(), but rasterizes a single page of a PDF
    #' first. Uses the `pdftoppm` CLI (poppler-utils) rather than
    #' magick::image_read_pdf(), which additionally requires the R
    #' `pdftools` package.
    if (!file.exists(path)) {
        stop(
            "missing ", path, " - run the notebook that saves this figure first."
        )
    }
    png_root <- tempfile()
    status <- system2(
        "pdftoppm",
        c("-f", page, "-l", page, "-r", "600", "-png", "-singlefile", path, png_root)
    )
    if (status != 0) stop("pdftoppm failed with status ", status)
    on.exit(unlink(paste0(png_root, ".png")), add = TRUE)
    img <- pad_image_to_aspect(
        magick::image_trim(magick::image_read(paste0(png_root, ".png"))),
        target_aspect, gravity
    )
    wrap_elements(
        full = grid::rasterGrob(
            as.raster(img), interpolate = TRUE,
            width = unit(1, "npc"), height = unit(1, "npc")
        )
    )
}

# Panel aspect ratios (width / height) are measured from the trimmed images, so
# re-rendered panels never need hand-updated numbers.
trimmed_aspect <- function(path, page = NULL) {
    #' Width / height of a PNG, or of one page of a PDF, after trimming its white border.
    if (is.null(page)) {
        img <- magick::image_read(path)
    } else {
        png_root <- tempfile()
        system2("pdftoppm", c("-f", page, "-l", page, "-r", "50", "-png", "-singlefile", path, png_root))
        on.exit(unlink(paste0(png_root, ".png")), add = TRUE)
        img <- magick::image_read(paste0(png_root, ".png"))
    }
    info <- magick::image_info(magick::image_trim(img))
    info$width / info$height
}

pathway_path <- file.path(root_dir, "figures/drugs_figure/figures/drug_targets_on_ras_mapk_pi3k_pathway.png")
heatmap_path <- file.path(figures_dir, "heatmap_panel.pdf")
umap_path <- file.path(figures_dir, "umap_image_frame_sc_mip.pdf")

aspect_a <- trimmed_aspect(pathway_path)
aspect_b <- trimmed_aspect(heatmap_path, page = 1)
aspect_c <- trimmed_aspect(umap_path, page = 1)

# Layout (tags follow position):
#   A pathway | B heatmap
#   C UMAP with 3D crops (full width)
# A sits in a narrow left column and B in the wide right column; C spans both.
# Column widths and row heights are set so A and B fill the top row and C the full
# figure width exactly.
fig_width <- 20
# A and B share the top row: split its width so both fill their cells at the same
# height (no letterbox padding around A)
left_col_width <- fig_width * aspect_a / (aspect_a + aspect_b)
right_col_width <- fig_width - left_col_width

row_top_height <- right_col_width / aspect_b
row_bottom_height <- fig_width / aspect_c

panel_a <- embed_png_panel(
    pathway_path, target_aspect = left_col_width / row_top_height, gravity = "west"
)
panel_b <- embed_pdf_page_panel(
    heatmap_path, page = 1, target_aspect = right_col_width / row_top_height
)
panel_c <- embed_pdf_page_panel(
    umap_path, page = 1, target_aspect = fig_width / row_bottom_height
)

fig_height <- row_top_height + row_bottom_height
dims <- set_plot_size(fig_width, fig_height)

assembled_figure <- (
    wrap_plots(
        A = panel_a, B = panel_b, C = panel_c,
        design = "AB\nCC",
        widths = c(left_col_width, right_col_width),
        heights = c(row_top_height, row_bottom_height)
    )
    + plot_annotation(tag_levels = "A")
    & theme(plot.margin = margin(2, 2, 2, 2), plot.tag = element_text(face = "bold", size = 18))
)

save_ggplot(
    assembled_figure,
    file.path(figures_dir, "assembled_figure_3.png"),
    width = dims$width, height = dims$height, dpi = 300, limitsize = FALSE
)
