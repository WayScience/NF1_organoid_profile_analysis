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

embed_pdf_page_panel <- function(path, page) {
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
    img <- magick::image_trim(magick::image_read(paste0(png_root, ".png")))
    wrap_elements(
        full = grid::rasterGrob(
            as.raster(img), interpolate = TRUE,
            width = unit(1, "npc"), height = unit(1, "npc")
        )
    )
}

panel_a <- embed_png_panel(file.path(root_dir, "figures/drugs_figure/figures/drug_targets_by_cellular_compartment_from_uniprot_go_annotations.png"))
panel_b <- embed_png_panel(file.path(source_panels_dir, "confusion_heatmap_Logistic_Regression.png"))
panel_c <- embed_pdf_page_panel(
    file.path(source_panels_dir, "3D_sc_consensus_profile_heatmaps.pdf"), page = 79
)
panel_d <- embed_pdf_page_panel(
    file.path(source_panels_dir, "morphology_vs_viability_cosine_similarity.pdf"), page = 1
)
panel_e <- embed_png_panel(file.path(media_dir, "Whole_montage.png"))

# Trimmed aspect ratios (width/height) of each panel.
aspect_a <- 2740 / 1656  # drug_targets_by_cellular_compartment_from_uniprot_go_annotations.png (trimmed)
aspect_b <- 2560 / 2120  # confusion_heatmap_Logistic_Regression.png
aspect_c <- 3227 / 1480  # 3D_sc_consensus_profile_heatmaps.pdf, page 79 (trimmed)
aspect_d <- 576 / 432  # morphology_vs_viability_cosine_similarity.pdf, page 1
aspect_e <- 12084 / 4887  # Whole_montage.png

row1 <- panel_a + panel_b + plot_layout(widths = c(aspect_a, aspect_b))
row2 <- panel_c + panel_d + plot_layout(widths = c(aspect_c, aspect_d))
row3 <- panel_e

row1_height <- 1 / (aspect_a + aspect_b)
row2_height <- 1 / (aspect_c + aspect_d)
row3_height <- 1 / aspect_e

fig_width <- 20
fig_height <- fig_width * (row1_height + row2_height + row3_height)
dims <- set_plot_size(fig_width, fig_height)

assembled_figure <- (
    (row1 / row2 / row3) + plot_layout(heights = c(row1_height, row2_height, row3_height))
    + plot_annotation(tag_levels = "A")
    & theme(plot.margin = margin(2, 2, 2, 2))
)

save_ggplot(
    assembled_figure,
    file.path(figures_dir, "assembled_figure_3.png"),
    width = dims$width, height = dims$height, dpi = 300
)
