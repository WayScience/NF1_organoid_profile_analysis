list_of_packages <- c("ggplot2", "dplyr", "arrow", "RColorBrewer", "scales")
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

results_dir <- file.path(root_dir, "1.EDA", "results", "area_vs_volume")
figures_dir <- file.path(root_dir, "1.EDA", "figures", "area_vs_volume")
dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)

plot_theme <- theme_bw() + theme(
    plot.title = element_text(hjust = 0.5, size = 13),
    axis.title.x = element_text(size = 13),
    axis.title.y = element_text(size = 13),
    axis.text.x = element_text(size = 9),
    axis.text.y = element_text(size = 9),
    legend.position = "right"
)

# Biological question: does 2D area track 3D volume the way simple organoid
# geometry would predict, and does that relationship hold consistently
# across patients and treatments? Restricted to max_projection -- the
# standard 2D representation -- rather than comparing all 3 projection
# methods side-by-side. Per kind (organoid/cell): one pooled view, one
# faceted by patient, one faceted by treatment (pooled across patients) --
# all 6 pages in a single PDF.
#
# Plotted in RAW (unnormalized) units, read directly from each modality's
# pre-normalization stage (script 17) -- not the z-scored values used
# elsewhere in this repo. Area came out of its upstream pipeline already
# z-scored per patient while volume didn't (previously mislabeled
# "z-scored" in this script when it wasn't); rather than patch that
# mismatch, we sidestep it entirely by plotting both in their native scale.
# Both area (px^2) and volume (voxels) span ~3-4 orders of magnitude, so
# both axes are log10.
#
# There's no shared organoid ID between the 2D and 3D pipelines (separate
# segmentations per modality), so individual area/volume records can't be
# paired per-object. We instead randomly pair a capped sample of each
# patient's area values with a capped sample of that patient's volume
# values. This shows the joint occupied range of the two distributions
# under an independence assumption -- NOT a per-organoid correlation -- and
# is capped per patient to keep the 2D density estimate tractable.
set.seed(0)
pair_sample_n <- 500

load_raw <- function(kind) {
    area <- read_parquet(file.path(results_dir, paste0("area_2D_", kind, "_raw.parquet")))
    volume <- read_parquet(file.path(results_dir, paste0("volume_3D_", kind, "_raw.parquet")))
    # A handful of rows (20, organoid volume only) come out of the pipeline
    # as NaN. Left in, a NaN drawn into any facet's sample (a facet can
    # inherit one from just a single contributing patient, since treatment
    # facets pool across patients) poisons that facet's density estimate
    # entirely, leaving it blank -- this is why facets were blank
    # inconsistently rather than uniformly.
    area <- area[is.finite(area$area), ]
    volume <- volume[is.finite(volume$volume), ]
    list(area = area, volume = volume)
}

# Random-pairs area/volume within each combination of group_cols (e.g. just
# patient, or patient x treatment), capping each side at sample_n before the
# cross join so the pair count stays tractable. Attaches group_cols back
# onto the output so the result can be faceted by them.
pair_within_group <- function(area_df, volume_df, group_cols, sample_n) {
    area_key <- do.call(paste, c(area_df[group_cols], sep = ""))
    volume_key <- do.call(paste, c(volume_df[group_cols], sep = ""))
    a_split <- split(area_df$area, area_key)
    v_split <- split(volume_df$volume, volume_key)
    shared_keys <- intersect(names(a_split), names(v_split))

    # named-vector lookup (not data.frame rownames -- tibbles don't support
    # row-name indexing) from key string -> row index of its first occurrence
    key_vals <- as.data.frame(volume_df[group_cols])
    key_row_for <- setNames(seq_len(nrow(key_vals)), volume_key)

    bind_rows(lapply(shared_keys, function(k) {
        a_sample <- sample(a_split[[k]], min(length(a_split[[k]]), sample_n))
        v_sample <- sample(v_split[[k]], min(length(v_split[[k]]), sample_n))
        paired <- expand.grid(area = a_sample, volume = v_sample)
        row_idx <- key_row_for[[k]]
        for (col in group_cols) paired[[col]] <- key_vals[row_idx, col]
        paired
    }))
}

# Facets in this data get small fast (patient x treatment groups run from
# ~1 to a few hundred records each -- see script comment above on group
# sizes), so a smaller sample cap is used for the treatment facet than the
# pooled/per-patient plots to keep total pair count in check; a facet with
# very few source records will simply be sparse, which is real signal about
# how much data supports that combination, not something to paper over.
treatment_sample_n <- 150

base_layers <- function() {
    list(
        # ndensity (each panel's density rescaled to its own 0-1 range) --
        # not raw density, which is shared/comparable across facets, so a
        # sparsely-populated facet's peak (however real within that facet)
        # reads as barely-above-white next to a heavily-populated facet's
        # peak. ndensity gives every facet its own full white -> purple
        # range so within-facet structure stays visible everywhere.
        #
        # geom = "tile" (not "raster") looked like the fix for the "uneven
        # horizontal intervals" warning on the log-scaled grid, but it
        # silently renders every tile at zero size -- and so completely
        # blank -- once there's more than one facet panel with free scales
        # (verified: identical code with 1 facet renders fine with "tile",
        # 2+ facets renders fine with "raster" but blank with "tile"). The
        # "uneven pixel" warning from raster is cosmetic on a log grid, not
        # a correctness issue, so raster is the right geom here.
        stat_density_2d(aes(fill = after_stat(ndensity)), geom = "raster", contour = FALSE),
        scale_fill_gradient(low = "white", high = "#3B0764", name = "Density", limits = c(0, 1)),
        geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "orange", linewidth = 0.4),
        # 3 breaks per axis (not the default ~5-6) so scientific-notation
        # labels have room and don't overlap.
        scale_x_log10(labels = label_scientific(), breaks = breaks_log(n = 3), expand = c(0, 0)),
        scale_y_log10(labels = label_scientific(), breaks = breaks_log(n = 3), expand = c(0, 0))
    )
}

pdf(file.path(figures_dir, "area_vs_volume_density.pdf"), width = 8, height = 7, onefile = TRUE)
for (kind in c("organoid", "cell")) {
    label <- if (kind == "organoid") "Organoid" else "Single-cell"
    d <- load_raw(kind)
    paired_by_patient <- pair_within_group(d$area, d$volume, "Metadata_patient_tumor", pair_sample_n)
    paired_by_treatment <- pair_within_group(
        d$area, d$volume, c("Metadata_patient_tumor", "Metadata_treatment"), treatment_sample_n
    )

    p_pooled <- (
        ggplot(paired_by_patient, aes(x = area, y = volume))
        + base_layers()
        + labs(
            title = paste0(label, ": area (2D, max projection) vs. volume (3D), raw units"),
            x = "Area (px^2, log10 scale)", y = "Volume (voxels, log10 scale)"
        )
        + plot_theme
    )
    print(p_pooled)

    p_by_patient <- (
        ggplot(paired_by_patient, aes(x = area, y = volume))
        + base_layers()
        + facet_wrap(~Metadata_patient_tumor, scales = "free")
        + labs(
            title = paste0(label, ": area vs. volume, by patient"),
            x = "Area (px^2, log10 scale)", y = "Volume (voxels, log10 scale)"
        )
        + plot_theme
        + theme(strip.text = element_text(size = 7))
    )
    print(p_by_patient)

    p_by_treatment <- (
        ggplot(paired_by_treatment, aes(x = area, y = volume))
        + base_layers()
        + facet_wrap(~Metadata_treatment, scales = "free")
        + labs(
            title = paste0(label, ": area vs. volume, by treatment (pooled across patients)"),
            x = "Area (px^2, log10 scale)", y = "Volume (voxels, log10 scale)"
        )
        + plot_theme
        + theme(strip.text = element_text(size = 7))
    )
    print(p_by_treatment)
}
dev.off()
