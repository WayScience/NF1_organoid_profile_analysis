list_of_packages <- c("ggplot2", "dplyr", "arrow", "RColorBrewer")
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

results_dir <- file.path(root_dir, "1.EDA", "results", "intensity")
figures_dir <- file.path(root_dir, "1.EDA", "figures", "intensity")
dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)

plot_theme <- theme_bw() + theme(
    plot.title = element_text(hjust = 0.5, size = 12),
    axis.title.x = element_text(size = 12),
    axis.title.y = element_text(size = 12),
    axis.text.x = element_text(size = 7),
    axis.text.y = element_text(size = 7),
    strip.text = element_text(size = 8),
    legend.position = "bottom",
    legend.title = element_text(size = 9),
    legend.text = element_text(size = 8)
)

# Biological question: how does intensity, per channel and per patient,
# differ between MEK1/2 inhibitors and DMSO control? Restricted to 3D and
# this treatment class, and split one compartment per plot (rather than
# dodging all compartments into one crowded plot) so each panel stays
# readable -- the full treatment panel and the 2D projection-method
# comparisons didn't serve this question and were dropped.
#
# `value` is z-scored PER PATIENT upstream, so Metadata_patient is always a
# facet (never an x-axis category) with free scales -- every panel is
# valid on its own terms. Treatments are overlapping density curves (color)
# within each panel rather than separate x-axis categories, so the shift
# between DMSO and each MEK inhibitor's distribution is visible directly.

mek_treatments <- names(treatment_moa_map)[treatment_moa_map == "MEK1/2 inhibitor"]
keep_treatments <- c("DMSO", mek_treatments)
keep_stats <- c("MeanIntensity", "MedianIntensity")
keep_compartments <- c("whole_organoid", "cell", "nucleus")

intensity_3d <- read_parquet(file.path(results_dir, "intensity_values_3D.parquet")) %>%
    filter(Metadata_treatment %in% keep_treatments,
           stat %in% keep_stats,
           compartment %in% keep_compartments)
treatment_levels <- intersect(custom_treatment_order, keep_treatments)
intensity_3d$Metadata_treatment <- factor(intensity_3d$Metadata_treatment, levels = treatment_levels)

# Diverging palette, scoped to this script only (not the shared
# custom_treatment_palette, which is categorical/MOA-hued) -- DMSO sits at
# the neutral midpoint so each inhibitor's shift reads as a divergence
# away from control rather than an arbitrary hue.
non_dmso <- setdiff(treatment_levels, "DMSO")
half <- length(non_dmso) / 2
ordered_for_palette <- c(non_dmso[seq_len(half)], "DMSO", non_dmso[(half + 1):length(non_dmso)])
diverging_colors <- setNames(brewer.pal(length(treatment_levels), "RdBu"), ordered_for_palette)
diverging_colors["DMSO"] <- "grey50"

pdf(file.path(figures_dir, "3D_intensity_MEK_vs_DMSO.pdf"), width = 16, height = 8, onefile = TRUE)
for (s in unique(intensity_3d$stat)) {
    for (cmp in unique(intensity_3d$compartment)) {
        d <- intensity_3d %>% filter(stat == s, compartment == cmp)
        # facet_wrap's default panel order sorts by the first facet variable
        # then the second, so this is already channel-major -- but that's an
        # implicit contract that breaks silently if facet_wrap's fill
        # behavior ever changes or the channel/patient sets become unbalanced.
        # Build the ordering explicitly via an ordered combined key instead
        # of relying on nrow alone to keep channels grouped by row.
        channel_order <- intersect(names(channel_palette), unique(d$channel))
        patient_order <- sort(unique(as.character(d$Metadata_patient)))
        d$channel <- factor(d$channel, levels = channel_order)
        d$Metadata_patient <- factor(d$Metadata_patient, levels = patient_order)
        d$facet_key <- interaction(
            d$channel, d$Metadata_patient, sep = " | ", lex.order = TRUE, drop = TRUE
        )
        p <- (
            ggplot(d, aes(x = value, color = Metadata_treatment, fill = Metadata_treatment))
            + geom_density(alpha = 0.25, linewidth = 0.5)
            + scale_color_manual(values = diverging_colors, na.value = "grey70")
            + scale_fill_manual(values = diverging_colors, na.value = "grey70")
            # facet_grid's "free" scales are still shared down each column/row,
            # so one extreme patient x channel combination distorts every
            # other panel in its row or column. facet_wrap frees each panel
            # independently; nrow pins the layout to one row per channel so
            # it still reads as a channel x patient grid.
            + facet_wrap(~ facet_key, scales = "free", nrow = length(channel_order), drop = FALSE)
            + labs(
                title = paste0("3D (", cmp, "): ", s, ", MEK inhibitors vs. DMSO, by patient x channel"),
                x = paste0(s, " (z-scored within patient)"), y = "Density",
                color = "Treatment", fill = "Treatment"
            )
            + plot_theme
        )
        print(p)
    }
}
dev.off()
