list_of_packages <- c("ggplot2", "dplyr", "arrow")

for (package in list_of_packages) {
    suppressPackageStartupMessages(
        suppressWarnings(
            library(
                package,
                character.only = TRUE,
                quietly = TRUE,
                warn.conflicts = FALSE
            )
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
source(file.path(root_dir, "utils/r_result_cache.r"))

# Caching: skip work whose outputs were built from unchanged inputs and code
CACHE_STEP <- "7.plot_shannon_entropy"
CODE_PATH <- file.path(root_dir, "2.2d_vs_3d_analysis", "scripts", "7.plot_shannon_entropy.r")
results_root <- file.path(root_dir, "2.2d_vs_3d_analysis", "results")
cat("Git root directory:", root_dir, "\n")

entropy_dir <- file.path(root_dir, "2.2d_vs_3d_analysis", "results", "entropy")
figures_dir <- file.path(root_dir, "2.2d_vs_3d_analysis", "figures", "entropy")
dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)

# Load data

all_files <- list.files(entropy_dir, pattern = "_entropy\\.parquet$", full.names = TRUE)
pooled_files <- all_files[!grepl("_per_patient_entropy\\.parquet$", all_files)]
per_patient_files <- all_files[grepl("_per_patient_entropy\\.parquet$", all_files)]

# Stale 3D aggregated files written before the from_fs / from_norm split have
# no mode in their name, so they are not read.
is_stale_3d_agg <- function(name) {
    grepl("^3D_.*_agg(_per_patient)?_entropy\\.parquet$", name)
}

# 3D aggregated profiles are written twice, from feature-selected and from
# normalized single cells. The file name says which one, so each gets its own
# stage_label (agg_from_fs / agg_from_norm). 2D aggregated profiles keep "agg".
read_entropy_files <- function(files) {
    files <- files[!vapply(basename(files), is_stale_3d_agg, logical(1))]
    bind_rows(lapply(files, function(path) {
        name <- basename(path)
        df <- as.data.frame(arrow::read_parquet(path))
        df$stage_label <- as.character(df$stage)
        if (grepl("_agg_from_(fs|norm)_", name)) {
            mode <- sub(".*_(agg_from_(fs|norm))_.*", "\\1", name)
            df$stage_label[df$imaging_modality == "3D" & df$stage == "agg"] <- mode
        }
        df
    }))
}

entropy_pooled <- read_entropy_files(pooled_files)
entropy_per_patient <- read_entropy_files(per_patient_files)

entropy_pooled$group_label <- factor(entropy_pooled$group_label)
entropy_per_patient$group_label <- factor(entropy_per_patient$group_label)

# Each PNG is redrawn only when the entropy files it is drawn from have changed
entropy_inputs <- all_files
save_plot_if_stale <- function(plot, filename, width, height) {
    if (result_cache_is_fresh(CACHE_STEP, results_root, CODE_PATH, filename, entropy_inputs)) {
        cat("Using cached result:", basename(filename), "\n")
        return(invisible(FALSE))
    }
    print(plot)
    ggsave(
        filename = filename,
        plot     = plot,
        width    = width,
        height   = height,
        dpi      = dpi,
        units    = "in"
    )
    result_cache_mark_fresh(CACHE_STEP, results_root, CODE_PATH, filename, entropy_inputs)
    invisible(TRUE)
}


# Plot styling
entropy_theme <- theme(
    plot.title   = element_text(hjust = 0.5, size = 14),
    axis.title.x = element_text(size = 16),
    axis.title.y = element_text(size = 16),
    axis.text.x  = element_text(size = 12, angle = 20, hjust = 1),
    axis.text.y  = element_text(size = 14),
    legend.title = element_text(size = 14),
    legend.text  = element_text(size = 12)
)

# agg is 2D only; the 3D aggregated stages are split by the mode they were built from
stage_titles <- c(
    normal = "Normal",
    fs = "Feature-Selected",
    agg = "Aggregated (2D)",
    agg_from_norm = "Aggregated (from Normalized)",
    agg_from_fs = "Aggregated (from Feature-Selected)"
)
resolution_titles <- c(organoid = "Organoid", sc = "Single-Cell")

make_entropy_plot <- function(df, title) {
    ggplot(df, aes(x = group_label, y = entropy, fill = group_label)) +
        geom_violin(alpha = 0.6, trim = FALSE) +
        geom_boxplot(width = 0.15, alpha = 0.85, outlier.size = 0.5, outlier.alpha = 0.4) +
        labs(
            title = title,
            x     = "Profile type",
            y     = "Shannon Entropy (bits)",
            fill  = "Profile type"
        ) +
        theme_bw() +
        entropy_theme +
        theme(legend.position = "none")
}

width <- 7
height <- 6
dpi <- 300

for (res in names(resolution_titles)) {
    for (stage in names(stage_titles)) {
        df_subset <- entropy_pooled %>% filter(resolution == res, stage_label == !!stage)
        if (nrow(df_subset) == 0) {
            cat("SKIPPED (no data):", res, stage, "\n")
            next
        }

        title <- paste0(
            "Shannon Entropy: ", resolution_titles[[res]],
            " Profiles (", stage_titles[[stage]], ")"
        )
        p <- make_entropy_plot(df_subset, title)
        save_plot_if_stale(p, file.path(figures_dir, paste0(res, "_", stage, "_entropy.png")), width, height)
    }
}

# Per patient plots

make_per_patient_entropy_plot <- function(df, title) {
    ggplot(df, aes(x = patient, y = entropy, fill = patient)) +
        geom_boxplot(alpha = 0.7, outlier.size = 0.4, outlier.alpha = 0.3) +
        facet_wrap(~group_label, scales = "free_x") +
        labs(
            title = title,
            x     = "Patient",
            y     = "Shannon Entropy (bits)",
            fill  = "Patient"
        ) +
        theme_bw() +
        entropy_theme +
        theme(
            legend.position = "none",
            axis.text.x = element_text(angle = 45, hjust = 1)
        )
}

pp_width  <- 12
pp_height <- 8

for (res in names(resolution_titles)) {
    for (stage in names(stage_titles)) {
        df_subset <- entropy_per_patient %>% filter(resolution == res, stage_label == !!stage)
        if (nrow(df_subset) == 0) {
            cat("SKIPPED (no data):", res, stage, "\n")
            next
        }

        title <- paste0(
            "Per-Patient Shannon Entropy: ", resolution_titles[[res]],
            " Profiles (", stage_titles[[stage]], ")"
        )
        p <- make_per_patient_entropy_plot(df_subset, title)
        save_plot_if_stale(p, file.path(figures_dir, paste0(res, "_", stage, "_per_patient_entropy.png")), pp_width, pp_height)
    }
}
