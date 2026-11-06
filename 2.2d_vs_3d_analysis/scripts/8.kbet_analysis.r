list_of_packages <- c("arrow", "dplyr", "ggplot2", "kBET")
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
cat("Git root directory:", root_dir, "\n")

figures_path <- file.path(root_dir, "2.2d_vs_3d_analysis/figures/kbet")
dir.create(figures_path, recursive = TRUE, showWarnings = FALSE)

# kBET (n_repeat = 1000) is slow; cache each profile's result here so reruns
# of this notebook don't recompute profiles that already finished.
cache_dir <- file.path(root_dir, "2.2d_vs_3d_analysis/results/kbet/cache")
dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)

profiles <- list(
    list(
        label   = "2D_organoid",
        path    = file.path(root_dir, "data/profiles_2D/all_patients/max_projection/organoid_fs_profiles.parquet"),
        is_3d   = FALSE
    ),
    list(
        label   = "2D_singlecell",
        path    = file.path(root_dir, "data/profiles_2D/all_patients/max_projection/sc_fs_profiles.parquet"),
        is_3d   = FALSE
    ),
    list(
        label   = "3D_organoid",
        path    = file.path(root_dir, "data/profiles_3D/all_patients/1.feature_selected_profiles/organoid_norm_fs_profiles.parquet"),
        is_3d   = TRUE
    ),
    list(
        label   = "3D_singlecell",
        path    = file.path(root_dir, "data/profiles_3D/all_patients/1.feature_selected_profiles/sc_norm_fs_profiles.parquet"),
        is_3d   = TRUE
    )
)

prep_features <- function(df) {
    meta_cols <- grep("^Metadata_", colnames(df), value = TRUE)
    metadata_df   <- df %>% select(all_of(meta_cols))
    morphology_df <- df %>% select(-all_of(meta_cols))

    # keep only numeric columns
    numeric_cols <- sapply(morphology_df, is.numeric)
    morphology_df <- morphology_df[, numeric_cols, drop = FALSE]

    # drop constant columns
    constant_cols <- sapply(morphology_df, function(x) length(unique(na.omit(x))) <= 1)
    morphology_df <- morphology_df[, !constant_cols, drop = FALSE]

    # drop all-NA columns, then mean-impute remaining NAs
    all_na_cols <- sapply(morphology_df, function(x) all(is.na(x)))
    morphology_df <- morphology_df[, !all_na_cols, drop = FALSE]
    morphology_df <- as.data.frame(lapply(morphology_df, function(x) {
        if (any(is.na(x))) { x[is.na(x)] <- mean(x, na.rm = TRUE) }
        x
    }))

    list(morphology = morphology_df, metadata = as.data.frame(metadata_df))
}

dmso_results <- list()

for (prof in profiles) {
    stats_cache_path <- file.path(
        cache_dir, paste0(prof$label, "_dmso_patient_stats.parquet")
    )
    summary_cache_path <- file.path(
        cache_dir, paste0(prof$label, "_dmso_patient_summary.parquet")
    )

    if (file.exists(stats_cache_path) && file.exists(summary_cache_path)) {
        plot.data   <- arrow::read_parquet(stats_cache_path)
        summary_row <- arrow::read_parquet(summary_cache_path)
    } else {
        df <- arrow::read_parquet(prof$path)

        # 3D profiles use the Biology_PatientTumor/Experiment_Treatment naming
        # convention; rename to the 2D-style columns the rest of this loop uses.
        if (prof$is_3d) {
            df <- df %>%
                rename(
                    Metadata_patient_tumor = Metadata_Biology_PatientTumor,
                    Metadata_treatment     = Metadata_Experiment_Treatment
                )

            # harmonize NF0037_T1 treatment names in 3D data
            df <- df %>%
                mutate(Metadata_treatment = ifelse(
                    Metadata_patient_tumor == "NF0037_T1",
                    gsub(" \\d+$", "", Metadata_treatment),
                    Metadata_treatment
                ))
        }

        # subset to DMSO across all patients
        df_dmso <- df %>% filter(Metadata_treatment == "DMSO")

        prepped      <- prep_features(df_dmso)
        batch_labels <- prepped$metadata[["Metadata_patient_tumor"]]

        set.seed(0)
        result <- tryCatch(
            kBET(prepped$morphology, batch_labels, n_repeat = 1000, plot = FALSE),
            error = function(e) {
                NULL
            }
        )

        if (is.null(result) || !is.list(result) || is.null(result$summary)) next

        plot.data <- data.frame(
            class = rep(c("observed", "expected"), each = length(result$stats$kBET.observed)),
            data  = c(result$stats$kBET.observed, result$stats$kBET.expected)
        )
        summary_row <- data.frame(
            profile_type            = prof$label,
            observed_rejection_rate = result$summary["mean", "kBET.observed"],
            expected_rejection_rate = result$summary["mean", "kBET.expected"],
            p_value                 = result$summary["mean", "kBET.signif"],
            stringsAsFactors        = FALSE
        )

        arrow::write_parquet(plot.data, stats_cache_path)
        arrow::write_parquet(summary_row, summary_cache_path)
    }

    # individual boxplot
    p <- ggplot(plot.data, aes(class, data)) +
        geom_boxplot() +
        labs(
            x     = "Test",
            y     = "Rejection rate",
            title = paste0(prof$label, "\nDMSO — patient batch")
        ) +
        theme_bw() +
        theme(text = element_text(size = 16)) +
        scale_y_continuous(limits = c(0, 1))
    print(p)

    ggsave(
        file.path(figures_path, paste0("kbet_dmso_", prof$label, "_patient.png")),
        p, width = 5, height = 7, units = "in", dpi = 300
    )

    dmso_results[[length(dmso_results) + 1]] <- summary_row
}

dmso_summary <- do.call(rbind, dmso_results)
head(dmso_summary)

dmso_summary$profile_type <- factor(
    dmso_summary$profile_type,
    levels = c("2D_organoid", "2D_singlecell", "3D_organoid", "3D_singlecell")
)

options(repr.plot.width = 8, repr.plot.height = 6)
dmso_plot <- ggplot(dmso_summary, aes(x = profile_type, y = observed_rejection_rate)) +
    geom_col(fill = "#882E8B") +
    geom_point(aes(y = expected_rejection_rate), shape = 95, size = 8, color = "gray40") + # shape 95 = horizontal dash
    labs(
        x        = "Profile type",
        y        = "Observed rejection rate",
        title    = "kBET (DMSO only): patient batch effect by modality",
        subtitle = "Gray dash = expected rejection rate"
    ) +
    theme_bw() +
    theme(
        text        = element_text(size = 16),
        axis.text.x = element_text(angle = 30, hjust = 1)
    ) +
    scale_y_continuous(limits = c(0, 1))

dmso_plot

ggsave(
    file.path(figures_path, "kbet_dmso_patient_summary.png"),
    dmso_plot, width = 7, height = 6, units = "in", dpi = 300
)
