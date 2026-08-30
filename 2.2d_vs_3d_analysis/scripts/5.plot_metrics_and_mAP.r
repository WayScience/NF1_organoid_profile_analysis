list_of_packages <- c("ggplot2", "dplyr", "tidyr", "arrow")
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

mAP_results_dir <- file.path(root_dir, "2.2d_vs_3d_analysis", "results", "mAP")
dist_results_dir <- file.path(root_dir, "2.2d_vs_3d_analysis", "results", "distance_metrics")
mAP_figures_dir <- file.path(root_dir, "2.2d_vs_3d_analysis", "figures", "mAP")
dist_figures_dir <- file.path(root_dir, "2.2d_vs_3d_analysis", "figures", "distance_metrics")
dir.create(mAP_figures_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(dist_figures_dir, recursive = TRUE, showWarnings = FALSE)

# Every profile type from notebooks 3/4, and the 2D-vs-3D pairs for cross-modality plots
profile_types <- tibble::tribble(
    ~label, ~slug,
    "2D Organoid", "2D_organoid",
    "2D Single-cell", "2D_sc",
    "3D Organoid - Handcrafted", "3D_organoid_handcrafted",
    "3D Organoid - DL (SAM-Med3D)", "3D_organoid_sammed",
    "3D Single-cell - Handcrafted", "3D_sc_handcrafted",
    "3D Single-cell - DL (SAM-Med3D)", "3D_sc_sammed",
    "3D Single-cell - Nucleocentric DL (SAM-Med3D)", "3D_sc_sammed_nucleocentric",
    "3D Single-cell - Nucleocentric DL (MorphEM)", "3D_sc_nucleocentric_morphem"
)

pairs_2d_3d <- tibble::tribble(
    ~label, ~slug_2d, ~slug_3d,
    "Organoid - Handcrafted", "2D_organoid", "3D_organoid_handcrafted",
    "Organoid - DL (SAM-Med3D)", "2D_organoid", "3D_organoid_sammed",
    "Single-cell - Handcrafted", "2D_sc", "3D_sc_handcrafted",
    "Single-cell - DL (SAM-Med3D)", "2D_sc", "3D_sc_sammed",
    "Single-cell - Nucleocentric DL (SAM-Med3D)", "2D_sc", "3D_sc_sammed_nucleocentric",
    "Single-cell - Nucleocentric DL (MorphEM)", "2D_sc", "3D_sc_nucleocentric_morphem"
)

# Drugs are colored by identity; dose is encoded as point shape (1 = circle, 10 = triangle)
custom_treatment_palette <- c(
    'DMSO' = "#A6A6A6",           # Control

    'Staurosporine' = "#3F468C",  # Broad-spectrum kinase inhibitor

    'Fimepinostat' = "#3DCCA8",   # HDAC/PI3K inhibitor (dual)
    'Copanlisib' = "#3DCACC",     # PI3K inhibitor
    'Everolimus' = "#3DA4CC",     # mTOR inhibitor
    'Rapamycin' = "#3D7DCC",      # mTOR inhibitor
    'Sapanisertib' = "#3D57CC",   # mTOR inhibitor
    'Vistusertib' = "#493DCC",    # mTOR inhibitor

    'Panobinostat' = "#CC8029",   # HDAC inhibitor
    'ARV-825' = "#CCAB29",        # BRD4 inhibitor

    'Imatinib' = "#6047CC",       # BCR-ABL/KIT inhibitor
    'Nilotinib' = "#8347CC",      # BCR-ABL/KIT inhibitor
    'Cabozantinib' = "#A647CC",   # Multi-kinase inhibitor
    'Linsitinib' = "#CA47CC",     # IGF-1R inhibitor

    'Binimetinib' = "#D92B7F",    # MEK inhibitor
    'Mirdametinib' = "#D92B51",   # MEK inhibitor
    'Trametinib' = "#D9342B",     # MEK inhibitor
    'Selumetinib' = "#D9622B",    # MEK inhibitor

    'Onalespib' = "#6CA642",      # HSP90 inhibitor
    'Digoxin' = "#BF3078",        # Na/K-ATPase inhibitor
    'Ketotifen' = "#238C83",      # Antihistamine
    'Trabectedin' = "#388C5B"     # DNA-binding agent
)

dose_shape_palette <- c("1" = 16, "10" = 17)  # 16 = circle, 17 = triangle

# Results only carry the combined "Drug_Dose" string; split it into bare drug
# (for color) and bare dose (for shape) before plotting.
split_treatment_dose <- function(df) {
    df$Metadata_treatment <- sub("_[0-9]+$", "", df$Metadata_treatment_dose)
    df$Metadata_dose <- sub(".*_([0-9]+)$", "\\1", df$Metadata_treatment_dose)
    df
}

plot_theme <- theme(
    plot.title   = element_text(hjust = 0.5, size = 14),
    axis.title.x = element_text(size = 16),
    axis.title.y = element_text(size = 16),
    axis.text.x  = element_text(size = 14),
    axis.text.y  = element_text(size = 14),
    legend.position = "bottom",
    legend.title = element_text(size = 12, face = "bold", hjust = 0.5),
    legend.text  = element_text(size = 10)
)

# mAP volcano-style scatter (mAP vs -log10(p)), one intra + one inter plot per profile type
make_mAP_plot <- function(df, title, faceted) {
    p <- (
        ggplot(df, aes(x = mean_average_precision, y = `-log10(p-value)`, color = Metadata_treatment, shape = Metadata_dose))
        + geom_point(size = 3, alpha = 0.7)
        + scale_color_manual(name = "Treatment", values = custom_treatment_palette,
                              guide = guide_legend(title.position = "top", title.hjust = 0.5, order = 1))
        + scale_shape_manual(name = "Dose", values = dose_shape_palette,
                              guide = guide_legend(title.position = "top", title.hjust = 0.5, order = 2))
        + geom_hline(yintercept = 1.3, linetype = "dashed", color = "red")
        + labs(x = "Mean Average Precision", y = "-log10(p-value)", title = title)
        + theme_bw()
        + plot_theme
        + xlim(0, 1)
    )
    if (faceted) {
        p <- p + facet_wrap(~Metadata_patient, ncol = 4)
    }
    p
}

for (i in seq_len(nrow(profile_types))) {
    label <- profile_types$label[i]
    slug  <- profile_types$slug[i]

    intra_df <- split_treatment_dose(arrow::read_parquet(file.path(mAP_results_dir, paste0(slug, "_intra_patient_mAP_by_dose.parquet"))))
    inter_df <- split_treatment_dose(arrow::read_parquet(file.path(mAP_results_dir, paste0(slug, "_inter_patient_mAP_by_dose.parquet"))))

    intra_plot <- make_mAP_plot(intra_df, paste0("Intra-patient mAP - ", label), faceted = TRUE)
    ggsave(file.path(mAP_figures_dir, paste0(slug, "_intra_patient_mAP.png")),
           plot = intra_plot, width = 12, height = 8, dpi = 600)
    print(intra_plot)

    inter_plot <- make_mAP_plot(inter_df, paste0("Inter-patient mAP - ", label), faceted = FALSE)
    ggsave(file.path(mAP_figures_dir, paste0(slug, "_inter_patient_mAP.png")),
           plot = inter_plot, width = 10, height = 6, dpi = 600)
    print(inter_plot)
}

# Ratio bar chart + inter-vs-intra scatter, one pair of plots per profile type
for (i in seq_len(nrow(profile_types))) {
    label <- profile_types$label[i]
    slug  <- profile_types$slug[i]

    intra_df <- arrow::read_parquet(file.path(mAP_results_dir, paste0(slug, "_intra_patient_mAP_by_dose.parquet"))) %>%
        select(Metadata_treatment_dose, Metadata_patient, intra_patient_mAP = mean_average_precision)
    inter_df <- arrow::read_parquet(file.path(mAP_results_dir, paste0(slug, "_inter_patient_mAP_by_dose.parquet"))) %>%
        select(Metadata_treatment_dose, inter_patient_mAP = mean_average_precision)

    merged_df <- merge(intra_df, inter_df, by = "Metadata_treatment_dose", all = TRUE)
    merged_df$intra_to_inter_ratio <- merged_df$intra_patient_mAP / merged_df$inter_patient_mAP
    merged_df <- split_treatment_dose(merged_df)

    # merged_df has one row per (patient, treatment) since intra is per-patient and inter
    # isn't; summarize to one ratio per treatment (median across patients) before plotting
    # a bar, otherwise geom_bar sums the per-patient duplicates into an inflated bar height.
    ratio_by_treatment <- merged_df %>%
        group_by(Metadata_treatment_dose, Metadata_treatment, Metadata_dose) %>%
        summarize(median_ratio = median(intra_to_inter_ratio, na.rm = TRUE), .groups = "drop")

    ratio_plot <- (
        ggplot(ratio_by_treatment, aes(x = Metadata_treatment_dose, y = median_ratio, fill = Metadata_treatment))
        + geom_bar(stat = "identity", position = "dodge")
        + scale_fill_manual(name = "Treatment", values = custom_treatment_palette,
                             guide = guide_legend(title.position = "top", title.hjust = 0.5))
        + labs(x = "Treatment (dose)", y = "Median Intra:Inter Patient mAP Ratio",
               title = paste0("Intra:Inter mAP Ratio - ", label))
        + theme_bw()
        + plot_theme
        + theme(axis.text.x = element_text(angle = 90, hjust = 1, size = 10), legend.position = "none")
        + geom_hline(yintercept = 1, linetype = "dashed", color = "red")
    )
    ggsave(file.path(mAP_figures_dir, paste0(slug, "_intra_to_inter_mAP_ratio_bar.png")),
           plot = ratio_plot, width = 8, height = 6, dpi = 600)
    print(ratio_plot)

    scatter_plot <- (
        ggplot(merged_df, aes(x = inter_patient_mAP, y = intra_patient_mAP, color = Metadata_treatment, shape = Metadata_dose))
        + geom_point(size = 2, alpha = 0.7)
        + scale_color_manual(name = "Treatment", values = custom_treatment_palette,
                              guide = guide_legend(title.position = "top", title.hjust = 0.5, order = 1))
        + scale_shape_manual(name = "Dose", values = dose_shape_palette,
                              guide = guide_legend(title.position = "top", title.hjust = 0.5, order = 2))
        + geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red")
        + labs(x = "Inter-patient mAP", y = "Intra-patient mAP", title = paste0("Inter vs Intra Patient mAP - ", label))
        + theme_bw()
        + plot_theme
        + xlim(0, 1)
    )
    ggsave(file.path(mAP_figures_dir, paste0(slug, "_mAP_inter_vs_intra.png")),
           plot = scatter_plot, width = 10, height = 6, dpi = 600)
    print(scatter_plot)
}

# 2D vs 3D mAP scatter, one intra (faceted by patient) + one inter plot per pair
for (i in seq_len(nrow(pairs_2d_3d))) {
    label   <- pairs_2d_3d$label[i]
    slug_2d <- pairs_2d_3d$slug_2d[i]
    slug_3d <- pairs_2d_3d$slug_3d[i]

    intra_2d <- arrow::read_parquet(file.path(mAP_results_dir, paste0(slug_2d, "_intra_patient_mAP_by_dose.parquet")))
    intra_3d <- arrow::read_parquet(file.path(mAP_results_dir, paste0(slug_3d, "_intra_patient_mAP_by_dose.parquet")))
    intra_merged <- inner_join(
        intra_2d %>% select(Metadata_treatment_dose, Metadata_patient, mAP_2D = mean_average_precision),
        intra_3d %>% select(Metadata_treatment_dose, Metadata_patient, mAP_3D = mean_average_precision),
        by = c("Metadata_treatment_dose", "Metadata_patient")
    )

    inter_2d <- arrow::read_parquet(file.path(mAP_results_dir, paste0(slug_2d, "_inter_patient_mAP_by_dose.parquet")))
    inter_3d <- arrow::read_parquet(file.path(mAP_results_dir, paste0(slug_3d, "_inter_patient_mAP_by_dose.parquet")))
    inter_merged <- inner_join(
        inter_2d %>% select(Metadata_treatment_dose, mAP_2D = mean_average_precision),
        inter_3d %>% select(Metadata_treatment_dose, mAP_3D = mean_average_precision),
        by = "Metadata_treatment_dose"
    )
    intra_merged <- split_treatment_dose(intra_merged)
    inter_merged <- split_treatment_dose(inter_merged)

    intra_plot <- (
        ggplot(intra_merged, aes(x = mAP_2D, y = mAP_3D, color = Metadata_treatment, shape = Metadata_dose))
        + geom_point(size = 3, alpha = 0.7)
        + scale_color_manual(name = "Treatment", values = custom_treatment_palette,
                              guide = guide_legend(title.position = "top", title.hjust = 0.5, order = 1))
        + scale_shape_manual(name = "Dose", values = dose_shape_palette,
                              guide = guide_legend(title.position = "top", title.hjust = 0.5, order = 2))
        + geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red")
        + labs(x = "2D mAP Score", y = "3D mAP Score", title = paste0("2D vs 3D Intra-patient mAP - ", label))
        + theme_bw()
        + plot_theme
        + xlim(0, 1) + ylim(0, 1)
        + facet_wrap(~Metadata_patient, ncol = 4)
    )
    ggsave(file.path(mAP_figures_dir, paste0(slug_3d, "_vs_", slug_2d, "_intra_mAP.png")),
           plot = intra_plot, width = 12, height = 8, dpi = 600)
    print(intra_plot)

    inter_plot <- (
        ggplot(inter_merged, aes(x = mAP_2D, y = mAP_3D, color = Metadata_treatment, shape = Metadata_dose))
        + geom_point(size = 4, alpha = 0.7)
        + scale_color_manual(name = "Treatment", values = custom_treatment_palette,
                              guide = guide_legend(title.position = "top", title.hjust = 0.5, order = 1))
        + scale_shape_manual(name = "Dose", values = dose_shape_palette,
                              guide = guide_legend(title.position = "top", title.hjust = 0.5, order = 2))
        + geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red")
        + labs(x = "2D mAP Score", y = "3D mAP Score", title = paste0("2D vs 3D Inter-patient mAP - ", label))
        + theme_bw()
        + plot_theme
        + xlim(0, 1) + ylim(0, 1)
    )
    ggsave(file.path(mAP_figures_dir, paste0(slug_3d, "_vs_", slug_2d, "_inter_mAP.png")),
           plot = inter_plot, width = 10, height = 8, dpi = 600)
    print(inter_plot)
}

# 2D vs 3D cosine distance scatter, one intra (faceted by patient) + one inter plot per pair
for (i in seq_len(nrow(pairs_2d_3d))) {
    label   <- pairs_2d_3d$label[i]
    slug_2d <- pairs_2d_3d$slug_2d[i]
    slug_3d <- pairs_2d_3d$slug_3d[i]

    intra_2d <- arrow::read_parquet(file.path(dist_results_dir, paste0(slug_2d, "_intra_patient_cosine_distance.parquet")))
    intra_3d <- arrow::read_parquet(file.path(dist_results_dir, paste0(slug_3d, "_intra_patient_cosine_distance.parquet")))
    intra_merged <- inner_join(
        intra_2d %>% select(Metadata_treatment_dose, Metadata_patient, cosine_2D = cosine_distance_mean),
        intra_3d %>% select(Metadata_treatment_dose, Metadata_patient, cosine_3D = cosine_distance_mean),
        by = c("Metadata_treatment_dose", "Metadata_patient")
    )

    inter_2d <- arrow::read_parquet(file.path(dist_results_dir, paste0(slug_2d, "_inter_patient_cosine_distance.parquet")))
    inter_3d <- arrow::read_parquet(file.path(dist_results_dir, paste0(slug_3d, "_inter_patient_cosine_distance.parquet")))
    inter_merged <- inner_join(
        inter_2d %>% select(Metadata_treatment_dose, cosine_2D = cosine_distance_mean),
        inter_3d %>% select(Metadata_treatment_dose, cosine_3D = cosine_distance_mean),
        by = "Metadata_treatment_dose"
    )
    intra_merged <- split_treatment_dose(intra_merged)
    inter_merged <- split_treatment_dose(inter_merged)

    intra_plot <- (
        ggplot(intra_merged, aes(x = cosine_2D, y = cosine_3D, color = Metadata_treatment, shape = Metadata_dose))
        + geom_point(size = 3, alpha = 0.7)
        + scale_color_manual(name = "Treatment", values = custom_treatment_palette,
                              guide = guide_legend(title.position = "top", title.hjust = 0.5, order = 1))
        + scale_shape_manual(name = "Dose", values = dose_shape_palette,
                              guide = guide_legend(title.position = "top", title.hjust = 0.5, order = 2))
        + geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red")
        + labs(x = "2D Cosine Distance", y = "3D Cosine Distance",
               title = paste0("2D vs 3D Intra-patient Cosine Distance - ", label))
        + theme_bw()
        + plot_theme
        + facet_wrap(~Metadata_patient, ncol = 4)
    )
    ggsave(file.path(dist_figures_dir, paste0(slug_3d, "_vs_", slug_2d, "_intra_cosine_distance.png")),
           plot = intra_plot, width = 12, height = 8, dpi = 600)
    print(intra_plot)

    inter_plot <- (
        ggplot(inter_merged, aes(x = cosine_2D, y = cosine_3D, color = Metadata_treatment, shape = Metadata_dose))
        + geom_point(size = 4, alpha = 0.7)
        + scale_color_manual(name = "Treatment", values = custom_treatment_palette,
                              guide = guide_legend(title.position = "top", title.hjust = 0.5, order = 1))
        + scale_shape_manual(name = "Dose", values = dose_shape_palette,
                              guide = guide_legend(title.position = "top", title.hjust = 0.5, order = 2))
        + geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red")
        + labs(x = "2D Cosine Distance", y = "3D Cosine Distance",
               title = paste0("2D vs 3D Inter-patient Cosine Distance - ", label))
        + theme_bw()
        + plot_theme
    )
    ggsave(file.path(dist_figures_dir, paste0(slug_3d, "_vs_", slug_2d, "_inter_cosine_distance.png")),
           plot = inter_plot, width = 10, height = 8, dpi = 600)
    print(inter_plot)
}
