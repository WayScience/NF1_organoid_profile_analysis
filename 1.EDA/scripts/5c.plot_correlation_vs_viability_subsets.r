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
source(file.path(root_dir, "utils", "r_plot_funcs.r"))

correlation_viability_file_path <- file.path(
    root_dir, "1.EDA", "results", "correlation",
    "3D_sc_correlation_pairs_sc_norm_agg_from_norm_with_meta_and_viability.parquet"
)
figures_dir <- file.path(root_dir, "1.EDA", "figures", "correlation_vs_viability")
if (!dir.exists(figures_dir)) {
    dir.create(figures_dir, recursive = TRUE)
}
output_pdf_path <- file.path(figures_dir, "correlation_vs_viability_subsets.pdf")

# Same pair-level data as 5b.plot_correlation_and_montages, annotated with
# pair-level descriptors used to subset the scatter and to color points.
correlation_viability_df <- arrow::read_parquet(correlation_viability_file_path)

correlation_viability_df <- correlation_viability_df %>%
    mutate(
        group1_tumor_type = unname(tumor_type_lookup[group1_Metadata_Biology_PatientTumor]),
        group2_tumor_type = unname(tumor_type_lookup[group2_Metadata_Biology_PatientTumor]),
        same_patient = group1_Metadata_Biology_PatientTumor == group2_Metadata_Biology_PatientTumor,
        same_tumor_type = group1_tumor_type == group2_tumor_type,
        patient_relationship = factor(
            case_when(
                same_patient ~ "Same patient",
                same_tumor_type ~ "Same tumor type",
                TRUE ~ "Different tumor type"
            ),
            levels = c("Same patient", "Same tumor type", "Different tumor type")
        ),
        # exactly one sample is DMSO: the other sample is the treatment being
        # compared against vehicle
        is_treatment_vs_dmso = xor(
            group1_Metadata_Experiment_Treatment == "DMSO",
            group2_Metadata_Experiment_Treatment == "DMSO"
        ),
        # a treatment is a drug at a specific dose, e.g. "Selumetinib 10 uM"
        group1_treatment_label = paste(
            group1_Metadata_Experiment_Treatment,
            group1_Metadata_Experiment_Dose,
            group1_Metadata_Experiment_Unit
        ),
        group2_treatment_label = paste(
            group2_Metadata_Experiment_Treatment,
            group2_Metadata_Experiment_Dose,
            group2_Metadata_Experiment_Unit
        ),
        treated_sample_label = if_else(
            group1_Metadata_Experiment_Treatment == "DMSO",
            group2_treatment_label,
            group1_treatment_label
        ),
        treated_sample_target = if_else(
            group1_Metadata_Experiment_Treatment == "DMSO",
            group2_Metadata_Experiment_Target,
            group1_Metadata_Experiment_Target
        ),
        # order-independent pair label, so "A + B" and "B + A" are one level
        tumor_type_pair = if_else(
            match(group1_tumor_type, names(tumor_type_palette)) <= match(group2_tumor_type, names(tumor_type_palette)),
            paste(group1_tumor_type, group2_tumor_type, sep = " + "),
            paste(group2_tumor_type, group1_tumor_type, sep = " + ")
        )
    )

# one hue per drug, light for the low dose and dark for the high dose, so
# the two doses of a drug read as a pair when all treatments share one plot
treatment_levels_df <- bind_rows(
    correlation_viability_df %>%
        distinct(
            treatment = group1_Metadata_Experiment_Treatment,
            dose = group1_Metadata_Experiment_Dose,
            label = group1_treatment_label
        ),
    correlation_viability_df %>%
        distinct(
            treatment = group2_Metadata_Experiment_Treatment,
            dose = group2_Metadata_Experiment_Dose,
            label = group2_treatment_label
        )
) %>%
    distinct() %>%
    filter(treatment != "DMSO") %>%
    arrange(treatment, dose) %>%
    mutate(
        drug_index = as.integer(factor(treatment)),
        hue = seq(15, 375, length.out = n_distinct(treatment) + 1)[drug_index],
        color = hcl(h = hue, c = 90, l = if_else(dose == 1, 75, 40))
    )
treatment_palette <- setNames(treatment_levels_df$color, treatment_levels_df$label)

# a single set of axis limits for every subset, so panels are directly
# comparable to each other and to the full-data plot in 5b
correlation_limits <- range(correlation_viability_df$correlation, na.rm = TRUE)
viability_diff_limits <- c(0, 1)

dim(correlation_viability_df)

# Ways to color the points, for subsets where every pair has a single
# treatment of interest (treatment vs DMSO, or the same treatment in two
# patients):
#   treatment_label      - which drug + dose each pair is
# and, for same-treatment cross-patient pairs:
#   tumor_type_pair      - which two tumor types each pair spans
color_metrics <- list(
    treatment_label = list(
        label = "Treatment (drug + dose)",
        scale = scale_color_manual(values = treatment_palette, drop = TRUE)
    ),
    tumor_type_pair = list(
        label = "Tumor type pair",
        scale = scale_color_brewer(palette = "Dark2")
    )
)

# fewer points per panel -> less overplotting -> more opaque points
alpha_for_panel_size <- function(median_points_per_panel) {
    min(0.8, max(0.05, 150 / median_points_per_panel))
}

# 2D density contours are noise below a few dozen points, so only draw them
# for panels with enough pairs
min_points_for_density <- 50

# rows of the groups that can carry a density contour. kde2d's bandwidth is
# IQR-based, so a group where either axis has zero IQR (e.g. every pair at
# viability difference 0) fails, and one failing group drops the contour
# layer from the whole plot
density_eligible_rows <- function(df, group_col) {
    df %>%
        group_by(.data[[group_col]]) %>%
        filter(
            n() >= min_points_for_density,
            IQR(correlation) > 0,
            IQR(group1_group2_viability_diff) > 0
        ) %>%
        ungroup()
}

# color_metric = NULL draws uncolored points; show_other_groups = TRUE draws
# every other facet's pairs in faint gray behind each panel, so each group
# reads against the full distribution
plot_correlation_vs_viability <- function(
    df, facet_col, color_metric, title, show_other_groups = FALSE
) {
    # facet_col = NULL pools every pair into one panel
    if (is.null(facet_col)) {
        df <- df %>% mutate(all_pairs = "All pairs")
        facet_col <- "all_pairs"
    }
    other_groups_layer <- NULL
    if (show_other_groups) {
        other_groups_df <- bind_rows(lapply(unique(df[[facet_col]]), function(group) {
            other_rows <- df[df[[facet_col]] != group, c("correlation", "group1_group2_viability_diff")]
            other_rows[[facet_col]] <- group
            other_rows
        }))
        other_groups_layer <- geom_point(
            data = other_groups_df,
            color = "gray70",
            alpha = 0.02,
            size = 0.6
        )
    }
    panel_sizes <- df %>% count(.data[[facet_col]])
    density_df <- density_eligible_rows(df, facet_col)
    # skipped when no panel has enough pairs (e.g. per-patient
    # treatment-vs-DMSO), since an empty contour layer only warns
    density_layer <- NULL
    if (nrow(density_df) > 0) {
        density_layer <- geom_density_2d(
            data = density_df,
            color = "black",
            linewidth = 0.4,
            alpha = 0.5,
            bins = 6
        )
    }
    point_alpha <- alpha_for_panel_size(median(panel_sizes$n))
    # label each facet with its pair count so sparse panels are obvious
    facet_labels <- setNames(
        paste0(panel_sizes[[facet_col]], " (n = ", panel_sizes$n, ")"),
        panel_sizes[[facet_col]]
    )
    n_panels <- nrow(panel_sizes)

    if (is.null(color_metric)) {
        # gray, not black, so the black contours stay visible in dense panels
        point_layer <- geom_point(color = "gray50", alpha = point_alpha, size = 1)
        color_scale <- NULL
        color_guide <- NULL
        color_label <- NULL
    } else {
        point_layer <- geom_point(aes(color = .data[[color_metric]]), alpha = point_alpha, size = 1)
        color_scale <- color_metrics[[color_metric]]$scale
        color_label <- color_metrics[[color_metric]]$label
        color_guide <- guides(color = guide_legend(
            ncol = if (n_distinct(df[[color_metric]]) > 12) 2 else 1,
            override.aes = list(alpha = 1, size = 4)
        ))
    }

    plot <- (
        ggplot(df, aes(x = correlation, y = group1_group2_viability_diff))
        + other_groups_layer
        + point_layer
        + density_layer
        + color_scale
        + facet_wrap(
            vars(.data[[facet_col]]),
            ncol = ceiling(sqrt(n_panels)),
            labeller = as_labeller(facet_labels)
        )
        + coord_cartesian(xlim = correlation_limits, ylim = viability_diff_limits)
        + labs(
            title = title,
            x = "Morphology profile correlation between two samples",
            y = "abs(Viability difference) between samples",
            color = color_label
        )
        + color_guide
        + theme_manuscript(base_size = 14)
        + theme(strip.text = element_text(size = 11))
    )
    list(plot = plot, n_panels = n_panels)
}

# every subset/color-metric plot becomes one page of a single combined PDF
# (written at the end of this notebook); each page is sized to its own
# facet grid
pdf_pages <- list()

# queues one page per color metric for a subset; metrics that are constant
# within the subset carry no information and are skipped, and a subset with
# no (informative) metric gets a single uncolored page
add_subset_plots <- function(
    df, facet_col, title, metrics = character(0), show_other_groups = FALSE
) {
    informative_metrics <- Filter(function(m) n_distinct(df[[m]]) >= 2, metrics)
    if (length(informative_metrics) == 0) {
        informative_metrics <- list(NULL)
    }
    for (color_metric in informative_metrics) {
        page_title <- if (is.null(color_metric)) {
            title
        } else {
            paste0(title, " (colored by ", color_metrics[[color_metric]]$label, ")")
        }
        result <- plot_correlation_vs_viability(
            df, facet_col, color_metric, page_title,
            show_other_groups = show_other_groups
        )
        n_cols <- ceiling(sqrt(result$n_panels))
        n_rows <- ceiling(result$n_panels / n_cols)
        # floor keeps single-panel pages wide enough for the treatment legend
        pdf_pages[[length(pdf_pages) + 1]] <<- list(
            plot = result$plot,
            width = max(3.5 * n_cols + 2.5, 11),
            height = max(3.2 * n_rows + 1.2, 8)
        )
    }
    invisible(NULL)
}

# one panel with every group's pairs together, plus one density contour per
# group in that group's color, so groups are compared on the same axes
# rather than across facets
add_overlay_plot <- function(df, group_col, title, color_scale) {
    n_groups <- n_distinct(df[[group_col]])
    plot <- (
        ggplot(
            df,
            aes(x = correlation, y = group1_group2_viability_diff, color = .data[[group_col]])
        )
        + geom_point(alpha = alpha_for_panel_size(nrow(df)), size = 1)
        + geom_density_2d(
            data = density_eligible_rows(df, group_col),
            linewidth = 0.7,
            bins = 5
        )
        + color_scale
        + coord_cartesian(xlim = correlation_limits, ylim = viability_diff_limits)
        + labs(
            title = paste0(title, " (all groups, per-group contours)"),
            x = "Morphology profile correlation between two samples",
            y = "abs(Viability difference) between samples",
            color = NULL
        )
        + guides(color = guide_legend(
            ncol = if (n_groups > 20) 2 else 1,
            override.aes = list(alpha = 1, size = 4, linewidth = 1)
        ))
        + theme_manuscript(base_size = 14)
    )
    pdf_pages[[length(pdf_pages) + 1]] <<- list(plot = plot, width = 11, height = 8)
    invisible(NULL)
}

# 1. per patient: pairs where both samples come from the same patient
per_patient_df <- correlation_viability_df %>%
    filter(same_patient) %>%
    mutate(patient = group1_Metadata_Biology_PatientTumor)
add_subset_plots(
    per_patient_df,
    facet_col = "patient",
    title = "Within-patient pairs"
)
add_overlay_plot(
    per_patient_df,
    group_col = "patient",
    title = "Within-patient pairs",
    color_scale = scale_color_manual(values = tab20_palette_for_patients)
)

# 2. treatment + DMSO: every treatment (drug + dose) paired against DMSO,
# pooled on one plot and colored by treatment (all patients, so
# cross-patient pairs are included -- see subset 3 for within-patient only)
treatment_vs_dmso_df <- correlation_viability_df %>%
    filter(is_treatment_vs_dmso) %>%
    mutate(treatment_label = treated_sample_label)
add_subset_plots(
    treatment_vs_dmso_df,
    facet_col = NULL,
    title = "Treatment vs DMSO pairs (all patients)",
    metrics = "treatment_label"
)

# 3. per patient x (treatment + DMSO): within-patient treatment-vs-DMSO
# pairs, one pooled plot per patient
patient_treatment_vs_dmso_df <- treatment_vs_dmso_df %>%
    filter(same_patient) %>%
    mutate(patient = group1_Metadata_Biology_PatientTumor)
for (patient_id in sort(unique(patient_treatment_vs_dmso_df$patient))) {
    add_subset_plots(
        patient_treatment_vs_dmso_df %>% filter(patient == patient_id),
        facet_col = NULL,
        title = paste0(patient_id, ": treatment vs DMSO pairs"),
        metrics = "treatment_label"
    )
}

# 4. per tumor type: pairs where both samples come from the same tumor type
# (includes within-patient and cross-patient pairs of that tumor type)
per_tumor_type_df <- correlation_viability_df %>%
    filter(same_tumor_type) %>%
    mutate(tumor_type = factor(group1_tumor_type, levels = names(tumor_type_palette)))
add_subset_plots(
    per_tumor_type_df,
    facet_col = "tumor_type",
    title = "Within-tumor-type pairs"
)
add_overlay_plot(
    per_tumor_type_df,
    group_col = "tumor_type",
    title = "Within-tumor-type pairs",
    color_scale = scale_color_manual(values = tumor_type_palette, drop = TRUE)
)

# 5. per tumor type x (treatment + DMSO): treatment-vs-DMSO pairs within a
# tumor type, one pooled plot per tumor type
tumor_type_treatment_vs_dmso_df <- treatment_vs_dmso_df %>%
    filter(same_tumor_type) %>%
    mutate(tumor_type = group1_tumor_type)
for (tumor_type_id in intersect(names(tumor_type_palette), unique(tumor_type_treatment_vs_dmso_df$tumor_type))) {
    add_subset_plots(
        tumor_type_treatment_vs_dmso_df %>% filter(tumor_type == tumor_type_id),
        facet_col = NULL,
        title = paste0(tumor_type_id, ": treatment vs DMSO pairs"),
        metrics = "treatment_label"
    )
}

# 6. per drug target + DMSO: pools treatments that share a target (e.g. the
# four MEK1/2 inhibitors) to ask whether target class, not individual drug,
# sets where treatment-vs-DMSO pairs land
add_subset_plots(
    treatment_vs_dmso_df,
    facet_col = "treated_sample_target",
    title = "Treatment vs DMSO pairs, grouped by drug target",
    show_other_groups = TRUE
)
add_overlay_plot(
    treatment_vs_dmso_df,
    group_col = "treated_sample_target",
    title = "Treatment vs DMSO pairs, grouped by drug target",
    color_scale = scale_color_hue(l = 50, c = 90)
)

# 7. same treatment across patients: pairs of the same treatment and dose
# from two different patients -- does a drug produce a consistent morphology
# (high correlation) regardless of patient?
same_treatment_across_patients_df <- correlation_viability_df %>%
    filter(
        !same_patient,
        group1_Metadata_Experiment_Treatment == group2_Metadata_Experiment_Treatment,
        group1_Metadata_Experiment_Dose == group2_Metadata_Experiment_Dose
    ) %>%
    mutate(treatment_label = group1_treatment_label)
add_subset_plots(
    same_treatment_across_patients_df,
    facet_col = NULL,
    title = "Same treatment (drug + dose), different patients",
    metrics = c("treatment_label", "tumor_type_pair")
)

# 8. by pair relationship: all pairs split into same patient / same tumor
# type / different tumor type, to compare patient-driven vs
# tumor-type-driven similarity side by side
add_subset_plots(
    correlation_viability_df,
    facet_col = "patient_relationship",
    title = "All pairs by patient relationship"
)
add_overlay_plot(
    correlation_viability_df,
    group_col = "patient_relationship",
    title = "All pairs by patient relationship",
    color_scale = scale_color_manual(values = c(
        "Same patient" = "#D55E00",
        "Same tumor type" = "#0072B2",
        "Different tumor type" = "#999999"
    ))
)

save_plots_pdf(
    lapply(pdf_pages, `[[`, "plot"),
    output_pdf_path,
    width = vapply(pdf_pages, `[[`, numeric(1), "width"),
    height = vapply(pdf_pages, `[[`, numeric(1), "height")
)
length(pdf_pages)
