list_of_packages <- c("ggplot2", "dplyr", "tidyr", "arrow", "RColorBrewer")
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

module_dir <- file.path(root_dir, "figures", "clinical_trial_data_plots")
data_dir <- file.path(module_dir, "data")
figures_dir <- file.path(module_dir, "figures")
dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)

dropout_summary_path <- file.path(data_dir, "trial_dropout_summary.parquet")
adverse_events_detail_path <- file.path(data_dir, "trial_adverse_events_detail.parquet")
dropout_figure_path <- file.path(figures_dir, "clinical_trial_dropout_rates.png")
adverse_events_figure_path <- file.path(figures_dir, "clinical_trial_adverse_events.png")

dropout_df <- read_parquet(dropout_summary_path)

dropout_long_df <- dropout_df %>%
    mutate(
        cohort_label = if_else(
            arm == "Placebo",
            paste0(trial, " (", population, ", Placebo, n=", n, ")"),
            paste0(trial, " (", population, ", n=", n, ")")
        ),
        cohort_label = sub("^SPRINT Phase ", "SPRINT-", cohort_label)
    ) %>%
    select(cohort_label, trial, `Any reason` = discontinued_any_reason_pct, `Due to adverse event` = discontinued_for_ae_pct) %>%
    pivot_longer(cols = c("Any reason", "Due to adverse event"), names_to = "reason", values_to = "pct") %>%
    mutate(
        cohort_label = factor(cohort_label, levels = unique(cohort_label)),
        reason = factor(reason, levels = c("Any reason", "Due to adverse event"))
    )

dropout_palette <- c("Any reason" = "#4D4D4D", "Due to adverse event" = "#B0B0B0")

dropout_plot <- (
    ggplot(dropout_long_df %>% filter(!is.na(pct)), aes(x = cohort_label, y = pct, fill = reason))
    + geom_col(position = position_dodge(width = 0.7), width = 0.6)
    + geom_text(
        aes(label = paste0(pct, "%")),
        position = position_dodge(width = 0.7),
        vjust = -0.5,
        size = 4
    )
    + scale_fill_manual(values = dropout_palette)
    + scale_y_continuous(limits = c(0, 50), expand = expansion(mult = c(0, 0.1)))
    + labs(
        x = NULL,
        y = "Patients who discontinued treatment (%)",
        fill = NULL,
        title = "Treatment discontinuation across NF1-plexiform neurofibroma MEK-inhibitor trials",
        subtitle = "'Any reason' is only shown where the paper reports an overall discontinuation count (not reported for ReNeu)"
    )
    + theme_manuscript(base_size = 14, x_text = "angled")
    + theme(plot.margin = margin(10, 10, 10, 40))
)
dropout_plot

save_ggplot(dropout_plot, dropout_figure_path, width = 12, height = 7)

adverse_events_df <- read_parquet(adverse_events_detail_path)

ae_heatmap_df <- adverse_events_df %>%
    mutate(
        arm_label = paste0(trial, "\n", population, ", ", arm),
        arm_label = factor(arm_label, levels = unique(arm_label[order(trial, drug != "Placebo")])),
        ae_term = factor(ae_term, levels = rev(sort(unique(ae_term))))
    )

ae_heatmap_plot <- (
    ggplot(ae_heatmap_df, aes(x = arm_label, y = ae_term, fill = pct_all_grades))
    + geom_tile(color = "white", linewidth = 0.6)
    + geom_text(aes(label = paste0(round(pct_all_grades), "%"), color = pct_all_grades > 50), size = 3.5, show.legend = FALSE)
    + scale_fill_gradient(low = "#FFFFFF", high = "#2B2B2B", limits = c(0, 100))
    + scale_color_manual(values = c("FALSE" = "black", "TRUE" = "white"))
    + labs(
        x = NULL,
        y = NULL,
        fill = "All-grade\nincidence (%)",
        title = "Most common adverse events by trial arm",
        subtitle = "ReNeu (mirdametinib) and KOMET (selumetinib vs placebo)",
        caption = "SPRINT II reports toxicity narratively, without a per-term frequency table, so it is excluded here."
    )
    + theme_manuscript(base_size = 14, x_text = "angled")
    + theme(panel.grid = element_blank())
)
ae_heatmap_plot

save_ggplot(ae_heatmap_plot, adverse_events_figure_path, width = 12, height = 8)
