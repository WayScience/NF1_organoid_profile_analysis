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
source(file.path(root_dir, "utils", "r_plot_funcs.r"))

module_dir <- file.path(root_dir, "figures", "clinical_trial_data_plots")
data_dir <- file.path(module_dir, "data")
figures_dir <- file.path(module_dir, "figures")
dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)

trial_summary_path <- file.path(data_dir, "trial_efficacy_safety_summary.parquet")
response_distribution_path <- file.path(data_dir, "trial_response_heterogeneity.parquet")
orr_figure_path <- file.path(figures_dir, "clinical_trial_orr_efficacy.png")
heterogeneity_figure_path <- file.path(figures_dir, "clinical_trial_response_heterogeneity.png")

drug_arm_palette <- c(
    "Selumetinib" = "#2B2B2B",
    "Mirdametinib" = "#7A7A7A",
    "Placebo" = "#C2C2C2"
)

trial_summary_df <- read_parquet(trial_summary_path)

orr_df <- trial_summary_df %>%
    mutate(
        cohort_label = paste0(trial, " (", population, ", n=", n, ")"),
        cohort_label = sub("^SPRINT Phase ", "SPRINT-", cohort_label),
        cohort_label = factor(
            cohort_label,
            levels = cohort_label[order(year, match(drug, c("Selumetinib", "Mirdametinib", "Placebo")), population)]
        )
    )

orr_plot <- (
    ggplot(orr_df, aes(x = cohort_label, y = orr_pct, fill = drug))
    + geom_col(width = 0.6)
    + geom_errorbar(aes(ymin = orr_ci_low, ymax = orr_ci_high), width = 0.2, linewidth = 0.6)
    + geom_text(aes(label = paste0(orr_pct, "%")), vjust = -0.6, size = 5)
    + scale_fill_manual(values = drug_arm_palette)
    + scale_y_continuous(limits = c(0, 100), expand = expansion(mult = c(0, 0.1)))
    + labs(
        x = NULL,
        y = "Objective response rate (%)",
        fill = "Treatment",
        title = "Objective response rate across NF1-plexiform neurofibroma MEK-inhibitor trials",
        subtitle = "Error bars are 95% CI where reported in the primary publication"
    )
    + theme_manuscript(base_size = 14, x_text = "angled")
    + theme(plot.margin = margin(10, 10, 10, 20))
)
orr_plot

save_ggplot(orr_plot, orr_figure_path, width = 12, height = 7)

response_distribution_df <- read_parquet(response_distribution_path)

heterogeneity_df <- response_distribution_df %>%
    mutate(
        cohort_label = paste0(trial, "\n", population, ", ", subgroup, " (n=", n, ")"),
        cohort_label = factor(cohort_label, levels = rev(cohort_label))
    )

heterogeneity_plot <- (
    ggplot(heterogeneity_df, aes(x = cohort_label, y = median_pct_change, color = drug))
    + geom_hline(yintercept = 0, linetype = "dashed", color = "grey50")
    + geom_hline(yintercept = -20, linetype = "dotted", color = "grey50")
    + geom_linerange(aes(ymin = min_pct_change, ymax = max_pct_change), linewidth = 1)
    + geom_point(size = 3)
    + scale_color_manual(values = drug_arm_palette)
    + coord_flip()
    + labs(
        x = NULL,
        y = "Best percent change in target plexiform neurofibroma volume (%)",
        color = "Treatment",
        title = "Heterogeneity in individual-patient tumor volume response",
        subtitle = "Point = median; whiskers = reported min-max range per trial x cohort x subgroup",
        caption = "Dashed line: no change. Dotted line: partial-response threshold (-20%)."
    )
    + theme_manuscript(base_size = 14)
)
heterogeneity_plot

save_ggplot(heterogeneity_plot, heterogeneity_figure_path, width = 12, height = 6)
