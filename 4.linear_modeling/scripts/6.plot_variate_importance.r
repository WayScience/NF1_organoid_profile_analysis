list_of_packages <- c("ggplot2", "dplyr", "tidyr", "patchwork", "arrow", "RColorBrewer", "ComplexHeatmap", "circlize", "grid")
for (package in list_of_packages) {
    suppressPackageStartupMessages(
        suppressWarnings(
            library(package, character.only = TRUE, quietly = TRUE, warn.conflicts = FALSE)
        )
    )
}

find_git_root <- function() {
    current_path <- getwd()
    while (!dir.exists(file.path(current_path, ".git"))) {
        parent_path <- dirname(current_path)
        if (parent_path == current_path) stop("No Git root directory found.")
        current_path <- parent_path
    }
    current_path
}
root_dir <- find_git_root()
source(file.path(root_dir, "utils/r_plot_themes.r"))
source(file.path(root_dir, "utils/r_plot_funcs.r"))

results_path <- file.path(root_dir, "4.linear_modeling/results/variate_importance")
figures_path <- file.path(root_dir, "4.linear_modeling/figures/variate_importance")
dir.create(figures_path, recursive = TRUE, showWarnings = FALSE)

venn_df <- arrow::read_parquet(file.path(results_path, "variate_hit_venn_regions_all_scopes.parquet"))
upset_df <- arrow::read_parquet(file.path(results_path, "variate_hit_upset_counts_all_scopes.parquet"))
sizes_df <- arrow::read_parquet(file.path(results_path, "variate_hit_set_sizes_all_scopes.parquet"))

# every scope: the columns that define one Venn + UpSet pair
scopes <- list(
    all_models = character(0),
    per_patient_treatment = c("patient", "treatment"),
    per_patient = c("patient"),
    per_treatment = c("treatment"),
    per_treatment_tumor_type = c("treatment", "tumor_type"),
    per_tumor_type = c("tumor_type")
)
model_sets <- c("original", "technical")


group_title <- function(group_cols, row) {
    if (length(group_cols) == 0) return("all models")
    paste(sprintf("%s=%s", group_cols, unlist(row[group_cols])), collapse = ", ")
}

# one multi-page pdf per scope (Venn then UpSet page for every group and model set); a pdf is only rebuilt when missing
for (scope in names(scopes)) {
    pdf_path <- if (scope == "all_models") {
        file.path(figures_path, "significant_variates_per_all_groups.pdf")
    } else {
        file.path(figures_path, scope, paste0(scope, ".pdf"))
    }
    if (file.exists(pdf_path)) {
        next
    }
    dir.create(dirname(pdf_path), recursive = TRUE, showWarnings = FALSE)
    group_cols <- scopes[[scope]]
    plots <- list()
    widths <- c()
    heights <- c()
    for (model in model_sets) {
        sizes_scope <- sizes_df |> filter(model_set == model, scope == !!scope)
        groups <- sizes_scope |> select(all_of(group_cols)) |> distinct() |> arrange(across(all_of(group_cols)))
        if (length(group_cols) == 0) groups <- data.frame(dummy = 1)
        for (i in seq_len(nrow(groups))) {
            in_group <- function(df) {
                df <- df |> filter(model_set == model, scope == !!scope)
                for (col in group_cols) df <- df[df[[col]] == groups[[col]][i], ]
                df
            }
            title <- group_title(group_cols, groups[i, , drop = FALSE])
            sizes_venn <- in_group(sizes_df) |> filter(plot == "venn")
            sizes_upset <- in_group(sizes_df) |> filter(plot == "upset")
            combos <- in_group(upset_df)
            plots[[length(plots) + 1]] <- plot_venn(in_group(venn_df), sizes_venn, title)
            widths <- c(widths, 7.5)
            heights <- c(heights, 6)
            if (nrow(combos) > 0) {
                plots[[length(plots) + 1]] <- plot_upset(combos, sizes_upset, title)
                dims <- upset_dims(nrow(combos), nrow(sizes_upset))
                widths <- c(widths, dims[["width"]])
                heights <- c(heights, dims[["height"]])
            }
        }
    }
    save_plots_pdf(plots, pdf_path, width = widths, height = heights)
}

lm_results_path <- file.path(root_dir, "4.linear_modeling/results/linear_modeling")
cooccurrence_figures_path <- file.path(figures_path, "treatment_only_cooccurrence")
dir.create(cooccurrence_figures_path, recursive = TRUE, showWarnings = FALSE)
cooccurrence_pdf <- file.path(cooccurrence_figures_path, "treatment_only_cooccurrence.pdf")

# profile -> (original model file, technical model file)
lm_files <- list(
    organoid = c(original = "organoid_norm", technical = "organoid_norm_technical_model"),
    sc = c(original = "sc_norm", technical = "sc_norm_technical_model"),
    organoid_agg = c(original = "organoid_agg", technical = "organoid_agg_technical_model"),
    sc_agg = c(original = "single_cell_agg", technical = "single_cell_agg_technical_model")
)
meki_drugs <- names(treatment_moa_map)[treatment_moa_map == "MEK1/2 inhibitor"]

read_significant <- function(file_name) {
    df <- arrow::read_parquet(file.path(lm_results_path, paste0(file_name, ".parquet")))
    # the technical models name the columns / treatment term differently
    df <- df |>
        rename(any_of(c(patient = "Metadata_Biology_PatientTumor", treatment = "Metadata_Experiment_Treatment"))) |>
        mutate(term = ifelse(term == "Metadata_Experiment_Treatment", "treatment", term))
    df |>
        filter(pvalue_fdr < 0.05) |>
        group_by(patient, treatment, feature) |>
        # treatment is significant and it is the only variate that is
        summarise(treatment_only = all(term == "treatment"), Feature_type = first(Feature_type), .groups = "drop") |>
        filter(treatment_only)
}

occurrence_tables <- list()
pages <- list()
for (profile in names(lm_files)) {
    for (model in model_sets) {
        sig <- read_significant(lm_files[[profile]][[model]])
        occurrence_tables[[length(occurrence_tables) + 1]] <- sig |>
            select(patient, treatment, feature, Feature_type, treatment_only) |>
            mutate(profile = profile, model_set = model)
        label <- paste0("Treatment-only significant: ", profile, ", ", model, " model")
        meki_sig <- sig |> filter(sub("_.*$", "", treatment) %in% meki_drugs)
        # the full matrix, then the same matrix subset to the MEKi treatments
        pages[[length(pages) + 1]] <- heatmap_grid_page(list(cooccurrence_heatmap(sig, label)))
        pages[[length(pages) + 1]] <- heatmap_grid_page(list(cooccurrence_heatmap(meki_sig, paste0(label, " (MEKi treatments only)"))))
    }
}
save_plots_pdf(pages, cooccurrence_pdf, width = 15, height = 10)
arrow::write_parquet(
    bind_rows(occurrence_tables),
    file.path(results_path, "treatment_only_cooccurrence.parquet")
)

multiresult_figure_subpanels_path <- file.path(figures_path, "multiresult_figure_subpanels")
dir.create(multiresult_figure_subpanels_path, recursive = TRUE, showWarnings = FALSE)

# panel: tumor-type UpSet, tumor_type = pNF, original model
pnf_sizes <- sizes_df |> filter(model_set == "original", scope == "per_tumor_type", tumor_type == "pNF")
pnf_upset_sizes <- pnf_sizes |> filter(plot == "upset")
pnf_combos <- upset_df |> filter(model_set == "original", scope == "per_tumor_type", tumor_type == "pNF")
p_pnf_upset <- plot_upset(pnf_combos, pnf_upset_sizes, show_title = FALSE)
pnf_dims <- upset_dims(nrow(pnf_combos), nrow(pnf_upset_sizes))
ggsave(
    file.path(multiresult_figure_subpanels_path, "tumor_type_pNF_upset.png"), p_pnf_upset,
    width = pnf_dims[["width"]], height = pnf_dims[["height"]], dpi = 600, bg = "white"
)

# panel: treatment-only co-occurrence heatmap, organoid, original model
sig_organoid <- read_significant(lm_files[["organoid"]][["original"]])
ht_organoid_notitle <- cooccurrence_heatmap(
    sig_organoid, title = NULL, show_title = FALSE, legend_title_size = 22, legend_label_size = 20
)
png(file.path(multiresult_figure_subpanels_path, "cooccurrence_organoid_original.png"), width = 15, height = 10, units = "in", res = 600)
ComplexHeatmap::draw(ht_organoid_notitle)
dev.off()

cat("multiresult-figure subpanels ->", multiresult_figure_subpanels_path, "\n")
