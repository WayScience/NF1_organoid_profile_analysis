packages <- c("dplyr", "ggplot2", "tidyr", "patchwork", "arrow", "coop", "RColorBrewer")
for (pkg in packages) {
    suppressPackageStartupMessages(
        suppressWarnings(
            library(pkg, character.only = TRUE)
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

git_root <- find_git_root()
source(file.path(git_root, "utils", "r_plot_themes.r"))
source(file.path(git_root, "utils", "r_plot_funcs.r"))

morphology_profile_path <- file.path(
    git_root, "data", "profiles_3D", "all_patients", "3.consensus_profiles",
    "sc_norm_sc_consensus_profiles.parquet"
)
viability_log2fc_path <- file.path(
    git_root, "5.differential_analysis", "results", "viability_log2fc.parquet"
)
results_dir <- file.path(git_root, "5.differential_analysis", "results")
figures_dir <- file.path(git_root, "5.differential_analysis", "figures")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)

metadata_cols <- c(
    "Metadata_Biology_PatientTumor", "Metadata_Experiment_Treatment",
    "Metadata_Experiment_Dose", "Metadata_Experiment_Class",
    "Metadata_Experiment_Target", "Metadata_Experiment_TherapeuticCategories",
    "Metadata_Experiment_Unit"
)

morphology_df <- arrow::read_parquet(morphology_profile_path) %>%
    dplyr::filter(!is.na(Metadata_Experiment_Treatment))
feature_cols <- setdiff(colnames(morphology_df), metadata_cols)
head(morphology_df)

compute_patient_cosine <- function(patient_df) {
    #' For one patient: cosine similarity (coop::cosine) of every
    #' treatment's morphology feature vector against that patient's own
    #' DMSO feature vector. Returns one row per non-DMSO treatment.
    dmso_row <- which(patient_df$Metadata_Experiment_Treatment == "DMSO")
    if (length(dmso_row) != 1) {
        warning(
            "expected exactly one DMSO row for ",
            unique(patient_df$Metadata_Biology_PatientTumor),
            ", found ", length(dmso_row), " - skipping patient"
        )
        return(patient_df[0, ])
    }
    # coop::cosine() compares matrix columns pairwise, so each sample
    # (treatment or DMSO) is one column and each feature is one row.
    feature_mat <- t(as.matrix(patient_df[, feature_cols]))
    similarity_to_dmso <- coop::cosine(feature_mat)[, dmso_row]
    patient_df$morphology_cosine_similarity <- similarity_to_dmso
    patient_df[-dmso_row, ]
}

morphology_cosine_df <- morphology_df %>%
    split(.$Metadata_Biology_PatientTumor) %>%
    lapply(compute_patient_cosine) %>%
    dplyr::bind_rows() %>%
    dplyr::select(dplyr::all_of(metadata_cols), morphology_cosine_similarity)
head(morphology_cosine_df)

viability_df <- arrow::read_parquet(viability_log2fc_path) %>%
    dplyr::select(
        Metadata_Biology_PatientTumor, Metadata_Experiment_Treatment, Metadata_Experiment_Dose,
        Metadata_Viability_Percentage, Metadata_DMSO_Viability_Percentage, viability_log2fc
    )

response_df <- morphology_cosine_df %>%
    dplyr::left_join(
        viability_df,
        by = c("Metadata_Biology_PatientTumor", "Metadata_Experiment_Treatment", "Metadata_Experiment_Dose")
    )

# tumor_type_lookup/patient_palette (utils/r_plot_themes.r) don't carry
# NF0037_T1_CQ1 - it's the same tumor re-imaged on a different instrument
# (CQ1), so it inherits NF0037_T1's tumor type and color.
local_tumor_type_lookup <- c(tumor_type_lookup, NF0037_T1_CQ1 = tumor_type_lookup[["NF0037_T1"]])
local_patient_palette <- c(patient_palette, NF0037_T1_CQ1 = patient_palette[["NF0037_T1"]])
# custom_MOA_palette's one multi-word key uses an embedded newline for
# compact legend wrapping; Metadata_Experiment_Target stores it as a plain
# space, so match palette names to the data instead of changing the data.
local_moa_palette <- setNames(custom_MOA_palette, gsub("\n", " ", names(custom_MOA_palette)))

response_df <- response_df %>%
    dplyr::mutate(
        Metadata_Biology_TumorType = local_tumor_type_lookup[Metadata_Biology_PatientTumor],
        Metadata_Biology_PatientTumor = factor(
            Metadata_Biology_PatientTumor, levels = names(local_patient_palette)
        ),
        Metadata_Biology_TumorType = factor(
            Metadata_Biology_TumorType, levels = names(tumor_type_palette)
        ),
        Metadata_Experiment_Treatment = factor(
            Metadata_Experiment_Treatment, levels = names(custom_treatment_palette)
        ),
        Metadata_Experiment_Target = factor(
            Metadata_Experiment_Target, levels = names(local_moa_palette)
        )
    )

arrow::write_parquet(
    response_df, file.path(results_dir, "morphology_viability_cosine_similarity.parquet")
)
response_df

width <- 8
height <- 6
dims <- set_plot_size(width, height)

plot_morphology_vs_viability <- function(color_by, palette, legend_title, plot_title) {
    #' Same morphology-cosine-similarity-vs-viability-log2FC scatter,
    #' recolored by a different grouping variable each call.
    plot_data <- response_df %>% dplyr::filter(!is.na(viability_log2fc))
    (
        ggplot(
            plot_data,
            aes(x = morphology_cosine_similarity, y = viability_log2fc, color = .data[[color_by]])
        )
        + geom_hline(yintercept = 0, linetype = "dashed", color = "grey50")
        + geom_point(alpha = 0.7, size = 2)
        + scale_color_manual(values = palette, name = legend_title, drop = TRUE, na.translate = FALSE)
        + labs(
            x = "Morphology cosine similarity to DMSO",
            y = expression(paste("Viability ", log[2], "(fold change vs DMSO)")),
            title = plot_title
        )
        + theme_manuscript(base_size = 14)
    )
}

morphology_vs_viability_plots <- list(
    plot_morphology_vs_viability(
        "Metadata_Biology_PatientTumor", local_patient_palette, "Patient", "Colored by patient"
    ),
    plot_morphology_vs_viability(
        "Metadata_Biology_TumorType", tumor_type_palette, "Tumor type", "Colored by tumor type"
    ),
    plot_morphology_vs_viability(
        "Metadata_Experiment_Treatment", custom_treatment_palette, "Treatment", "Colored by treatment"
    ),
    plot_morphology_vs_viability(
        "Metadata_Experiment_Target", local_moa_palette, "Target", "Colored by target"
    )
)

save_plots_pdf(
    morphology_vs_viability_plots,
    file.path(figures_dir, "morphology_vs_viability_cosine_similarity.pdf"),
    width = dims$width, height = dims$height
)
morphology_vs_viability_plots
