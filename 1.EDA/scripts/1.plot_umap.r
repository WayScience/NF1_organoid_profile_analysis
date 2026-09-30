list_of_packages <- c("ggplot2", "dplyr", "tidyr", "RColorBrewer")
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

# Get the current working directory and find Git root
find_git_root <- function() {
  # Get current working directory
  cwd <- getwd()

  # Check if current directory has .git
  if (dir.exists(file.path(cwd, ".git"))) {
    return(cwd)
  }

  # If not, search parent directories
  current_path <- cwd
  while (dirname(current_path) != current_path) { # While not at root
    parent_path <- dirname(current_path)
    if (dir.exists(file.path(parent_path, ".git"))) {
      return(parent_path)
    }
    current_path <- parent_path
  }

  # If no Git root directory found, stop with error
  stop("No Git root directory found.")
}

# Find the Git root directory
root_dir <- find_git_root()


# Shared color palettes/themes (e.g. custom_treatment_palette, custom_MOA_palette,
# tab20_palette_for_patients, tumor_type_palette, theme_manuscript) live in
# utils/r_plot_themes.r so that colors stay consistent across notebooks.
source(file.path(root_dir, "utils/r_plot_funcs.r"))
source(file.path(root_dir, "utils/r_plot_themes.r"))

figures_path <- file.path(root_dir, "1.EDA/figures/umaps")
if (!dir.exists(figures_path)) {
  dir.create(figures_path, recursive = TRUE)
}

figures_patient_specific_path <- file.path(figures_path, "patient_specific")
if (!dir.exists(figures_patient_specific_path)) {
  dir.create(figures_patient_specific_path, recursive = TRUE)
}

umap_results_dir <- file.path(root_dir, "1.EDA/results/umap")

# Load 3D single-cell UMAP results
sc_3D_umap_results <- arrow::read_parquet(
  file.path(umap_results_dir, "3D_scfs_umap.parquet")
)

dim(sc_3D_umap_results)
head(sc_3D_umap_results)

# Load 2D max projection single-cell UMAP results
# Harmonize column names to the 3D metadata naming convention
# (Metadata_Experiment_Treatment, Metadata_Biology_PatientTumor)
max_projection_2D_sc_umap_results <- arrow::read_parquet(
  file.path(umap_results_dir, "2D_maxproj_scfs_umap.parquet")
) %>%
  rename(
    Metadata_Experiment_Treatment = Metadata_treatment,
    Metadata_Biology_PatientTumor = Metadata_patient_tumor
  )

dim(max_projection_2D_sc_umap_results)
head(max_projection_2D_sc_umap_results)

# Load 2D middle slice single-cell UMAP results
# Harmonize column names to the 3D metadata naming convention
# (Metadata_Experiment_Treatment, Metadata_Biology_PatientTumor)
middle_slice_2D_sc_umap_results <- arrow::read_parquet(
  file.path(umap_results_dir, "2D_midslice_scfs_umap.parquet")
) %>%
  rename(
    Metadata_Experiment_Treatment = Metadata_treatment,
    Metadata_Biology_PatientTumor = Metadata_patient_tumor
  )

dim(middle_slice_2D_sc_umap_results)
head(middle_slice_2D_sc_umap_results)

# Define all possible patients from data/patient_IDs.txt. Each row is a
# distinct patient+tumor combination (e.g. NF0014_T1 and NF0014_T2 are
# treated as separate patients), matching the granularity of the 3D metadata
# (Metadata_Biology_PatientTumor is the only patient-identity column 3D has).
all_patient_tumor_ids <- readLines(file.path(root_dir, "data/patient_IDs.txt"))
all_patient_tumor_ids <- unique(all_patient_tumor_ids[all_patient_tumor_ids != ""])

# Create master palette from the shared tab20 palette (utils/r_plot_themes.r)
master_patient_palette <- setNames(tab20_palette_for_patients[1:length(all_patient_tumor_ids)], all_patient_tumor_ids)

patient_color_palette <- master_patient_palette[
  names(master_patient_palette) %in% unique(max_projection_2D_sc_umap_results$Metadata_Biology_PatientTumor)
]

sc_3D_umap_results$Metadata_Biology_TumorType <- tumor_type_lookup[sc_3D_umap_results$Metadata_Biology_PatientTumor]
max_projection_2D_sc_umap_results$Metadata_Biology_TumorType <- tumor_type_lookup[max_projection_2D_sc_umap_results$Metadata_Biology_PatientTumor]
middle_slice_2D_sc_umap_results$Metadata_Biology_TumorType <- tumor_type_lookup[middle_slice_2D_sc_umap_results$Metadata_Biology_PatientTumor]

# Patient legend labels with tumor type in parentheses (e.g. "NF0014_T1
# (cNF)"), used only for the unfaceted by-patient plots (which show a
# legend); facet plots use the plain patient ID as their strip label and are
# left unchanged.
patient_legend_labels <- setNames(
  paste0(names(tumor_type_lookup), " (", tumor_type_lookup, ")"),
  names(tumor_type_lookup)
)

sc_3D_umap_results$Metadata_Biology_PatientTumorLabel <- patient_legend_labels[sc_3D_umap_results$Metadata_Biology_PatientTumor]
max_projection_2D_sc_umap_results$Metadata_Biology_PatientTumorLabel <- patient_legend_labels[max_projection_2D_sc_umap_results$Metadata_Biology_PatientTumor]
middle_slice_2D_sc_umap_results$Metadata_Biology_PatientTumorLabel <- patient_legend_labels[middle_slice_2D_sc_umap_results$Metadata_Biology_PatientTumor]

patient_color_palette_labeled <- patient_color_palette
names(patient_color_palette_labeled) <- patient_legend_labels[names(patient_color_palette)]

plot_umap(
  data = sc_3D_umap_results,
  color_by = "Metadata_Biology_PatientTumorLabel",
  palette = patient_color_palette_labeled,
  title = "All patients: 3D Single Cell FS Profiles",
  legend_title = "Patient",
  alpha = 0.15
)

plot_umap(
  data = sc_3D_umap_results,
  color_by = "Metadata_Biology_TumorType",
  palette = tumor_type_palette,
  title = "All patients: 3D Single Cell FS Profiles",
  legend_title = "Tumor type",
  facet_by = "Metadata_Biology_PatientTumor",
  facet_nrow = 3,
  facet_legend = TRUE,
  alpha = 0.3,
  point_size = 0.3
)

plot_umap(
  data = sc_3D_umap_results,
  color_by = "Metadata_Experiment_Treatment",
  palette = custom_treatment_palette,
  title = "All patients: 3D Single Cell FS Profiles",
  legend_title = "Treatment",
  alpha = 0.15
)

plot_umap(
  data = sc_3D_umap_results,
  color_by = "Metadata_Biology_PatientTumor",
  palette = patient_color_palette,
  title = "All patients: 3D Single Cell FS Profiles",
  legend_title = "Patient",
  facet_by = "Metadata_Experiment_Treatment",
  facet_nrow = 4,
  facet_legend = TRUE,
  alpha = 0.3,
  point_size = 0.3,
  height = 8
)

plot_umap(
  data = sc_3D_umap_results,
  color_by = "Metadata_Biology_TumorType",
  palette = tumor_type_palette,
  title = "All patients: 3D Single Cell FS Profiles",
  legend_title = "Tumor type",
  alpha = 0.08,
  point_size = 0.5
)

plot_umap(
  data = sc_3D_umap_results,
  color_by = "Metadata_Biology_TumorType",
  palette = tumor_type_palette,
  title = "All patients: 3D Single Cell FS Profiles",
  facet_by = "Metadata_Biology_TumorType",
  facet_nrow = 2,
  alpha = 0.15,
  point_size = 0.15
)

# Filter for DMSO only
sc_umap_results_dmsos <- sc_3D_umap_results %>%
  filter(Metadata_Experiment_Treatment == "DMSO")

plot_umap(
  data = sc_umap_results_dmsos,
  color_by = "Metadata_Biology_PatientTumorLabel",
  palette = patient_color_palette_labeled,
  title = "All patients: 3D Single Cell FS Profiles (DMSO Controls)",
  legend_title = "Patient",
  alpha = 0.15
)

plot_umap(
  data = sc_umap_results_dmsos,
  color_by = "Metadata_Biology_PatientTumor",
  palette = patient_color_palette,
  title = "All patients: 3D Single Cell FS Profiles (DMSO Controls)",
  facet_by = "Metadata_Biology_PatientTumor",
  facet_nrow = 3,
  alpha = 0.3,
  point_size = 0.3
)

pdf_plots_3D <- list(
  by_patient = plot_umap(
    data = sc_3D_umap_results,
    color_by = "Metadata_Biology_PatientTumorLabel",
    palette = patient_color_palette_labeled,
    title = "All patients: 3D Single Cell FS Profiles",
    legend_title = "Patient",
    alpha = 0.15,
    rasterize_dpi = 150
  ),
  facet_by_patient = plot_umap(
    data = sc_3D_umap_results,
    color_by = "Metadata_Biology_TumorType",
    palette = tumor_type_palette,
    title = "All patients: 3D Single Cell FS Profiles",
    legend_title = "Tumor type",
    facet_by = "Metadata_Biology_PatientTumor",
    facet_nrow = 3,
    facet_legend = TRUE,
    alpha = 0.3,
    point_size = 0.3,
    rasterize_dpi = 150
  ),
  by_treatment = plot_umap(
    data = sc_3D_umap_results,
    color_by = "Metadata_Experiment_Treatment",
    palette = custom_treatment_palette,
    title = "All patients: 3D Single Cell FS Profiles",
    legend_title = "Treatment",
    alpha = 0.15,
    rasterize_dpi = 150
  ),
  facet_by_treatment = plot_umap(
    data = sc_3D_umap_results,
    color_by = "Metadata_Biology_PatientTumor",
    palette = patient_color_palette,
    title = "All patients: 3D Single Cell FS Profiles",
    legend_title = "Patient",
    facet_by = "Metadata_Experiment_Treatment",
    facet_nrow = 4,
    facet_legend = TRUE,
    alpha = 0.3,
    point_size = 0.3,
    height = 8,
    rasterize_dpi = 150
  ),
  by_tumor_type = plot_umap(
    data = sc_3D_umap_results,
    color_by = "Metadata_Biology_TumorType",
    palette = tumor_type_palette,
    title = "All patients: 3D Single Cell FS Profiles",
    legend_title = "Tumor type",
    alpha = 0.08,
    point_size = 0.5,
    rasterize_dpi = 150
  ),
  facet_by_tumor_type = plot_umap(
    data = sc_3D_umap_results,
    color_by = "Metadata_Biology_TumorType",
    palette = tumor_type_palette,
    title = "All patients: 3D Single Cell FS Profiles",
    facet_by = "Metadata_Biology_TumorType",
    facet_nrow = 2,
    alpha = 0.15,
    point_size = 0.15,
    rasterize_dpi = 150
  ),
  dmso_only = plot_umap(
    data = sc_umap_results_dmsos,
    color_by = "Metadata_Biology_PatientTumorLabel",
    palette = patient_color_palette_labeled,
    title = "All patients: 3D Single Cell FS Profiles (DMSO Controls)",
    legend_title = "Patient",
    alpha = 0.15,
    rasterize_dpi = 150
  ),
  dmso_facet_by_patient = plot_umap(
    data = sc_umap_results_dmsos,
    color_by = "Metadata_Biology_PatientTumor",
    palette = patient_color_palette,
    title = "All patients: 3D Single Cell FS Profiles (DMSO Controls)",
    facet_by = "Metadata_Biology_PatientTumor",
    facet_nrow = 3,
    alpha = 0.3,
    point_size = 0.3,
    rasterize_dpi = 150
  )
)
save_umap_pca_plots_pdf(pdf_plots_3D, file.path(figures_path, "3D_scfs_all_patients.pdf"))

plot_umap(
  data = max_projection_2D_sc_umap_results,
  color_by = "Metadata_Biology_PatientTumorLabel",
  palette = patient_color_palette_labeled,
  title = "All patients: 2D MIP Single Cell FS Profiles",
  legend_title = "Patient",
  alpha = 0.15
)

plot_umap(
  data = max_projection_2D_sc_umap_results,
  color_by = "Metadata_Biology_TumorType",
  palette = tumor_type_palette,
  title = "All patients: 2D MIP Single Cell FS Profiles",
  legend_title = "Tumor type",
  facet_by = "Metadata_Biology_PatientTumor",
  facet_nrow = 3,
  facet_legend = TRUE,
  alpha = 0.3,
  point_size = 0.3
)

plot_umap(
  data = max_projection_2D_sc_umap_results,
  color_by = "Metadata_Experiment_Treatment",
  palette = custom_treatment_palette,
  title = "All patients: 2D MIP Single Cell FS Profiles",
  legend_title = "Treatment",
  alpha = 0.15
)

plot_umap(
  data = max_projection_2D_sc_umap_results,
  color_by = "Metadata_Biology_PatientTumor",
  palette = patient_color_palette,
  title = "All patients: 2D MIP Single Cell FS Profiles",
  legend_title = "Patient",
  facet_by = "Metadata_Experiment_Treatment",
  facet_nrow = 4,
  facet_legend = TRUE,
  alpha = 0.3,
  point_size = 0.3,
  height = 8
)

plot_umap(
  data = max_projection_2D_sc_umap_results,
  color_by = "Metadata_Biology_TumorType",
  palette = tumor_type_palette,
  title = "All patients: 2D MIP Single Cell FS Profiles",
  legend_title = "Tumor type",
  alpha = 0.08,
  point_size = 0.5
)

plot_umap(
  data = max_projection_2D_sc_umap_results,
  color_by = "Metadata_Biology_TumorType",
  palette = tumor_type_palette,
  title = "All patients: 2D MIP Single Cell FS Profiles",
  facet_by = "Metadata_Biology_TumorType",
  facet_nrow = 2,
  alpha = 0.15,
  point_size = 0.15
)

# Filter for DMSO only
sc_umap_results_dmsos <- max_projection_2D_sc_umap_results %>%
  filter(Metadata_Experiment_Treatment == "DMSO")

plot_umap(
  data = sc_umap_results_dmsos,
  color_by = "Metadata_Biology_PatientTumorLabel",
  palette = patient_color_palette_labeled,
  title = "All patients: 2D MIP Single Cell FS Profiles (DMSO Controls)",
  legend_title = "Patient",
  alpha = 0.15
)

plot_umap(
  data = sc_umap_results_dmsos,
  color_by = "Metadata_Biology_PatientTumor",
  palette = patient_color_palette,
  title = "All patients: 2D MIP Single Cell FS Profiles (DMSO Controls)",
  facet_by = "Metadata_Biology_PatientTumor",
  facet_nrow = 3,
  alpha = 0.3,
  point_size = 0.3
)

pdf_plots_2D_maxproj <- list(
  by_patient = plot_umap(
    data = max_projection_2D_sc_umap_results,
    color_by = "Metadata_Biology_PatientTumorLabel",
    palette = patient_color_palette_labeled,
    title = "All patients: 2D MIP Single Cell FS Profiles",
    legend_title = "Patient",
    alpha = 0.15,
    rasterize_dpi = 150
  ),
  facet_by_patient = plot_umap(
    data = max_projection_2D_sc_umap_results,
    color_by = "Metadata_Biology_TumorType",
    palette = tumor_type_palette,
    title = "All patients: 2D MIP Single Cell FS Profiles",
    legend_title = "Tumor type",
    facet_by = "Metadata_Biology_PatientTumor",
    facet_nrow = 3,
    facet_legend = TRUE,
    alpha = 0.3,
    point_size = 0.3,
    rasterize_dpi = 150
  ),
  by_treatment = plot_umap(
    data = max_projection_2D_sc_umap_results,
    color_by = "Metadata_Experiment_Treatment",
    palette = custom_treatment_palette,
    title = "All patients: 2D MIP Single Cell FS Profiles",
    legend_title = "Treatment",
    alpha = 0.15,
    rasterize_dpi = 150
  ),
  facet_by_treatment = plot_umap(
    data = max_projection_2D_sc_umap_results,
    color_by = "Metadata_Biology_PatientTumor",
    palette = patient_color_palette,
    title = "All patients: 2D MIP Single Cell FS Profiles",
    legend_title = "Patient",
    facet_by = "Metadata_Experiment_Treatment",
    facet_nrow = 4,
    facet_legend = TRUE,
    alpha = 0.3,
    point_size = 0.3,
    height = 8,
    rasterize_dpi = 150
  ),
  by_tumor_type = plot_umap(
    data = max_projection_2D_sc_umap_results,
    color_by = "Metadata_Biology_TumorType",
    palette = tumor_type_palette,
    title = "All patients: 2D MIP Single Cell FS Profiles",
    legend_title = "Tumor type",
    alpha = 0.08,
    point_size = 0.5,
    rasterize_dpi = 150
  ),
  facet_by_tumor_type = plot_umap(
    data = max_projection_2D_sc_umap_results,
    color_by = "Metadata_Biology_TumorType",
    palette = tumor_type_palette,
    title = "All patients: 2D MIP Single Cell FS Profiles",
    facet_by = "Metadata_Biology_TumorType",
    facet_nrow = 2,
    alpha = 0.15,
    point_size = 0.15,
    rasterize_dpi = 150
  ),
  dmso_only = plot_umap(
    data = sc_umap_results_dmsos,
    color_by = "Metadata_Biology_PatientTumorLabel",
    palette = patient_color_palette_labeled,
    title = "All patients: 2D MIP Single Cell FS Profiles (DMSO Controls)",
    legend_title = "Patient",
    alpha = 0.15,
    rasterize_dpi = 150
  ),
  dmso_facet_by_patient = plot_umap(
    data = sc_umap_results_dmsos,
    color_by = "Metadata_Biology_PatientTumor",
    palette = patient_color_palette,
    title = "All patients: 2D MIP Single Cell FS Profiles (DMSO Controls)",
    facet_by = "Metadata_Biology_PatientTumor",
    facet_nrow = 3,
    alpha = 0.3,
    point_size = 0.3,
    rasterize_dpi = 150
  )
)
save_umap_pca_plots_pdf(pdf_plots_2D_maxproj, file.path(figures_path, "2D_maxproj_scfs_all_patients.pdf"))

plot_umap(
  data = middle_slice_2D_sc_umap_results,
  color_by = "Metadata_Biology_PatientTumorLabel",
  palette = patient_color_palette_labeled,
  title = "All patients: 2D Middle Slice Single Cell FS Profiles",
  legend_title = "Patient",
  alpha = 0.15
)

plot_umap(
  data = middle_slice_2D_sc_umap_results,
  color_by = "Metadata_Biology_TumorType",
  palette = tumor_type_palette,
  title = "All patients: 2D Middle Slice Single Cell FS Profiles",
  legend_title = "Tumor type",
  facet_by = "Metadata_Biology_PatientTumor",
  facet_nrow = 3,
  facet_legend = TRUE,
  alpha = 0.3,
  point_size = 0.3
)

plot_umap(
  data = middle_slice_2D_sc_umap_results,
  color_by = "Metadata_Experiment_Treatment",
  palette = custom_treatment_palette,
  title = "All patients: 2D Middle Slice Single Cell FS Profiles",
  legend_title = "Treatment",
  alpha = 0.15
)

plot_umap(
  data = middle_slice_2D_sc_umap_results,
  color_by = "Metadata_Biology_PatientTumor",
  palette = patient_color_palette,
  title = "All patients: 2D Middle Slice Single Cell FS Profiles",
  legend_title = "Patient",
  facet_by = "Metadata_Experiment_Treatment",
  facet_nrow = 4,
  facet_legend = TRUE,
  alpha = 0.3,
  point_size = 0.3,
  height = 8
)

plot_umap(
  data = middle_slice_2D_sc_umap_results,
  color_by = "Metadata_Biology_TumorType",
  palette = tumor_type_palette,
  title = "All patients: 2D Middle Slice Single Cell FS Profiles",
  legend_title = "Tumor type",
  alpha = 0.08,
  point_size = 0.5
)

plot_umap(
  data = middle_slice_2D_sc_umap_results,
  color_by = "Metadata_Biology_TumorType",
  palette = tumor_type_palette,
  title = "All patients: 2D Middle Slice Single Cell FS Profiles",
  facet_by = "Metadata_Biology_TumorType",
  facet_nrow = 2,
  alpha = 0.15,
  point_size = 0.15
)

# Filter for DMSO only
sc_umap_results_dmsos <- middle_slice_2D_sc_umap_results %>%
  filter(Metadata_Experiment_Treatment == "DMSO")

plot_umap(
  data = sc_umap_results_dmsos,
  color_by = "Metadata_Biology_PatientTumorLabel",
  palette = patient_color_palette_labeled,
  title = "All patients: 2D Middle Slice Single Cell FS Profiles (DMSO Controls)",
  legend_title = "Patient",
  alpha = 0.15
)

plot_umap(
  data = sc_umap_results_dmsos,
  color_by = "Metadata_Biology_PatientTumor",
  palette = patient_color_palette,
  title = "All patients: 2D Middle Slice Single Cell FS Profiles (DMSO Controls)",
  facet_by = "Metadata_Biology_PatientTumor",
  facet_nrow = 3,
  alpha = 0.3,
  point_size = 0.3
)

pdf_plots_2D_midslice <- list(
  by_patient = plot_umap(
    data = middle_slice_2D_sc_umap_results,
    color_by = "Metadata_Biology_PatientTumorLabel",
    palette = patient_color_palette_labeled,
    title = "All patients: 2D Middle Slice Single Cell FS Profiles",
    legend_title = "Patient",
    alpha = 0.15,
    rasterize_dpi = 150
  ),
  facet_by_patient = plot_umap(
    data = middle_slice_2D_sc_umap_results,
    color_by = "Metadata_Biology_TumorType",
    palette = tumor_type_palette,
    title = "All patients: 2D Middle Slice Single Cell FS Profiles",
    legend_title = "Tumor type",
    facet_by = "Metadata_Biology_PatientTumor",
    facet_nrow = 3,
    facet_legend = TRUE,
    alpha = 0.3,
    point_size = 0.3,
    rasterize_dpi = 150
  ),
  by_treatment = plot_umap(
    data = middle_slice_2D_sc_umap_results,
    color_by = "Metadata_Experiment_Treatment",
    palette = custom_treatment_palette,
    title = "All patients: 2D Middle Slice Single Cell FS Profiles",
    legend_title = "Treatment",
    alpha = 0.15,
    rasterize_dpi = 150
  ),
  facet_by_treatment = plot_umap(
    data = middle_slice_2D_sc_umap_results,
    color_by = "Metadata_Biology_PatientTumor",
    palette = patient_color_palette,
    title = "All patients: 2D Middle Slice Single Cell FS Profiles",
    legend_title = "Patient",
    facet_by = "Metadata_Experiment_Treatment",
    facet_nrow = 4,
    facet_legend = TRUE,
    alpha = 0.3,
    point_size = 0.3,
    height = 8,
    rasterize_dpi = 150
  ),
  by_tumor_type = plot_umap(
    data = middle_slice_2D_sc_umap_results,
    color_by = "Metadata_Biology_TumorType",
    palette = tumor_type_palette,
    title = "All patients: 2D Middle Slice Single Cell FS Profiles",
    legend_title = "Tumor type",
    alpha = 0.08,
    point_size = 0.5,
    rasterize_dpi = 150
  ),
  facet_by_tumor_type = plot_umap(
    data = middle_slice_2D_sc_umap_results,
    color_by = "Metadata_Biology_TumorType",
    palette = tumor_type_palette,
    title = "All patients: 2D Middle Slice Single Cell FS Profiles",
    facet_by = "Metadata_Biology_TumorType",
    facet_nrow = 2,
    alpha = 0.15,
    point_size = 0.15,
    rasterize_dpi = 150
  ),
  dmso_only = plot_umap(
    data = sc_umap_results_dmsos,
    color_by = "Metadata_Biology_PatientTumorLabel",
    palette = patient_color_palette_labeled,
    title = "All patients: 2D Middle Slice Single Cell FS Profiles (DMSO Controls)",
    legend_title = "Patient",
    alpha = 0.15,
    rasterize_dpi = 150
  ),
  dmso_facet_by_patient = plot_umap(
    data = sc_umap_results_dmsos,
    color_by = "Metadata_Biology_PatientTumor",
    palette = patient_color_palette,
    title = "All patients: 2D Middle Slice Single Cell FS Profiles (DMSO Controls)",
    facet_by = "Metadata_Biology_PatientTumor",
    facet_nrow = 3,
    alpha = 0.3,
    point_size = 0.3,
    rasterize_dpi = 150
  )
)
save_umap_pca_plots_pdf(pdf_plots_2D_midslice, file.path(figures_path, "2D_midslice_scfs_all_patients.pdf"))

individual_patients <- readLines(file.path(root_dir, "data/patient_IDs.txt"))
individual_patients <- individual_patients[individual_patients != ""]

# NF0037_T1 has no 2D per-patient data (the 2D pipeline never produced output
# for this plate); skip only its 2D plots, not 3D (which is fully populated).
individual_patients_2D <- individual_patients[individual_patients != "NF0037_T1"]

# Consolidated per-patient files: one file per projection/variant combo,
# with all patients' independently-fit UMAP results concatenated together
# and patient identity kept as a metadata column
# (Metadata_Biology_PatientTumor, stamped explicitly at generation time).
# Harmonize the 2D treatment column name to the 3D metadata convention, as
# done for the pooled results above.
patient_specific_2D_maxproj_scfs <- arrow::read_parquet(
  file.path(umap_results_dir, "patient_specific", "patient_specific_2D_maxproj_scfs_umap.parquet")
) %>%
  rename(Metadata_Experiment_Treatment = Metadata_treatment)

patient_specific_2D_midslice_scfs <- arrow::read_parquet(
  file.path(umap_results_dir, "patient_specific", "patient_specific_2D_midslice_scfs_umap.parquet")
) %>%
  rename(Metadata_Experiment_Treatment = Metadata_treatment)

patient_specific_3D_scfs <- arrow::read_parquet(
  file.path(umap_results_dir, "patient_specific", "patient_specific_3D_scfs_umap.parquet")
)

plots_2D_maxproj <- list()
plots_2D_midslice <- list()
plots_3D <- list()

for (patient in individual_patients) {
  if (patient %in% individual_patients_2D) {
    # UMAP1/UMAP2 are only comparable within a patient (each patient has its
    # own independent UMAP fit), so filter to one patient's rows before
    # plotting rather than plotting the consolidated file directly.
    # fixed_coord = FALSE: each patient's independent UMAP fit has its own
    # arbitrary data aspect ratio, unrelated to the page's fixed width/height,
    # so enforcing a 1:1 UMAP1:UMAP2 scale here left some pages fully filling
    # the page and others letterboxed with margins depending on how close
    # that patient's own aspect ratio happened to match the page's.
    plots_2D_maxproj[[patient]] <- plot_umap(
      data = patient_specific_2D_maxproj_scfs %>% filter(Metadata_Biology_PatientTumor == patient),
      color_by = "Metadata_Experiment_Treatment",
      palette = custom_treatment_palette,
      title = paste0(patient, " - Single-cells MIP FS Profiles"),
      legend_title = "Treatment",
      alpha = 0.15,
      rasterize_dpi = 150,
      fixed_coord = FALSE
    )

    plots_2D_midslice[[patient]] <- plot_umap(
      data = patient_specific_2D_midslice_scfs %>% filter(Metadata_Biology_PatientTumor == patient),
      color_by = "Metadata_Experiment_Treatment",
      palette = custom_treatment_palette,
      title = paste0(patient, " - Single-cells Middle Slice FS Profiles"),
      legend_title = "Treatment",
      alpha = 0.15,
      rasterize_dpi = 150,
      fixed_coord = FALSE
    )
  }

  # 3D per-patient data is fully populated for every patient (including NF0037_T1)
  plots_3D[[patient]] <- plot_umap(
    data = patient_specific_3D_scfs %>% filter(Metadata_Biology_PatientTumor == patient),
    color_by = "Metadata_Experiment_Treatment",
    palette = custom_treatment_palette,
    title = paste0(patient, " - Single-cells 3D FS Profiles"),
    legend_title = "Treatment",
    alpha = 0.15,
    rasterize_dpi = 150,
    fixed_coord = FALSE
  )
}

# One combined, multi-page PDF per projection (one page per patient) instead
# of individual PNGs.
save_umap_pca_plots_pdf(plots_2D_maxproj, file.path(figures_patient_specific_path, "patient_specific_2D_maxproj_scfs_by_treatment.pdf"))
save_umap_pca_plots_pdf(plots_2D_midslice, file.path(figures_patient_specific_path, "patient_specific_2D_midslice_scfs_by_treatment.pdf"))
save_umap_pca_plots_pdf(plots_3D, file.path(figures_patient_specific_path, "patient_specific_3D_scfs_by_treatment.pdf"))
