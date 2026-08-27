packages <- c("ggplot2", "dplyr", "RColorBrewer", "patchwork")
for (pkg in packages) {
  suppressPackageStartupMessages(
    suppressWarnings(
      library(pkg, character.only = TRUE)
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
    while (dirname(current_path) != current_path) {  # While not at root
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

source(file.path(root_dir, "utils/r_plot_themes.r"))
source(file.path(root_dir, "utils/r_plot_funcs.r"))

cell_counts_path_dir <- file.path(root_dir, "1.EDA/results/cell_counts/cell_counts.parquet")
cell_counts_figure_path <- file.path(root_dir, "1.EDA/figures/cell_counts")
if (!dir.exists(cell_counts_figure_path)) {
    dir.create(cell_counts_figure_path, recursive = TRUE)
}


# read in all the cell counts files and combine them into a single data frame
cell_counts_df <- arrow::read_parquet(cell_counts_path_dir)
dim(cell_counts_df)
# drop the CQ1 data
cell_counts_df <- cell_counts_df %>%
  dplyr::filter(Metadata_Biology_PatientTumor != "NF0037_T1_CQ1")


add_feature_type <- function(df, x) {
    #' Add a column to the input data frame that classifies each profile by feature type.
    #'
    #' Adds a `Metadata_feature_type` column classifying each row (based on
    #' keywords found in the `x` column, typically `Metadata_profile_name`)
    #' into one of: nucleocentric/single-cell/organoid deep-learning
    #' embeddings (SAMMed3D or MorphEM), handcrafted 3D features
    #' (ZedProfiler), or 2D CellProfiler features (per slice strategy).
    #' Rows matching none of these fall into "other".
    col_vals <- df[[x]]
    df %>%
        mutate(
            Metadata_feature_type = dplyr::case_when(
                # nucleocentric and sammed
                grepl("sammed", col_vals, ignore.case = TRUE) 
                & grepl("nucleocentric", col_vals, ignore.case = TRUE) 
                ~ "Nucleocentric\nDL (SAMMed3D)",

                grepl("morphem", col_vals, ignore.case = TRUE) 
                & grepl("nucleocentric", col_vals, ignore.case = TRUE) 
                ~ "Nucleocentric\nDL (MorphEM)",

                grepl("sammed", col_vals, ignore.case = TRUE) 
                & !grepl("nucleocentric", col_vals, ignore.case = TRUE) 
                & grepl("sc", col_vals, ignore.case = TRUE)
                ~ "Single cell\nDL (SAMMed3D)",

                grepl("sammed", col_vals, ignore.case = TRUE) 
                & !grepl("nucleocentric", col_vals, ignore.case = TRUE) 
                & grepl("organoid", col_vals, ignore.case = TRUE)
                ~ "Organoid\nDL (SAMMed3D)",

                !grepl("sammed", col_vals, ignore.case = TRUE) 
                & !grepl("nucleocentric", col_vals, ignore.case = TRUE) 
                & grepl("3D", col_vals, ignore.case = TRUE) 
                & grepl("sc", col_vals, ignore.case = TRUE)
                ~ "Single cell\nhandcrafted (ZedProfiler)",

                !grepl("sammed", col_vals, ignore.case = TRUE) 
                & !grepl("nucleocentric", col_vals, ignore.case = TRUE) 
                & grepl("3D", col_vals, ignore.case = TRUE) 
                & grepl("organoid", col_vals, ignore.case = TRUE)
                ~ "Organoid\nhandcrafted (ZedProfiler)",

            

                grepl("2D", col_vals, ignore.case = TRUE) 
                & grepl("max", col_vals, ignore.case = TRUE) 
                & grepl("sc", col_vals, ignore.case = TRUE)
                ~ "Single cell\n2D max projection\n(CellProfiler)",
                
                grepl("2D", col_vals, ignore.case = TRUE) 
                & grepl("max", col_vals, ignore.case = TRUE) 
                & grepl("Organoid", col_vals, ignore.case = TRUE)
                ~ "Organoid\n2D max projection\n(CellProfiler)",

                grepl("2D", col_vals, ignore.case = TRUE) 
                & grepl("middle_slice", col_vals, ignore.case = TRUE) 
                & grepl("sc", col_vals, ignore.case = TRUE)
                ~ "Single cell\n2D middle slice\n(CellProfiler)",
                
                grepl("2D", col_vals, ignore.case = TRUE) 
                & grepl("middle_slice", col_vals, ignore.case = TRUE) 
                & grepl("Organoid", col_vals, ignore.case = TRUE)
                ~ "Organoid\n2D middle slice\n(CellProfiler)",

                grepl("2D", col_vals, ignore.case = TRUE) 
                & grepl("middle_n", col_vals, ignore.case = TRUE) 
                & grepl("sc", col_vals, ignore.case = TRUE)
                ~ "Single cell\n2D middle 3 slices\n(CellProfiler)",

                grepl("2D", col_vals, ignore.case = TRUE) 
                & grepl("middle_n", col_vals, ignore.case = TRUE) 
                & grepl("Organoid", col_vals, ignore.case = TRUE)
                ~ "Organoid\n2D middle 3 slices\n(CellProfiler)",
                TRUE ~ "other"
            )
        )
}


cell_counts_df <- add_feature_type(cell_counts_df, "Metadata_profile_name")
cell_counts_df$sc_or_organoid <- "other"
cell_counts_df$sc_or_organoid <- dplyr::case_when(
    grepl("organoid", cell_counts_df$Metadata_profile_name, ignore.case = TRUE) ~ "Organoid",
    grepl("(^|_)sc(_|$)", cell_counts_df$Metadata_profile_name, ignore.case = TRUE) ~ "Single cell",
    grepl("(^|_)nucleocentric(_|$)", cell_counts_df$Metadata_profile_name, ignore.case = TRUE) ~ "Nucleocentric",
    TRUE ~ "other"
)

# tumor_type_lookup/tumor_type_palette live in utils/r_plot_themes.r. Set as
# a factor with tumor_type_palette's level order (cNF, pNF, MPNST, Other) so
# every downstream plot that uses this column for an axis or a fill/legend
# sorts consistently without needing its own explicit factor() call.
cell_counts_df$Metadata_Biology_TumorType <- factor(
    tumor_type_lookup[cell_counts_df$Metadata_Biology_PatientTumor],
    levels = names(tumor_type_palette)
)

# get only the ZedProfiler (handcrafted 3D) profiles, split into single-cell and organoid subsets
zedprofiler_only_df <- cell_counts_df %>%
    filter(Metadata_feature_type %in% c("Single cell\nhandcrafted (ZedProfiler)", "Organoid\nhandcrafted (ZedProfiler)"))
sc_counts_df <- zedprofiler_only_df %>% filter(sc_or_organoid == "Single cell")
organoid_counts_df <- zedprofiler_only_df %>% filter(sc_or_organoid == "Organoid")

# Single-cell y-axis label, shared by every plot in this section.
sc_y_lab <- "Cell counts per well FOV"

# build_tumor_type_count_plots() (utils/r_plot_funcs.r) builds the standard
# 6-plot tumor-type-centric summary: distribution overlaid by tumor type
# (+ pooled "All"), a tumor-type boxplot with a pooled "Total" box
# (outliers shown so a minority subgroup's, e.g. MPNST's, pull on the
# pooled box stays visible), by patient (colored by tumor type), by
# treatment split into one horizontal plot per dose (axes swapped so
# tumor-type fill colors are legible), and by treatment faceted by patient
# (axes swapped so the 22 treatment labels read horizontally, legend moved
# below). Shared with the organoid biology section below.
sc_plots <- build_tumor_type_count_plots(
    sc_counts_df,
    y_col = "Metadata_n_cells_norm_by_well_fov",
    y_lab = sc_y_lab
)

# Each page gets its own size (save_plots_pdf combines them with pdfunite)
# instead of a single shared page size, so a simple single-panel plot
# isn't stretched to match the much busier facet plots elsewhere in the
# file: distribution/tumor-type-box/by-patient are simple single-panel
# charts; by_treatment_dose1 needs real height for its 22 horizontal rows,
# while by_treatment_dose10 only has ~5 treatments at dose 10 and would
# look sparse at the same height; by_treatment_patient is a 12-panel facet.
sc_widths <- c(10, 8, 10, 10, 10, 13)
sc_heights <- c(5, 6, 6, 10, 4, 8)
save_plots_pdf(
    sc_plots,
    output_path = file.path(cell_counts_figure_path, "single_cell_biology.pdf"),
    width = sc_widths,
    height = sc_heights
)

organoid_plots <- build_tumor_type_count_plots(
    organoid_counts_df,
    y_col = "Metadata_n_cells_norm_by_well_fov",
    y_lab = "Normalized organoid counts per well FOV"
)

organoid_cell_count_file_path <- file.path(root_dir, "1.EDA/results/cell_counts/organoid_cell_counts.parquet")
organoid_cell_counts <- arrow::read_parquet(organoid_cell_count_file_path)
organoid_cell_counts <- organoid_cell_counts %>% dplyr::filter(Metadata_Biology_PatientTumor != "NF0037_T1_CQ1")

# tumor_type_lookup/tumor_type_palette live in utils/r_plot_themes.r; factor
# level order (cNF, pNF, MPNST, Other) keeps every plot below consistently
# ordered.
organoid_cell_counts$Metadata_Biology_TumorType <- factor(
    tumor_type_lookup[organoid_cell_counts$Metadata_Biology_PatientTumor],
    levels = names(tumor_type_palette)
)

head(organoid_cell_counts)

# Mean cells per organoid ("organoid cellularity"/size -- how many cells
# make up each organoid on average -- distinct from the organoid *counts*
# per FOV plotted in the summary above), broken down by patient and
# colored by tumor type.
mean_cells_per_organoid_by_patient_plot <- plot_boxplot(
    organoid_cell_counts,
    x_col = "Metadata_Biology_PatientTumor",
    y_col = "mean_cells_per_organoid",
    fill_col = "Metadata_Biology_TumorType",
    x_lab = "Patient tumor",
    y_lab = "Mean cells per organoid",
    fill_lab = "Tumor type",
    custom_palette = tumor_type_palette,
    x_text = "angled",
    base_size = 14
)

# Distribution of cells per organoid, overlaid by tumor type (+ pooled
# "All"), with each group's overall MEAN cells-per-organoid annotated
# top-right (mean, not sum, since this column is already a per-organoid
# average).
mean_cells_per_organoid_density_plot <- plot_group_density_with_stats(
    organoid_cell_counts,
    x_col = "mean_cells_per_organoid", x_lab = "Mean cells per organoid",
    stat_col = "mean_cells_per_organoid",
    stat_fn = function(x) round(mean(x, na.rm = TRUE), 1),
    stat_label = "Mean", stat_fmt = function(v) format(v, nsmall = 1),
    base_size = 14
)

# reshape to long format so cells with and without a parent organoid can be
# plotted side by side
organoid_cell_counts_long <- dplyr::bind_rows(
    organoid_cell_counts %>%
        mutate(
            Parent_Organoid_Status = "Inside organoid",
            n_cells_normalized_per_FOV = n_cells_with_parent_organoid_normalized_per_FOV
        ),
    organoid_cell_counts %>%
        mutate(
            Parent_Organoid_Status = "Outside organoid",
            n_cells_normalized_per_FOV = n_cells_without_parent_organoid_normalized_per_FOV
        )
)
parent_organoid_status_palette <- c(
    "Inside organoid" = "#1FAD23",
    "Outside organoid" = "#CC29CC"
)

# Distribution of cells with vs. without a parent organoid, faceted by
# tumor type (one row each, cNF/pNF/MPNST/Other) with an added "Total" row
# pooling all tumor types together.
parentage_facet_order <- c(names(tumor_type_palette), "Total")
cells_with_without_parent_organoid_density_df <- dplyr::bind_rows(
    organoid_cell_counts_long %>% mutate(FacetTumorType = "Total"),
    organoid_cell_counts_long %>% mutate(FacetTumorType = Metadata_Biology_TumorType)
)
cells_with_without_parent_organoid_density_df$FacetTumorType <- factor(
    cells_with_without_parent_organoid_density_df$FacetTumorType, levels = parentage_facet_order
)
cells_with_without_parent_organoid_density_plot <- (
    ggplot(cells_with_without_parent_organoid_density_df, aes(x = n_cells_normalized_per_FOV, fill = Parent_Organoid_Status))
    + geom_density(color = "black", alpha = 0.6)
    + scale_fill_manual(values = parent_organoid_status_palette)
    + facet_wrap(~ FacetTumorType, ncol = 1, scales = "free_y")
    + labs(x = "Cells per FOV (normalized)", y = "Density", fill = "Organoid parentage")
    + theme_manuscript(base_size = 14)
)

# Enrichment of patients for cells without (vs. with) a parent organoid
cells_without_parent_organoid_by_patient_plot <- plot_boxplot(
    organoid_cell_counts_long,
    x_col = "Metadata_Biology_PatientTumor",
    y_col = "n_cells_normalized_per_FOV",
    fill_col = "Parent_Organoid_Status",
    x_lab = "Patient tumor",
    y_lab = "Cells per FOV (normalized)",
    fill_lab = "Organoid parentage",
    custom_palette = parent_organoid_status_palette,
    x_text = "angled",
    base_size = 14
)

# Enrichment of treatments for cells without (vs. with) a parent organoid
cells_without_parent_organoid_by_treatment_plot <- plot_boxplot(
    organoid_cell_counts_long,
    x_col = "Metadata_Experiment_Treatment",
    y_col = "n_cells_normalized_per_FOV",
    fill_col = "Parent_Organoid_Status",
    x_lab = "Treatment",
    y_lab = "Cells per FOV (normalized)",
    fill_lab = "Organoid parentage",
    custom_palette = parent_organoid_status_palette,
    x_text = "angled",
    base_size = 14
)
cells_without_parent_organoid_by_treatment_plot <- cells_without_parent_organoid_by_treatment_plot + ylim(0, 30)

# Enrichment of treatments for cells without (vs. with) a parent organoid,
# faceted by patient. Axes swapped (treatment on the vertical axis) so the
# 22 treatment labels read horizontally instead of overlapping in a narrow
# angled x-axis per panel, matching the single-cell by-treatment-by-patient
# fix above. Legend moved below to save horizontal space.
cells_without_parent_organoid_by_treatment_patient_plot <- plot_boxplot_horizontal(
    organoid_cell_counts_long,
    x_col = "Metadata_Experiment_Treatment",
    y_col = "n_cells_normalized_per_FOV",
    fill_col = "Parent_Organoid_Status",
    fill_palette = parent_organoid_status_palette,
    x_lab = "Treatment",
    y_lab = "Cells per FOV (normalized)",
    fill_lab = "Organoid parentage",
    facet_formula = as.formula("~ Metadata_Biology_PatientTumor"), facet_ncol = 4,
    ylim_max = 30,
    base_size = 8
) + theme(legend.position = "bottom")

# Proportion of cells with a parent organoid, by treatment. Axes swapped
# (treatment on the vertical axis) and split by tumor type (dodged boxes
# per treatment row) instead of a single pooled box.
cells_without_parent_organoid_by_treatment_proportion_plot <- plot_boxplot_horizontal(
    organoid_cell_counts,
    x_col = "Metadata_Experiment_Treatment",
    y_col = "proportion_of_cells_with_parent_organoid_compared_to_n_cells_without_parent_organoid",
    fill_col = "Metadata_Biology_TumorType",
    fill_palette = tumor_type_palette,
    x_lab = "Treatment",
    y_lab = "Proportion of cells with parent organoid",
    fill_lab = "Tumor type",
    base_size = 14
)

organoid_qc_plots <- list(
    mean_cells_per_organoid_by_patient = mean_cells_per_organoid_by_patient_plot,
    mean_cells_per_organoid_density = mean_cells_per_organoid_density_plot,
    cells_with_without_parent_organoid_density = cells_with_without_parent_organoid_density_plot,
    cells_without_parent_organoid_by_patient = cells_without_parent_organoid_by_patient_plot,
    cells_without_parent_organoid_by_treatment = cells_without_parent_organoid_by_treatment_plot,
    cells_without_parent_organoid_by_treatment_patient = cells_without_parent_organoid_by_treatment_patient_plot,
    cells_without_parent_organoid_by_treatment_proportion = cells_without_parent_organoid_by_treatment_proportion_plot
)

# Each page gets its own size (see the matching comment in the single-cell
# save above): organoid_plots reuses the same sizes as sc_plots (identical
# structure); the QC plots are single-panel except the 5-row tumor-type
# parentage facet and the by-treatment proportion plot (dodged by up to 4
# tumor types per treatment row), both of which need real height.
organoid_widths <- c(10, 8, 10, 10, 10, 13, 10, 10, 10, 10, 14, 13, 12)
organoid_heights <- c(5, 6, 6, 10, 4, 8, 6, 5, 12, 6, 6, 8, 12)
save_plots_pdf(
    c(organoid_plots, organoid_qc_plots),
    output_path = file.path(cell_counts_figure_path, "organoid_biology.pdf"),
    width = organoid_widths,
    height = organoid_heights
)

# plot the cell counts by 2D and 3D profiles where the y axis is the 3D counts and the x axis is the 2D counts,
# and each point is a patient/treatment/dose combination, colored by tumor type
extract_single_cell_counts <- function(df, name_patterns, output_col, feature_type_pattern = NULL) {
    #' Filter cell_counts_df down to single-cell rows for one profile
    #' type/slice (every pattern in name_patterns must match somewhere in
    #' Metadata_profile_name; feature_type_pattern, if given, is an
    #' additional required match against Metadata_feature_type -- e.g. 3D
    #' has multiple feature-extraction variants sharing "3D" in their name,
    #' so it also needs "zed" to isolate the handcrafted ZedProfiler ones),
    #' then reduces to one count column (output_col) for merging across
    #' profile types.
    #'
    #' Note: "!= 'Organoid'" below is deliberately case-sensitive to match
    #' sc_or_organoid's actual values (see add_feature_type() above).
    for (pattern in name_patterns) {
        df <- df %>% filter(grepl(pattern, Metadata_profile_name, ignore.case = TRUE))
    }
    df <- df %>% filter(sc_or_organoid != "Organoid")
    if (!is.null(feature_type_pattern)) {
        df <- df %>% filter(grepl(feature_type_pattern, Metadata_feature_type, ignore.case = TRUE))
    }
    df %>%
        select(Metadata_Biology_PatientTumor, Metadata_Experiment_Treatment, Metadata_Experiment_Dose, Metadata_n_cells_norm_by_well_fov) %>%
        rename(!!output_col := Metadata_n_cells_norm_by_well_fov) %>%
        distinct()
}

ZP_profiles_3D <- extract_single_cell_counts(cell_counts_df, "3D", "n_cells_3D", feature_type_pattern = "zed")
CP_profiles_2D_max <- extract_single_cell_counts(cell_counts_df, c("2D", "max"), "n_cells_2D_max")
CP_profiles_2D_middle <- extract_single_cell_counts(cell_counts_df, c("2D", "middle_slice"), "n_cells_2D_middle_slice")
CP_profiles_2D_middle_n <- extract_single_cell_counts(cell_counts_df, c("2D", "middle_n"), "n_cells_2D_middle_n")

# merge the four data frames by patient tumor, treatment, and dose
merged_df <- ZP_profiles_3D %>%
    inner_join(CP_profiles_2D_max, by = c("Metadata_Biology_PatientTumor", "Metadata_Experiment_Treatment", "Metadata_Experiment_Dose")) %>%
    inner_join(CP_profiles_2D_middle, by = c("Metadata_Biology_PatientTumor", "Metadata_Experiment_Treatment", "Metadata_Experiment_Dose")) %>%
    inner_join(CP_profiles_2D_middle_n, by = c("Metadata_Biology_PatientTumor", "Metadata_Experiment_Treatment", "Metadata_Experiment_Dose")) %>%
    select(Metadata_Biology_PatientTumor, Metadata_Experiment_Treatment, Metadata_Experiment_Dose, n_cells_3D, n_cells_2D_max, n_cells_2D_middle_slice, n_cells_2D_middle_n)
merged_df <- distinct(merged_df)
# drop the values of the 2D counts above 100
merged_df <- merged_df %>%
    filter(n_cells_2D_max <= 100) %>%
    filter(n_cells_2D_middle_slice <= 100) %>%
    filter(n_cells_2D_middle_n <= 100)

# tumor_type_lookup/tumor_type_palette live in utils/r_plot_themes.r
merged_df$Metadata_Biology_TumorType <- factor(
    tumor_type_lookup[merged_df$Metadata_Biology_PatientTumor],
    levels = names(tumor_type_palette)
)

# One unfaceted overview plot per 2D slice strategy, then the same three
# comparisons faceted by treatment (22 panels) so per-treatment agreement
# between 2D and 3D counts is visible directly.
scatter_specs <- list(
    list(
        x_col = "n_cells_2D_middle_slice",
        x_lab = "Normalized cell counts per well FOV (2D middle slice)",
        facet = FALSE
    ),
    list(
        x_col = "n_cells_2D_middle_n",
        x_lab = "Normalized cell counts per well FOV (2D middle 3 slices)",
        facet = FALSE
    ),
    list(
        x_col = "n_cells_2D_max",
        x_lab = "Normalized cell counts per well FOV (2D max projection)",
        facet = FALSE
    ),
    list(
        x_col = "n_cells_2D_middle_slice",
        x_lab = "Normalized cell counts per well FOV (2D middle slice)",
        facet = TRUE
    ),
    list(
        x_col = "n_cells_2D_middle_n",
        x_lab = "Normalized cell counts per well FOV (2D middle 3 slices)",
        facet = TRUE
    ),
    list(
        x_col = "n_cells_2D_max",
        x_lab = "Normalized cell counts per well FOV (2D max projection)",
        facet = TRUE
    )
)

# diagonal_shading (on by default in plot_2d_vs_3d_scatter()) fills each
# panel with a white-to-red gradient by perpendicular distance to y = x,
# computed per treatment for the faceted specs (facet_col) so it stays
# meaningful even in a free-scaled panel whose visible range happens not
# to include any point where 2D == 3D -- the case where the dashed
# reference line itself would otherwise just not be drawn at all.
technical_plots <- lapply(scatter_specs, function(spec) {
    plot_2d_vs_3d_scatter(
        merged_df,
        x_col = spec$x_col,
        y_col = "n_cells_3D",
        color_col = "Metadata_Biology_TumorType",
        shape_col = "Metadata_Experiment_Dose",
        color_palette = tumor_type_palette,
        x_lab = spec$x_lab,
        y_lab = "Normalized cell counts per well FOV (3D ZEDProfiler)",
        color_lab = "Tumor type",
        facet_formula = if (spec$facet) as.formula(. ~ Metadata_Experiment_Treatment) else NULL,
        facet_ncol = 5,
        facet_col = if (spec$facet) "Metadata_Experiment_Treatment" else NULL,
        shape_lab = "Dose (uM)",
        base_size = 14
    )
})

# Each page gets its own size (see the matching comment in the single-cell
# save above). The three unfaceted overview scatters are sized close to a
# single panel's scale within the 22-panel treatment-faceted grid below --
# same base_size/point size as those facets, just a small canvas around
# them, so they read the same way instead of looking blown up with excess
# white space. The three faceted plots keep the larger page they need for
# 5 columns x 5 rows of panels plus their own axis titles/legend.
tech_widths <- c(6, 6, 6, 16, 16, 16)
tech_heights <- c(5.5, 5.5, 5.5, 12, 12, 12)
save_plots_pdf(
    technical_plots,
    output_path = file.path(cell_counts_figure_path, "technical_considerations.pdf"),
    width = tech_widths,
    height = tech_heights
)
