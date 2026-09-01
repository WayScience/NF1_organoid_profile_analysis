packages <- c("ggplot2", "dplyr", "RColorBrewer", "ggpattern", "patchwork")
for (pkg in packages) {
  suppressPackageStartupMessages(
    suppressWarnings(
      library(pkg, character.only = TRUE)
    )
  )
}

figures_path <- file.path("../figures/NF0014/")
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

    # If no Git root found, stop with error
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
    #' This function takes a data frame and adds a new column called `Metadata_feature_type`
    #' that classifies each profile based on the values in the `Metadata_profile_type` column. The classification is done using a series of conditions that check for specific keywords in the `Metadata_profile
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

width <- 12
height <- 14
options(repr.plot.width = width, repr.plot.height = height)

# compute the max density across all facet levels so y-axis can be shared
max_density <- cell_counts_df %>%
    filter(!is.na(Metadata_n_cells_norm_by_well_fov)) %>%
    group_by(sc_or_organoid, Metadata_feature_type) %>%
    summarise(
        dens = list(density(Metadata_n_cells_norm_by_well_fov, na.rm = TRUE)$y),
        .groups = "drop"
    ) %>%
    pull(dens) %>%
    unlist() %>%
    max()

facet_levels <- unique(cell_counts_df$sc_or_organoid)

plot_list <- lapply(
    facet_levels,
    function(fv) plot_density_by_facet(
        cell_counts_df,
        facet_col = "sc_or_organoid",
        facet_val = fv,
        x_col = "Metadata_n_cells_norm_by_well_fov",
        fill_col = "Metadata_feature_type",
        y_max = max_density,
        x_max = 50,
        palette = "PuOr"
    )
)

cell_count_density_plot <- wrap_plots(plot_list, ncol = 1) +
    plot_layout(axis_titles = "collect")
save_ggplot(
    cell_count_density_plot,
    path = file.path(cell_counts_figure_path, "cell_count_density_plot.png"),
    width = width,
    height = height
)

width <- 12
height <- 8
options(repr.plot.width = width, repr.plot.height = height)
# summary_boxplot per feature type and sc_or_organoid
summary_plot <- plot_boxplot(
    cell_counts_df,
    x_col = "Metadata_feature_type",
    y_col = "Metadata_n_cells_norm_by_well_fov",
    fill_col = "Metadata_feature_type",
    x_lab = "Profile type",
    y_lab = "Normalized cell counts per well FOV",
    fill_lab = "Profile type",
    ylim_max = 35,
    x_text = "blank",
    custom_palette=RColorBrewer::brewer.pal(n = 12, name = "Paired"),
    guides_ncol = 2
)
save_ggplot(
    summary_plot,
    path = file.path(cell_counts_figure_path, "cell_count_summary_boxplot.png"),
    width = width,
    height = height
)


# get only thre ZedProfiler profiles
zedprofiler_only_df <- cell_counts_df %>%
    filter(Metadata_feature_type %in% c("Single cell\nhandcrafted (ZedProfiler)", "Organoid\nhandcrafted (ZedProfiler)"))

cell_count_per_patient_plot <- plot_boxplot(
    zedprofiler_only_df,
    x_col = "Metadata_Biology_PatientTumor",
    y_col = "Metadata_n_cells_norm_by_well_fov",
    fill_col = "Metadata_Biology_PatientTumor",
    x_lab = "Patient tumor",
    y_lab = "Normalized object counts per well FOV",
    fill_lab = "Patient tumor sample",
    ylim_max = 50,
    x_text = "blank",
    facet_formula = as.formula(sc_or_organoid ~ .)
)
save_ggplot(
    cell_count_per_patient_plot,
    path = file.path(cell_counts_figure_path, "cell_count_per_patient_boxplot.png"),
    width = width,
    height = height
)

no_nucleocentric_organoid_df <- zedprofiler_only_df %>%
    filter(sc_or_organoid == "Organoid")
no_nucleocentric_sc_df <- zedprofiler_only_df %>%
    filter(sc_or_organoid == "Single cell")

width <- 24
height <- 18
options(repr.plot.width = width, repr.plot.height = height)
organoid_count_by_treatment_plot <- plot_bar_horizontal(
    no_nucleocentric_organoid_df,
    x_col = "Metadata_Experiment_Treatment",
    y_col = "Metadata_n_cells_norm_by_well_fov",
    fill_col = "Metadata_Experiment_Dose",
    fill_palette = dose_palette,
    x_lab = "Patient treatment",
    y_lab = "Normalized organoid counts per well FOV",
    fill_lab = "Dose (uM)",
    facet_formula = as.formula(. ~ Metadata_Biology_PatientTumor)
)
organoid_count_by_treatment_plot <- (
    organoid_count_by_treatment_plot
    + theme(legend.position = "bottom")
)
save_ggplot(
    organoid_count_by_treatment_plot,
    path = file.path(cell_counts_figure_path, "organoid_count_by_treatment_barplot.png"),
    width = width,
    height = height
)

width <- 24
height <- 18
options(repr.plot.width = width, repr.plot.height = height)
cell_count_by_treatment_plot <- plot_bar_horizontal(
    no_nucleocentric_sc_df,
    x_col = "Metadata_Experiment_Treatment",
    y_col = "Metadata_n_cells_norm_by_well_fov",
    fill_col = "Metadata_Experiment_Dose",
    fill_palette = dose_palette,
    x_lab = "Patient treatment",
    y_lab = "Normalized cell counts per well FOV",
    fill_lab = "Dose (uM)",
    facet_formula = as.formula(. ~ Metadata_Biology_PatientTumor)
)
cell_count_by_treatment_plot <- cell_count_by_treatment_plot + theme(legend.position = "bottom")
save_ggplot(
    cell_count_by_treatment_plot,
    path = file.path(cell_counts_figure_path, "cell_count_by_treatment_barplot.png"),
    width = width,
    height = height
)

width <- 12
height <- 12
options(repr.plot.width = width, repr.plot.height = height)
cell_count_by_treatment_plot <- plot_boxplot_horizontal(
    no_nucleocentric_sc_df,
    x_col = "Metadata_Experiment_Treatment",
    y_col = "Metadata_n_cells_norm_by_well_fov",
    fill_col = "Metadata_Experiment_Dose",
    fill_palette = dose_palette,
    x_lab = "Patient treatment",
    y_lab = "Normalized cell counts per well FOV",
    fill_lab = "Dose (uM)",
    ylim_max = 40
    )
cell_count_by_treatment_plot <- cell_count_by_treatment_plot + theme(legend.position = "bottom")
save_ggplot(
    cell_count_by_treatment_plot,
    path = file.path(cell_counts_figure_path, "cell_count_by_treatment_boxplot.png"),
    width = width,
    height = height
)

width <- 12
height <- 12
options(repr.plot.width = width, repr.plot.height = height)
organoid_count_by_treatment_plot <- plot_boxplot_horizontal(
    no_nucleocentric_organoid_df,
    x_col = "Metadata_Experiment_Treatment",
    y_col = "Metadata_n_cells_norm_by_well_fov",
    fill_col = "Metadata_Experiment_Dose",
    fill_palette = dose_palette,
    x_lab = "Patient treatment",
    y_lab = "Normalized organoid counts per well FOV",
    fill_lab = "Dose (uM)",
    ylim_max = 20,
)
organoid_count_by_treatment_plot <- organoid_count_by_treatment_plot + theme(legend.position = "bottom")
save_ggplot(
    organoid_count_by_treatment_plot,
    path = file.path(cell_counts_figure_path, "organoid_count_by_treatment_boxplot.png"),
    width = width,
    height = height
)

# plot the cell counts by 2D and 3D profiles where the y axis is the 3D counts and the x axis is the 2D counts,
# and each point is a patient tumor, colored by treatment
ZP_profiles_3D <- cell_counts_df %>%
    filter(grepl("3D", Metadata_profile_name, ignore.case = TRUE)) %>%
    filter(grepl("zed", Metadata_feature_type, ignore.case = TRUE)) %>%
    filter(sc_or_organoid != "organoid") %>%
    select(Metadata_Biology_PatientTumor, Metadata_Experiment_Treatment,Metadata_Experiment_Dose, Metadata_n_cells_norm_by_well_fov,Metadata_feature_type) %>%
    rename(n_cells_3D = Metadata_n_cells_norm_by_well_fov)
CP_profiles_2D_max <- cell_counts_df %>%
    filter(grepl("2D", Metadata_profile_name, ignore.case = TRUE)) %>%
    filter(grepl("max", Metadata_profile_name, ignore.case = TRUE)) %>%
    filter(sc_or_organoid != "organoid") %>%
    select(Metadata_Biology_PatientTumor, Metadata_Experiment_Treatment,Metadata_Experiment_Dose, Metadata_n_cells_norm_by_well_fov,Metadata_feature_type) %>%
    rename(n_cells_2D_max = Metadata_n_cells_norm_by_well_fov)
CP_profiles_2D_max <- distinct(CP_profiles_2D_max)
CP_profiles_2D_middle <- cell_counts_df %>%
    filter(grepl("2D", Metadata_profile_name, ignore.case = TRUE)) %>%
    filter(grepl("middle_slice", Metadata_profile_name, ignore.case = TRUE)) %>%
    filter(sc_or_organoid != "organoid") %>%
    select(Metadata_Biology_PatientTumor, Metadata_Experiment_Treatment,Metadata_Experiment_Dose, Metadata_n_cells_norm_by_well_fov,Metadata_feature_type) %>%
    rename(n_cells_2D_middle_slice = Metadata_n_cells_norm_by_well_fov)
CP_profiles_2D_middle <- distinct(CP_profiles_2D_middle)
CP_profiles_2D_middle_n <- cell_counts_df %>%
    filter(grepl("2D", Metadata_profile_name, ignore.case = TRUE)) %>%
    filter(grepl("middle_n", Metadata_profile_name, ignore.case = TRUE)) %>%
    filter(sc_or_organoid != "organoid") %>%
    select(Metadata_Biology_PatientTumor, Metadata_Experiment_Treatment,Metadata_Experiment_Dose, Metadata_n_cells_norm_by_well_fov,Metadata_feature_type) %>%
    rename(n_cells_2D_middle_n = Metadata_n_cells_norm_by_well_fov)
CP_profiles_2D_middle_n <- distinct(CP_profiles_2D_middle_n)

# merge the four data frames by patient tumor, treatment, and dose
merged_df <- ZP_profiles_3D %>%
    inner_join(CP_profiles_2D_max, by = c("Metadata_Biology_PatientTumor", "Metadata_Experiment_Treatment", "Metadata_Experiment_Dose")) %>%
    inner_join(CP_profiles_2D_middle, by = c("Metadata_Biology_PatientTumor", "Metadata_Experiment_Treatment", "Metadata_Experiment_Dose")) %>%
    inner_join(CP_profiles_2D_middle_n, by = c("Metadata_Biology_PatientTumor", "Metadata_Experiment_Treatment", "Metadata_Experiment_Dose")) %>%
    select(Metadata_Biology_PatientTumor, Metadata_Experiment_Treatment,Metadata_Experiment_Dose, n_cells_3D, n_cells_2D_max, n_cells_2D_middle_slice, n_cells_2D_middle_n)
merged_df <- distinct(merged_df)
# drop the values of the 2D counts above 100
merged_df <- merged_df %>%
    filter(n_cells_2D_max <= 100) %>%
    filter(n_cells_2D_middle_slice <= 100) %>%
    filter(n_cells_2D_middle_n <= 100)


width <- 12
height <- 8
options(repr.plot.width = width, repr.plot.height = height)
comparison_plot_of_2D_and_3D_counts <- plot_2d_vs_3d_scatter(
    merged_df,
    x_col = "n_cells_2D_middle_slice",
    y_col = "n_cells_3D",
    color_col = "Metadata_Experiment_Treatment",
    shape_col = "Metadata_Experiment_Dose",
    color_palette = custom_treatment_palette,
    x_lab = "Normalized cell counts per well FOV (2D middle slice)",
    y_lab = "Normalized cell counts per well FOV (3D ZedProfiler)",
    shape_lab = "Dose (uM)"
)
save_ggplot(
    comparison_plot_of_2D_and_3D_counts,
    path = file.path(cell_counts_figure_path, "comparison_plot_of_2D_and_3D_counts.png"),
    width = width,
    height = height
)

width <- 12
height <- 8
options(repr.plot.width = width, repr.plot.height = height)
comparison_plot_of_2D_and_3D_counts <- plot_2d_vs_3d_scatter(
    merged_df,
    x_col = "n_cells_2D_middle_n",
    y_col = "n_cells_3D",
    color_col = "Metadata_Experiment_Treatment",
    shape_col = "Metadata_Experiment_Dose",
    color_palette = custom_treatment_palette,
    x_lab = "Normalized cell counts per well FOV (2D middle 3 slices)",
    y_lab = "Normalized cell counts per well FOV (3D ZedProfiler)",
    shape_lab = "Dose (uM)"
)
save_ggplot(
    comparison_plot_of_2D_and_3D_counts,
    path = file.path(cell_counts_figure_path, "comparison_plot_of_2D_and_3D_counts_middle_n.png"),
    width = width,
    height = height
)

width <- 12
height <- 8
options(repr.plot.width = width, repr.plot.height = height)
comparison_plot_of_2D_and_3D_counts <- plot_2d_vs_3d_scatter(
    merged_df,
    x_col = "n_cells_2D_max",
    y_col = "n_cells_3D",
    color_col = "Metadata_Experiment_Treatment",
    shape_col = "Metadata_Experiment_Dose",
    color_palette = custom_treatment_palette,
    x_lab = "Normalized cell counts per well FOV (2D max projection)",
    y_lab = "Normalized cell counts per well FOV (3D ZedProfiler)",
    shape_lab = "Dose (uM)"
)
save_ggplot(
    comparison_plot_of_2D_and_3D_counts,
    path = file.path(cell_counts_figure_path, "comparison_plot_of_2D_and_3D_counts_max.png"),
    width = width,
    height = height
)

width <- 16
height <- 16
options(repr.plot.width = width, repr.plot.height = height)
comparison_plot_of_2D_and_3D_counts <- plot_2d_vs_3d_scatter(
    merged_df,
    x_col = "n_cells_2D_max",
    y_col = "n_cells_3D",
    color_col = "Metadata_Experiment_Treatment",
    shape_col = "Metadata_Experiment_Dose",
    color_palette = custom_treatment_palette,
    x_lab = "Normalized cell counts per well FOV (2D max projection)",
    y_lab = "Normalized cell counts per well FOV (3D ZedProfiler)",
    facet_formula = as.formula(. ~ Metadata_Biology_PatientTumor),
    shape_lab = "Dose (uM)"
)
save_ggplot(
    comparison_plot_of_2D_and_3D_counts,
    path = file.path(cell_counts_figure_path, "comparison_plot_of_2D_and_3D_counts_max_facet.png"),
    width = width,
    height = height
)

organid_cell_count_file_path <- file.path(root_dir, "1.EDA/results/cell_counts/organoid_cell_counts.parquet")
organid_cell_counts <- arrow::read_parquet(organid_cell_count_file_path)
organid_cell_counts <- organid_cell_counts %>% dplyr::filter(Metadata_Biology_PatientTumor != "NF0037_T1_CQ1")

head(organid_cell_counts)

width <- 12
height <- 8
options(repr.plot.width = width, repr.plot.height = height)
# Mean cells per organoid, broken down by patient
mean_cells_per_organoid_by_patient_plot <- plot_boxplot(
    organid_cell_counts,
    x_col = "Metadata_Biology_PatientTumor",
    y_col = "mean_cells_per_organoid",
    fill_col = "Metadata_Biology_PatientTumor",
    x_lab = "Patient tumor",
    y_lab = "Mean cells per organoid",
    fill_lab = "Patient tumor sample",
    x_text = "angled",
    show_legend = FALSE
)
save_ggplot(
    mean_cells_per_organoid_by_patient_plot,
    path = file.path(cell_counts_figure_path, "mean_cells_per_organoid_by_patient_boxplot.png"),
    width = width,
    height = height
)

width <- 10
height <- 6
options(repr.plot.width = width, repr.plot.height = height)
# Distribution of cell counts per FOV across the dataset
cell_counts_per_fov_density_plot <- (
    ggplot(organid_cell_counts, aes(x = total_cells_normalized_per_FOV))
    + geom_density(fill = "#3D7DCC", color = "black", alpha = 0.7)
    + labs(x = "Total cells per FOV", y = "Density")
    + theme_manuscript(base_size = 18)
)
save_ggplot(
    cell_counts_per_fov_density_plot,
    path = file.path(cell_counts_figure_path, "cell_counts_per_fov_density.png"),
    width = width,
    height = height
)

width <- 10
height <- 6
options(repr.plot.width = width, repr.plot.height = height)
# Distribution of cell counts per organoid across the dataset
mean_cells_per_organoid_density_plot <- (
    ggplot(organid_cell_counts, aes(x = mean_cells_per_organoid))
    + geom_density(fill = "#CC8029", color = "black", alpha = 0.7)
    + labs(x = "Mean cells per organoid", y = "Density")
    + theme_manuscript(base_size = 18)
)
save_ggplot(
    mean_cells_per_organoid_density_plot,
    path = file.path(cell_counts_figure_path, "mean_cells_per_organoid_density.png"),
    width = width,
    height = height
)

# reshape to long format so cells with and without a parent organoid can be
# plotted side by side
organid_cell_counts_long <- dplyr::bind_rows(
    organid_cell_counts %>%
        mutate(
            Parent_Organoid_Status = "Inside organoid",
            n_cells_normalized_per_FOV = n_cells_with_parent_organoid_normalized_per_FOV
        ),
    organid_cell_counts %>%
        mutate(
            Parent_Organoid_Status = "Outside organoid",
            n_cells_normalized_per_FOV = n_cells_without_parent_organoid_normalized_per_FOV
        )
)
parent_organoid_status_palette <- c(
    "Inside organoid" = "#1FAD23",
    "Outside organoid" = "#CC29CC"
)

width <- 10
height <- 6
options(repr.plot.width = width, repr.plot.height = height)
# Distribution of cells with vs. without a parent organoid across the dataset
cells_with_without_parent_organoid_density_plot <- (
    ggplot(organid_cell_counts_long, aes(x = n_cells_normalized_per_FOV, fill = Parent_Organoid_Status))
    + geom_density(color = "black", alpha = 0.6)
    + scale_fill_manual(values = parent_organoid_status_palette)
    + labs(x = "Cells per FOV (normalized)", y = "Density", fill = "Organoid parentage")
    + theme_manuscript(base_size = 18)
)
save_ggplot(
    cells_with_without_parent_organoid_density_plot,
    path = file.path(cell_counts_figure_path, "cells_with_without_parent_organoid_density.png"),
    width = width,
    height = height
)

width <- 12
height <- 8
options(repr.plot.width = width, repr.plot.height = height)
# Enrichment of patients for cells without (vs. with) a parent organoid
cells_without_parent_organoid_by_patient_plot <- plot_boxplot(
    organid_cell_counts_long,
    x_col = "Metadata_Biology_PatientTumor",
    y_col = "n_cells_normalized_per_FOV",
    fill_col = "Parent_Organoid_Status",
    x_lab = "Patient tumor",
    y_lab = "Cells per FOV (normalized)",
    fill_lab = "Organoid parentage",
    custom_palette = parent_organoid_status_palette,
    x_text = "angled"
)
save_ggplot(
    cells_without_parent_organoid_by_patient_plot,
    path = file.path(cell_counts_figure_path, "cells_without_parent_organoid_by_patient_boxplot.png"),
    width = width,
    height = height
)

width <- 20
height <- 10
options(repr.plot.width = width, repr.plot.height = height)
# Enrichment of treatments for cells without (vs. with) a parent organoid
cells_without_parent_organoid_by_treatment_plot <- plot_boxplot(
    organid_cell_counts_long,
    x_col = "Metadata_Experiment_Treatment",
    y_col = "n_cells_normalized_per_FOV",
    fill_col = "Parent_Organoid_Status",
    x_lab = "Treatment",
    y_lab = "Cells per FOV (normalized)",
    fill_lab = "Organoid parentage",
    custom_palette = parent_organoid_status_palette,
    x_text = "angled"
)
cells_without_parent_organoid_by_treatment_plot <- cells_without_parent_organoid_by_treatment_plot + ylim(0,30)
save_ggplot(
    cells_without_parent_organoid_by_treatment_plot,
    path = file.path(cell_counts_figure_path, "cells_without_parent_organoid_by_treatment_boxplot.png"),
    width = width,
    height = height
)

width <- 26
height <- 16
options(repr.plot.width = width, repr.plot.height = height)
# Enrichment of treatments for cells without (vs. with) a parent organoid, faceted by patient
cells_without_parent_organoid_by_treatment_patient_plot <- plot_boxplot(
    organid_cell_counts_long,
    x_col = "Metadata_Experiment_Treatment",
    y_col = "n_cells_normalized_per_FOV",
    fill_col = "Parent_Organoid_Status",
    x_lab = "Treatment",
    y_lab = "Cells per FOV (normalized)",
    fill_lab = "Organoid parentage",
    custom_palette = parent_organoid_status_palette,
    x_text = "angled",
    facet_formula = as.formula(. ~ Metadata_Biology_PatientTumor)
)
cells_without_parent_organoid_by_treatment_patient_plot <- cells_without_parent_organoid_by_treatment_patient_plot + ylim(0, 30)
save_ggplot(
    cells_without_parent_organoid_by_treatment_patient_plot,
    path = file.path(cell_counts_figure_path, "cells_without_parent_organoid_by_treatment_patient_boxplot.png"),
    width = width,
    height = height
)

# plot the proportion_of_cells_with_parent_organoid_compared_to_n_cells_without_parent_organoid
width <- 20
height <- 10
options(repr.plot.width = width, repr.plot.height = height)
# Proportion of cells with a parent organoid, by treatment
cells_without_parent_organoid_by_treatment_proportion_plot <- plot_boxplot(
    organid_cell_counts,
    x_col = "Metadata_Experiment_Treatment",
    y_col = "proportion_of_cells_with_parent_organoid_compared_to_n_cells_without_parent_organoid",
    x_lab = "Treatment",
    y_lab = "Proportion of cells with parent organoid",
    x_text = "angled"
)
save_ggplot(
    cells_without_parent_organoid_by_treatment_proportion_plot,
    path = file.path(cell_counts_figure_path, "cells_without_parent_organoid_by_treatment_proportion_boxplot.png"),
    width = width,
    height = height
)
