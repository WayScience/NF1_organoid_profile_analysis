library(RColorBrewer)
# set custom colors for each MOA
custom_MOA_palette <- c(
    'BRD4 inhibitor' = "#93152A",  # Dark red
    'receptor tyrosine kinase inhibitor' = "#BA3924",  # Red
    'tyrosine kinase inhibitor' = "#D08543",  # Orange
    'MEK1/2 inhibitor' = "#A1961A",  # Yellow-green/olive

    'IGF-1R inhibitor' = "#9FC62A",  # Yellow-green
    'mTOR inhibitor' = "#1FAD23",  # Green
    'PI3K inhibitor' = "#32D06A",  # Light green
    'PI3K and HDAC inhibitor' = "#15937C",  # Teal/dark green
    'HDAC inhibitor' = "#24A5BA",  # Light blue/cyan

    'Apoptosis induction' = "#438CD0",  # Medium blue
    'DNA binding' = "#1A24A1",  # Dark blue
    'HSP90 inhibitor' = "#532AC6",  # Blue-purple

    'histamine H1\nreceptor antagonist' = "#AD1FA6",  # Purple/magenta
    'Na+/K+ pump inhibitor' = "#D03294",  # Pink/magenta

    'Control' = "#444444"  # Gray
)
treatment_moa_map <- c(
    'DMSO' = 'Control',
    'Staurosporine' = 'Apoptosis induction',
    'Fimepinostat' = 'PI3K and HDAC inhibitor',
    'Copanlisib' = 'PI3K inhibitor',
    'Everolimus' = 'mTOR inhibitor',
    'Rapamycin' = 'mTOR inhibitor',
    'Sapanisertib' = 'mTOR inhibitor',
    'Vistusertib' = 'mTOR inhibitor',
    'Panobinostat' = 'HDAC inhibitor',
    'ARV-825' = 'BRD4 inhibitor',
    'Imatinib' = 'tyrosine kinase inhibitor',
    'Nilotinib' = 'tyrosine kinase inhibitor',
    'Cabozantinib' = 'receptor tyrosine kinase inhibitor',
    'Linsitinib' = 'IGF-1R inhibitor',
    'Binimetinib' = 'MEK1/2 inhibitor',
    'Mirdametinib' = 'MEK1/2 inhibitor',
    'Trametinib' = 'MEK1/2 inhibitor',
    'Selumetinib' = 'MEK1/2 inhibitor',
    "Onalespib" = "HSP90 inhibitor",
    "Digoxin" = "Na+/K+ pump inhibitor",
    "Ketotifen" = "histamine H1\nreceptor antagonist",
    "Trabectedin" = "DNA binding"
)

custom_MOA_order <- c(
    'Control',
    'BRD4 inhibitor',
    'receptor tyrosine kinase inhibitor',
    'tyrosine kinase inhibitor',
    'MEK1/2 inhibitor',
    'IGF-1R inhibitor',
    'mTOR inhibitor',
    'PI3K inhibitor',
    'PI3K and HDAC inhibitor',
    'HDAC inhibitor',
    'Apoptosis induction',
    'DNA binding',
    'HSP90 inhibitor',
    'histamine H1\nreceptor antagonist',
    'Na+/K+ pump inhibitor'
)

# Set custom colors for each treatment
# Hue = MOA family (similar mechanisms share a hue range); Lightness/Saturation = dose
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

dose_palette <- c(
    "1" = "#CFCFCF",
    "10" = "#4D4D4D"
)
custom_treatment_order <- c(
    'DMSO',
    'Staurosporine',
    'Fimepinostat',
    'Copanlisib',
    'Everolimus',
    'Rapamycin',
    'Sapanisertib',
    'Vistusertib',
    'Imatinib',
    'Nilotinib',
    'Cabozantinib',
    'Linsitinib',
    'Panobinostat',
    'ARV-825',
    'Onalespib',
    'Digoxin',
    'Ketotifen',
    'Trabectedin',
    'Binimetinib',
    'Mirdametinib',
    'Trametinib',
    'Selumetinib'
)
channel_palette = c(
    "DNA" = "#0000AB",
    "AGP" = "#b1001a",
    "Mito" = "#B000B0",
    "ER" = "#00D55B",
    "BF" = "#FFFF00",
    "NoChannel" = "#B09FB0"
)

sc_compartment_palette = c(
    "Cell" = "#B000B0",
    "Cytoplasm" = "#00D55B",
    "Nuclei" = "#0000AB"
)

organoid_compartment_palette = c(
    "Organoid" = "#B09FB0"
)

feature_type_palette = c(
    "AreaSizeShape" = brewer.pal(8, "Paired")[1],
    # V3 3D profiles name the shape features VolumeSizeShape (was AreaSizeShape)
    "VolumeSizeShape" = brewer.pal(8, "Paired")[1],
    "Neighbors" = brewer.pal(8, "Paired")[4],
    "Colocalization" = brewer.pal(8, "Paired")[2],
    "Granularity" = brewer.pal(8, "Paired")[3],
    "Intensity" = brewer.pal(8, "Paired")[5],
    "Texture" = brewer.pal(8, "Paired")[8]
)

# Shared theme for manuscript-quality plots.
# x_text controls how the x axis tick labels are rendered:
#   "default" - leave tick labels as ggplot's default
#   "blank"   - hide tick labels/ticks (e.g. many-category axes with a legend)
#   "angled"  - 45-degree tick labels (e.g. few-category axes without a legend)
theme_manuscript <- function(base_size = 18, x_text = c("default", "blank", "angled")) {
    x_text <- match.arg(x_text)

    x_axis_theme <- switch(
        x_text,
        blank = theme(
            axis.text.x = element_blank(),
            axis.ticks.x = element_blank()
        ),
        angled = theme(
            axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = base_size)
        ),
        default = theme()
    )

    (
        theme_bw()
        + theme(
            axis.text = element_text(size = base_size),
            axis.title = element_text(size = base_size),
            strip.text = element_text(size = base_size),
            plot.title = element_text(size = base_size, face = "bold"),
            legend.title = element_text(size = base_size),
            legend.text = element_text(size = base_size)
        )
        + x_axis_theme
    )
}
tab20_palette_for_patients <- c(
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
    "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
    "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
    "#17becf", "#9edae5"
)

# cNF = cutaneous/subcutaneous neurofibroma, pNF = plexiform neurofibroma,
# MPNST = malignant peripheral nerve sheath tumor. NF0030_T1 (myopericytoma)
# and NF0040_T1 (schwannoma) are not NF1 nerve-sheath tumors and are grouped
# as "Other".
# Source: https://github.com/WayScience/NF1_3D_organoid_profiling_pipeline/blob/4072be16543851063df9bcd16500498f269f45fd/figures/table1_patients_and_counts/results/table1_patients_and_counts_results.tsv
tumor_type_lookup <- c(
    "NF0014_T1" = "cNF",
    "NF0014_T2" = "pNF",
    "NF0016_T1" = "pNF",
    "NF0018_T6" = "cNF",
    "NF0021_T1" = "cNF",
    "NF0030_T1" = "Other",
    "NF0035_T1" = "cNF",
    "NF0037_T1" = "cNF",
    "NF0040_T1" = "Other",
    "NF0055_T1" = "pNF",
    "SARCO219_T2" = "MPNST",
    "SARCO361_T1" = "MPNST"
)

tumor_type_palette <- c(
    "cNF" = "#1B9E77",
    "pNF" = "#D95F02",
    "MPNST" = "#7570B3",
    "Other" = "#999999"
)

viability_model_eval_split_colors <- c(
    train = "#3B6FA0",
    test  = "#E08214"
)

viability_shuffle_status_colors <- c(
    not_shuffled = "#8f3ba0",
    shuffled      = "#70e014"
)

linear_modeling_term_palette<- c(
    treatment = "#d95f02", cell_count = "#1b9e77", organoid_count = "#7570b3",
    cell_per_organoid_count = "#e7298a", manhattan_distance_from_center = "#66a61e",
    cell_x_position = "#e6ab02", cell_y_position = "#a6761d",
    cell_z_position = "#666666", cell_z_depth = "#1f78b4"
)

# Short display labels for the 3D normalization variants, for use in plot
# titles/legends instead of the raw snake_case identifiers (e.g.
# "nucleocentric_morphem_norm" -> "Nucleocentric (MorphEm)"). First-pass
# scheme: level (Organoid / Single-cell / Nucleocentric) + a parenthetical
# naming the feature-extraction method (ZEDProfiler, MorphEm/CHAMMI, or
# SAM-Med3D). See
# https://github.com/WayScience/NF1_organoid_profile_analysis/issues/33 for
# expanding this to a full cross-notebook labeling standard.
normalization_variant_labels <- c(
    organoid_norm = "Organoid (ZEDProfiler)",
    sammed_organoid_norm = "Organoid (SAM-med)",
    sc_norm = "Single-cell (ZEDProfiler)",
    sammed_sc_norm = "Single-cell (SAM-med)",
    nucleocentric_morphem_norm = "Nucleocentric (MorphEm)",
    sammed_nucleocentric_norm = "Nucleocentric (SAM-med)"
)

# ---------------------------------------------------------------------------
# Linear modeling figures (4.linear_modeling)
# ---------------------------------------------------------------------------

# class letters -> model variates (as coded by 7.calculate_variate_class_upsets_and_clustermap)
variate_letters <- c(
    T = "treatment", O = "organoid_count", C = "cell_per_organoid_count", N = "cell_count",
    M = "manhattan_distance_from_center", X = "cell_x_position", Y = "cell_y_position",
    Z = "cell_z_position", D = "cell_z_depth"
)

# the four linear-model profiles: organoid / single cell, and their well-aggregated versions
profiles <- c("organoid", "sc", "organoid_agg", "sc_agg")
main_profiles <- c("organoid", "sc")
profile_palette <- setNames(c("#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3"), profiles)

# annotation colours shared by every heatmap / clustermap: patient, dose (drug and tumor type
# palettes are defined above)
patient_palette <- setNames(tab20_palette_for_patients[seq_along(tumor_type_lookup)], names(tumor_type_lookup))
dose_label_palette <- c("1uM" = dose_palette[["1"]], "10uM" = dose_palette[["10"]], "10nM" = "#8da0cb")
feature_type_palette_with_other <- c(feature_type_palette, Other = "#999999")

# UpSet plots
treatment_specific_palette <- c("TRUE" = linear_modeling_term_palette[["treatment"]], "FALSE" = "#555555")
upset_set_size_colour <- "#555555"
upset_class_bar_colour <- "#333333"
upset_dot_off_colour <- "#dddddd"
upset_stripe_colour <- "#f2f2f2"

# treatment-only significance (TRUE = the treatment is the only significant variate)
treatment_only_palette <- c("TRUE" = "#d62728", "FALSE" = "#7f7f7f")
treatment_only_fill <- "#d62728"
all_vs_top_palette <- c("#bdbdbd", "#d62728")
treatment_only_heatmap_palette <- c(no = "white", yes = linear_modeling_term_palette[["treatment"]])
variate_presence_palette <- c(no = "white", yes = "#333333")

# direction / enrichment of effects
direction_palette <- c(down = "#4575b4", up = "#d73027")
enrichment_palette <- c(enriched = "#d73027", depleted = "#4575b4", "n.s." = "grey")

# diverging heatmap colours (low, mid, high): correlations and treatment coefficients
correlation_diverging_colours <- c("#4575b4", "white", "#d73027")
coefficient_diverging_colours <- c("#3b4cc0", "white", "#b40426")

# theme_manuscript() for panels that have no axes (Venn diagrams, UpSet dot matrices, label
# columns): drops the axes, panel and grid but keeps the manuscript text sizes and bold titles
theme_manuscript_void <- function(base_size = 18) {
    (
        theme_void()
        + theme(
            plot.title = element_text(size = base_size, face = "bold"),
            legend.title = element_text(size = base_size),
            legend.text = element_text(size = base_size)
        )
    )
}
