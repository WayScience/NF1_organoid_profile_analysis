packages <- c("ggplot2", "dplyr", "arrow", "ComplexHeatmap", "circlize", "scales", "RColorBrewer", "ggrastr")
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

    # If no Git root found, stop with error
    stop("No Git root directory found.")
}

# Find the Git root directory
root_dir <- find_git_root()
source(file.path(root_dir, "utils", "r_plot_themes.r"))

figures_dir <- file.path(root_dir, "1.EDA", "figures", "profile_heatmaps")
dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)

load_tumor_type_map <- function() {
    # 2D profiles never carried a real Metadata_Biology_TumorType column
    # (only 3D's per-patient files do, e.g.
    # data/profiles_3D/*/5.normalized_profiles/*), and both modalities lose
    # it once aggregated to all_patients. Rather than relying on the
    # hand-maintained tumor_type_lookup in r_plot_themes.r, read the real
    # (patient, tumor type) pairs from one 3D feature-selected file (any
    # variant has the same pairs) and reuse that single mapping for both
    # 2D and 3D samples below.
    path <- file.path(
        root_dir, "data", "profiles_3D", "all_patients", "1.feature_selected_profiles",
        "sc_norm_fs_profiles.parquet"
    )
    df <- arrow::read_parquet(
        path, col_select = c("Metadata_Biology_PatientTumor", "Metadata_Biology_TumorType")
    )
    df <- df[!is.na(df$Metadata_Biology_TumorType), ]
    df <- unique(df)
    setNames(df$Metadata_Biology_TumorType, df$Metadata_Biology_PatientTumor)
}
tumor_type_map <- load_tumor_type_map()

PROJECTION_FILE_PREFIX <- c(
    max_projection = "max_projected",
    middle_slice = "middle_slice",
    middle_n_slice = "middle_n_slice"
)

list_2d_patient_dirs <- function() {
    # Real per-patient directories under data/profiles_2D (excludes the
    # all_patients pooled directory and the separately-analyzed
    # NF0037_T1_CQ1, matching clean_profile()'s exclusion elsewhere).
    dirs <- list.dirs(file.path(root_dir, "data", "profiles_2D"), recursive = FALSE, full.names = FALSE)
    setdiff(dirs, c("all_patients", "NF0037_T1_CQ1"))
}

load_dose_unit_map <- function() {
    # Per-patient 2D aggregated files (data/profiles_2D/{patient}/7.aggregated/)
    # carry dose but not its unit -- (treatment, dose) determines unit
    # deterministically in the pooled 2D data (verified: no (treatment,
    # dose) pair spans more than one unit), so recover it here from the
    # all_patients file, which does have it, and join it back onto the
    # per-patient files below.
    path <- file.path(root_dir, "data", "profiles_2D", "all_patients", "max_projection", "sc_agg_profiles.parquet")
    df <- arrow::read_parquet(path, col_select = c("Metadata_treatment", "Metadata_dose", "Metadata_dose_unit"))
    unique(df)
}
dose_unit_map <- load_dose_unit_map()

load_2d_per_patient_feature_matrices <- function(slice_strategy, profile_source, feature_parser) {
    # The pooled all_patients 2D files (used for the all-patients page)
    # collapse every row's patient identity into a single dummy
    # "2D_analysis" value -- there's no real per-row patient ID to recover
    # from them. The real per-patient breakdown only exists one step
    # earlier, at data/profiles_2D/{patient}/7.aggregated/, so this reads
    # those files directly instead, tagging each with its own directory
    # name as patient ID. Only exists for the agg profile type -- 2D has
    # no per-patient consensus step on disk at all.
    #
    # Returns a named list (one entry per patient) of
    # list(mat, metadata, feature_annotations) rather than one combined
    # table: each patient's own file can carry a *different* surviving
    # feature set (feature selection upstream runs per patient), so
    # concatenating them first and then dropping any row with an NA
    # feature -- as clean_profile does -- would drop every row (each
    # patient is missing some other patient's features). Since each
    # patient gets its own separate heatmap anyway, there's no need to
    # force a shared feature set across them; only within one patient's
    # own data does clean_profile's row-wise NA drop make sense.
    file_prefix <- PROJECTION_FILE_PREFIX[[slice_strategy]]
    result <- list()
    for (patient_id in list_2d_patient_dirs()) {
        input_path <- file.path(
            root_dir, "data", "profiles_2D", patient_id, "7.aggregated",
            paste0(file_prefix, "_", profile_source, ".parquet")
        )
        if (!file.exists(input_path)) next
        df <- arrow::read_parquet(input_path)
        df <- df %>% left_join(dose_unit_map, by = c("Metadata_treatment", "Metadata_dose"))
        df$Metadata_Biology_PatientTumor <- patient_id
        df <- df %>% rename(
            Metadata_Experiment_Treatment = Metadata_treatment,
            Metadata_Experiment_Dose = Metadata_dose,
            Metadata_Experiment_Unit = Metadata_dose_unit
        )

        cleaned <- clean_profile(df, "Metadata_Biology_PatientTumor")
        patient_df <- cleaned$df
        feature_columns <- cleaned$feature_columns
        if (nrow(patient_df) == 0 || length(feature_columns) == 0) next

        mat <- as.matrix(patient_df[, feature_columns])
        rownames(mat) <- NULL
        metadata <- patient_df[, cleaned$metadata_columns]
        metadata$Metadata_Biology_TumorType <- tumor_type_map[metadata$Metadata_Biology_PatientTumor]

        result[[patient_id]] <- list(
            mat = mat, metadata = metadata, feature_annotations = feature_parser(feature_columns)
        )
    }
    result
}

clean_profile <- function(df, patient_col) {
    # Exclude NF0037_T1_CQ1: this is a separate analysis and should not be
    # included anywhere in this EDA (matches 0.generate_umap.ipynb /
    # 4.calculate_correlation_matrix.ipynb).
    df <- df[df[[patient_col]] != "NF0037_T1_CQ1", ]

    # A small number of samples are missing core experimental metadata
    # (treatment, dose, tumor type) entirely -- an incomplete metadata join,
    # not a real treatment condition. Drop them.
    if ("Metadata_Experiment_Treatment" %in% colnames(df)) {
        df <- df[!is.na(df$Metadata_Experiment_Treatment), ]
    }

    metadata_columns <- grep("^Metadata_", colnames(df), value = TRUE)
    feature_columns <- setdiff(colnames(df), metadata_columns)

    # Some non-feature-selected profiles (*_from_norm) carry a handful of
    # features that are NA for every sample (e.g. degenerate/undefined for
    # that normalization) -- feature-selected profiles never have this
    # since selection already drops them. Left in, complete.cases() below
    # would drop every row instead of just those features. Drop them first
    # so an otherwise-complete row only loses NA-prone individual features,
    # not its only chance to survive at all.
    all_na_features <- feature_columns[vapply(df[feature_columns], function(x) all(is.na(x)), logical(1))]
    feature_columns <- setdiff(feature_columns, all_na_features)

    df <- df[stats::complete.cases(df[, feature_columns]), ]
    list(df = df, metadata_columns = metadata_columns, feature_columns = feature_columns)
}

# 2D's own naming for a few categories differs from the shared palettes in
# r_plot_themes.r (which were defined against 3D's naming) -- normalize to
# match so the same palettes (sc_compartment_palette, feature_type_palette_2D,
# channel_palette) apply to both.
CHANNELS_2D <- c("AGP", "Brightfield", "ER", "Hoechst", "Mito", "DNA")
CHANNEL_RENAME_2D <- c(Brightfield = "BF", Hoechst = "DNA")
COMPARTMENT_RENAME_2D <- c(Cells = "Cell")
FEATURE_TYPE_RENAME_2D <- c(AreaShape = "AreaSizeShape", Correlation = "Colocalization")

parse_feature_names_2d <- function(feature_columns) {
    # 2D CellProfiler naming: <Compartment>_<FeatureType>_<Measurement...>[_<Channel>].
    # FeatureType is always the 2nd underscore-separated token; channel (if
    # any) is whichever remaining token matches a known channel name --
    # its position varies (e.g. right after FeatureType for Granularity,
    # but before trailing texture-scale numbers for Texture), so every
    # remaining token is scanned rather than just checking the last one.
    split <- strsplit(feature_columns, "_")
    compartment <- vapply(split, `[`, character(1), 1)
    feature_type <- vapply(split, `[`, character(1), 2)
    channel <- vapply(split, function(parts) {
        rest <- parts[-c(1, 2)]
        hit <- rest[rest %in% CHANNELS_2D]
        if (length(hit) > 0) hit[1] else "NoChannel"
    }, character(1))

    compartment <- ifelse(compartment %in% names(COMPARTMENT_RENAME_2D), COMPARTMENT_RENAME_2D[compartment], compartment)
    feature_type <- ifelse(feature_type %in% names(FEATURE_TYPE_RENAME_2D), FEATURE_TYPE_RENAME_2D[feature_type], feature_type)
    channel <- ifelse(channel %in% names(CHANNEL_RENAME_2D), CHANNEL_RENAME_2D[channel], channel)

    data.frame(
        feature = feature_columns, compartment = compartment,
        feature_type = feature_type, channel = channel,
        stringsAsFactors = FALSE
    )
}

FEATURE_TYPE_RENAME_3D <- c(VolumeSizeShape = "AreaSizeShape")

parse_feature_names_3d <- function(feature_columns) {
    # 3D naming: <Compartment>_<Channel>_<FeatureType>_<Measurement...>,
    # a fixed 4-token-minimum layout (verified across every sc/organoid
    # normalization variant) -- channel and feature type are always the
    # 2nd and 3rd tokens, so this is a direct positional split rather than
    # the token-scanning 2D parser needs. A colocalization feature's
    # channel token is a "Channel1-Channel2" pair (e.g. "AGP-Mito") --
    # channel_palette only has single-channel colors, so this uses just
    # the first channel of the pair for color-matching purposes.
    split <- strsplit(feature_columns, "_")
    compartment <- vapply(split, `[`, character(1), 1)
    channel <- vapply(split, `[`, character(1), 2)
    feature_type <- vapply(split, `[`, character(1), 3)

    channel <- vapply(strsplit(channel, "-"), `[`, character(1), 1)
    feature_type <- ifelse(feature_type %in% names(FEATURE_TYPE_RENAME_3D), FEATURE_TYPE_RENAME_3D[feature_type], feature_type)

    data.frame(
        feature = feature_columns, compartment = compartment,
        feature_type = feature_type, channel = channel,
        stringsAsFactors = FALSE
    )
}

compartment_palette <- c(sc_compartment_palette, organoid_compartment_palette)

lookup_colors <- function(labels, palette, fallback = "#999999") {
    # Colors for annotation values from an established palette
    # (channel_palette / compartment_palette / feature_type_palette[_2D]),
    # falling back to a flat grey for any value the palette doesn't cover
    # (e.g. the 2D-only "Number"/"Children" measurement groups, or the
    # SAMMed3D feature-extraction method, which none of the palettes name).
    labels <- sort(unique(labels))
    colors <- setNames(character(length(labels)), labels)
    known <- intersect(labels, names(palette))
    colors[known] <- palette[known]
    colors[setdiff(labels, known)] <- fallback
    colors
}

load_feature_matrix <- function(input_path, patient_col, feature_parser, rename_map = NULL) {
    # Loads one profile file, cleans it, and returns the sample-by-feature
    # matrix (rows = samples, columns = features) plus per-sample metadata
    # (Treatment/Dose/Patient/TumorType) and per-feature annotations
    # (Compartment/FeatureType/Channel, from feature_parser). Profile
    # values are used as-is: they're already normalized earlier in the
    # pipeline, so no additional scaling happens here.
    df <- arrow::read_parquet(input_path)
    if (!is.null(rename_map)) {
        df <- df %>% rename(!!!rename_map)
    }

    cleaned <- clean_profile(df, patient_col)
    df <- cleaned$df
    feature_columns <- cleaned$feature_columns

    mat <- as.matrix(df[, feature_columns])
    rownames(mat) <- NULL

    metadata <- df[, cleaned$metadata_columns]
    metadata$Metadata_Biology_TumorType <- tumor_type_map[metadata[[patient_col]]]

    list(mat = mat, metadata = metadata, feature_annotations = feature_parser(feature_columns))
}

build_profile_heatmap <- function(mat, metadata, feature_annotations, feature_type_palette, cluster = TRUE,
                                   title, legend_fontsize = 8, column_title_fontsize = 10) {
    # Build a single feature-by-sample profile Heatmap: rows are features,
    # columns are samples (mat comes in as samples x features from the
    # loaders -- transposed here). Column annotations (Treatment/Dose/
    # Patient/TumorType, omitted individually if constant, e.g. Patient/
    # TumorType for a single-patient matrix) sit on top; row annotations
    # (Compartment/FeatureType/Channel) are drawn on the right -- the
    # top-left corner is where the column annotation names sit, and a
    # left-side row annotation would otherwise overlap them there.
    # feature_type_palette is feature_type_palette_2D for 2D callers or
    # feature_type_palette (r_plot_themes.r) for 3D ones -- their 2D/3D
    # naming conventions use different feature-type vocabularies (e.g. 2D's
    # "AreaShape" vs 3D's "VolumeSizeShape", both normalized to
    # "AreaSizeShape" by the parsers above), so the matching palette has to
    # be threaded in by the caller rather than picked here. When
    # cluster = FALSE, columns (samples) are instead sorted by tumor type,
    # then patient, then dose, then treatment, and rows (features) by
    # compartment, then feature type, then channel -- clustering a single
    # column isn't possible, so it's disabled automatically when n < 2.
    mat <- t(mat)
    n_samples <- ncol(mat)
    cluster_cols <- cluster && n_samples >= 2
    cluster_rows <- cluster

    col_perm <- seq_len(n_samples)
    if (!cluster_cols) {
        sort_keys <- list()
        if (length(unique(metadata$Metadata_Biology_TumorType)) > 1) sort_keys$tumor_type <- metadata$Metadata_Biology_TumorType
        if (length(unique(metadata$Metadata_Biology_PatientTumor)) > 1) sort_keys$patient <- metadata$Metadata_Biology_PatientTumor
        sort_keys$dose <- metadata$dose_label
        sort_keys$treatment <- metadata$Metadata_Experiment_Treatment
        col_perm <- do.call(order, sort_keys)
    }
    row_perm <- seq_len(nrow(mat))
    if (!cluster_rows) {
        row_perm <- order(
            feature_annotations$compartment, feature_annotations$feature_type, feature_annotations$channel
        )
    }

    mat <- mat[row_perm, col_perm, drop = FALSE]
    metadata <- metadata[col_perm, ]
    feature_annotations <- feature_annotations[row_perm, ]

    col_annotations <- list(Treatment = metadata$Metadata_Experiment_Treatment, Dose = metadata$dose_label)
    col_anno_cols <- list(Treatment = custom_treatment_palette, Dose = dose_colors_for(metadata$dose_label))
    col_show_legend <- c(Treatment = TRUE, Dose = TRUE)
    col_legend_params <- list(
        Treatment = list(title = "Treatment", ncol = 1, labels_gp = gpar(fontsize = legend_fontsize)),
        Dose = list(title = "Dose", labels_gp = gpar(fontsize = legend_fontsize))
    )

    unique_patients <- sort(unique(metadata$Metadata_Biology_PatientTumor))
    if (length(unique_patients) > 1) {
        patient_colors <- setNames(tab20_palette_for_patients[seq_along(unique_patients)], unique_patients)
        col_annotations$Patient <- metadata$Metadata_Biology_PatientTumor
        col_anno_cols$Patient <- patient_colors
        col_show_legend <- c(col_show_legend, Patient = TRUE)
    }

    unique_tumor_types <- unique(metadata$Metadata_Biology_TumorType)
    if (length(unique_tumor_types) > 1) {
        col_annotations$TumorType <- metadata$Metadata_Biology_TumorType
        col_anno_cols$TumorType <- tumor_type_palette
        col_show_legend <- c(col_show_legend, TumorType = TRUE)
        col_legend_params$TumorType <- list(title = "Tumor type", labels_gp = gpar(fontsize = legend_fontsize))
    }

    top_annotation <- do.call(HeatmapAnnotation, c(
        col_annotations,
        list(
            col = col_anno_cols,
            show_legend = col_show_legend,
            annotation_legend_param = col_legend_params,
            annotation_name_side = "left",
            annotation_name_gp = gpar(fontsize = column_title_fontsize)
        )
    ))

    compartment_colors <- lookup_colors(feature_annotations$compartment, compartment_palette)
    feature_type_colors <- lookup_colors(feature_annotations$feature_type, feature_type_palette)
    channel_colors <- lookup_colors(feature_annotations$channel, channel_palette)

    right_annotation <- rowAnnotation(
        Compartment = feature_annotations$compartment,
        FeatureType = feature_annotations$feature_type,
        Channel = feature_annotations$channel,
        col = list(Compartment = compartment_colors, FeatureType = feature_type_colors, Channel = channel_colors),
        annotation_legend_param = list(
            Compartment = list(title = "Compartment", labels_gp = gpar(fontsize = legend_fontsize)),
            FeatureType = list(title = "Feature type", labels_gp = gpar(fontsize = legend_fontsize)),
            Channel = list(title = "Channel", labels_gp = gpar(fontsize = legend_fontsize))
        ),
        annotation_name_side = "top",
        annotation_name_gp = gpar(fontsize = column_title_fontsize),
        annotation_name_rot = 45,
        simple_anno_size = unit(3, "mm")
    )

    # Profile values are already normalized upstream, but not necessarily
    # to the same scale in every group/variant -- cap the color scale at
    # this matrix's own 99th percentile absolute value (clipping rarer,
    # more extreme cells to the end colors) rather than the true min/max,
    # so a handful of outliers don't wash out the rest of the heatmap.
    cap <- quantile(abs(mat), 0.99, na.rm = TRUE)
    if (!is.finite(cap) || cap == 0) cap <- 1
    col_fun <- colorRamp2(c(-cap, 0, cap), c("#2166AC", "white", "#B2182B"))

    na_safe_dist <- function(m) {
        m[is.na(m)] <- 0
        stats::dist(m)
    }

    Heatmap(
        mat,
        name = "Value",
        col = col_fun,
        na_col = "grey80",
        show_row_names = FALSE,
        show_column_names = FALSE,
        show_row_dend = cluster_rows,
        show_column_dend = cluster_cols,
        cluster_rows = cluster_rows,
        cluster_columns = cluster_cols,
        clustering_distance_rows = na_safe_dist,
        clustering_distance_columns = na_safe_dist,
        use_raster = TRUE,
        raster_quality = 1,
        right_annotation = right_annotation,
        top_annotation = top_annotation,
        column_title = title,
        column_title_gp = gpar(fontsize = column_title_fontsize, fontface = "bold"),
        heatmap_legend_param = list(
            title = "Normalized\nfeature value",
            title_gp = gpar(fontsize = legend_fontsize, fontface = "bold"),
            labels_gp = gpar(fontsize = legend_fontsize)
        )
    )
}

dose_colors_for <- function(dose_labels) {
    # dose_label_palette (r_plot_themes.r) doesn't cover every dose label
    # seen here (e.g. DMSO's "1%" vehicle dose) -- lookup_colors() falls
    # back to a flat grey for anything missing from it instead of erroring.
    lookup_colors(dose_labels, dose_label_palette)
}

save_heatmaps_pdf <- function(heatmaps, output_path, width, height) {
    # Under the Jupyter/IRkernel evaluate() context, ComplexHeatmap's own
    # draw() unconditionally re-renders into a grid.grabExpr() canvas sized
    # by getOption("repr.plot.width"/"height") -- the notebook's default
    # plot size -- ignoring this function's own (much larger) pdf() device
    # dimensions. For a wide heatmap (e.g. hundreds of samples) that
    # mismatch collapses an internal annotation viewport scale to zero
    # width, erroring "invalid 'xscale' in viewport". Matching these options
    # to the real page size for the duration of the draw avoids it; this is
    # a no-op outside Jupyter (plain Rscript never hits that code path).
    old_opts <- options(repr.plot.width = width, repr.plot.height = height)
    on.exit(options(old_opts), add = TRUE)

    pdf(output_path, width = width, height = height)
    on.exit(dev.off(), add = TRUE)
    for (ht in heatmaps) {
        # draw()'s own newpage=TRUE default silently fails to advance the
        # PDF page when run under the Jupyter/IRkernel evaluate() context
        # (verified: identical code in plain Rscript advances pages fine).
        # Calling grid.newpage() ourselves and telling draw() not to is a
        # reliable workaround in both contexts.
        grid::grid.newpage()
        draw(ht, newpage = FALSE, merge_legend = TRUE)
    }
}

page_size_for <- function(n_samples) {
    # Samples are columns now, so page width (not height) needs to scale
    # with how many there are.
    list(width = max(8, min(30, n_samples / 15 + 4)), height = 10)
}

add_dose_label <- function(metadata) {
    # A few (treatment, dose) combinations exist in some patients' own
    # per-patient files but not in the all_patients file dose_unit_map is
    # built from (e.g. ARV-825 at some patients only) -- treat a missing
    # unit the same as a missing dose rather than let paste0() silently
    # turn it into a literal "NA" suffix (e.g. "1NA").
    missing <- is.na(metadata$Metadata_Experiment_Dose) | is.na(metadata$Metadata_Experiment_Unit)
    metadata$dose_label <- ifelse(
        missing, "None",
        paste0(metadata$Metadata_Experiment_Dose, metadata$Metadata_Experiment_Unit)
    )
    metadata
}

build_all_patients_from_per_patient <- function(per_patient_data) {
    # Builds a combined all-patients matrix from
    # load_2d_per_patient_feature_matrices()'s per-patient entries, keeping
    # only the feature columns common to every patient (each patient's own
    # file can carry a *different* surviving feature set -- see that
    # function). This is the only way to get a real per-row Patient/
    # TumorType breakdown on the 2D all-patients page: the pooled
    # all_patients file itself has no real per-row patient ID.
    common_features <- Reduce(intersect, lapply(per_patient_data, function(e) colnames(e$mat)))
    mat <- do.call(rbind, lapply(per_patient_data, function(e) e$mat[, common_features, drop = FALSE]))
    metadata <- do.call(rbind, lapply(per_patient_data, function(e) e$metadata))

    feature_annotations <- per_patient_data[[1]]$feature_annotations
    feature_annotations <- feature_annotations[match(common_features, feature_annotations$feature), ]

    list(mat = mat, metadata = metadata, feature_annotations = feature_annotations)
}

plot_one_file <- function(input_path, patient_col, feature_parser, rename_map, feature_type_palette,
                           key_label, profile_label, output_prefix, heatmaps, max_n,
                           per_patient_data = "same", all_patients_data = NULL) {
    # Builds the all-patients-pooled clustered/unclustered page pair, plus
    # one clustered/unclustered page pair per patient, for a single profile
    # file -- appending them into the running `heatmaps`/`max_n`
    # accumulators the caller passes in (so one PDF can hold every key's
    # pages).
    #
    # all_patients_data overrides the all-patients page's data source: NULL
    # reads input_path as usual (every 3D file, and 2D consensus, carry
    # real Patient/TumorType there already); a list(mat, metadata,
    # feature_annotations) -- see build_all_patients_from_per_patient --
    # is used instead for 2D agg, whose pooled all_patients file has no
    # real per-row patient ID at all (see load_2d_per_patient_feature_matrices).
    #
    # per_patient_data controls where the per-patient breakdown comes from:
    # "same" derives it from this same file by subsetting rows on
    # patient_col (correct when it already carries a real per-row patient
    # ID, as every 3D file does); a named list (one entry per patient) of
    # list(mat, metadata, feature_annotations) uses those directly instead
    # (2D's pooled all_patients files collapse patient identity to a dummy
    # value, so their per-patient pages are built separately from
    # data/profiles_2D/{patient}/7.aggregated/ -- see
    # load_2d_per_patient_feature_matrices, whose per-patient entries can
    # each have a *different* feature set, hence not just a subset of one
    # shared matrix); NULL skips per-patient pages entirely (2D has no
    # per-patient consensus data on disk at all).
    loaded <- if (!is.null(all_patients_data)) all_patients_data else load_feature_matrix(input_path, patient_col, feature_parser, rename_map)
    loaded$metadata <- add_dose_label(loaded$metadata)
    max_n <- max(max_n, nrow(loaded$mat))

    title_for <- function(suffix, cluster_label) {
        paste0(output_prefix, " ", key_label, " - ", profile_label, " Profiles", suffix, " (", cluster_label, ")")
    }

    # Namespaced by key_label (e.g. the slice strategy or 3D normalization
    # variant): the caller accumulates `heatmaps` across several calls to
    # this function for the SAME patient IDs under different key_labels
    # (plot_2d_group loops over slice strategies, plot_3d_group over
    # normalization variants), and without this prefix each call's pages
    # would silently overwrite the previous call's same-named entries,
    # leaving only the last key_label's pages in the final PDF.
    key_prefix <- gsub("[^A-Za-z0-9]+", "_", key_label)

    heatmaps[[paste0(key_prefix, "_all_patients_clustered")]] <- build_profile_heatmap(
        loaded$mat, loaded$metadata, loaded$feature_annotations, feature_type_palette, cluster = TRUE,
        title = title_for(" - All Patients", "Clustered")
    )
    heatmaps[[paste0(key_prefix, "_all_patients_unclustered")]] <- build_profile_heatmap(
        loaded$mat, loaded$metadata, loaded$feature_annotations, feature_type_palette, cluster = FALSE,
        title = title_for(" - All Patients", "Unclustered")
    )

    add_patient_pages <- function(patient_id, mat, metadata, feature_annotations) {
        # TumorType's own color bar is omitted for a single patient's page
        # (constant, see build_profile_heatmap) -- put it in the title
        # instead, where it's still informative.
        tumor_type <- unique(metadata$Metadata_Biology_TumorType)
        tumor_type <- if (length(tumor_type) == 1 && !is.na(tumor_type)) tumor_type else "Unknown"
        suffix <- paste0(" - ", patient_id, " (", tumor_type, ")")

        heatmaps[[paste0(key_prefix, "_", patient_id, "_clustered")]] <<- build_profile_heatmap(
            mat, metadata, feature_annotations, feature_type_palette, cluster = TRUE,
            title = title_for(suffix, "Clustered")
        )
        heatmaps[[paste0(key_prefix, "_", patient_id, "_unclustered")]] <<- build_profile_heatmap(
            mat, metadata, feature_annotations, feature_type_palette, cluster = FALSE,
            title = title_for(suffix, "Unclustered")
        )
    }

    if (identical(per_patient_data, "same")) {
        for (patient_id in sort(unique(loaded$metadata[[patient_col]]))) {
            keep <- loaded$metadata[[patient_col]] == patient_id
            add_patient_pages(
                patient_id, loaded$mat[keep, , drop = FALSE], loaded$metadata[keep, ], loaded$feature_annotations
            )
        }
    } else if (!is.null(per_patient_data)) {
        for (patient_id in sort(names(per_patient_data))) {
            entry <- per_patient_data[[patient_id]]
            entry$metadata <- add_dose_label(entry$metadata)
            add_patient_pages(patient_id, entry$mat, entry$metadata, entry$feature_annotations)
        }
    }

    list(heatmaps = heatmaps, max_n = max_n)
}

plot_2d_group <- function(group_name, profile_source, key_display_names, output_prefix) {
    # Plots one PDF per profile_type (agg, consensus) for a 2D group (e.g.
    # "2D_sc"): all-patients-pooled clustered/unclustered pages plus
    # per-patient clustered/unclustered pages, for every slice strategy in
    # key_display_names. Reads each slice-strategy/profile_type profile
    # file directly (no precomputed intermediate).
    profile_type_labels <- c(agg = "Aggregate", consensus = "Consensus")
    rename_map_2d <- list(
        Metadata_Experiment_Treatment = "Metadata_treatment",
        Metadata_Biology_PatientTumor = "Metadata_patient_tumor",
        Metadata_Experiment_Dose = "Metadata_dose",
        Metadata_Experiment_Unit = "Metadata_dose_unit"
    )

    for (profile_type in names(profile_type_labels)) {
        profile_label <- profile_type_labels[[profile_type]]
        heatmaps <- list()
        max_n <- 0

        for (slice_strategy in names(key_display_names)) {
            input_path <- file.path(
                root_dir, "data", "profiles_2D", "all_patients", slice_strategy,
                paste0(profile_source, "_", profile_type, "_profiles.parquet")
            )
            # Per-patient pages: for agg, read the real per-patient data
            # (the pooled file above has no real per-row patient ID -- see
            # load_2d_per_patient_feature_matrix); for consensus, there's
            # no per-patient data on disk at all, so skip those pages. The
            # same per-patient data also stands in for the all-patients
            # page's data source for agg, since it's the only way to get a
            # real Patient/TumorType breakdown there too.
            per_patient_data <- if (profile_type == "agg") {
                load_2d_per_patient_feature_matrices(slice_strategy, profile_source, parse_feature_names_2d)
            } else {
                NULL
            }
            all_patients_data <- if (!is.null(per_patient_data) && length(per_patient_data) > 0) {
                build_all_patients_from_per_patient(per_patient_data)
            } else {
                NULL
            }
            result <- plot_one_file(
                input_path, "Metadata_Biology_PatientTumor", parse_feature_names_2d, rename_map_2d,
                feature_type_palette_2D,
                key_display_names[[slice_strategy]], profile_label, output_prefix, heatmaps, max_n,
                per_patient_data = per_patient_data, all_patients_data = all_patients_data
            )
            heatmaps <- result$heatmaps
            max_n <- result$max_n
        }

        size <- page_size_for(max_n)
        output_path <- file.path(figures_dir, paste0(group_name, "_", profile_type, "_profile_heatmaps.pdf"))
        save_heatmaps_pdf(heatmaps, output_path, size$width, size$height)
        cat("Saved:", basename(output_path), "(", length(heatmaps), "pages)\n")
    }
}

plot_3d_group <- function(group_name, variants, variant_display_names, output_prefix) {
    # Same as plot_2d_group, but for a 3D group (e.g. "3D_sc") -- one
    # normalization variant per PDF section instead of one slice strategy.
    profile_type_labels <- c(agg = "Aggregate", consensus = "Consensus")
    profile_dirs_3d <- c(agg = "2.aggregated_profiles", consensus = "3.consensus_profiles")

    for (profile_type in names(profile_type_labels)) {
        profile_label <- profile_type_labels[[profile_type]]
        heatmaps <- list()
        max_n <- 0

        for (variant in variants) {
            input_path <- file.path(
                root_dir, "data", "profiles_3D", "all_patients", profile_dirs_3d[[profile_type]],
                paste0(variant, "_", profile_type, "_from_norm.parquet")
            )
            result <- plot_one_file(
                input_path, "Metadata_Biology_PatientTumor", parse_feature_names_3d, NULL, feature_type_palette,
                variant_display_names[[variant]], profile_label, output_prefix, heatmaps, max_n
            )
            heatmaps <- result$heatmaps
            max_n <- result$max_n
        }

        size <- page_size_for(max_n)
        output_path <- file.path(figures_dir, paste0(group_name, "_", profile_type, "_profile_heatmaps.pdf"))
        save_heatmaps_pdf(heatmaps, output_path, size$width, size$height)
        cat("Saved:", basename(output_path), "(", length(heatmaps), "pages)\n")
    }
}

slice_display_names <- c(
    max_projection = "2D Max Projection",
    middle_slice = "2D Middle Slice",
    middle_n_slice = "2D Middle N Slice"
)

sc_3d_variant_display_names <- c(
    nucleocentric_morphem_norm = "3D Nucleocentric Morphem Norm",
    sammed_nucleocentric_norm = "3D Sammed Nucleocentric Norm",
    sammed_sc_norm = "3D Sammed Sc Norm",
    sc_norm = "3D Sc Norm"
)

organoid_3d_variant_display_names <- c(
    organoid_norm = "3D Organoid Norm",
    sammed_organoid_norm = "3D Sammed Organoid Norm"
)

plot_2d_group("2D_sc", "sc", slice_display_names, "Single-Cell")
plot_2d_group("2D_organoid", "organoid", slice_display_names, "Organoid")
plot_3d_group("3D_sc", names(sc_3d_variant_display_names), sc_3d_variant_display_names, "Single-Cell")
plot_3d_group("3D_organoid", names(organoid_3d_variant_display_names), organoid_3d_variant_display_names, "Organoid")
