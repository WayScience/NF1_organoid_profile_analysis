list_of_packages <- c("arrow", "dplyr", "tidyr", "PMA")

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
cat("Git root directory:", root_dir, "\n")

results_dir <- file.path(root_dir, "2.2d_vs_3d_analysis/results/sparse_cca")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

# Data pathing
path_2d_organoid <- file.path(root_dir, "data/profiles_2D/all_patients/max_projection/organoid_agg_profiles.parquet")
path_2d_sc <- file.path(root_dir, "data/profiles_2D/all_patients/max_projection/sc_agg_profiles.parquet")
dir_3d_agg <- file.path(root_dir, "data/profiles_3D/all_patients/2.aggregated_profiles")

profile_pairs <- data.frame(
    label = c(
        "Organoid - Handcrafted",
        "Organoid - Deep Learning (SAM-Med3D)",
        "Single-cell - Handcrafted",
        "Single-cell - Deep Learning (SAM-Med3D)",
        "Single-cell - Nucleocentric DL (SAM-Med3D)",
        "Single-cell - Nucleocentric DL (MorphEM)"
    ),
    slug = c(
        "organoid_handcrafted",
        "organoid_sammed",
        "sc_handcrafted",
        "sc_sammed",
        "sc_sammed_nucleocentric",
        "sc_nucleocentric_morphem"
    ),
    path_2d = c(
        path_2d_organoid,
        path_2d_organoid,
        path_2d_sc,
        path_2d_sc,
        path_2d_sc,
        path_2d_sc
    ),
    path_3d = c(
        file.path(dir_3d_agg, "organoid_norm_sc_agg_profiles.parquet"),
        file.path(dir_3d_agg, "sammed_organoid_norm_sc_agg_profiles.parquet"),
        file.path(dir_3d_agg, "sc_norm_sc_agg_profiles.parquet"),
        file.path(dir_3d_agg, "sammed_sc_norm_sc_agg_profiles.parquet"),
        file.path(dir_3d_agg, "sammed_nucleocentric_norm_sc_agg_profiles.parquet"),
        file.path(dir_3d_agg, "nucleocentric_morphem_norm_sc_agg_profiles.parquet")
    ),
    stringsAsFactors = FALSE
)

# Standardizes 2D/3D metadata column names and matches samples present in both views.
load_and_match <- function(path_2d, path_3d) {
    df_2d <- arrow::read_parquet(path_2d)
    df_3d <- arrow::read_parquet(path_3d)

    df_3d <- df_3d %>%
        rename(
            Metadata_patient_tumor = Metadata_Biology_PatientTumor,
            Metadata_Well = Metadata_Experiment_Well
        )

    df_2d$Metadata_patient <- sub("_T[0-9].*$", "", df_2d$Metadata_patient_tumor)
    df_3d$Metadata_patient <- sub("_T[0-9].*$", "", df_3d$Metadata_patient_tumor)

    merge_cols <- c("Metadata_patient_tumor", "Metadata_Well")

    keys_2d <- df_2d %>% select(all_of(merge_cols)) %>% distinct()
    keys_3d <- df_3d %>% select(all_of(merge_cols)) %>% distinct()
    shared_keys <- inner_join(keys_2d, keys_3d, by = merge_cols)

    if (nrow(shared_keys) == 0) {
        stop("No matched patient_tumor-well combinations between 2D and 3D.")
    }

    df_2d_matched <- df_2d %>%
        inner_join(shared_keys, by = merge_cols) %>%
        arrange(across(all_of(merge_cols)))
    df_3d_matched <- df_3d %>%
        inner_join(shared_keys, by = merge_cols) %>%
        arrange(across(all_of(merge_cols)))

    stopifnot(identical(
        df_2d_matched %>% select(all_of(merge_cols)),
        df_3d_matched %>% select(all_of(merge_cols))
    ))

    metadata <- df_2d_matched %>% select(Metadata_patient, all_of(merge_cols))

    list(df_2d = df_2d_matched, df_3d = df_3d_matched, metadata = metadata)
}

# TEMPORARY - remove once ZedProfiler's Texture-feature computation is fixed
# upstream. Both handcrafted pairs currently have a single sample with
# astronomically large corrupted feature values (visible as a single outlier
# point at ~1e23-1e94 in the score scatter plots), which by itself drags the
# canonical correlation around. This masks those values to NA so prep_features'
# existing mean-imputation covers them instead; for every other pair (DL,
# nucleocentric) this is a no-op.
mask_extreme_values <- function(df) {
    meta_cols <- grep("^Metadata_", colnames(df), value = TRUE)
    feat_cols <- setdiff(colnames(df), meta_cols)
    df[feat_cols] <- lapply(df[feat_cols], function(x) {
        if (is.numeric(x)) x[abs(x) > 1e10] <- NA
        x
    })
    df
}

# Cleans a profile dataframe into a numeric feature matrix ready for CCA:
# drops metadata/constant/all-NA columns, then mean-imputes remaining NAs.
prep_features <- function(df) {
    meta_cols <- grep("^Metadata_", colnames(df), value = TRUE)
    feat_df <- df %>% select(-all_of(meta_cols))

    numeric_cols <- sapply(feat_df, is.numeric)
    feat_df <- feat_df[, numeric_cols, drop = FALSE]

    # drop constant columns (zero variance, uninformative for CCA)
    constant_cols <- sapply(feat_df, function(x) length(unique(na.omit(x))) <= 1)
    feat_df <- feat_df[, !constant_cols, drop = FALSE]

    # drop all-NA columns, then mean-impute remaining NAs
    # (PMA::CCA requires complete matrices, no NA support)
    all_na_cols <- sapply(feat_df, function(x) all(is.na(x)))
    feat_df <- feat_df[, !all_na_cols, drop = FALSE]
    feat_df <- as.data.frame(lapply(feat_df, function(x) {
        if (any(is.na(x))) x[is.na(x)] <- mean(x, na.rm = TRUE)
        x
    }))

    feat_df
}

# Fits sparse CCA for one profile-type pair (penalty chosen via permutation,
# K=1 since only the first component gets a validated p-value)
run_sparse_cca <- function(mat_2d, mat_3d, metadata, slug, results_dir) {
    set.seed(0)
    n_components <- 1

    perm_out <- CCA.permute(
        x = mat_2d,
        z = mat_3d,
        typex = "standard",
        typez = "standard",
        standardize = TRUE,
        nperms = 25 # CCA.permute's own package default
    )

    permute_summary <- data.frame(
        penaltyx = perm_out$penaltyxs,
        penaltyz = perm_out$penaltyzs,
        cor = perm_out$cors,
        zstat = perm_out$zstats,
        pval = perm_out$pvals,
        n_nonzero_2d = perm_out$nnonzerous,
        n_nonzero_3d = perm_out$nnonzerovs
    )
    permute_summary$is_best <- (permute_summary$penaltyx == perm_out$bestpenaltyx) &
        (permute_summary$penaltyz == perm_out$bestpenaltyz)
    arrow::write_parquet(permute_summary, file.path(results_dir, paste0(slug, "_cca_permute_summary.parquet")))

    cca_out <- CCA(
        x = mat_2d,
        z = mat_3d,
        typex = "standard",
        typez = "standard",
        penaltyx = perm_out$bestpenaltyx,
        penaltyz = perm_out$bestpenaltyz,
        K = n_components,
        standardize = TRUE
    )

    scores_2d <- mat_2d %*% cca_out$u
    scores_3d <- mat_3d %*% cca_out$v
    colnames(scores_2d) <- paste0("CC", seq_len(n_components), "_2D")
    colnames(scores_3d) <- paste0("CC", seq_len(n_components), "_3D")

    canonical_scores <- cbind(metadata, as.data.frame(scores_2d), as.data.frame(scores_3d))
    arrow::write_parquet(canonical_scores, file.path(results_dir, paste0(slug, "_canonical_scores.parquet")))

    canonical_cors <- sapply(seq_len(n_components), function(k) {
        cor(scores_2d[, k], scores_3d[, k])
    })
    best_idx <- which(permute_summary$is_best)[1]
    cor_summary <- data.frame(
        component = paste0("CC", seq_len(n_components)),
        canonical_correlation = canonical_cors,
        perm_pvalue = c(permute_summary$pval[best_idx], rep(NA, n_components - 1))
    )
    arrow::write_parquet(cor_summary, file.path(results_dir, paste0(slug, "_canonical_correlations.parquet")))

    loadings_2d <- as.data.frame(cca_out$u)
    colnames(loadings_2d) <- paste0("CC", seq_len(n_components))
    loadings_2d$feature <- colnames(mat_2d)
    loadings_2d_long <- loadings_2d %>%
        tidyr::pivot_longer(cols = starts_with("CC"), names_to = "component", values_to = "weight")
    arrow::write_parquet(loadings_2d_long, file.path(results_dir, paste0(slug, "_loadings_2d.parquet")))

    loadings_3d <- as.data.frame(cca_out$v)
    colnames(loadings_3d) <- paste0("CC", seq_len(n_components))
    loadings_3d$feature <- colnames(mat_3d)
    loadings_3d_long <- loadings_3d %>%
        tidyr::pivot_longer(cols = starts_with("CC"), names_to = "component", values_to = "weight")
    arrow::write_parquet(loadings_3d_long, file.path(results_dir, paste0(slug, "_loadings_3d.parquet")))

    cor_summary
}

# Run sparse CCA and save results
summary_rows <- list()

for (i in seq_len(nrow(profile_pairs))) {
    pair <- profile_pairs[i, ]
    cat("===", pair$label, "===\n")

    matched <- load_and_match(pair$path_2d, pair$path_3d)
    mat_2d <- as.matrix(prep_features(mask_extreme_values(matched$df_2d)))  # TEMPORARY - remove once upstream data is fixed
    mat_3d <- as.matrix(prep_features(mask_extreme_values(matched$df_3d)))  # TEMPORARY - remove once upstream data is fixed
    cat("Matched samples:", nrow(mat_2d), " | 2D features:", ncol(mat_2d), " | 3D features:", ncol(mat_3d), "\n")

    result <- run_sparse_cca(mat_2d, mat_3d, matched$metadata, pair$slug, results_dir)
    result$label <- pair$label
    result$slug <- pair$slug
    summary_rows[[pair$slug]] <- result

    cat("r =", round(result$canonical_correlation, 3), " p =", result$perm_pvalue, "\n\n")
}

all_pairs_summary <- do.call(rbind, summary_rows)
rownames(all_pairs_summary) <- NULL
arrow::write_parquet(all_pairs_summary, file.path(results_dir, "all_pairs_summary.parquet"))

cat("Sparse CCA results for all pairs saved to:", results_dir, "\n")
print(all_pairs_summary)
