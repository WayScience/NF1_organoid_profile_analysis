# Skip-if-unchanged result caching for analysis steps (R counterpart of
# utils/src/notebook_init_utils/result_cache.py).
#
# A cached output is reused only when every one of the following holds:
#   - the output file exists,
#   - the step's code file (its mirrored script) has not changed, and
#   - every input file has the same path, size and modification time as when
#     the output was last written.
#
# Each step keeps <results_dir>/_cache_manifests/<step>.parquet, which maps
# each output to the fingerprint it was produced from. Requires the arrow
# package. Set RESULTS_CACHE_FORCE=1 to ignore every cache and recompute.

result_cache_fingerprint <- function(code_path, inputs) {
    input_paths <- sort(normalizePath(inputs, mustWork = TRUE))
    info <- file.info(input_paths)
    signatures <- sprintf(
        "%s|%s|%.6f",
        input_paths,
        info$size,
        as.numeric(info$mtime)
    )
    code_md5 <- unname(tools::md5sum(code_path))

    text_path <- tempfile()
    on.exit(unlink(text_path), add = TRUE)
    writeLines(c(code_md5, signatures), text_path)
    unname(tools::md5sum(text_path))
}

result_cache_manifest_path <- function(step, results_dir) {
    file.path(results_dir, "_cache_manifests", paste0(step, ".parquet"))
}

result_cache_read_manifest <- function(manifest_path) {
    if (!file.exists(manifest_path)) {
        return(setNames(character(0), character(0)))
    }
    manifest <- as.data.frame(arrow::read_parquet(manifest_path))
    setNames(manifest$fingerprint, manifest$output)
}

#' Return TRUE if all outputs exist and were produced from the current code and inputs.
result_cache_is_fresh <- function(step, results_dir, code_path, outputs, inputs) {
    if (identical(Sys.getenv("RESULTS_CACHE_FORCE"), "1")) {
        return(FALSE)
    }
    recorded <- result_cache_read_manifest(result_cache_manifest_path(step, results_dir))
    current <- result_cache_fingerprint(code_path, inputs)
    recorded_for_outputs <- unname(recorded[outputs])
    all(file.exists(outputs) & !is.na(recorded_for_outputs) & recorded_for_outputs == current)
}

#' Record that the outputs were just written from the current code and inputs.
#' Call this immediately after each output is written.
result_cache_mark_fresh <- function(step, results_dir, code_path, outputs, inputs) {
    manifest_path <- result_cache_manifest_path(step, results_dir)
    recorded <- result_cache_read_manifest(manifest_path)
    current <- result_cache_fingerprint(code_path, inputs)
    recorded[outputs] <- current

    dir.create(dirname(manifest_path), recursive = TRUE, showWarnings = FALSE)
    arrow::write_parquet(
        data.frame(
            output = names(recorded),
            fingerprint = unname(recorded),
            stringsAsFactors = FALSE
        ),
        manifest_path
    )
}
