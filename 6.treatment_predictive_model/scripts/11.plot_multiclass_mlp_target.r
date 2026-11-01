list_of_packages <- c("ggplot2", "dplyr", "tidyr", "arrow", "RColorBrewer")
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

module_dir <- file.path(root_dir, "6.treatment_predictive_model")
results_dir <- file.path(module_dir, "model_results")
figures_dir <- file.path(module_dir, "figures")
dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)

TASK_PREFIX <- "multiclass_mlp_target"

options(repr.plot.width = 7, repr.plot.height = 5)

training_curve_df <- read_parquet(
    file.path(results_dir, paste0(TASK_PREFIX, "_training_curve.parquet"))
) %>%
    pivot_longer(cols = c(train_loss, val_loss), names_to = "split", values_to = "loss")

training_curve_plot <- (
    ggplot(training_curve_df, aes(x = epoch, y = loss, colour = split))
    + geom_line(linewidth = 0.8)
    + scale_colour_manual(values = c(train_loss = "#3B6FA0", val_loss = "#E08214"), name = NULL)
    + labs(x = "Epoch", y = "Cross-entropy loss", title = "Target/MOA MLP training curve")
    + theme_manuscript(base_size = 14)
)

ggsave(
    file.path(figures_dir, paste0(TASK_PREFIX, "_training_curve.png")),
    training_curve_plot, width = 7, height = 5, dpi = 600, bg = "white"
)
training_curve_plot

options(repr.plot.width = 7, repr.plot.height = 5)

top_k_df <- read_parquet(
    file.path(results_dir, paste0(TASK_PREFIX, "_top_k_accuracy.parquet"))
) %>%
    mutate(
        k = factor(k),
        Metadata_split = factor(Metadata_split, levels = c("train", "test"))
    )

top_k_plot <- (
    ggplot(top_k_df, aes(x = k, y = top_k_accuracy, fill = Metadata_split))
    + geom_col(position = position_dodge(width = 0.7), width = 0.6)
    + geom_text(
        aes(label = sprintf("%.2f", top_k_accuracy)),
        position = position_dodge(width = 0.7), vjust = -0.5, size = 3.5
    )
    + scale_fill_manual(values = c(train = "#3B6FA0", test = "#E08214"), name = "Split")
    + scale_y_continuous(limits = c(0, 1), labels = scales::percent)
    + labs(x = "k", y = "Top-k accuracy", title = "Target/MOA top-k accuracy (train vs test)")
    + theme_manuscript(base_size = 14)
)

ggsave(
    file.path(figures_dir, paste0(TASK_PREFIX, "_top_k_accuracy.png")),
    top_k_plot, width = 7, height = 5, dpi = 600, bg = "white"
)
top_k_plot

options(repr.plot.width = 9, repr.plot.height = 6)

per_class_df <- read_parquet(
    file.path(results_dir, paste0(TASK_PREFIX, "_classification_report.parquet"))
) %>%
    filter(!Metadata_class %in% c("macro avg", "weighted avg"))

class_order <- per_class_df %>%
    filter(Metadata_split == "test") %>%
    arrange(`f1-score`) %>%
    pull(Metadata_class)

per_class_df <- per_class_df %>%
    mutate(
        Metadata_class = factor(Metadata_class, levels = class_order),
        Metadata_split = factor(Metadata_split, levels = c("train", "test"))
    )

per_class_f1_plot <- (
    ggplot(per_class_df, aes(x = Metadata_class, y = `f1-score`, fill = Metadata_split))
    + geom_col(position = position_dodge(width = 0.7), width = 0.6)
    + coord_flip()
    + scale_fill_manual(values = c(train = "#3B6FA0", test = "#E08214"), name = "Split")
    + labs(x = NULL, y = "F1 score", title = "Target/MOA per-class F1 (train vs test)")
    + theme_manuscript(base_size = 12)
)

ggsave(
    file.path(figures_dir, paste0(TASK_PREFIX, "_per_class_f1.png")),
    per_class_f1_plot, width = 9, height = 6, dpi = 600, bg = "white"
)
per_class_f1_plot

options(repr.plot.width = 16, repr.plot.height = 9)

predictions_df <- read_parquet(
    file.path(results_dir, paste0(TASK_PREFIX, "_predictions.parquet"))
)
class_levels <- sort(unique(c(
    predictions_df$Metadata_true_label, predictions_df$Metadata_predicted_label
)))

confusion_df <- predictions_df %>%
    count(Metadata_split, Metadata_true_label, Metadata_predicted_label, name = "n") %>%
    tidyr::complete(
        Metadata_split, Metadata_true_label = class_levels, Metadata_predicted_label = class_levels,
        fill = list(n = 0)
    ) %>%
    mutate(
        Metadata_true_label = factor(Metadata_true_label, levels = class_levels),
        Metadata_predicted_label = factor(Metadata_predicted_label, levels = class_levels),
        Metadata_split = factor(Metadata_split, levels = c("train", "test"))
    )

confusion_matrix_plot <- (
    ggplot(confusion_df, aes(x = Metadata_predicted_label, y = Metadata_true_label, fill = n))
    + geom_tile()
    + geom_text(aes(label = n), size = 2.5)
    + scale_fill_distiller(palette = "Blues", direction = 1, name = "Count")
    + scale_y_discrete(limits = rev(class_levels))
    + facet_wrap(~Metadata_split)
    + labs(
        x = "Predicted label", y = "True label",
        title = "Target/MOA confusion matrix (raw counts, train vs test)"
    )
    + theme_manuscript(base_size = 10, x_text = "angled")
    + theme(panel.grid = element_blank())
)

ggsave(
    file.path(figures_dir, paste0(TASK_PREFIX, "_confusion_matrix.png")),
    confusion_matrix_plot, width = 16, height = 9, dpi = 600, bg = "white"
)
confusion_matrix_plot
