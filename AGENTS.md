# Coding-agents guidance (co-developer)
You are responsible for reading, reviewing, and pointing out any issues in the code provided.

Do not suggest reintroducing code that has been deliberately deleted — treat deletions in the diff/history as final, not as omissions to restore.

Provide feedback on the code, including potential improvements, best practices, and any errors you notice.

## Python
- Code must be linted and formatted according to PEP 8 standards.
- Use `pathlib.Path` to define file paths instead of string concatenation or `os`.
- Paths should be defined relative to the project's Git root directory, as returned by `init_notebook()` in `notebook_init_utils.py`.
- Paths should be defined at the top of the notebook/script, not in the middle of the code.

## Data and figures
- All results, intermediate files, data, etc. should be saved as parquet files, not any other format.
- Figures should be saved as `.png` only, with `dpi=600`.

## R / ggplot2
Preferred style: each layer indented on its own line, with the `+` operator at the start of the line.
```R
plot_name <- (
    ggplot(data, aes(x = x_variable, y = y_variable))
    + geom_point()
)
```

## Notebook/script pairing
Every analysis step exists as both a Jupyter notebook (`notebooks/`) and a mirrored plain-text script (`scripts/`, `.py` or `.r`) with the same base name. Keep both in sync — changes to one should be reflected in the other.

## File naming and ordering
Analysis files are numbered by pipeline order (`0.`, `1.`, `2.`, ...), not alphabetically. A new analysis step gets the next unused integer prefix, and calculation steps are typically followed by a corresponding plotting step (e.g. `N.calculate_x.py` / `N+1.plot_x.r`).
