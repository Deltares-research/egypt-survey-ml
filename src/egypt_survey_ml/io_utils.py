"""Common IO helpers shared across notebooks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_excel_file(df, excel_path, sheet_name, append=False):
    """Write a dataframe to an Excel sheet, replacing an existing sheet when needed."""
    path = Path(excel_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        writer_kwargs = {"engine": "openpyxl"}
        if path.exists():
            writer_kwargs["mode"] = "a"
            writer_kwargs["if_sheet_exists"] = "replace"
        else:
            writer_kwargs["mode"] = "w"

        with pd.ExcelWriter(path, **writer_kwargs) as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    except Exception:
        # Keep the original notebook fallback behavior if ExcelWriter cannot be used.
        df.to_excel(path, sheet_name=sheet_name, index=False)
