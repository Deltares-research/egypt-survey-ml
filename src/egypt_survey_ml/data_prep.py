"""Shared survey filtering and household-preparation helpers."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .catboost import convert_irregular


def build_household_features(df, hh, avg_hours_by_ind=None):
    """Build the repeated household-level wage and demographic features used across notebooks."""
    if avg_hours_by_ind is None:
        avg_hours_by_ind = df[df["hrswk"] > 0].groupby("ind")["hrswk"].mean()

    household_size = df.groupby("caseser").size()
    n_employed_in_hh = df[df["emps"].notna()].groupby("caseser").size()

    if "rel" in df.columns:
        head_rows = df[df["rel"] == 1].drop_duplicates("caseser")
    else:
        head_rows = df.iloc[0:0].copy()
    head_info = (
        head_rows.set_index("caseser")
        if "caseser" in head_rows.columns
        else head_rows.copy()
    )
    head_fields = [c for c in ["educ", "sex"] if c in head_info.columns]
    if head_fields:
        head_info = head_info[head_fields].copy()
        head_info.columns = [
            "head_educ" if c == "educ" else "head_sex" for c in head_fields
        ]
    else:
        head_info = pd.DataFrame(index=household_size.index)

    head_hours = (
        head_rows.set_index("caseser")["hrswk"]
        if "hrswk" in head_rows.columns and "caseser" in head_rows.columns
        else pd.Series(dtype=float)
    )

    spouses_ind = (
        df[df["rel"] == 2].copy() if "rel" in df.columns else df.iloc[0:0].copy()
    )
    if "pnum" in spouses_ind.columns:
        spouses_ind = spouses_ind.sort_values(["caseser", "pnum"]).copy()
        spouses_ind["spouse_order"] = spouses_ind.groupby("caseser").cumcount() + 1
    else:
        spouses_ind = spouses_ind.copy()
        spouses_ind["spouse_order"] = pd.Series(dtype=int, index=spouses_ind.index)
    spouse_hours = {
        n: (
            spouses_ind[spouses_ind["spouse_order"] == n].set_index("caseser")["hrswk"]
            if "hrswk" in spouses_ind.columns
            else pd.Series(dtype=float)
        )
        for n in [1, 2, 3]
    }

    hh_wage = hh[hh["rururb"] == 0].copy()
    hh_wage["head_hrswk"] = hh_wage["caseser"].map(head_hours)
    hh_wage["irrwagehd"] = convert_irregular(
        (
            hh_wage["irrgwaghd"].fillna(0)
            if "irrgwaghd" in hh_wage.columns
            else pd.Series(0.0, index=hh_wage.index)
        ),
        (
            hh_wage["indhd"]
            if "indhd" in hh_wage.columns
            else pd.Series(0, index=hh_wage.index)
        ),
        hh_wage["head_hrswk"],
        avg_hours_by_ind=avg_hours_by_ind,
    )

    for n in [1, 2, 3]:
        hh_wage[f"sp_{n}_hrswk"] = hh_wage["caseser"].map(spouse_hours[n])
        if f"irrgwagsp_{n}" in hh_wage.columns and f"indsp_{n}" in hh_wage.columns:
            hh_wage[f"irrwagesp_{n}"] = convert_irregular(
                hh_wage[f"irrgwagsp_{n}"].fillna(0),
                hh_wage[f"indsp_{n}"],
                hh_wage[f"sp_{n}_hrswk"],
                avg_hours_by_ind=avg_hours_by_ind,
            )
        else:
            hh_wage[f"irrwagesp_{n}"] = 0.0

    wagehd_cols = [
        c
        for c in ["empinchd", "sempinchd", "totwaghd", "irrwagehd"]
        if c in hh_wage.columns
    ]
    hh_wage["wagehd"] = (
        hh_wage[wagehd_cols].sum(axis=1, skipna=True) if wagehd_cols else 0.0
    )
    sp_cols = [
        c
        for n in [1, 2, 3]
        for p in ["empinc", "sempinc", "totwag", "irrwage"]
        for c in [f"{p}sp_{n}"]
        if c in hh_wage.columns
    ]
    hh_wage["wagesp"] = hh_wage[sp_cols].sum(axis=1, skipna=True) if sp_cols else 0.0
    hh_wage["household_wage_total"] = hh_wage["wagehd"] + hh_wage["wagesp"]
    hh_wage["wage_ratio"] = (
        hh_wage["wagesp"] / hh_wage["household_wage_total"]
    ).fillna(0.5)

    return household_size, n_employed_in_hh, head_info, hh_wage


def filter_rural_agriculture_population(
    df,
    min_working_age=15,
    industry_code=10,
    rural_code=0,
    require_emps=True,
):
    """Filter to working-age rural households in the agriculture sector."""
    working_age = df[df["age"] >= min_working_age].copy()
    if require_emps:
        rural = working_age[
            (working_age["rururb"] == rural_code) & (working_age["emps"].notna())
        ].copy()
    else:
        rural = working_age[working_age["rururb"] == rural_code].copy()
    return rural[rural["ind"] == industry_code].copy()


def split_train_test(pool, test_size=0.2, seed=42):
    """Split a survey pool into train and test sets while preserving the sex stratification."""
    train_idx, test_idx = train_test_split(
        pool.index,
        test_size=test_size,
        random_state=seed,
        stratify=pool["sex"],
    )
    test_ids = set(test_idx)
    train_pool = pool.loc[~pool.index.isin(test_ids)].copy()
    test_pool = pool.loc[list(test_ids)].copy()
    return train_pool, test_pool, test_ids
