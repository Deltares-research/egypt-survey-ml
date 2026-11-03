"""Build the rural-agriculture wage dataset used by the prediction notebooks.

All functions take their inputs explicitly (no notebook globals). Scenario settings
come from :class:`egypt_survey_ml.config.DatasetConfig`; :func:`prepare_dataset` runs the
whole pipeline and returns a :class:`Dataset` holding every intermediate object.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DatasetConfig
from .paths import DataPaths, resolve_data_paths

# Columns that are never model features.
TARGET_COLS = ("log1p_wage", "totwag_all", "is_paid", "command_unit")

SEX_CODES = {"male": 1, "female": 2}


# --------------------------------------------------------------------------- loading
def load_survey(paths: DataPaths):
    """Read the individual-level and household-level LFS files."""
    df = pd.read_stata(paths.lfs_dta, convert_categoricals=False)
    hh = pd.read_stata(paths.hh_dta, convert_categoricals=False)
    print("individual shape:", df.shape, " household shape:", hh.shape)
    return df, hh


# ------------------------------------------------------------------ household context
def convert_irregular(daily_wage, ind_code, real_hours, avg_hours_by_ind):
    """Convert a daily irregular wage to a monthly amount, falling back to industry-average hours."""
    used_real = real_hours.notna() & (real_hours > 0)
    hours = real_hours.where(used_real, ind_code.map(avg_hours_by_ind))
    return daily_wage * (hours / 8) * 4


@dataclass
class HouseholdContext:
    household_size: pd.Series
    n_employed_in_hh: pd.Series
    head_info: pd.DataFrame
    avg_hours_by_ind: pd.Series
    hh_wage: pd.DataFrame


def build_household_context(df, hh) -> HouseholdContext:
    """Household size, household head info, and household wage totals / spouse wage ratio."""
    household_size = df.groupby("caseser").size()
    n_employed_in_hh = df[df["emps"].notna()].groupby("caseser").size()
    head_info = (
        df[df["rel"] == 1]
        .drop_duplicates("caseser")
        .set_index("caseser")[["educ", "sex"]]
    )
    head_info.columns = ["head_educ", "head_sex"]

    avg_hours_by_ind = df[df["hrswk"] > 0].groupby("ind")["hrswk"].mean()
    head_hours = (
        df[df["rel"] == 1].drop_duplicates("caseser").set_index("caseser")["hrswk"]
    )
    spouses_ind = df[df["rel"] == 2].sort_values(["caseser", "pnum"]).copy()
    spouses_ind["spouse_order"] = spouses_ind.groupby("caseser").cumcount() + 1
    spouse_hours = {
        n: spouses_ind[spouses_ind["spouse_order"] == n].set_index("caseser")["hrswk"]
        for n in [1, 2, 3]
    }

    hh_wage = hh[hh["rururb"] == 0].copy()
    hh_wage["head_hrswk"] = hh_wage["caseser"].map(head_hours)
    hh_wage["irrwagehd"] = convert_irregular(
        hh_wage["irrgwaghd"].fillna(0),
        hh_wage["indhd"],
        hh_wage["head_hrswk"],
        avg_hours_by_ind,
    )
    for n in [1, 2, 3]:
        hh_wage[f"sp_{n}_hrswk"] = hh_wage["caseser"].map(spouse_hours[n])
        hh_wage[f"irrwagesp_{n}"] = convert_irregular(
            hh_wage[f"irrgwagsp_{n}"].fillna(0),
            hh_wage[f"indsp_{n}"],
            hh_wage[f"sp_{n}_hrswk"],
            avg_hours_by_ind,
        )
    hh_wage["wagehd"] = hh_wage[["empinchd", "sempinchd", "totwaghd", "irrwagehd"]].sum(
        axis=1, skipna=True
    )
    sp_cols = [
        f"{p}sp_{n}"
        for n in [1, 2, 3]
        for p in ["empinc", "sempinc", "totwag", "irrwage"]
    ]
    hh_wage["wagesp"] = hh_wage[sp_cols].sum(axis=1, skipna=True)
    hh_wage["household_wage_total"] = hh_wage["wagehd"] + hh_wage["wagesp"]
    hh_wage["wage_ratio"] = (
        hh_wage["wagesp"] / hh_wage["household_wage_total"]
    ).fillna(0.5)
    print(
        "household_wage_total / wage_ratio computed for",
        len(hh_wage),
        "rural households",
    )
    return HouseholdContext(
        household_size, n_employed_in_hh, head_info, avg_hours_by_ind, hh_wage
    )


# ------------------------------------------------------------- population / split
def _sex_counts(frame):
    return f"({(frame['sex'] == 1).sum()} male, {(frame['sex'] == 2).sum()} female)"


def select_rural_agriculture(df, min_working_age):
    """Rural, working-age, employed respondents in agriculture (ind == 10)."""
    n_underage = (df["age"] < min_working_age).sum()
    working_age = df[df["age"] >= min_working_age].copy()
    rural = working_age[
        (working_age["rururb"] == 0) & (working_age["emps"].notna())
    ].copy()
    print(
        f"Dropped {n_underage} under age {min_working_age}. "
        f"Rural, working-age, employed: {len(rural)} rows {_sex_counts(rural)}"
    )

    agri_pool = rural[rural["ind"] == 10].copy()
    print(
        f"Rural + agriculture population: {len(agri_pool)} rows {_sex_counts(agri_pool)}"
    )
    return agri_pool


def split_train_test(agri_pool, seed, test_size=0.2):
    """Sex-stratified train/test split. Returns (train_pool, test_pool, test_ids)."""
    _, test_idx = train_test_split(
        agri_pool.index,
        test_size=test_size,
        random_state=seed,
        stratify=agri_pool["sex"],
    )
    test_ids = set(test_idx)
    test_pool = agri_pool.loc[list(test_ids)].copy()
    train_pool = agri_pool.loc[~agri_pool.index.isin(test_ids)].copy()

    assert test_ids.isdisjoint(
        set(train_pool.index)
    ), "test/train overlap -- must never happen"
    print(
        f"\nTEST (rural + agriculture): {len(test_pool)} rows {_sex_counts(test_pool)}"
    )
    print(
        f"TRAIN (rural + agriculture, test rows excluded): {len(train_pool)} rows "
        f"{_sex_counts(train_pool)}"
    )
    return train_pool, test_pool, test_ids


# ----------------------------------------------------------------- wage target
def attach_household_context(pool, ctx: HouseholdContext, hh):
    pool = pool.copy()
    pool["household_size"] = pool["caseser"].map(ctx.household_size)
    pool["n_other_employed"] = pool["caseser"].map(ctx.n_employed_in_hh) - 1
    pool = pool.merge(ctx.head_info, on="caseser", how="left")
    pool = pool.merge(
        hh[["caseser", "malinlf", "feminlf", "occhd"]], on="caseser", how="left"
    )
    pool = pool.merge(
        ctx.hh_wage[["caseser", "household_wage_total", "wage_ratio"]],
        on="caseser",
        how="left",
    )
    return pool


def build_wage_target(pool):
    """Monthly wage target (`totwag_all`), paid flag and log wage, with outlier cap at 10,000 EGP."""
    pool = pool.copy()
    valid_hours = pool["hrswk"].notna() & (pool["hrswk"] > 0)
    pool = pool[valid_hours].copy()

    # immigr: 3 = "No" migration (modal category); disabl: 0 = no disability.
    # mart_d has no natural default, so rows missing it are dropped.
    if "immigr" in pool.columns:
        pool["immigr"] = pool["immigr"].fillna(3)
    if "disabl" in pool.columns:
        pool["disabl"] = pool["disabl"].fillna(0)
    if "mart_d" in pool.columns:
        n_before = len(pool)
        pool = pool.dropna(subset=["mart_d"])
        print(f"  dropped {n_before - len(pool)} rows missing mart_d")

    # Same wage formula as the train-catboost*.ipynb notebooks: totwag + sempinc + irrgwag*22.
    pool["sempinc"] = pool["sempinc"].fillna(0)
    pool["irrgwag"] = pool["irrgwag"].fillna(0)
    pool["totwag"] = pool["totwag"].fillna(0)
    pool["totwag_all"] = pool["totwag"] + pool["sempinc"] + pool["irrgwag"] * 22

    # Without the cap a handful of extreme outliers (up to ~108,705 EGP/month) dominate R2.
    n_before = len(pool)
    pool = pool[pool["totwag_all"] < 10000].copy()
    print(
        f"  dropped {n_before - len(pool)} rows with totwag_all >= 10,000 EGP (outlier cap)"
    )

    pool["is_paid"] = (pool["totwag_all"] > 0).astype(int)
    # Only used by lognormal_synthetic_augment; not the CatBoost fit target.
    pool["log1p_wage"] = np.log1p(pool["totwag_all"])
    return pool


def describe_pools(train_pool, test_pool):
    for label, pool in [("TRAIN", train_pool), ("TEST  (rural+agri only)", test_pool)]:
        print(
            f"{label}: {len(pool)} rows, {pool['is_paid'].mean():.1%} paid overall "
            f"(male {pool.loc[pool['sex'] == 1, 'is_paid'].mean():.1%}, "
            f"female {pool.loc[pool['sex'] == 2, 'is_paid'].mean():.1%})"
        )


# ------------------------------------------------------------------ command units
def attach_governorate(pool, crosswalk):
    return pool.merge(
        crosswalk[["reg_code", "OBJECTID", "NAME1_"]],
        left_on="reg",
        right_on="reg_code",
        how="left",
    )


def assign_command_units(pool, seed, distribution_df):
    """Draw one random command unit per respondent from the (rural-population-weighted)
    governorate distribution. Rows landing on 'other' (outside every command area) are
    dropped -- matches assign-survey-data-to-command-units.ipynb."""
    rng = np.random.default_rng(seed)
    pool = pool.copy()
    pool["command_unit"] = None
    for gov, idx in pool.groupby("NAME1_").groups.items():
        if gov not in distribution_df.index:
            continue
        probs = distribution_df.loc[gov].drop("other")
        if probs.sum() == 0:
            continue
        probs = probs / probs.sum()

        pool.loc[idx, "command_unit"] = rng.choice(
            probs.index,
            size=len(idx),
            p=probs.values,
        )
    pool = pool[pool["command_unit"] != "other"]
    return pool[pool["command_unit"].notna()].copy()


# ---------------------------------------------------------------- feature selection
@dataclass
class FeatureSpec:
    base_cols: list
    categorical_columns: list
    male_extra: list
    female_extra: list
    extra_features: list

    def columns_for(self, sex_code):
        return self.base_cols + (
            self.male_extra if sex_code == 1 else self.female_extra
        )


def select_features(masking_df, agri_pool, config: DatasetConfig) -> FeatureSpec:
    base_cols = masking_df.loc[masking_df["correl"] == 1, "variable"].tolist()
    # hrswk/hrswksc are added separately below, so strip them here to avoid duplicates.
    base_cols = [
        c for c in base_cols if c not in ("emps", "hrswk", "hrswksc", "totwag")
    ]
    if not config.include_empstab:
        base_cols = [c for c in base_cols if c != "empstab"]

    # rel, attsch, occ aren't flagged in the mask file but carry real, non-wage-derived
    # signal; added for both sexes.
    extra_features = [c for c in ["rel", "attsch", "occ"] if c in agri_pool.columns]
    base_cols = base_cols + [c for c in extra_features if c not in base_cols]

    hrs = ["hrswk", "hrswksc"] if config.include_hrswk else []

    categorical_columns = masking_df[
        (masking_df["type"].isin(["binary", "categorical"]))
        & (masking_df["variable"].isin(base_cols))
    ]["variable"].tolist()
    # rel, attsch, occ are nominal codes; the mask file lists their type as "-".
    categorical_columns = categorical_columns + [
        c for c in extra_features if c not in categorical_columns
    ]

    print(f"Base feature list: {len(base_cols)} columns")
    print(f"Categorical: {categorical_columns}")
    print(
        f"INCLUDE_EMPSTAB={config.include_empstab}  INCLUDE_HRSWK={config.include_hrswk}  "
        f"ZERO_HANDLING={config.zero_handling!r}"
    )
    return FeatureSpec(
        base_cols=base_cols,
        categorical_columns=categorical_columns,
        male_extra=list(hrs),
        female_extra=list(hrs),
        extra_features=extra_features,
    )


# -------------------------------------------------------------- zero-wage handling
def drop_random_zeros(frame, frac, seed):
    rng = np.random.default_rng(seed)
    zero_idx = frame.index[frame["is_paid"] == 0]
    n_drop = int(len(zero_idx) * frac)
    drop_idx = rng.choice(zero_idx, size=n_drop, replace=False)
    return frame.drop(index=drop_idx)


def perturbate_wage(frame, seed, mu=0.0, sigma=0.03):
    rng = np.random.default_rng(seed)
    wage = frame["totwag_all"]
    frame["totwag_all"] = wage * (1 + rng.normal(mu, sigma, size=wage.shape))
    return frame


def geo_columns(excel_df):
    return ["command_unit"] + [c for c in excel_df.columns if c != "area"]


def lognormal_synthetic_augment(
    frame, feature_cols, cat_cols, n_synthetic, geo_cols, k=5, seed=42
):
    """Turn real zero-wage rows into hypothetical paid rows.

    - command_unit and every command-area covariate (``geo_cols``): copied AS-IS from the
      donor row's own real command unit. Never blended across command units.
    - Other numeric individual-level columns: SMOGN-style k-NN interpolation between two
      REAL zero-wage donor rows, restricted to neighbors within the SAME command unit.
    - Categorical individual-level columns: copied from whichever of the two donors the
      interpolation weight favors (can't blend a category).
    - Wage: NOT interpolated. Drawn from a log-normal fit conditioned on the donor's own
      command unit (median real log-wage of PAID people there + a residual drawn from the
      overall real residual spread).
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    paid = frame[frame["is_paid"] == 1]
    if len(paid) < 5:
        raise ValueError("too few real paid rows to fit a conditional log-normal")

    overall_resid_std = (
        paid["log1p_wage"]
        - paid.groupby("command_unit")["log1p_wage"].transform("median")
    ).std()
    cu_median_log_wage = paid.groupby("command_unit")["log1p_wage"].median()
    global_median_log_wage = paid["log1p_wage"].median()

    donor_pool = frame[frame["is_paid"] == 0].reset_index(drop=True)
    geo_cols_present = [c for c in geo_cols if c in donor_pool.columns]
    num_cols = [
        c for c in feature_cols if c not in cat_cols and c not in geo_cols_present
    ]

    synth_rows = []
    for cu, group in donor_pool.groupby("command_unit"):
        if len(group) < 2:
            continue
        X_num = group[num_cols].fillna(np.nanmedian(group[num_cols])).to_numpy()
        X_scaled = StandardScaler().fit_transform(X_num)
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(group))).fit(X_scaled)
        _, neighbor_idx = nn.kneighbors(X_scaled)

        n_this_cu = max(1, round(n_synthetic * len(group) / len(donor_pool)))
        for _ in range(n_this_cu):
            i = rng.integers(0, len(group))
            j = rng.choice(neighbor_idx[i][1:]) if len(group) > 1 else i
            lam = rng.uniform(0, 1)
            # geo + categorical columns come from this donor, untouched
            row = group.iloc[i if lam < 0.5 else j].copy()
            row[num_cols] = group.iloc[i][num_cols].to_numpy() + lam * (
                group.iloc[j][num_cols].to_numpy() - group.iloc[i][num_cols].to_numpy()
            )
            mu = cu_median_log_wage.get(cu, global_median_log_wage)
            synth_log_wage = mu + rng.normal(0, overall_resid_std)
            row["log1p_wage"] = max(synth_log_wage, 0.0)
            row["totwag_all"] = np.expm1(row["log1p_wage"])
            row["is_paid"] = 1
            synth_rows.append(row)

    synth_df = pd.DataFrame(synth_rows)
    print(
        f"  lognormal_synthetic_augment: generated {len(synth_df)} synthetic paid rows "
        f"\nfrom {donor_pool['command_unit'].nunique()} command units"
        f"\ntotal amount of orginal rows: {len(frame)} "
        f"({len(synth_df) / len(frame) * 100:.1f}% synthetic data)"
    )
    return pd.concat([frame, synth_df], ignore_index=True)


# ------------------------------------------------------------------------ pipeline
@dataclass
class Realization:
    """The test set of one sex under one random command-unit draw (one seed).

    The respondents, their wages and survey weights are the same for every seed; only
    ``command_unit`` (and the command-area covariates in ``frame``) change between draws.
    """

    seed: int
    sex_code: int
    frame: pd.DataFrame  # model features + targets, outlier cap applied
    weight: np.ndarray  # survey weight per respondent
    command_unit: np.ndarray  # drawn command unit per respondent


@dataclass
class Dataset:
    """Everything the notebooks need, plus helpers pre-bound to this dataset's inputs."""

    config: DatasetConfig
    paths: DataPaths
    df: pd.DataFrame
    hh: pd.DataFrame
    household: HouseholdContext
    agri_pool: pd.DataFrame
    train_pool: pd.DataFrame
    test_pool: pd.DataFrame
    test_ids: set
    survey_columns: pd.Index
    crosswalk: pd.DataFrame
    distribution_df: pd.DataFrame
    excel_df: pd.DataFrame
    masking_df: pd.DataFrame
    features: FeatureSpec
    geo_cols: list = field(default_factory=list)

    def assign_command_units(self, pool, seed):
        return assign_command_units(pool, seed, self.distribution_df)

    def build_feature_frame(
        self, pool_with_geo, sex_code, excel_df=None, extra_cols=(), extra_cats=()
    ):
        """Per-sex model frame. Returns (frame, categorical columns, extra_info with pweight/command_unit).

        Columns of the mask-based ``features`` that are not in the survey or command-area data
        are skipped. ``extra_cols`` / ``extra_cats`` add columns (e.g. those a saved model
        was trained on) on top of the mask selection; ``extra_cats`` are cast to str.
        """
        if excel_df is None:
            excel_df = self.excel_df

        cols = list(
            dict.fromkeys(self.features.columns_for(sex_code) + list(extra_cols))
        )
        frame = (
            pool_with_geo[pool_with_geo["sex"] == sex_code]
            .merge(excel_df, left_on="command_unit", right_on="area", how="inner")
            .drop(columns=["area"])  # duplicate of command_unit
        )
        keep = [c for c in cols if c in frame.columns] + [
            "log1p_wage",
            "totwag_all",
            "is_paid",
            "command_unit",
        ]
        frame = frame.dropna(subset=["log1p_wage"])
        extra_info = frame[["pweight", "command_unit"]]
        frame = frame[keep]
        cats = [
            c
            for c in dict.fromkeys(
                list(self.features.categorical_columns) + list(extra_cats)
            )
            if c in frame.columns
        ]
        # With pandas 3 .astype(str) keeps NaN as a missing value, but CatBoost's cat_features
        # cannot contain NaN, so fill explicitly first.
        frame[cats] = frame[cats].fillna("missing").astype(str)
        return frame, cats, extra_info

    def outlier_cap(self, sex):
        return (
            self.config.male_outlier_cap
            if sex == "male"
            else self.config.female_outlier_cap
        )

    def realization(
        self, sex, seed, excel_df=None, columns=(), categorical=(), test_data=True
    ) -> Realization:
        """Test set of ``sex`` ("male"/"female") for the command-unit draw given by ``seed``.

        ``excel_df`` swaps in other command-area covariates (e.g. another RIBASIM run).
        ``columns`` / ``categorical`` add the features a saved model needs on top of the
        feature mask (see :func:`egypt_survey_ml.predict.model_columns`).
        """
        sex_code = SEX_CODES[sex]
        geo = self.assign_command_units(
            self.test_pool if test_data else self.train_pool, seed
        )
        frame, _, extra = self.build_feature_frame(
            geo, sex_code, excel_df, columns, categorical
        )
        keep = (frame["totwag_all"] < self.outlier_cap(sex)).to_numpy()
        if self.config.zero_handling != "none":
            keep = (
                (frame["totwag_all"] < self.outlier_cap(sex))
                & (frame["totwag_all"] > 0)
            ).to_numpy()
            frame = self.apply_zero_handling(
                frame, categorical, columns, mode=self.config.zero_handling, seed=seed
            )
            frame = frame[frame["totwag_all"] < self.outlier_cap(sex)]
        return Realization(
            seed=seed,
            sex_code=sex_code,
            frame=frame,
            weight=extra["pweight"].to_numpy()[keep],
            command_unit=extra["command_unit"].to_numpy()[keep],
        )

    def apply_zero_handling(self, frame, cat_cols, feature_cols, mode=None, seed=None):
        if mode is None:
            mode = self.config.zero_handling
        if seed is None:
            seed = self.config.rng_seed
        if mode == "none":
            return frame
        if mode == "drop_random":
            return drop_random_zeros(frame, self.config.drop_random_frac, seed)
        if mode == "lognormal_synthetic":
            n_zero = (frame["is_paid"] == 0).sum()
            n_pos = (frame["is_paid"] == 1).sum()
            n_synthetic = max(
                0, n_zero - n_pos
            )  # toward parity, not necessarily exact 50/50
            return lognormal_synthetic_augment(
                frame, feature_cols, cat_cols, n_synthetic, self.geo_cols, seed=seed
            )
        raise ValueError(mode)

    def perturbate_wage(self, frame, seed, mu=0.0, sigma=None):
        return perturbate_wage(
            frame,
            seed,
            mu,
            self.config.wage_perturbation_sigma if sigma is None else sigma,
        )

    def lognormal_synthetic_augment(
        self, frame, feature_cols, cat_cols, n_synthetic, k=5, seed=42
    ):
        return lognormal_synthetic_augment(
            frame, feature_cols, cat_cols, n_synthetic, self.geo_cols, k=k, seed=seed
        )


def prepare_dataset(
    config: DatasetConfig | None = None,
    paths: DataPaths | None = None,
    *,
    mask_path: str | Path | None = None,
) -> Dataset:
    """Run the full dataset pipeline for the given scenario settings.

    ``mask_path`` overrides only the column-selection workbook (default:
    ``Result/correlation.xlsx``), not the command-area data workbook. Its ``Sheet1`` must
    contain ``variable``, ``correl`` (1 selects a feature) and ``type`` (``binary`` /
    ``categorical`` mark categorical features). Selected columns that exist in neither the
    survey nor the command-area data are skipped, as in the training notebooks.
    """
    config = config or DatasetConfig()
    paths = paths or resolve_data_paths(config.data_filename)
    if mask_path is not None:
        paths = replace(paths, mask_xlsx=Path(mask_path).expanduser())
    print(f"data source: {paths.source}")
    paths.report()

    masking_df = pd.read_excel(paths.mask_xlsx, sheet_name="variable")
    missing = {"variable", "correl", "type"} - set(masking_df.columns)
    if missing:
        raise ValueError(
            f"Feature mask {paths.mask_xlsx} (Sheet1) is missing required columns: "
            f"{sorted(missing)}"
        )

    df, hh = load_survey(paths)
    household = build_household_context(df, hh)

    agri_pool = select_rural_agriculture(df, config.min_working_age)
    train_pool, test_pool, test_ids = split_train_test(agri_pool, seed=config.rng_seed)

    train_pool = build_wage_target(attach_household_context(train_pool, household, hh))
    test_pool = build_wage_target(attach_household_context(test_pool, household, hh))
    survey_columns = train_pool.columns
    describe_pools(train_pool, test_pool)

    crosswalk = pd.read_csv(paths.crosswalk_csv)
    distribution_df = pd.read_csv(paths.pop_density_csv, index_col="NAME1_")
    train_pool = attach_governorate(train_pool, crosswalk)
    test_pool = attach_governorate(test_pool, crosswalk)

    excel_df = pd.read_excel(paths.excel_path, sheet_name="command_unit")
    print(
        f"Loaded command_unit: {excel_df.shape[0]} areas, {excel_df.shape[1]} columns "
        f"(already scaled -- no rural-population normalization needed for this sheet)"
    )

    features = select_features(masking_df, agri_pool, config)
    unavailable = [
        c
        for c in features.base_cols
        if c not in train_pool.columns and c not in excel_df.columns
    ]
    if unavailable:
        print(f"Skipped (selected in mask, not in the data): {unavailable}")

    return Dataset(
        config=config,
        paths=paths,
        df=df,
        hh=hh,
        household=household,
        agri_pool=agri_pool,
        train_pool=train_pool,
        test_pool=test_pool,
        test_ids=test_ids,
        survey_columns=survey_columns,
        crosswalk=crosswalk,
        distribution_df=distribution_df,
        excel_df=excel_df,
        masking_df=masking_df,
        features=features,
        geo_cols=geo_columns(excel_df),
    )
