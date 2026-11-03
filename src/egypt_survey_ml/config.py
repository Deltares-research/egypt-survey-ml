"""Scenario settings that control how the survey dataset is built and modelled."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace

ZERO_HANDLING_MODES = ("none", "drop_random", "lognormal_synthetic")
COLUMN_LISTS = (
    "male_extra_columns",
    "female_extra_columns",
    "male_exclude_columns",
    "female_exclude_columns",
)


@dataclass
class DatasetConfig:
    """All user-changeable scenario settings in one place.

    Create one with keyword overrides (``DatasetConfig(include_ribasim=True)``),
    derive variants with :meth:`replace`, or build one from a plain dict / TOML file
    with :meth:`from_dict` / :meth:`from_toml`.

    The feature columns start from the ``correl == 1`` variables of the feature mask.
    ``male_extra_columns`` / ``female_extra_columns`` add columns for one sex (also
    mask variables with ``correl == 0``), ``male_exclude_columns`` /
    ``female_exclude_columns`` remove columns for one sex; exclusion wins over everything
    else. Extra columns that are not in the loaded data are skipped with a warning.
    """

    rng_seed: int = 42
    min_working_age: int = 15
    weeks_per_month: float = 365.25 / 7 / 12  # ~4.348

    # feature toggles
    include_empstab: bool = True
    include_hrswk: bool = True
    include_ribasim: bool = False

    # per-sex feature columns, on top of / removed from the feature mask
    male_extra_columns: list[str] = field(default_factory=list)
    female_extra_columns: list[str] = field(default_factory=list)
    male_exclude_columns: list[str] = field(default_factory=list)
    female_exclude_columns: list[str] = field(default_factory=list)

    # zero-wage handling: one of ZERO_HANDLING_MODES
    zero_handling: str = "drop_random"
    drop_random_frac: float = 1.0  # only used when zero_handling == "drop_random"

    # target / perturbation settings
    predict_log_wage: bool = False
    wage_perturbation: bool = False
    wage_perturbation_sigma: float = 0.03

    n_reps: int = 100
    male_outlier_cap: float = 5000
    female_outlier_cap: float = 4000

    data_filename: str = "data_correlation.xlsx"

    def __post_init__(self):
        for name in COLUMN_LISTS:
            value = getattr(self, name)
            if isinstance(value, str):
                raise ValueError(f"{name} must be a list of column names, got {value!r}")
            setattr(self, name, list(value))
        if self.zero_handling not in ZERO_HANDLING_MODES:
            raise ValueError(
                f"zero_handling must be one of {ZERO_HANDLING_MODES}, "
                f"got {self.zero_handling!r}"
            )

    def replace(self, **changes) -> "DatasetConfig":
        """Return a copy with the given settings changed."""
        return replace(self, **changes)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "DatasetConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise KeyError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**values)

    @classmethod
    def from_toml(cls, path) -> "DatasetConfig":
        import tomllib

        with open(path, "rb") as f:
            return cls.from_dict(tomllib.load(f))
