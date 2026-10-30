"""Scenario settings that control how the survey dataset is built and modelled."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace

ZERO_HANDLING_MODES = ("none", "drop_random", "lognormal_synthetic")


@dataclass
class DatasetConfig:
    """All user-changeable scenario settings in one place.

    Create one with keyword overrides (``DatasetConfig(include_ribasim=True)``),
    derive variants with :meth:`replace`, or build one from a plain dict / TOML file
    with :meth:`from_dict` / :meth:`from_toml`.
    """

    rng_seed: int = 42
    min_working_age: int = 15
    weeks_per_month: float = 365.25 / 7 / 12  # ~4.348

    # feature toggles
    include_empstab: bool = True
    include_hrswk: bool = True
    include_ribasim: bool = False

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
