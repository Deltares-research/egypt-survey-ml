# ERF_Data

## What's here

- `Data_Cleaning/`: the two active notebooks (exploratory analysis, then preprocessing), plus `crosswalk/`, the two scripts that build the geographic lookup tables (see below).
- `Data/`: **not in this repo** (gitignored, restricted survey microdata). See "Data you need" below for what to get and where to put it.

## Full folder tree

```
ERF_Data/
├── Data/                                                                    (gitignored, build locally)
│   ├── crosswalk/
│   │   ├── command_area_governorate_crosswalk.csv
│   │   └── reg_governorate_crosswalk.csv
│   ├── gis_inputs/
│   │   ├── command_area/
│   │   │   └── Final2_Command_Area.shp   (+ .dbf, .prj, .shx, .cpg, .sbn, .sbx, .shp.xml)
│   │   └── governorates/
│   │       └── Governorates.shp          (+ .dbf, .prj, .shx, .cpg, .qmd)
│   ├── Labor Force Survey, LFS 2022 - Egypt, Arab Rep., 2022/
│   │   ├── Egypt 2022-LFS HH-V1.dta
│   │   └── Egypt 2022-LFS IND-V1.dta
│   ├── LFS2022_data_dictionary_extracted.csv
│   ├── preprocessed_data.xlsx
│   └── preprocessed_data_explain.xlsx
│
├── Data_Cleaning/                                                           (tracked in git)
│   ├── crosswalk/
│   │   ├── build_command_area_governorate_crosswalk.py
│   │   └── build_reg_governorate_mapping.py
│   ├── 00_EDA_explore_lfs2022.ipynb
│   ├── 01_preprocess_lfs2022_individual.ipynb
│   └── lfs2022_individual_preprocessed.csv                                 (generated, gitignored)
│
└── README.md                                                                (this file)
```

Everything under `Data/` is gitignored: none of it gets pushed to GitHub. Everything under `Data_Cleaning/` except the generated CSV is tracked.

## A note on terminology: "crosswalk"

"Crosswalk" isn't a standard term documented anywhere else in this project's source material. It's just what we've been calling the lookup/mapping tables built specifically for this project, so if you haven't been in these conversations it won't mean anything on its own. Concretely, it refers to two small CSV files:

- `reg_governorate_crosswalk.csv`: maps the LFS survey's numeric region codes to actual Egyptian governorate names.
- `command_area_governorate_crosswalk.csv`: maps RIBASIM's 61 water-management "command areas" to Egypt's 27 governorates. Their boundaries don't match, so this took a GIS overlay to build.

Both exist because the survey data and the water-management model use two different, incompatible geographic schemes, and something has to translate between them.

## Pipeline, in order

### 1. Crosswalk (`Data_Cleaning/crosswalk/`)

Two independent scripts, each producing one CSV under `Data/crosswalk/`. **What each one needs as input:**

| Script | Reads | Produces |
|---|---|---|
| `build_reg_governorate_mapping.py` | The LFS individual `.dta` file's own value labels for the `reg` column, plus `preprocessed_data.xlsx` (the RIBASIM-linked biophysical spreadsheet) | `reg_governorate_crosswalk.csv` |
| `build_command_area_governorate_crosswalk.py` | Two shapefile bundles: `gis_inputs/command_area/Final2_Command_Area.shp` and `gis_inputs/governorates/Governorates.shp` (each with their sidecar files: `.dbf`, `.prj`, `.shx`, etc.) | `command_area_governorate_crosswalk.csv` |

These two CSVs are what let survey respondents be connected to governorate-level water-management data at all. Both scripts are re-runnable: they overwrite their own output each time.

### 2. `00_EDA_explore_lfs2022.ipynb`

Exploratory pass through the raw LFS 2022 individual and household files before committing to any modeling choice. Answers: is there a wage gap, how does employment status/hours/occupation differ by gender, how prevalent is female headship, what can only combining both files tell us. Also the core finding in Part 4: restricted to rural agriculture, men work meaningfully more hours than women while the hourly wage gap is close to zero. That's the empirical basis for treating hours (not wage) as the primary outcome downstream.

### 3. `01_preprocess_lfs2022_individual.ipynb`

Builds the actual model-ready table: one row per rural, working-age, employed agricultural worker, with two targets (`hrswk` primary, `hourly_wage` secondary, paid subsample only), household-context features (including a head:spouse wage-ratio bargaining-power measure), and the governorate-level RIBASIM linkage via the crosswalk. Output: `lfs2022_individual_preprocessed.csv`, not committed, regenerate by running the notebook.

## Data you need (not included in this repo)

`Data/` is gitignored: the raw survey microdata is restricted-access and shouldn't be public. Build it locally to match the tree above, under `ERF_Data/Data/`. The two crosswalk CSVs don't need to be downloaded separately: running the two scripts in `Data_Cleaning/crosswalk/` regenerates them from everything else.

### Where to get each file, from the team's shared folder

_(Fill in the actual shared-folder link here once you have it. The file names below are exact, so search for these.)_

| File | Where it goes under `Data/` |
|---|---|
| `Egypt 2022-LFS IND-V1.dta` | `Labor Force Survey, LFS 2022 - Egypt, Arab Rep., 2022/` |
| `Egypt 2022-LFS HH-V1.dta` | `Labor Force Survey, LFS 2022 - Egypt, Arab Rep., 2022/` |
| `LFS2022_data_dictionary_extracted.csv` | directly under `Data/` |
| `preprocessed_data.xlsx` | directly under `Data/` |
| `preprocessed_data_explain.xlsx` | directly under `Data/` |
| `Final2_Command_Area.*` (all files with this name, different extensions) | `gis_inputs/command_area/` |
| `Governorates.*` (all files with this name, different extensions) | `gis_inputs/governorates/` |

Once `Data/` is in place, run the two crosswalk scripts first, then both notebooks in `Data_Cleaning/` top to bottom.
