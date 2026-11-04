from egypt_survey_ml.paths import DataPaths


def test_from_onedrive_accepts_custom_directory_names(tmp_path):
    onedrive = tmp_path / "OneDrive"
    egypt_dir = "Egypt files"
    erf_dir = "Egypt ERF"
    lfs_dir = "LFS 2022"
    (onedrive / egypt_dir).mkdir(parents=True)
    (onedrive / erf_dir).mkdir()

    paths = DataPaths.from_onedrive(
        onedrive_dir=onedrive,
        egypt_dir=egypt_dir,
        erf_dir=erf_dir,
        lfs_dir=lfs_dir,
        data_filename="custom_data.xlsx",
    )

    data_root = onedrive / egypt_dir / "04_Data/2026_data"
    crosswalk_dir = data_root.parent / "cleaned_data/crosswalk"
    result_dir = data_root.parent / "Result"
    assert paths.data_root == data_root
    assert paths.excel_path == onedrive / erf_dir / "custom_data.xlsx"
    assert paths.crosswalk_csv == crosswalk_dir / "reg_governorate_crosswalk.csv"
    assert paths.command_area_crosswalk_csv == (
        crosswalk_dir / "command_area_governorate_crosswalk.csv"
    )
    assert paths.lfs_dta == data_root / lfs_dir / "Egypt 2022-LFS IND-V1.dta"
    assert paths.hh_dta == data_root / lfs_dir / "Egypt 2022-LFS HH-V1.dta"
    assert paths.mask_xlsx == result_dir / "correlation.xlsx"
    assert paths.pop_density_csv == result_dir / "population_density_matrix.csv"
