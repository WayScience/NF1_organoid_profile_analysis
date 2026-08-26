import pathlib

import numpy as np
import pandas as pd
import skimage.io
from microfilm.microplot import Micropanel, microshow


def filter_out_self_correlations(df):
    """
    Filter out self-correlations from a correlation matrix dataframe.
    Self-correlations are where group1 == group2.
    """
    return df.loc[df["group1"] != df["group2"]]


def filter_out_diagonal_correlations(df):
    df = df.copy()
    df = df[df["group1"] != df["group2"]]

    # vectorized: put the lexicographically smaller value first, larger second
    g1 = df["group1"].astype(str)
    g2 = df["group2"].astype(str)
    lo = np.minimum(g1, g2)
    hi = np.maximum(g1, g2)
    df["pair_key"] = lo + "_" + hi

    df = df.drop_duplicates(subset="pair_key", keep="first")
    df = df.drop(columns=["pair_key"])
    return df


def retrieve_quadrant_info(
    df: pd.DataFrame,
) -> list:
    """
    Returns a list of strings, each representing a unique combination of metadata values for group1 and group2 in the dataframe.
    The combinations are constructed by joining the following metadata columns with "__":
    - group1_Metadata_Biology_PatientTumor
    - group1_Metadata_Experiment_Well
    - group1_Metadata_Experiment_Treatment
    - group1_Metadata_Experiment_Dose
    - group2_Metadata_Biology_PatientTumor
    - group2_Metadata_Experiment_Well
    - group2_Metadata_Experiment_Treatment
    """
    df["combinations"] = df.apply(
        lambda row: "__".join(
            str(x)
            for x in (
                row["group1_Metadata_Biology_PatientTumor"],
                row["group1_Metadata_Experiment_Well"],
                row["group1_Metadata_Experiment_Treatment"],
                row["group1_Metadata_Experiment_Dose"],
                row["group2_Metadata_Biology_PatientTumor"],
                row["group2_Metadata_Experiment_Well"],
                row["group2_Metadata_Experiment_Treatment"],
                row["group2_Metadata_Experiment_Dose"],
            )
        ),
        axis=1,
    )
    return df["combinations"].to_list()


def generate_image_paths_from_combination_string(
    image_base_dir: pathlib.Path, combination_string: str
) -> list:
    """
    Given a combination string, return a list of image paths for group1 and group2.
    The combination string is expected to be in the format:
    "group1_Metadata_Biology_PatientTumor__group1_Metadata_Experiment_Well__group1_Metadata_Experiment_Treatment__group1_Metadata_Experiment_Dose__group2_Metadata_Biology_PatientTumor__group2_Metadata_Experiment_Well__group2_Metadata_Experiment_Treatment__group2_Metadata_Experiment_Dose"
    """
    parts = combination_string.split("__")
    if len(parts) != 8:
        raise ValueError(
            "Combination string must have exactly 8 parts separated by '__'"
        )

    (
        group1_patient_tumor,
        group1_well,
        group1_treatment,
        group1_dose,
        group2_patient_tumor,
        group2_well,
        group2_treatment,
        group2_dose,
    ) = parts
    image1_path = pathlib.Path(
        f"{image_base_dir}/{group1_patient_tumor}",
        "zstack_images",
        f"{group1_well}-1",
    ).resolve(strict=True)
    image2_path = pathlib.Path(
        f"{image_base_dir}/{group2_patient_tumor}",
        "zstack_images",
        f"{group2_well}-1",
    ).resolve(strict=True)
    return [image1_path, image2_path]


def make_multi_channel_image_array(image_path: pathlib.Path) -> np.ndarray:
    """
    Given a path to a zstack image directory, return a 4D numpy array of shape (channels, z, y, x).
    The channels are expected to be in the order: DAPI, GFP, RFP.
    """
    channel_image_paths = sorted(image_path.glob("*.tif*"))
    channel_images = []
    for channel_path in channel_image_paths:
        if "TRANS" in channel_path.name:
            continue
        zstack_array = skimage.io.imread(channel_path)
        # get the middle slice of the zstack for each channel
        zstack_array = zstack_array[zstack_array.shape[0] // 2, :, :]
        channel_images.append(zstack_array)
    multi_channel_array = np.stack(channel_images, axis=0)
    return multi_channel_array


def create_and_save_two_image_panel(
    image_1_array: np.ndarray,
    image_2_array: np.ndarray,
    image_1_label: str,
    image_2_label: str,
    output_path: pathlib.Path,
) -> pathlib.Path:
    """
    Given two 4D numpy arrays of shape (channels, z, y, x), create a two-panel image and save it to the specified output path.
    The output image will be a 2D projection of the maximum intensity across the z-axis for each channel.

    Parameters:
    ----------
    image_1_array : np.ndarray
        A 4D numpy array representing the first image (channels, z, y, x).
    image_2_array : np.ndarray
        A 4D numpy array representing the second image (channels, z, y, x).
    output_path : pathlib.Path
        The path where the output image will be saved.

    Returns:
    -------
    pathlib.Path
        The path to the saved output image.
    """

    microim1 = microshow(
        images=image_1_array[:, :, :],
        fig_scaling=5,
        cmaps=["pure_cyan", "pure_green", "pure_magenta", "pure_red"],
        unit="um",
        scalebar_size_in_units=10,
        scalebar_unit_per_pix=0.1,
        scalebar_font_size=10,
        # label_text='A',
        label_text=image_1_label,
        label_color="white",
        label_font_size=20,
        label_location="upper right",
    )
    microim2 = microshow(
        images=image_2_array[:, :, :],
        fig_scaling=5,
        cmaps=["pure_cyan", "pure_green", "pure_magenta", "pure_red"],
        unit="um",
        scalebar_size_in_units=10,
        scalebar_unit_per_pix=0.1,
        scalebar_font_size=10,
        # label_text='B',
        label_text=image_2_label,
        label_color="white",
        label_font_size=20,
        label_location="upper right",
    )

    panel = Micropanel(rows=1, cols=2)
    panel.add_element(pos=[0, 0], microim=microim1)
    panel.add_element(pos=[0, 1], microim=microim2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.savefig(
        output_path,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    return output_path
