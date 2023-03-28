"""
This file contains methods related to input-output operations, such as
loading and saving models.
"""
import sys
import pickle
from typing import Tuple, Any
import pandas as pd  # type: ignore
from yacs.config import CfgNode  # type: ignore

from src._typing import PathLike, ArrayLike


def load_data(conf: CfgNode, which: str = 'train') -> Tuple[ArrayLike, ArrayLike]:
    """
    Loads data located in folder conf.DATA.<TRAIN|TEST>_DATA_PATH stored in csv format.

    Parameters
    ----------
    conf: CfgNode
        A Configuration node storing configuration information.
    which: str (default = 'train')
        Which dataset to load.

    Returns
    -------
    Tuple[ArrayLike, ArrayLike]
        A tuple of features and ground-truths corresponding to
        the specified dataset.
    """
    if which == 'test':
        data_path = conf.DATA.TEST_DATA_PATH
    elif which == 'external':
        data_path = conf.DATA.EXTERNAL_DATA_PATH
    else:
        data_path = conf.DATA.TRAIN_DATA_PATH

    data = pd.read_csv(data_path)

    X, y = data.drop(conf.DATA.TARGET, axis=1), data[conf.DATA.TARGET]

    X.loc[:, "gender"] = X.loc[:, "gender"].astype("category")
    # X.loc[:, "BMI"] = X.loc[:, "weight"] / (X.loc[:, "height"] ** 2)

    X = X.drop(conf.DATA.COLS_TO_DROP, axis=1)

    # subset data if necessary
    if conf.DATA.SUBSET_DATA:
        if conf.DATA.WHICH_SUBSET == "ventricle_strain":
            X = X[conf.DATA.VENTRICLE_STRAIN_SUBSET]
        elif conf.DATA.WHICH_SUBSET == "atrium_strain":
            X = X[conf.DATA.ATRIUM_STRAIN_SUBSET]
        elif conf.DATA.WHICH_SUBSET == "strain":
            X = X[conf.DATA.STRAIN_SUBSET]
        elif conf.DATA.WHICH_SUBSET == "ventricle_function":
            X = X[conf.DATA.FUNCTION_SUBSET]
        elif conf.DATA.WHICH_SUBSET == "demographic":
            X = X[conf.DATA.DEMO_SUBSET]

    # X["diagnosis"] = y
    # X.to_csv("./input/df_final_with_indexes.csv")
    # sys.exit(1)

    return X, y


def save_obj(obj: Any, path: PathLike):
    """
    Dump the model to the specified path.
    Parameters
    ----------
    obj: Estimator
        The model to be saved
    path: PathLike
        The path where to save the model
    """
    with open(path, "wb") as f_w:
        pickle.dump(obj, f_w)


def load_obj(path: PathLike) -> Any:
    """
    Load the model located at the specified path.

    Parameters
    ----------
    path: PathLike
        The path where to find the model.

    Returns
    -------
    Estimator
        The loaded model
    """
    with open(path, "rb") as f_r:
        obj = pickle.load(f_r)

    return obj
