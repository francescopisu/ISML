"""
Default configuration values for the experiments.
"""
import os
from yacs.config import CfgNode as CN

_C = CN()

# data
_C.DATA = CN()
_C.DATA.BASE_INPUT_PATH = os.path.join(os.getcwd(), 'input')
_C.DATA.TRAIN_DATA_PATH = os.path.join(_C.DATA.BASE_INPUT_PATH, 'train.csv')
_C.DATA.TEST_DATA_PATH = os.path.join(_C.DATA.BASE_INPUT_PATH, 'test.csv')
_C.DATA.EXTERNAL_DATA_PATH = os.path.join(_C.DATA.BASE_INPUT_PATH, 'external.csv')
_C.DATA.APPLY_OHE = False
_C.DATA.SCALE = False
_C.DATA.TEST_SIZE = 0.2
_C.DATA.TARGET = "diagnosis"
_C.DATA.TO_SCALE = []
_C.DATA.COLS_TO_DROP = ["height", "weight"]
_C.DATA.CAT_FEATURES = []
_C.DATA.SUBSET_DATA = False
_C.DATA.WHICH_SUBSET = "strain"
_C.DATA.DEMO_SUBSET = ['gender', 'age', 'weight', 'height', 'BSA']
_C.DATA.VENTRICLE_STRAIN_SUBSET = ['LVRS_mid', 'LVCS_mid', 'LVLS_mid', 'LVRS_apical', 'LVCS_apical', 'LVLS_apical', 'LVRS_global',
       'LVCS_global', 'LVLS_global', 'RVRS_basal', 'RVCS_basal', 'RVRS_mid',
       'RVCS_mid', 'RVRS_apical', 'RVCS_apical', 'RVRS_global', 'RVCS_global',
       'RVLS_global']
_C.DATA.ATRIUM_STRAIN_SUBSET = ['reservoir', 'reservoir_rate', 'conduit', 'conduit_rate', 
        'booster', 'booster_rate']
_C.DATA.STRAIN_SUBSET = ['reservoir', 'reservoir_rate', 'conduit', 'conduit_rate', 
        'booster', 'booster_rate', 'LVRS_basal', 'LVCS_basal', 'LVLS_basal', 
        'LVRS_mid', 'LVCS_mid', 'LVLS_mid', 'LVRS_apical', 'LVCS_apical', 'LVLS_apical', 'LVRS_global',
       'LVCS_global', 'LVLS_global', 'RVRS_basal', 'RVCS_basal', 'RVRS_mid',
       'RVCS_mid', 'RVRS_apical', 'RVCS_apical', 'RVRS_global', 'RVCS_global',
       'RVLS_global']
_C.DATA.FUNCTION_SUBSET = ['EF', 'HR', 'LV_MASS', 'LVEDV/BSA', 'LVESV/BSA', 'SV/BSA',
       'LV MASS/BSA', 'RVEF', 'RVEDV/BSA', 'RVESV/BSA', 'RVSV/BSA']

# output
_C.OUTPUT = CN()
_C.OUTPUT.BASE_OUTPUT_PATH = os.path.join(os.getcwd(), "output")
_C.OUTPUT.FITTED_MODELS_PATH = os.path.join(os.getcwd(), "output/fitted_models")
_C.OUTPUT.RESULTS_PATH = os.path.join(os.getcwd(), "output/results")
_C.OUTPUT.PARAMS_PATH = os.path.join(os.getcwd(), "output/parameters")
_C.OUTPUT.PREDS_PATH = os.path.join(os.getcwd(), "output/predictions")
_C.OUTPUT.PLOTS_PATH = os.path.join(os.getcwd(), "output/plots")
_C.OUTPUT.FITTED_MODEL_PATH = os.path.join(_C.OUTPUT.FITTED_MODELS_PATH, "model.pkl")
_C.OUTPUT.BEST_MODEL = os.path.join(_C.OUTPUT.FITTED_MODELS_PATH, "final.pkl")
_C.OUTPUT.BEST_PARAMS = os.path.join(_C.OUTPUT.PARAMS_PATH, "best_params.pkl")
_C.OUTPUT.TEST_RESULTS = os.path.join(_C.OUTPUT.RESULTS_PATH, "test_results.csv")

# training
_C.TRAIN = CN()
_C.TRAIN.WHICH = "ebm"

# feature selection
_C.FS = CN()
_C.FS.FRAC = 1.25

# evaluation
_C.EVAL = CN()
_C.EVAL.N_INNER_SPLITS = 5
_C.EVAL.N_OUTER_SPLITS = 10
_C.EVAL.N_REPEATS = 10
_C.EVAL.SHUFFLE = True
_C.EVAL.ALPHA = 0.05
_C.EVAL.BOOT_ROUNDS = 5000
_C.EVAL.CALIBRATE = True
_C.EVAL.CALIB_FRAC = 0.15
# _C.EVAL.THRESHOLD = 0.5729885
_C.EVAL.THRESHOLD = 0.23723600934486413
# _C.EVAL.THRESHOLD = 0.5
_C.EVAL.SET_NAMES = ["Train", "Valid"]
_C.EVAL.ALGO_SHORT_NAMES = ["EBM"]
_C.EVAL.ALGO_LONG_NAMES = ["Explainable Boosting Classifier"]
_C.EVAL.METRIC_NAMES = [
    #"Sensitivity", "Specificity", "FPR", "FNR",
    #"Precision", "PPV", "NPV", "F1", "F2", 
    "ROCAUC", "PRAUC", "Brier"]

# misc
_C.MISC = CN()
_C.MISC.SEED = 1303


def get_defaults():
    return _C.clone()
