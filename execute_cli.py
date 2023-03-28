"""
This file implements a CLI to start the training, tuning and testing of the model.
"""
from typing import Dict, Optional, List
import os, sys
import time
import numpy as np
import pandas as pd
import click
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

from config.defaults import get_defaults
from src.utils.io import load_data, load_obj, save_obj
from src.utils.misc import add_extension, show_cross_val_results, set_all_seeds
from src.evaluate import cross_validate, get_evaluation_metrics, test_performance
from src.train import train_model
from src.utils.plotting import plot_average_roc_curves, plot_average_pr_curves


def bootstrap(new_options: Optional[List] = None,
              mode: str = "train") -> Dict:
    """
    This function is responsible for the bootstrap phase prior to
    training or testing the model.
    It is responsible for:
        - loading the default configuration values
        - updating defaults by merging CLI argumentsc
        - loading either train or test dataset
        - instantiating or loading a model

    Parameters
    ----------
    new_options: new options coming from CLI arguments. They will be merged
        with defaults.
    mode: str (default = "train")
        Modality of execution. Options are train, test, cv and tune.
    Returns
    -------
    Dict
        A dictionary containing preprocessed data, a model and configuration data.
    """
    defaults = get_defaults()
    if new_options:
        defaults.merge_from_list(new_options)
    defaults.freeze()
    
    set_all_seeds(defaults.MISC.SEED)

    # load datasets
    which = 'train'  # default
    if mode in ['train', 'cv']:
        which = 'train'
    elif mode == 'internal_test':
        which = 'test'
    elif mode == 'external_test':  # mode == 'external_test'
        which = 'external'

    X, y = load_data(defaults, which=which)

    # a. Get fixed parameters and search spaces for nested cross-validation
    # b. Get fixed parameters and search space for a specific algorithm to be tuned
    # c. load an already fitted model (after optimization/tuning)

    model = None
    params = None
    algorithms = None

    if mode in ['internal_test', 'external_test']:
        # load model from defaults.OUTPUT.FITTED_MODEL_PATH
        model = load_obj(defaults.OUTPUT.FITTED_MODEL_PATH)

    return {
        "data": (X, y),
        "defaults": defaults,
        "model": model,
    }


@click.group()
def cli():
    pass


@click.command()
@click.option('--fitted_model_filename', default="model.pkl")
def train(fitted_model_filename):
    """
    Train the model on the training set and save the fitted model to
    output/fitted_models/<fitted_model_filename>.pkl directory.

    fitted_model_filename is used as follows:
        a) to update the fitted_model_path from output/fitted_models/model.pkl (default)
        to output/fitted_models/<fitted_model_filename> in order to dump the fitted model.
    """
    click.echo("Mode: training.\n")
    defaults = get_defaults()

    fitted_model_filename = add_extension(fitted_model_filename)

    # derive final path for fitted model as base output path for fitted models + model filename
    fitted_model_path = os.path.join(defaults.OUTPUT.FITTED_MODELS_PATH, fitted_model_filename)

    new_options = ["OUTPUT.FITTED_MODEL_PATH", fitted_model_path]

    boot_data = bootstrap(new_options, mode="train")
    defaults = boot_data['defaults']

    X_train, y_train = boot_data['data']
    fitted_model = train_model(X_train, y_train, defaults)

    # dump fitted model
    os.makedirs(defaults.OUTPUT.FITTED_MODELS_PATH, exist_ok=True)
    save_obj(fitted_model, defaults.OUTPUT.FITTED_MODEL_PATH)


@click.command()
@click.option("--which", default='internal')
@click.option("--fitted_model_filename", default='model.pkl')
def test(which, fitted_model_filename):
    """
    Test a fitted model on the test set.
    By default, we look for a fitted model in the output/fitted_model
    directory.

    fitted_model_filename is used as follows:
        a) to update the fitted_model_path from output/fitted_models/model.pkl (default)
        to output/fitted_models/<fitted_model_filename> in order to load the already fitted model.
        b) to derive the model name (part preceding the .extension) which is then used to
        save the test results in output/results/test_results_<model name>.csv
    """
    click.echo("Mode: test.")
    defaults = get_defaults()

    # bootstrap input
    fitted_model_path = os.path.join(defaults.OUTPUT.FITTED_MODELS_PATH, fitted_model_filename)
    new_options = ["OUTPUT.FITTED_MODEL_PATH", fitted_model_path]

    mode = "{}_test".format(which)
    boot_data = bootstrap(new_options, mode=mode)

    model = boot_data['model']
    X_test, y_test = boot_data['data']
    defaults = boot_data['defaults']

    eval_metrics = get_evaluation_metrics()
    # model = RandomForestClassifier(random_state=defaults.MISC.SEED, class_weight='balanced')

    # X_train, y_train = load_data(defaults, which='train')
    # scaler = StandardScaler()
    # numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    # X_train.loc[:, numeric_cols] = scaler.fit_transform(X_train[numeric_cols])

    # model.fit(X_train, y_train)

    test_results = test_performance(conf=defaults,
                                    model=model,
                                    X_test=X_test, y_test=y_test,
                                    eval_metrics=eval_metrics)
    results = pd.DataFrame(test_results.values(), index=test_results.keys(), columns=["test"])

    results_filename = "{}_results_{}.csv".format(mode, fitted_model_filename.split(".")[0])
    results_path = os.path.join(defaults.OUTPUT.RESULTS_PATH, results_filename)
    results.to_csv(results_path)


@click.command()
@click.option('--exp_name', default="exp_1")
def cross_validation(exp_name):
    """
    Model selection and evaluation by means of nested cross-validation.

    fitted_model_filename is used as follows:
        a) to update the fitted_model_path from output/fitted_models/model.pkl (default)
        to output/fitted_models/<fitted_model_filename> in order to dump the fitted model.
    """
    click.echo("Mode: Cross-validation.")
    # defaults = get_defaults()

    # fitted_model_filename = add_extension(fitted_model_filename)

    # derive final path for fitted model as base output path for fitted models + model filename
    # fitted_model_path = os.path.join(defaults.OUTPUT.FITTED_MODELS_PATH, fitted_model_filename)
    # new_options = ["OUTPUT.FITTED_MODEL_PATH", fitted_model_path]

    # don't reserve dev set at this point since we need to do it in each cv fold
    boot_data = bootstrap(new_options=None, mode="cv")

    defaults = boot_data['defaults']
    X_train, y_train = boot_data['data']

    inner_cv = StratifiedKFold(n_splits=defaults.EVAL.N_INNER_SPLITS,
                                 shuffle=defaults.EVAL.SHUFFLE,
                                 random_state=defaults.MISC.SEED)
    outer_cv = StratifiedKFold(n_splits=defaults.EVAL.N_OUTER_SPLITS,
                                shuffle=defaults.EVAL.SHUFFLE,
                                random_state=defaults.MISC.SEED)                                 

    s = time.time()
    outer_results, outer_preds = cross_validate(X=X_train, y=y_train,
                                                outer_cv=outer_cv,
                                                inner_cv=inner_cv,
                                                conf=defaults)
    print("Execution time: %s seconds." % (time.time() - s))

    # dump results
    # fitted_model_best_params_path = os.path.join(defaults.OUTPUT.PARAMS_PATH,
    #                                              "best_params_{}.pkl".format(fitted_model_filename.split('.')[0]))

    outer_results_formatted = show_cross_val_results(outer_results, conf=defaults)

    cv_results_path = os.path.join(defaults.OUTPUT.RESULTS_PATH, "cv_results_{}.csv".format(exp_name))
    outer_results_formatted.to_csv(cv_results_path)

    # save predictions
    outer_preds_path = os.path.join(defaults.OUTPUT.PREDS_PATH, "cv_pooled_preds_{}.pkl".format(exp_name))
    save_obj(outer_preds, outer_preds_path)


@click.command()
@click.option('--fitted_model_filename', default="model.pkl")
def get_predictions(fitted_model_filename):
    """
    Use a fitted model to predict probabilities and save it
    in the results folder.
    """
    click.echo("Mode: predicting probabilities.\n")
    defaults = get_defaults()

    fitted_model_filename = add_extension(fitted_model_filename)
    fitted_model_path = os.path.join(defaults.OUTPUT.FITTED_MODELS_PATH, fitted_model_filename)
    new_options = ["OUTPUT.FITTED_MODEL_PATH", fitted_model_path]

    # boot_data = bootstrap(new_options, mode="internal_test")
    # model = boot_data['model']
    #
    # X_test_int, y_test_int = boot_data['data']
    # internal_test_proba = model.predict_proba(X_test_int)
    # internal_test_proba = np.c_[y_test_int, internal_test_proba[:, 1]]

    boot_data = bootstrap(new_options, mode="external_test")
    model = boot_data['model']
    X_test_ext, y_test_ext = boot_data['data']

    # fit scaler on train data and transform test data
    scaler = StandardScaler()
    X_train, y_train = load_data(defaults, which='train')

    numeric_cols = X_train.select_dtypes(include=np.float64).columns.tolist()
    scaler.fit(X_train[numeric_cols])
    X_test_ext.loc[:, numeric_cols] = scaler.transform(X_test_ext[numeric_cols])

    external_test_proba = model.predict_proba(X_test_ext)
    external_test_proba = np.c_[y_test_ext, external_test_proba[:, 1]]

    # internal_test_results_path = os.path.join(defaults.OUTPUT.PREDS_PATH, "internal_test_preds.csv")
    external_test_results_path = os.path.join(defaults.OUTPUT.PREDS_PATH,
                                              f"external_test_preds_{fitted_model_filename.replace('.pkl', '')}.csv")
    # pd.DataFrame(internal_test_proba, columns=['target', 'proba']).to_csv(internal_test_results_path, index=False)
    pd.DataFrame(external_test_proba, columns=['target', 'proba']).to_csv(external_test_results_path, index=False)


# cli.add_command(data_bootstrap)
cli.add_command(train)
cli.add_command(test)
cli.add_command(cross_validation)
cli.add_command(get_predictions)

if __name__ == '__main__':
    cli()
