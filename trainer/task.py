import datetime
import os
import subprocess
import sys
import argparse
import xgboost as xgb
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import numpy as np


from . import utils


def get_args():
    """Argument parser

    Returns:
        Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--job-dir',
        type=str,
        required=True,
        help='GCS location to write checkpoints and export models')
    parser.add_argument(
        '--train-file',
        type=str,
        required=True,
        help='Training file local or GCS')
    parser.add_argument(
        '--bucket-name',
        type=str,
        default='project-4-nlp-mlengine',
        help='The bucket name in which the finished model is saved'
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=100,
        help='number of times to go through the data, default=100')
    parser.add_argument(
        '--learning-rate',
        default=0.1,
        type=float,
        help='learning rate for XGBoost')
    parser.add_argument(
        '--n-estimators',
        default=50,
        type=int,
        help='number of gradient boosted decision trees')


    return parser.parse_args()

def train_and_evaluate(hparams):
    """Helper function: Trains and evaluates model.

    Args:
        args: a dictionary of arguments - see get_args() for details

        """
    # load and preprocess data
    (x_train, y_train), (x_test, y_test) = utils.preprocess(hparams.train_file)

    # convert data to a DMatrix for XGBoost
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)

    # configure XGBoost parameters
    param = {'max_depth' : 4, 'eta': hparams.learning_rate,
             'objective': 'binary:logistic', 'eval_metric' : 'logloss',
             'n_estimators':hparams.n_estimators}
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    num_round = hparams.num_epochs

    # train XGBoost
    bst = xgb.train(param, dtrain, num_round, evallist)

    # evaluate on test set and calculate RMSE
    preds = bst.predict(dtest)
    rmse = round(np.sqrt(mean_squared_error(y_test, preds)), 3)

    # Save the model locally
    model_filename = '{}_RMSE.bst'.format(rmse)
    bst.save_model(model_filename)
    model_folder = datetime.datetime.now().strftime('imdb_%Y%m%d_%H%M%S')

    # Upload the saved model to the bucket.
    gcs_model_path = os.path.join('gs://', hparams.bucket_name, model_folder, model_filename)
    subprocess.check_call(['gsutil', 'cp', model_filename, gcs_model_path], stderr=sys.stdout)


if __name__ == '__main__':

    args = get_args()
    hparams = tf.contrib.training.HParams(**args.__dict__)
    train_and_evaluate(hparams)