import json
import numpy as np
import os
import subprocess
import sklearn.feature_extraction

WORKING_DIR = os.getcwd()
IMDB_FILE = 'imdb.npz'

def download_file_from_gcs(source, destination):
    """Download files from GCS to WORKING_DIR/.

    Args:
        source: GCS path to the training data
        destination: GCS path to the validation data.
    Returns:
        The local data paths where the data is downloaded.
    """

    local_file_names = [destination]
    print("Local File Names: ", local_file_names)
    gcs_input_paths = [source]
    print("GCS Input Paths: ", gcs_input_paths)

    raw_local_files_data_paths = [os.path.join(WORKING_DIR, local_file_name)
                                  for local_file_name in local_file_names]
    print("Raw Local Files Data Paths: ", raw_local_files_data_paths)

    for i, gcs_input_path in enumerate(gcs_input_paths):
        if gcs_input_path:
            subprocess.check_call(['gsutil', 'cp', gcs_input_path, raw_local_files_data_paths[i]])

    return raw_local_files_data_paths

def _load_data(path):
    """Loads the IMDB Dataset in npz format.

    Args:
        path: (str) The GCS URL to download from (e.g. 'gs://bucket/file.csv')

    Returns:
        A tuple of numpy arrays: '(_train, y_train), (x_test, y_test)'.

    Raises:
        ValueError: In case path is not defined.
    """

    if not path:
        raise ValueError('No training file defined')
    if path.startswith('gs://'):
        download_file_from_gcs(path, destination=IMDB_FILE)
        path = IMDB_FILE
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

    np.random.seed(42)
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]

    indices = np.arange(len(x_test))
    np.random.shuffle(indices)
    x_test = x_test[indices]
    y_test = y_test[indices]

    return (x_train, y_train), (x_test, y_test)

def preprocess(data_file):
    """Loads IMDB Sentiment npz file, and applies an Sklearn CountVectorizer.

    Args:
        data_file: (str) Location of file.

    Returns:
        A tuple of training and test data.
    """

    (train_data, train_labels), (test_data, test_labels) = _load_data(path=data_file)
    vect = sklearn.feature_extraction.text.CountVectorizer(min_df=5, ngram_range=(2,2))
    x_train = vect.fit(train_data).transform(train_data)
    x_test = vect.transform(test_data)

    return (x_train, train_labels), (x_test, test_labels)