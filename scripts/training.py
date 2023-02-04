"""
This script is for training the Decision Tree.

@author: Ujjwal Garg
@email: ujjwalg3@gmail.com
"""
import os
import numpy as np
import pandas as pd
import pickle
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from constants import (
    FEATURES, LABEL, DATA_FILE_PATH, TEST_FILE_PATH, MODEL_PATH,
    MAX_DEPTH, MIN_SAMPLES_LEAF, MINS_SAMPLE_SPLIT
)
from sklearn.tree import DecisionTreeRegressor
from utils.utils import r2_score_adjusted
from utils.logger import train_logger


def read_data(train_size=0.7, validation_size=0.2, test_size=0.1, shuffle=True):
    """
    Here we are splitting data after shuffling and
    saving some data (e.g. 10%) for testing using flask API
    :param train_size:
    :param validation_size:
    :param test_size:
    :param shuffle:
    :return: training_dataframe, validation_dataframe
    """
    data = pd.read_csv(DATA_FILE_PATH)
    train_logger.info("Total data - %s", data.shape[0])

    # shuffle data
    if shuffle:
        data = data.sample(frac=1)

    # splitting data into three sets
    df = data[FEATURES + LABEL]
    va_size = train_size + validation_size
    _split = [int(train_size * len(df)), int(va_size * len(df))]
    df_train, df_validate, df_test = np.split(df, _split)
    # write test data locally and return train and validation data
    df_test.to_csv(TEST_FILE_PATH, index=False)
    train_logger.info("Test data file created of lenght - %s", df_test.shape[0])
    return df_train, df_validate


def train():
    train_logger.info("starting training...")
    train_data, validation_data = read_data()
    train_x, train_y = train_data[FEATURES], train_data[LABEL]
    valid_x, valid_y = validation_data[FEATURES], validation_data[LABEL]
    tree_regressor = DecisionTreeRegressor(max_depth=MAX_DEPTH,
                                           min_samples_leaf=MIN_SAMPLES_LEAF,
                                           min_samples_split=MINS_SAMPLE_SPLIT)
    tree_regressor.fit(X=train_x, y=train_y)
    train_logger.info("Decision Tree Regressor trained.")
    pred_y = tree_regressor.predict(valid_x)
    r2 = r2_score_adjusted(y_pred=pred_y, y_true=valid_y)
    train_logger.info("Adjusted R2 - %s", r2)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(tree_regressor, f)
    train_logger.info("Model saved.")


if __name__ == '__main__':
    train()
