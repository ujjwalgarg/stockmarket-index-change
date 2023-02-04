import os
import requests
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from constants import FEATURES, TEST_FILE_PATH, MIN_ADJUSTED_R2
from utils.logger import test_logger
from utils.utils import r2_score_adjusted
from scripts.training import train as retrain


def get_prediction_results(test_df):
    result = []
    url = 'http://0.0.0.0:5000/api/predict'
    headers = {
        'content-type': 'application/json',
        'Accept': 'application/json'
    }

    for i, row in test_df.iterrows():
        data = {f: row[f] for f in FEATURES}
        res = requests.post(url, json=data, headers=headers)
        if res.status_code == 200:
            result.append(res.json()['pct_change'])
        else:
            raise requests.HTTPError(
                'Invalid Status Code: {}'.format(res.status_code))

    return result


def main():
    test_df = pd.read_csv(TEST_FILE_PATH)
    test_logger.info('Test data loaded - %s', test_df.shape[0])
    y_test = test_df['pct_change'].values

    results = get_prediction_results(test_df)
    test_logger.info('Got prediction results.')
    y_pred = np.array(results)
    adjusted_r2 = r2_score_adjusted(y_test, y_pred)
    test_logger.info("Adjusted R2 - %s", adjusted_r2)

    if adjusted_r2 < MIN_ADJUSTED_R2:
        test_logger.error("Poor model performance. CALL SENTRY NOW!!!")
        test_logger.info("Re-trainnig model...")
        retrain()
    else:
        test_logger.info("Results are fine. Want me to email them?")

    test_logger.info("Done.")


if __name__ == '__main__':
    main()
