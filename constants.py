from os.path import join, dirname, realpath

REPO_DIR = realpath(dirname(realpath(__file__)))

DATA_DIR = join(REPO_DIR, 'data')
MODEL_DIR = join(REPO_DIR, 'models')

DATA_FILE_PATH = join(DATA_DIR, 'train.csv')
TEST_FILE_PATH = join(DATA_DIR, 'test_data.csv')
MODEL_PATH = join(MODEL_DIR, 'tree.pkl')
LOGGER_PATH = join(REPO_DIR, 'logs')

# these features are hard-coded here because we've selected them after analysis
FEATURES = ['Nifty500_PE', 'Nifty500_PB',
            'Earning_Yields_%', '3M_Yield_India',
            '2Y_Yield_India', '10Y_Yield_India']
NO_OF_FEATURES = len(FEATURES)
LABEL = ['pct_change']

# parameter for training Decision Tree Regressor
# these parameter are hard-coded here because we've selected them after analysis
MAX_DEPTH = 10
MIN_SAMPLES_LEAF = 1
MINS_SAMPLE_SPLIT = 5

# R2 Threshold
MIN_ADJUSTED_R2 = 0.95
