import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data',
                    default='./df.pkl',
                    help='preprocessed data path')
parser.add_argument('--test',
                   default='./test.csv',
                   help='input testing data file path')
parser.add_argument('--output',
                   default='./submission.csv',
                   help='output file path')

args = parser.parse_args()
print(args)

print("start to read data...", end="", flush=True);
try:
    df = pd.read_pickle(args.data)
except:
    print("\nError: can't find pickle data")
    quit()
# df.info()

X_train = df[df.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = df[df.date_block_num < 33]['item_cnt_month']
X_valid = df[df.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = df[df.date_block_num == 33]['item_cnt_month']
X_test = df[df.date_block_num == 34].drop(['item_cnt_month'], axis=1)
del df
print("...Done!")

# ready to train

feature_name = X_train.columns.tolist()

params = {
    'objective': 'mse',
    'metric': 'rmse',
    'num_leaves': 2 ** 7 - 1,
    'learning_rate': 0.005,
    'feature_fraction': 0.75,
    'bagging_fraction': 0.75,
    'bagging_freq': 5,
    'seed': 1,
    'verbose': 1
}

feature_name_indexes = [ 
                        'item_category_code', 
                        'city_code',
]

lgb_train = lgb.Dataset(X_train[feature_name], Y_train)
lgb_eval = lgb.Dataset(X_valid[feature_name], Y_valid, reference=lgb_train)

evals_result = {}

# Train
gbm = lgb.train(
        params, 
        lgb_train,
        num_boost_round=3000,
        valid_sets=(lgb_train, lgb_eval), 
        feature_name = feature_name,
        categorical_feature = feature_name_indexes,
        verbose_eval=5, 
        evals_result = evals_result,
        early_stopping_rounds = 100)

print("\n\n========== Done ==========\n")
# plot 
# lgb.plot_importance(
#     gbm, 
#     max_num_features=50, 
#     importance_type='gain', 
#     figsize=(12,8));

# Save submission.csv
print("start to save submission.csv...", end="", flush=True)
test = pd.read_csv(args.test)
Y_test = gbm.predict(X_test[feature_name]).clip(0, 20)

submission = pd.DataFrame({
    "ID": test.index, 
    "item_cnt_month": Y_test
})
submission.to_csv(args.output, index=False)
print("...Done")

