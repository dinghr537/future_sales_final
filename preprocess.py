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
parser.add_argument('--train',
                    default='./sales_train.csv',
                    help='input training file path')
parser.add_argument('--test',
                   default='./test.csv',
                   help='input testing data file path')
parser.add_argument('--shops',
                    default='./shops.csv',
                    help='input shops file path')
parser.add_argument('--items',
                    default='./items.csv',
                    help='input items file path')
parser.add_argument('--item_categories',
                    default='./item_categories.csv',
                    help='input items file path')
parser.add_argument('--output',
                    default='./df.pkl',
                    help='output pickle file path')

args = parser.parse_args()

print(f"start to read data...", end="", flush=True)
try:
    test = pd.read_csv(args.test)
    sales = pd.read_csv(args.train)
    shops = pd.read_csv(args.shops)
    items = pd.read_csv(args.items)
    item_cats = pd.read_csv(args.item_categories)
except:
    print("\nError: Can't find some of the input files, please check the file path below")
    print(args)
    quit()

print("...Done.(1/5)")
# test:  	ID 	shop_id 	item_id
# sales: date 	date_block_num 	shop_id 	item_id 	item_price 	item_cnt_day
#shops: shop_name shop_id
# items: item_name item_id item_category_id
# item_cats: item_category_name item_category_id



# Data cleaning
# sns.boxplot(x=sales.item_cnt_day)
# sns.boxplot(x=sales.item_price)
print(f"start to clean the data...", end="", flush=True)
# Remove outlier
train = sales[(sales.item_price < 300000 )& (sales.item_cnt_day < 1000)]
# remove negative item price
train = sales[sales.item_price > 0].reset_index(drop = True)


# Detect same shops
# print(shops[shops.shop_id.isin([0, 57])]['shop_name'])
# print(shops[shops.shop_id.isin([1, 58])]['shop_name'])
# print(shops[shops.shop_id.isin([40, 39])]['shop_name'])

train.loc[train.shop_id == 0, 'shop_id'] = 57
test.loc[test.shop_id == 0, 'shop_id'] = 57

train.loc[train.shop_id == 1, 'shop_id'] = 58
test.loc[test.shop_id == 1, 'shop_id'] = 58

train.loc[train.shop_id == 40, 'shop_id'] = 39
test.loc[test.shop_id == 40, 'shop_id'] = 39

print("...Done.(2/5)")

# train: date 	date_block_num 	shop_id 	item_id 	item_price 	item_cnt_day
# test:  	ID 	shop_id 	item_id

print(f"start to prepare existed features...", end="", flush=True)

index_cols = ['shop_id', 'item_id', 'date_block_num']
# 對於每一個月，找到這一個月有的所有的 item_id 和 shop_id
df = [] 
for block_num in train['date_block_num'].unique():
    cur_shops = train.loc[sales['date_block_num'] == block_num, 'shop_id'].unique()
    cur_items = train.loc[sales['date_block_num'] == block_num, 'item_id'].unique()
    df.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))

df = pd.DataFrame(np.vstack(df), columns = index_cols,dtype=np.int32)

# 針對上面要的每一項，把 item_cnt_day 給 sum 起來，得到一整個月的數據
group = train.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_cnt_day': ['sum']})
group.columns = ['item_cnt_month']
group.reset_index(inplace=True)

df = pd.merge(df, group, on=index_cols, how='left')
df['item_cnt_month'] = (df['item_cnt_month']
                                .fillna(0)
                                .clip(0,20)
                                .astype(np.float16))
# df: shop_id 	item_id 	date_block_num 	item_cnt_month

# 將 test 裡的 data 加到 df 裡
test['date_block_num'] = 34
test['date_block_num'] = test['date_block_num'].astype(np.int8)
test['shop_id'] = test['shop_id'].astype(np.int8)
test['item_id'] = test['item_id'].astype(np.int16)
df = pd.concat([df, test], ignore_index=True, sort=False, keys=index_cols)
df.fillna(0, inplace=True)


# 從 shop_name 中取出所在城市的名字
shops['city'] = shops['shop_name'].apply(lambda x: x.split()[0].lower())
# 處理例外城市情況
shops.loc[shops.city == '!якутск', 'city'] = 'якутск'
# 將城市名字 encode 後加入到 data 中
shops['city_code'] = LabelEncoder().fit_transform(shops['city'])
shops = shops[['shop_id', 'city_code']]
df = pd.merge(df, shops, on=['shop_id'], how='left')
# df: shop_id 	item_id 	date_block_num 	item_cnt_month 	ID 	city_code


# 將商品的類別整理出來
items = pd.merge(items, item_cats, on='item_category_id')
items['item_category_code'] = LabelEncoder().fit_transform(items['item_category_name'])
items = items[['item_id', 'item_category_code']]
df = pd.merge(df, items, on=['item_id'], how='left')

print("...Done.(3/5)")

print(f"start to prepare first interaction features...", end="", flush=True)
# 新增一項 feature，標記商品是否是第一次售出，與商品第一次售出的時間
first_item_block = df.groupby(['item_id'])['date_block_num'].min().reset_index()
first_item_block['item_first_interaction'] = 1
# first_item_block: item_id 	date_block_num 	item_first_interaction

df = pd.merge(df, first_item_block[['item_id', 'date_block_num', 'item_first_interaction']], on=['item_id', 'date_block_num'], how='left')
df['item_first_interaction'].fillna(0, inplace=True)
df['item_first_interaction'] = df['item_first_interaction'].astype('int8')  
print("...Done.(4/5)")

print(f"start to prepare lag features...", end="", flush=True)
# **lag features**
def lag_feature(df, lags, col):
    tmp = df[['date_block_num','shop_id','item_id',col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)]
        shifted['date_block_num'] += i
        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')
        df[col+'_lag_'+str(i)] = df[col+'_lag_'+str(i)].astype('float16')
    return df

#Add sales lags for last 4 months
df = lag_feature(df, [1, 2, 3, 4], 'item_cnt_month')
print("...Done.(5/5)")

# Remove data for the first three months
print("Ready.", flush=True)
df.fillna(0, inplace=True)
df = df[(df['date_block_num'] > 2)]

print(df.columns)

#Save dataset
df.drop(['ID'], axis=1, inplace=True, errors='ignore')
df.to_pickle(args.output)
print("All Done.")

