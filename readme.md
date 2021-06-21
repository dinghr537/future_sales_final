## DSAI HW4

python version: 3.7.10

16G RAM is recommended

### 執行方式

```shell
# 準備環境
pip install -r requirements.txt
# 資料前處理
python preprocess.py --train=./sales_train.csv --test=./test.csv --shops=./shops.csv --items=./items.csv --item_categories=./item_categories.csv --output=./df.pkl
# 跑model
python train.py --data=./df.pkl --test=./test.csv --output=./submission.csv
```

### Report link

> https://docs.google.com/presentation/d/1s3DJAFT36DrpRfO-XDy6-7hYQ0RFCzJ6WPss8Rn28EE/edit?usp=sharing

