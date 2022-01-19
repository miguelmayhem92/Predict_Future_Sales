import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import seaborn as sns; sns.set()

daily_train = pd.read_csv('sales_train.csv')
items = pd.read_csv('items.csv')
shops = pd.read_csv('shops.csv')
categories = pd.read_csv('item_categories.csv')
sample_submission = pd.read_csv('sample_submission.csv')
test = pd.read_csv('test.csv')

def merge_basic(train_data, test_data):
    train_merge = train_data.merge(items, on = 'item_id', how = 'left')\
        .merge(categories, on = 'item_category_id', how = 'left')\
        .merge(test_data, on = ['shop_id','item_id'], how = 'inner')\
        .drop(columns = ['item_category_id'])

    train_merge['SaleDate'] = pd.to_datetime(train_merge['date'], format = '%d.%m.%Y')
    
    return train_merge

def convert_bronze(data):
    data_result = data.assign(TransactionType = np.where(data.item_cnt_day <= 0, 'SaleBack', 'Sale'))
    data_result['SaleDateMY'] = data_result['SaleDate'].dt.to_period('M').dt.to_timestamp()
    result = data_result.pivot_table(index = ['SaleDateMY', 'date_block_num', 'shop_id', 'item_id', 'TransactionType'], values = 'item_cnt_day',aggfunc = 'sum')
    result = result.reset_index()
    return result