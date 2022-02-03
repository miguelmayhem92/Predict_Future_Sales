import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import seaborn as sns; sns.set()
from dateutil.relativedelta import relativedelta

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

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

    train_merge['SaleDate'] = pd.to_datetime(train_merge['date'], format = '%d.%m.%Y')
    
    return train_merge

def convert_bronze(data):
    data_result = data.assign(TransactionType = np.where(data.item_cnt_day <= 0, 'SaleBack', 'Sale'))
    data_result['SaleDateMY'] = data_result['SaleDate'].dt.to_period('M').dt.to_timestamp()
    result = data_result.pivot_table(index = ['SaleDateMY', 'date_block_num', 'shop_id', 'item_id', 'TransactionType'], values = 'item_cnt_day',aggfunc = 'sum')
    result = result.reset_index()
    return result

def convert_silver(data):
    data_result = data.assign(TransactionType = np.where(data.item_cnt_day <= 0, 'SaleBack', 'Sale'))
    data_result['SaleDateMY'] = data_result['SaleDate'].dt.to_period('M').dt.to_timestamp()
    result = data_result.pivot_table(index = ['SaleDateMY', 'date_block_num', 'shop_id', 'item_id','item_category_name','item_category_id'], values = 'item_cnt_day',aggfunc = 'sum')
    result = result.reset_index()
    result = result.assign(shopitem = result.shop_id.astype('str') +'-'+ result.item_id.astype('str') )
    return result

def completion_semi_gold(data):

    max_date = data.SaleDateMY.max()
    min_date = max_date - relativedelta(months=30)

    ### completing with 0 items with 0 in no dates
    result_data = data[(data.SaleDateMY >= min_date) & (data.SaleDateMY <= max_date)].pivot_table(index = ['shop_id', 'item_id','shopitem','item_category_id'], columns ='SaleDateMY', values = 'item_cnt_day', aggfunc = 'sum').reset_index().fillna(0)
    group_data = result_data.melt(id_vars=['shop_id', 'item_id','shopitem','item_category_id'], value_vars = result_data.columns[4:],  var_name='Date', value_name='Sale')

    ### get a map of the dates
    date_list =[group_data.Date.min()]
    date = group_data.Date.min()
    while date <= max_date:
        date = date_list[-1] + relativedelta(months=1)
        date_list.append(date)
    order_list = range(1,len(date_list)+1)
    date_map = pd.DataFrame({'Date':date_list, 'Order':order_list})
    data_maped = group_data.merge(date_map, on = 'Date', how = 'left')

    return data_maped

#def scaler(dataset, features, target, scaler=None):
#    if scaler:
#        df = dataset[features + [target]]
#        df_scaled = scaler.transform(df)
#        dataset_scaled = pd.DataFrame(df_scaled, columns = df.columns)
#        return dataset_scaled
#    
#    else:
#        df  = dataset[features + [target]]
#        scaler = MinMaxScaler()
#        scaler.fit(df)
#
#        dataset_scaled = scaler.transform(df)
#        dataset_scaled = pd.DataFrame(dataset_scaled, columns = df.columns)
#        return dataset_scaled, scaler

def inversed_scale(scaler, data, target_name, y_pred = np.array([])):
    data_new = data.copy()
    if y_pred.any():
        data_new[target_name] = y_pred
    
    unscaled_data = scaler.inverse_transform(data_new)
    unscaled_data = pd.DataFrame(unscaled_data, columns = data_new.columns)
    return unscaled_data

def consolidation_prediction(data, prediction ):
    pdf_plot = data.copy()
    pdf_plot['PredictedVar'] = prediction
    pdf_plot['RealValueVar'] = pdf_plot.L0M_L1M
    pdf_plot = pdf_plot.assign(PredSale = pdf_plot.PredictedVar + pdf_plot.SaleL1M)
    pdf_plot = pdf_plot[['Date','Sale','PredSale', 'PredictedVar', 'RealValueVar', 'SaleL1M']]
    return pdf_plot

def plot_prediction(dfplot):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20,8))

    ax = sns.scatterplot(ax = axs[0], x="RealValueVar", y="PredictedVar", data=dfplot)
    ax = sns.scatterplot(ax = axs[1], x="Sale", y="PredSale", data=dfplot)

    x0, x1 = axs[0].get_xlim()
    y0, y1 = axs[0].get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    axs[0].plot(lims, lims, '-r')

    x0, x1 = axs[1].get_xlim()
    y0, y1 = axs[1].get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    axs[1].plot(lims, lims, '-r')

    axs[0].set(xlabel='Real Variation', ylabel='Predicted Variation', title='Scatter plot of variation predictions')
    
def preparing_to_predict(val_data, train_data):
    val_data_go = val_data.copy()
    val_data_go['date_block_num'] = np.nan
    val_data_go['shopitem'] = val_data_go['shop_id'].astype('str') + '-' + val_data_go['item_id'].astype('str')
    val_data_go = val_data_go[train_data.columns]
    result_data = pd.concat([train_data,val_data_go])
    return result_data

def plot_metrics(data):
    resultdict = {'Date':list(), 'RMSE':list()}
    for datex in data.Date.unique():
        df = data[data.Date == datex]
        rmse = (mean_squared_error(df.Sale, df.PredSale))**(1/2)
        resultdict['Date'].append(datex)
        resultdict['RMSE'].append(rmse)
    result = pd.DataFrame(resultdict)
    return result
#########################################################
### The following functions follow a fixed sstructure ###
#########################################################

test_dates = [datetime.datetime(2014, 11, 1), datetime.datetime(2015, 4, 1), datetime.datetime(2015, 5, 1), datetime.datetime(2015, 6, 1), datetime.datetime(2015, 7, 1), datetime.datetime(2015, 8, 1),
datetime.datetime(2015, 9, 1), datetime.datetime(2015, 10, 1),]

to_drop_columns = ['Date','Sale','SaleL1M']
numericals = ['L1M_L2M', 'L2M_L3M', 'L3M_L4M','L1M_L11M', 'L1M_L12M', 'L1M_L13M', 'possCounts','maxSection', 'L11M', 'L12M', 'L13M', 'Roll0count',
       'Roll0L1', 'Roll0L1L6', 'Roll0L1L12', 'Roll0countL6', 'Roll0countL12','month', 'maxSaleL1M', 'noVar', 'countNoVar', 'FlagShop','monthToPredict',
       'FlagCategory', 'SumBefExplotion', 'labeling', 'meanVarExploItem','meanVarExploShop', 'countMonth', 'medianVarL12', 'SeasonalVariation','HighVarL12' ]
categoricals = ['categVolume', 'categSale']
my_features = to_drop_columns + numericals + categoricals
my_target = 'L0M_L1M'

my_columns_to_drops = ['noVar', 'categSale_A','labeling','categSale_D','categVolume_G','categVolume_F','categVolume_F','categVolume_H','Roll0count','categSale_C','categVolume_C','categVolume_D','FlagShop','maxSaleL1M']
dummy_columns = ['categVolume_A','categVolume_B','categVolume_C','categVolume_D','categVolume_F','categVolume_G',
                 'categSale_A','categSale_B','categSale_C','categSale_D']
final_features = dummy_columns +  numericals
final_features = [x for x in final_features if x not in my_columns_to_drops]

def my_kfold_crossval_and_Hptunning(models_toTrain,frac = 1.0):

    results = {'machine label': list(),
        'machine result': list()}

    for machine,i in zip(models_toTrain, range(1,len(models_toTrain) + 1)):
        label = f'machine-{i}'
        results['machine label'].append(label)
        df_result = My_ML_prediction_on_test( dates_vector = test_dates, model = machine, frac = frac)
        results['machine result'].append(df_result)
        print(label + ' is done')
        
    return results

def My_ML_prediction_on_test(dates_vector, model, frac = 0.80):

    data_result = list()

    for datex in dates_vector:
        datex_str = datex.strftime('%Y-%m-%d')
        #### reading files
        train_selection = pd.read_csv(f'generated_datasets/data_{datex_str}/train_selection.csv')
        val_selection = pd.read_csv(f'generated_datasets/data_{datex_str}/val_selection.csv')
        
        train_scaled, my_scaler = scaler(train_selection, numericals, my_target, scaler=None, drop_columns = my_columns_to_drops)
        train_dummies = pd.get_dummies(train_selection[categoricals])
        train_scaled = pd.concat([train_dummies,train_scaled],axis = 1).sample(frac = frac)

        ### ML train
        X_train = train_scaled[final_features]
        Y_train = train_scaled[my_target]
        model.fit(X_train, Y_train)
        
        ### Test data prepa
        val_scaled = scaler(val_selection, numericals, my_target, scaler=my_scaler,drop_columns = my_columns_to_drops )
        val_dummies = pd.get_dummies(val_selection[categoricals])
        val_scaled_full = pd.concat([val_dummies,val_scaled],axis = 1)
        
        X_val = val_scaled_full[final_features]
        
        ## Prediction
        Y_pred = model.predict(X_val)
        
        ## Saving Result
        predicted_val = inversed_scale(scaler = my_scaler, data = val_scaled, target_name = my_target, y_pred = Y_pred)
        my_lm_plot = consolidation_prediction(data = val_selection, prediction = predicted_val.L0M_L1M.values)
        data_result.append(my_lm_plot)
        #print(f'the prediction over the {datex} data is done')
        
    return pd.concat(data_result)

def scaler(dataset, features, target, scaler=None, drop_columns = []):
    if scaler:
        features = [x for x in features if x not in drop_columns]
        df = dataset[features + [target]]
        df_scaled = scaler.transform(df)
        dataset_scaled = pd.DataFrame(df_scaled, columns = df.columns, index = df.index)
        return dataset_scaled
    
    else:
        features = [x for x in features if x not in drop_columns]
        df  = dataset[features + [target]]
        scaler = MinMaxScaler()
        scaler.fit(df)

        dataset_scaled = scaler.transform(df)
        dataset_scaled = pd.DataFrame(dataset_scaled, columns = df.columns, index = df.index)
        return dataset_scaled, scaler
    
def consolidated_metrics_MSE(data):
    labels = list()
    RMSEs = list()
    for i in range(len(data['machine label'])):
        label = data['machine label'][i]
        result_detail = data['machine result'][i]
        pred = result_detail['Sale']
        real = result_detail['PredSale']
        rmse = mean_squared_error(real,pred)**(1/2)
        labels.append(label)
        RMSEs.append(rmse)
    
    return pd.DataFrame({'machine':labels,'RMSE': RMSEs}).sort_values('RMSE')

def consolidated_plot_metrics(data):
    list_dfplot = list()
    for i in range(len(data['machine label'])):
        label = data['machine label'][i]
        result_detail = data['machine result'][i]
        df_plot = plot_metrics(result_detail)
        df_plot['machine'] = label
        list_dfplot.append(df_plot)
    
    return pd.concat(list_dfplot)