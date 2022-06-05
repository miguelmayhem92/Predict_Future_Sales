import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import random
import seaborn as sns; sns.set()
from dateutil.relativedelta import relativedelta

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from especial_functions import augmentation_reduction

def exploiting_words(data):
    items_vector = list(data.item_name.unique())
    items_vector = pd.DataFrame({'item_name': items_vector})
    items_vector['lenstr'] = items_vector.item_name.str.len() ## this can be a feature
    items_vector['NWords'] = items_vector.item_name.str.split().str.len()
    items_vector['ItemWord0'] = items_vector.item_name.str.split().str.get(0)
    items_vector['ItemWord_1'] = items_vector.item_name.str.split().str.get(-1)
    
    def my_tokengen(data, columnx):
        unique_words = list()
        for string in  list(data[columnx].values):
            for word in string.split():
                unique_words.append(word)
        unique_words = set(unique_words)
        unique_words = list(unique_words)
        unique_words.sort()
        number = [idx[0] +1 for idx in enumerate(unique_words)]
        map_word = {x:y for x,y in zip(unique_words, number)}

        data[columnx] = data[columnx].map(map_word)
        return data
    
    items_vector = my_tokengen(items_vector, 'ItemWord0')
    items_vector = my_tokengen(items_vector, 'ItemWord_1')
    items_vector = items_vector.merge(data[['item_name','item_id']], on = 'item_name', how = 'left').drop(columns = 'item_name')
    
    return {'item_name_explo':items_vector}

def cleaning_shop_categs(shops, categories):
    shops[['shop_comp1','shop_comp2']] = shops.shop_name.str.split(' "',expand=True,)
    shops['shop_comp1'] = shops.shop_comp1.replace({r'\([^)]*\)' : ''}, regex=True)
    shops['shop_comp1'] = shops['shop_comp1'].str.split(' ').str[0]

    shops['shop_comp2'] = shops.shop_comp2.replace({r'\([^)]*\)' : ''}, regex=True)
    shops['shop_comp2'] = shops.shop_comp2.replace({'"' :''}, regex=True)

    categories[['categ_comp1','categ_comp2']] = categories.item_category_name.str.split(" - ",expand=True,)
    categories['categ_comp1'] = categories.categ_comp1.replace({r'\([^)]*\)' : ''}, regex=True)
    categories['categ_comp2'] = categories.categ_comp2.replace({r'\([^)]*\)' : ''}, regex=True)
    
    return shops, categories

def indexing_shop_categs(data, column):
    my_map = dict()
    for index, value in enumerate(data[column].unique()):
        my_map[value] = index
    data[f'map_{column}'] = data[column].map(my_map)
    return data

def merge_basic(train_data, test_data, test_how = 'inner'):
    train_merge = train_data.merge(items, on = 'item_id', how = 'left')\
        .merge(categories, on = 'item_category_id', how = 'left')\
        .merge(shops, on = 'shop_id', how = 'left')\
        .merge(test_data, on = ['shop_id','item_id'], how = test_how)\

    train_merge['SaleDate'] = pd.to_datetime(train_merge['date'], format = '%d.%m.%Y')
    
    return train_merge

def convert_bronze(data, train = False, test_order = 34):
    if train:
        data['date'] = pd.to_datetime(data['date'], format = '%d.%m.%Y')
        
    data['SaleDateMY'] = data['date'].dt.to_period('M').dt.to_timestamp()
    data['item_cnt_day'] = data['item_cnt_day'].clip(0,20)
    
    data = data.groupby(['SaleDateMY', 'date_block_num', 'shop_id', 'item_id'],
                                ).agg( Sale = ('item_cnt_day', 'sum'),
                                   item_price = ('item_price', 'mean'))
    data = data.reset_index().rename(columns = {'SaleDateMY':'Date'})
    
    data['min_date'] = data.groupby(['shop_id', 'item_id'])['Date'].transform('min')

    ### completing with 0 items with 0 in no dates
    
    map_aggg = data[['Date','item_id','shop_id','min_date','Sale']]\
        .pivot_table(index = ['item_id','shop_id','min_date'], columns = ['Date'], values = 'Sale', aggfunc = 'count').reset_index()
    map_aggg = map_aggg.melt(id_vars=['item_id','shop_id','min_date'], value_vars = map_aggg.columns[2:],  var_name='Date', value_name='Nothing')
    map_aggg = map_aggg[map_aggg.Date >= map_aggg.min_date]
    map_aggg = map_aggg[['item_id','shop_id','Date']]
    
    group_data = map_aggg.merge(data, on = ['item_id','shop_id','Date'], how = 'left').fillna(0)
    group_data['Sale'] = group_data['Sale'].clip(0,20)
    
    del map_aggg
    
    if train:
        ### get a map of the dates
        date_list =[group_data.Date.min()]
        date = group_data.Date.min()
        max_date = group_data.Date.max()
        while date <= max_date:
            date = date_list[-1] + relativedelta(months=1)
            date_list.append(date)
        order_list = range(1,len(date_list)+1)
        date_map = pd.DataFrame({'Date':date_list, 'Order':order_list})
        group_data = group_data.merge(date_map, on = 'Date', how = 'left')
    group_data = group_data.drop(columns = ['date_block_num', 'min_date'])
    return group_data

def merge_basic(train_data, items, items_feature ,categories, shops):
    train_merge = train_data.merge(items, on = 'item_id', how = 'left')
    train_merge = train_merge.merge(items_feature, on = 'item_id', how = 'left')
    train_merge = train_merge.merge(categories, on = 'item_category_id', how = 'left')
    train_merge = train_merge.merge(shops, on = 'shop_id', how = 'left')

    return train_merge

def supagg(data):
    data['lit'] = data.groupby(['item_id'])['shop_id'].transform('nunique')
    agge = data.groupby(['item_id','shop_id']).agg(lit = ('lit','max')).reset_index()
    return agge

def  reduce_sample(data, frac, single = False ):
    if single:
        data = data.groupby(['item_id','shop_id']).agg(lit = ('lit','max')).reset_index().sample(frac = frac, random_state = 123)
    else:
        data = data.sample(frac = frac, random_state = 123)
        data = data[['item_id','shop_id','lit']]
    return data

def get_val_window(data, date, items_to_drop_all):
    val_window_1 = data[data.Date == date ]
    already_found = val_window_1.shopitem.unique()
    
    val_window_2 = data[ (data.Date != date) & (data.shopitem.isin(items_to_drop_all)) ]
    val_window_2['Order'] = val_window_2.groupby(['shop_id','item_id']).cumcount() + 1
    val_window_2 = val_window_2[val_window_2.Order == 1]
    val_window_2['Date'] = date
    val_window_2 = val_window_2[~val_window_2.shopitem.isin(already_found)]
    val_window_2['Sale'] = 0
    print(len(val_window_1),len(val_window_2))
    val_window = pd.concat([val_window_1,val_window_2])
    return val_window


def inversed_scale(scaler, data, target_name, y_pred = np.array([])):
    data_new = data.copy()
    if y_pred.any():
        data_new[target_name] = y_pred
    
    unscaled_data = scaler.inverse_transform(data_new)
    unscaled_data = pd.DataFrame(unscaled_data, columns = data_new.columns)
    return unscaled_data

def consolidation_prediction(data, prediction ):
    pdf_plot = data.copy()
    #pdf_plot['PredictedVar'] = prediction
    pdf_plot['PredSale'] = prediction
    #pdf_plot['RealValueVar'] = pdf_plot.L0M_L1M
    #pdf_plot = pdf_plot.assign(PredSale = pdf_plot.PredictedVar + pdf_plot.SaleL1M)
    pdf_plot = pdf_plot[['Date','shop_id','item_id','Sale','PredSale']]
    return pdf_plot

def plot_prediction(dfplot):
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10,8))

    ax = sns.scatterplot(ax = axs, x="Sale", y="PredSale", data=dfplot)

    x0, x1 = axs.get_xlim()
    y0, y1 = axs.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    axs.plot(lims, lims, '-r')

    axs.set(xlabel='Real Sale', ylabel='Predicted Sale', title='Scatter plot of variation predictions')
    
def preparing_to_predict(val_data, train_data):
    val_data_go = val_data.copy()
    val_data_go['date_block_num'] = np.nan
    val_data_go['shopitem'] = val_data_go['shop_id'].astype('str') + '-' + val_data_go['item_id'].astype('str')
    val_data_go = val_data_go[train_data.columns]
    items = val_data_go.shopitem.unique()
    result_data = pd.concat([train_data[train_data.shopitem.isin(items)],val_data_go])
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

"""test_dates = [datetime.datetime(2014, 11, 1), datetime.datetime(2015, 4, 1), datetime.datetime(2015, 5, 1), datetime.datetime(2015, 6, 1), datetime.datetime(2015, 7, 1), datetime.datetime(2015, 8, 1),
datetime.datetime(2015, 9, 1), datetime.datetime(2015, 10, 1),]


to_drop_columns = ['Date']
ids_columns = ['item_category_id',
       'ItemIdPart1','ItemIdPart2', 'CategIdItem3A', 'CategIdItem3B'] #'item_id'
numericals = [ 'Quarter', 'OrderGot', 'countItemPossitives',
       'countItemZero', 'L1maxLev1Item', 'L2maxLev1Item', 'L3maxLev1Item',
       'L4maxLev1Item', 'L5maxLev1Item', 'lenstr', 'NWords',
       'Word0', 'Word_1', 'L1maxLev1Shop', 'meanLaggedVars',
       'item_id_mean_LastMonth', 'item_id_max_LastMonth',
       'item_id_mean_LastOrder', 'item_id_max_LastOrder', 'meanL1Price',
       'SpaceOrder'] + ids_columns

categoricals = ['SpaceOrderActivator', 'CategoryL1Item']
my_features = to_drop_columns + numericals + categoricals
my_target = 'Sale'

my_columns_to_drops = ['Quarter','CategoryL1Item_C', 'CategoryL1Item_A', 'CategoryL1Item_D',
       'SpaceOrderActivator_A', 'CategIdItem3B', 'ItemIdPart1',
       'CategIdItem3A']

categoricals_features = ['SpaceOrderActivator_A', 'SpaceOrderActivator_B',
                         'CategoryL1Item_A','CategoryL1Item_B','CategoryL1Item_C','CategoryL1Item_D']

if len(categoricals) != 0:
    final_features = categoricals_features +  numericals
else:
    final_features = numericals
final_features = [x for x in final_features if x not in my_columns_to_drops]


"""

def My_ML_prediction_on_test( data_dict , dates_vector, model, features_dict, zero_vector, frac = 0.80):
    
    numericals = features_dict['numericals']
    categoricals = features_dict['categoricals']
    my_columns_to_drops = features_dict['features_to_drop']
    my_target = features_dict['my_target']
    
    data_result = list()

    for datex in dates_vector:
        datex_str = datex.strftime('%Y-%m-%d')
        #### reading files
        train_selection = data_dict['Train Data'][datex_str]
        val_selection = data_dict['Validation Data'][datex_str]
        train_augmented = augmentation_reduction(train_selection, fracs = zero_vector)
        
        train_scaled, my_scaler = scaler(train_augmented, numericals, my_target, scaler=None, drop_columns = my_columns_to_drops)
        #train_scaled = train_augmented[ numericals + [my_target] ]
        if len(categoricals) != 0:
            train_dummies = pd.get_dummies(train_augmented[categoricals])
            train_scaled = pd.concat([train_dummies,train_scaled],axis = 1)
            
        if len(categoricals) != 0:
            final_features = list(train_dummies.columns) +  numericals
        else:
            final_features = numericals
        final_features = [x for x in final_features if x not in my_columns_to_drops]
        
        train_scaled_sampled = train_scaled.sample(frac = frac, random_state = 12489)
        
        ### ML train
        X_train = train_scaled_sampled[final_features]
        Y_train = train_scaled_sampled[my_target]
        model.fit(X_train, Y_train)
        
        ### Test data prepa
        val_scaled = scaler(val_selection, numericals, my_target, scaler=my_scaler,drop_columns = my_columns_to_drops )
        
        if len(categoricals) != 0:
            val_dummies = pd.get_dummies(val_selection[categoricals])
            val_scaled_full = pd.concat([val_dummies,val_scaled],axis = 1)
        else:
            val_scaled_full = val_scaled
        
        X_val = val_scaled_full[final_features]
        
        ## Prediction
        Y_pred = model.predict(X_val)
        
        ## Saving Result
        predicted_val = inversed_scale(scaler = my_scaler, data = val_scaled, target_name = my_target, y_pred = Y_pred)
        my_lm_plot = consolidation_prediction(data = val_selection, prediction = predicted_val.Sale.values)
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

def get_splited_data(dates_vector):
    train_data = dict()
    validation_data = dict()
    
    for datex in dates_vector:
        datex_str = datex.strftime('%Y-%m-%d')
        train_selection = pd.read_csv(f'generated_datasets/data_{datex_str}/train_selection.csv')
        val_selection = pd.read_csv(f'generated_datasets/data_{datex_str}/val_selection.csv')
        
        train_data[datex_str] = train_selection
        validation_data[datex_str] = val_selection
        print(datex_str)
    return { 'Train Data' : train_data, 
            'Validation Data': validation_data}