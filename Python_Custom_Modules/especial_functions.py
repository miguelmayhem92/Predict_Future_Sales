import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
from sklearn.cluster import KMeans
import random

items = pd.read_csv('items.csv')

def balance_items_test(data, date_to_take , gen_sample = 0.70, seed = 1234):
    
    data = data[data.Date < date_to_take]
    data['shopitem'] = data.shop_id.astype('str') + ('_') + data.item_id.astype('str')
    ## for general sample
    to_hide = int( round(len(data.shopitem.unique())*gen_sample,0 ))
    items = list(data.shopitem.unique())
    items = [str(x) for x in items]
    random.seed(seed)
    random.shuffle(items)
    
    keep_items = items[0:to_hide]
    data_result = data[data.shopitem.astype('str').isin(keep_items)].drop(columns = 'shopitem')
    
    return data_result

def get_full_lags(whole, date_to_take):
    
    data = whole[ whole.Date < date_to_take]
    data['itemShopMax'] = data.groupby(['shop_id','item_id']).Sale.transform('max')
    data['possnumb'] = np.where(data.Sale > 0 , 1,0)
    data['possCounts'] = data.groupby(['item_id','shop_id']).possnumb.transform('sum')
    
    features_dictionary = dict()
    #########################################
    ### only item Columns  ##################
    ########################################

    date_max = data.Date.max()
    date_max = datetime.datetime(date_max.year, date_max.month, date_max.day)
    
    data['OrderGot'] = data.groupby(['item_id']).Order.transform('min')
    data['SpaceOrder'] = data['Order'] - data['OrderGot']
    
    #########################################
    ### starts  ##################
    ########################################

    starts = data.groupby('item_id').agg(OrderGot = ('OrderGot','max')).reset_index()
    
    features_dictionary['Starts'] = {'data':starts, 'keys':['item_id']}
    
    ########################################
    #### explosionts ###############
    #########################################
    
    def explosion_feature(data, columns_feature, variable_name, lags ):
        df_feature = data[data.SpaceOrder >= 0]
        df_feature['Sale'] = np.where(df_feature['Sale'] == 0, np.nan, df_feature['Sale'])
        df_feature = df_feature.groupby( columns_feature + ['SpaceOrder']).agg( meanx = ('Sale','mean')).reset_index()
        df_feature = df_feature.rename(columns = {'meanx' : f'Explosition_{variable_name}'})
        if lags:
            for L in lags:
                df_feature[f'L{L}_Explosition_{variable_name}'] = df_feature.sort_values('SpaceOrder')\
                    .groupby(columns_feature)[f'Explosition_{variable_name}'].shift(L)

        return {'data' : df_feature, 'keys': columns_feature + ['SpaceOrder']}
        
    features_dictionary['Item explosion'] = explosion_feature(data = data, columns_feature = ['item_id'], variable_name = 'item', lags = [1])
    #features_dictionary['ItemWord explosion'] = explosion_feature(data = data, columns_feature = ['ItemWord0'], variable_name = 'ItemWord0', lags = [1,2])
    
    ####################### ####
    ####### counting ##########
    ############################
    
    def counting_go(data,variable, column, variable_name):
        data_to_use = data
        df_feature = data_to_use.groupby([variable] + ['Date']).agg(countx = (column,'nunique')).reset_index()
        df_feature = df_feature.rename(columns = {'countx': f'count_unique_{variable_name}'})
        df_feature['Date'] = df_feature.Date + pd.DateOffset(months=1)
        del data_to_use
        return  {'data': df_feature, 'keys':[variable,'Date']}
        
    features_dictionary['count item by shop'] = counting_go(data = data,variable = 'shop_id', column = 'item_id', variable_name = 'item')
    features_dictionary['count shop by item'] = counting_go(data = data,variable = 'item_id', column = 'shop_id', variable_name = 'shop_id')
    features_dictionary['count shop by itemword'] = counting_go(data = data,variable = 'ItemWord0', column = 'shop_id', variable_name = 'shop_itemword')
    features_dictionary['count itemword by shop'] = counting_go(data = data,variable = 'shop_id', column = 'ItemWord0', variable_name = 'itemword_shop')
    
    #########################
    #### level  function lags  ######
    #########################
    
    def get_lags_feature_go(data, columns_feature, list_lags, variable_name, category = False, lag_category = None, Null_activator = True):
        data_to_use = data
        if Null_activator:
            data_to_use['Sale'] = np.where(data_to_use['Sale'] == 0, np.nan, data_to_use['Sale'])
        else:
            data_to_use['Sale'] = data_to_use['Sale'].fillna(0)
        begin_date = data_to_use.Date.max() - relativedelta(months = 5)
        data_to_use = data_to_use[data_to_use.Date >= begin_date]
        
        df_feature = data_to_use\
            .groupby(columns_feature + ['Date'])\
            .agg( lag = ('Sale', 'median') )\
            .reset_index()
        
        df_feature['Date'] = df_feature.Date + pd.DateOffset(months=1)
        
        for li in list_lags:
            lu = li - 1
            df_feature[f'L{li}_{variable_name}'] = df_feature.sort_values('Date').groupby(columns_feature)['lag'].shift(lu)
        df_feature = df_feature.drop(columns = ['lag']).fillna(0)
        
        if category:
            df_feature[f'CategoryL{lag_category}{variable_name}'] = np.where(df_feature[f'L{lag_category}_{variable_name}'] < 2,'A',
                np.where(df_feature[f'L{lag_category}_{variable_name}'] < 5,'B',
                np.where(df_feature[f'L{lag_category}_{variable_name}'] < 10,'C','D')))

        del data_to_use
        return {'data':df_feature, 'keys': columns_feature + ['Date']}
    
    features_dictionary['Focus_item_features'] = get_lags_feature_go(data = data, columns_feature = ['item_id'], 
                                                                list_lags = [1,2], variable_name = 'Item',
                                                                category = True, lag_category = 1)
    
    features_dictionary['Focus_shop_features']  = get_lags_feature_go(data = data, columns_feature = ['shop_id'], 
                                                                      list_lags = [1,2,3], variable_name = 'Shop')
    
    features_dictionary['Focus_item_word_0_features'] = get_lags_feature_go(data = data, columns_feature = ['ItemWord0'],
                                                                            list_lags = [1,2], variable_name = 'ItemWord0')
        
    features_dictionary['Focus_item_word_1_features'] = get_lags_feature_go(data = data, columns_feature = ['ItemWord_1'],
                                                                            list_lags = [1,2], variable_name = 'ItemWord_1')
    
    features_dictionary['Focus_shopcomp_features']  = get_lags_feature_go(data = data, columns_feature = ['map_shop_comp1'],
                                                                      list_lags = [1,2], variable_name = 'shopcomp')
    
    features_dictionary['Focus_shop_comp1_features']  = get_lags_feature_go(data = data, columns_feature = ['map_shop_comp1'],
                                                                      list_lags = [1,2], variable_name = 'shop_comp1')
    
    features_dictionary['Focus_itemword_shop_features'] = get_lags_feature_go(data = data, columns_feature = ['shop_id', 'ItemWord0'],
                                                                             list_lags = [1,2,3], variable_name = 'ShopItemWord')
    
    features_dictionary['Focus_null_itemword_shop_features'] = get_lags_feature_go(data = data, columns_feature = ['shop_id', 'ItemWord0'],
                                                                             list_lags = [1,2,3], variable_name = 'noNull_ShopItemWord', Null_activator = False )
    
    features_dictionary['Focus_itemword_shopcomp_features'] = get_lags_feature_go(data = data, columns_feature = ['map_shop_comp1', 'ItemWord0'],
                                                                             list_lags = [1,2,3], variable_name = 'ShopCompItemWord')
    
    features_dictionary['Focus_item_shopcomp_features'] = get_lags_feature_go(data = data, columns_feature = ['map_shop_comp1', 'item_id'],
                                                                             list_lags = [1,2,3], variable_name = 'ShopCompItem')
    
    
    
    def get_count_integer(data, columns_feature, list_lags, variable_name, Null_activator = True, interval = [0,20]):
        
        data_to_use = data
        
        if Null_activator:
            data_to_use['Sale'] = np.where(data_to_use['Sale'] == 0, np.nan, data_to_use['Sale'])
        else:
            data_to_use['Sale'] = data_to_use['Sale'].fillna(0)
        
        data_to_use['integer'] = np.where((data_to_use.SpaceOrder >= 0) & ( (data_to_use.Sale >= interval[0]) & (data_to_use.Sale <= interval[1]) ), 1,0)
        begin_date = data_to_use.Date.max() - relativedelta(months = 5)
        data_to_use = data_to_use[data_to_use.Date >= begin_date]
        
        df_feature = data_to_use\
            .groupby(columns_feature + ['Date'])\
            .agg( integer_count = ('integer','sum') )\
            .reset_index()
        
        df_feature['Date'] = df_feature.Date + pd.DateOffset(months=1)
        
        for li in list_lags:
            lu = li - 1
            df_feature[f'L{li}_ceros_{variable_name}'] = df_feature.sort_values('Date').groupby(columns_feature)['integer_count'].shift(lu)
        df_feature = df_feature.drop(columns = ['integer_count']).fillna(0)

        del data_to_use
        return {'data':df_feature, 'keys': columns_feature + ['Date']}
    
    features_dictionary['Focus_positives_item_id_features']  = get_count_integer(data = data, columns_feature = ['item_id'], list_lags = [1,2], interval = [1,10],
                                                                                   variable_name = 'positives_item_id', Null_activator = False)
    
    features_dictionary['Focus_positives_shop_id_features']  = get_count_integer(data = data, columns_feature = ['shop_id'], list_lags = [1,2], interval = [1,10],
                                                                                   variable_name = 'positives_shop_id', Null_activator = False)
    
    features_dictionary['Focus_positives_ItemWord0_features']  = get_count_integer(data = data, columns_feature = ['ItemWord0'], list_lags = [1,2], interval = [1,10],
                                                                                     variable_name = 'positives_ItemWord0', Null_activator = False)
    
    features_dictionary['Focus_positives_shop_itemword_features']  = get_count_integer(data = data, columns_feature = ['shop_id','ItemWord0'], list_lags = [1,2], interval = [1,5],
                                                                                         variable_name = 'positives_shop_itemword', Null_activator = False)

    features_dictionary['Focus_ceros_item_id_features']  = get_count_integer(data = data, columns_feature = ['item_id'], interval = [0,0],
                                                                      list_lags = [1,2], variable_name = 'Ceros_item_id', Null_activator = False)
    
    features_dictionary['Focus_ceros_shop_id_features']  = get_count_integer(data = data, columns_feature = ['shop_id'], interval = [0,0],
                                                                      list_lags = [1,2], variable_name = 'Ceros_shop_id', Null_activator = False)
    
    features_dictionary['Focus_ceros_ItemWord0_features']  = get_count_integer(data = data, columns_feature = ['ItemWord0'], interval = [0,0],
                                                                      list_lags = [1,2], variable_name = 'Ceros_ItemWord0', Null_activator = False)
    
    features_dictionary['Focus_ceros_shop_ItemWord0_features']  = get_count_integer(data = data, columns_feature = ['shop_id','ItemWord0'], interval = [0,0],
                                                                      list_lags = [1,2], variable_name = 'Ceros_shop_ItemWord0', Null_activator = False)

    
    #########################
    #### last month and order sold ######
    #########################

    df_feature = data[['shop_id','item_id','Order','Date','Sale','map_shop_comp1', 'map_shop_comp2', 'ItemWord0', 'ItemWord_1']]
    df_feature['Month'] = df_feature.Date.dt.month
    df_feature['SaleFlag'] = np.where(df_feature['Sale'] > 0,1,0)
    df_feature['Sale'] = np.where(df_feature['Sale'] == 0, np.nan, df_feature['Sale'])
    df_feature['LastOrder'] = df_feature.groupby(['shop_id','item_id','SaleFlag']).Order.transform('max')
    df_feature['LastOrder'] = np.where(df_feature['SaleFlag'] == 1 , df_feature['LastOrder'], 0)
    df_feature['LastOrder'] = df_feature.groupby(['shop_id','item_id']).LastOrder.transform('max')

    df_feature['LastMonth'] = np.where(df_feature['LastOrder'] == df_feature['Order'], df_feature['Month'],0)
    df_feature['LastMonth'] = df_feature.groupby(['shop_id','item_id']).LastMonth.transform('max')
    
    def get_lasts(data, variable, variable_y):
        
        variable_prefix = '_'.join(variable)
        df_last = data.groupby(['shop_id','item_id','map_shop_comp1', 'map_shop_comp2', 'ItemWord0', 'ItemWord_1',
                                'LastOrder','LastMonth']).agg(maxSale = ('Sale','max')).reset_index()
        df_last = df_last.groupby(variable).agg(
            mean = (variable_y,'mean'),
            #Max = (variable_y,'max')
            ).reset_index()
        vector = df_last.rename(columns = {
            'mean':f'{variable_prefix}_mean_{variable_y}',
            #'Max':f'{variable}_max_{variable_y}'
                }
                               )
            
        return {'data':vector, 'keys': variable}
    
    features_dictionary['lasthOrder_item'] = get_lasts(data = df_feature, variable = ['item_id'], variable_y = 'LastOrder')
    features_dictionary['lasthOrder_shop'] = get_lasts(data = df_feature, variable = ['shop_id'], variable_y = 'LastOrder')
    features_dictionary['lasthOrder_mapshop'] = get_lasts(data = df_feature, variable = ['map_shop_comp1'], variable_y = 'LastOrder')
    features_dictionary['lasthOrder_shopword0'] = get_lasts(data = df_feature, variable = ['shop_id','ItemWord0'], variable_y = 'LastOrder')
    features_dictionary['lasthOrder_shopcompitem'] = get_lasts(data = df_feature, variable = ['map_shop_comp1','item_id'], variable_y = 'LastOrder')
    
    ####################
    ### item price #####
    ####################
    
    def item_price_features(data, variables, variable_name):
    
        df_feature = data[['shop_id','item_id','Date','Order','Sale','item_price','map_shop_comp1', 'map_shop_comp2', 'ItemWord0', 'ItemWord_1']]
        df_feature['Date'] = df_feature.Date + pd.DateOffset(months=1)
        df_feature['Sale'] = np.where(df_feature['Sale'] == 0, np.nan, df_feature['Sale'])
        df_feature['avgPrice'] = df_feature.groupby( variables + ['Order']).item_price.transform('mean')
        df_feature['L1Price'] = df_feature.sort_values('Date').groupby(['item_id','shop_id'])['item_price'].shift(1)
    
        df_feature = df_feature.groupby( variables + ['Date']).agg(meanL1Price = ('L1Price','mean')).reset_index()
        df_feature = df_feature.rename(columns = {'meanL1Price': f'{variable_name}_meanL1Price'})
        
        return {'data':df_feature, 'keys': variables + ['Date']}
        
    features_dictionary['meanL1Price_item'] = item_price_features(data = data, variables = ['item_id'], variable_name = 'ItemId')
    features_dictionary['meanL1Price_shop'] = item_price_features(data = data, variables = ['shop_id'], variable_name = 'shopid')
    features_dictionary['meanL1Price_itemword0'] = item_price_features(data = data, variables = ['ItemWord0'], variable_name = 'ItemWord0')
    features_dictionary['meanL1Price_shopitemWord0'] = item_price_features(data = data, variables = ['shop_id', 'ItemWord0'], variable_name = 'shopid_itemWord0')
        
    return features_dictionary


def feature_silver(data, date_to_take, Train = True):
    data = data
    
    if Train:
        data = data[ data.Date < date_to_take] 
    else:
        data = data[data.Date >= (date_to_take - relativedelta(months=1))]
    
    ## further quasi id
    data['shop_id_term_5'] =  data['shop_id'].astype('int') % 5
    data['shop_id_term_10'] =  data['shop_id'].astype('int') % 10
    
    data['map1_shop_term_5'] =  data['map_shop_comp1'].astype('int') % 5
    data['map1_shop_term_10'] =  data['map_shop_comp1'].astype('int') % 10
    
    data['map2_shop_term_5'] =  data['map_shop_comp2'].astype('int') % 5
    data['map2_shop_term_10'] =  data['map_shop_comp2'].astype('int') % 10
    
    return data

def integration_new_features_map(data, dict_feature, filter_ceros = True):
    data_wow = data

    ## first merge
    data_wow = data_wow.merge(dict_feature['Starts']['data'], on = dict_feature['Starts']['keys'], how = 'left')
    
    ### further features
    data_wow['SpaceOrder'] = data_wow['Order'] - data_wow['OrderGot']
    data_wow['SpaceOrder'] = data_wow['SpaceOrder'].fillna(0)
    data_wow['SpaceOrder'] = np.where(data_wow['SpaceOrder'].isnull(), 0, data_wow['SpaceOrder'] )
    data_wow['SpaceOrderActivator'] = np.where(data_wow['SpaceOrder'] >= 3,'A' , 'B' ) ### this is q dummy
    
    ## second merge
    
    for keyx in list(dict_feature.keys())[1:]:
        data_left = dict_feature[keyx]['data']
        onx = dict_feature[keyx]['keys']
        data_wow = data_wow.merge(data_left, on = onx, how = 'left')
    
    if filter_ceros:
        data_wow = data_wow[data_wow.SpaceOrder >= 0]
    return data_wow

def augmentation_reduction(data, fracs = [0.7, 0.1, 0.1, 0.8]):
    data_wow = data
    random.seed(1256)
    zeros = np.array(data_wow[data_wow.Sale == 0].index)
    nx = int(round(zeros.shape[0]*fracs[0], 0))
    indexes = np.random.choice(zeros.shape[0], nx, replace=False)
    zeros_selected = zeros[indexes]
    data_wow = data_wow[ ~data_wow.index.isin(zeros_selected)]
    
    ones = np.array(data_wow[data_wow.Sale == 1].index)
    nx = int(round(ones.shape[0]*fracs[1], 0))
    indexes = np.random.choice(ones.shape[0], nx, replace=False)
    ones_selected = ones[indexes]
    data_wow = data_wow[ ~data_wow.index.isin(ones_selected)]
    
    twoes = np.array(data_wow[data_wow.Sale == 2].index)
    nx = int(round(twoes.shape[0]*fracs[2], 0))
    indexes = np.random.choice(twoes.shape[0], nx, replace=False)
    twoes_selected = twoes[indexes]
    data_wow = data_wow[ ~data_wow.index.isin(twoes_selected)]
    
    twenties = np.array(data_wow[data_wow.Sale == 20].index)
    nx = int(round(twenties.shape[0]*fracs[3], 0))
    indexes = np.random.choice(twenties.shape[0], nx, replace=False)
    twenties_selected = twenties[indexes]
    data_wow = data_wow[ ~data_wow.index.isin(twenties_selected)]
    
    #data_augmented_tens = data_wow[data_wow.Sale > 10]
    data_augmented = pd.concat([data_wow, 
                               ],axis= 0).reset_index()
    
    return data_augmented

def features_rows_train_silver(data, features, target, sample_gen = None, validation = None, dates_back = 3, limit = 5,additional_val_cols = [] ):
    data = data
    if validation:
        if 'shop_id' in additional_val_cols or 'item_id' in additional_val_cols:
            additional_val_cols = []
            
        data_result = data[ data.Date == validation][additional_val_cols +features + [target]]
        features_new = list(data_result.columns)
        features_new.remove(target)
        data_result = data_result[ features_new + [target]].fillna(0)
        
    else:
        ### splits
        ### conventional for regular time series
        dates_back = 12
        ### Date cqpture ####################
        date_max = date_val - relativedelta(months=1)
        months_to_get = [4,5,6,10,11]
        minyear = 2014
        min_date = date_max - relativedelta(months=dates_back)

        year_i = date_max.year

        dates_result = list()

        while year_i >= minyear:
            for monthi in months_to_get:
                dates_result.append(datetime.datetime(year_i, monthi, 1))
                dates_result = [x for x in dates_result if x <= date_max]
            year_i = year_i - 1

        dates_result = dates_result + [date_max - relativedelta(months=i) for i in range(0,dates_back +1)]
        dates_result = list(set(dates_result))
        dates_result.sort()
        dates_result = dates_result[::-1]

        list_dfs = list()
        for datex,i in zip(dates_result[0:limit], range(len(dates_result[0:limit]))):
            datex_plus1 = datex + relativedelta(months=1)
            print(datex, datex_plus1)

            features_dictionary = get_full_lags(whole = train_full, date_to_take = date_val) ## for train but result easy in val
            full_features_dictionary = {**features_dictionary}
            
            train_full_selected = balance_items_test(data = train_full, date_to_take = date_val , gen_sample = sample_gen, seed = 1234)
            train_feature_tocomplete = feature_silver(train_full_selected , date_to_take = datex_plus1, Train = True) ## for both
            train_feature = integration_new_features_map(data = train_feature_tocomplete, dict_feature = full_features_dictionary)  ## for both not prob
            df = train_feature[train_feature.Date == datex]
            
            list_dfs.append(df)
            del df, train_feature, train_feature_tocomplete, full_features_dictionary
            del features_dictionary
            
        data_result = pd.concat(list_dfs).fillna(0).reset_index(drop = True)

    return data_result

def get_maps_from(datex_str, maps):
    features_mapping = dict()
    for key in maps:
        mapsx = pd.read_csv(f'generated_datasets/maps/data_{datex_str}/{key}.csv')
        if 'Date' in mapsx.columns:
            mapsx['Date'] = pd.to_datetime(mapsx['Date'])
        
        joinx = pd.read_csv(f'./generated_datasets/join_of_maps/{key}.csv')
        joinx = list(joinx['keys'].values)

        features_mapping[key] = {'data': mapsx, 'keys': joinx }
    return features_mapping