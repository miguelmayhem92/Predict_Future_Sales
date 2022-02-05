import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta

def feature_silver(data):
    data = data.copy()
    ## feature of diff using one lag
    data['SaleL1M'] = data.sort_values('Date').groupby(['shopitem'])['Sale'].shift(1)
    data['SaleL2M'] = data.sort_values('Date').groupby(['shopitem'])['Sale'].shift(2)
    data['SaleL3M'] = data.sort_values('Date').groupby(['shopitem'])['Sale'].shift(3)
    data['SaleL4M'] = data.sort_values('Date').groupby(['shopitem'])['Sale'].shift(4)
    data['SaleL11M'] = data.sort_values('Date').groupby(['shopitem'])['Sale'].shift(11)
    data['SaleL12M'] = data.sort_values('Date').groupby(['shopitem'])['Sale'].shift(12)
    data['SaleL13M'] = data.sort_values('Date').groupby(['shopitem'])['Sale'].shift(13)
    
    data_lagdiff = data.assign(L0M_L1M = data.Sale - data.SaleL1M)  ### This is new target ???? but yeaaaaaHHH
    data_lagdiff = data_lagdiff.assign(L1M_L2M = data_lagdiff.SaleL1M - data_lagdiff.SaleL2M) 
    data_lagdiff = data_lagdiff.assign(L2M_L3M = data_lagdiff.SaleL2M - data_lagdiff.SaleL3M)
    data_lagdiff = data_lagdiff.assign(L3M_L4M = data_lagdiff.SaleL3M - data_lagdiff.SaleL4M)
    
    data_lagdiff = data_lagdiff.assign(L1M_L11M = data_lagdiff.SaleL1M - data_lagdiff.SaleL11M)
    data_lagdiff = data_lagdiff.assign(L1M_L12M = data_lagdiff.SaleL1M - data_lagdiff.SaleL12M)
    data_lagdiff = data_lagdiff.assign(L1M_L13M = data_lagdiff.SaleL1M - data_lagdiff.SaleL13M)
    
    data_lagdiff = data_lagdiff.drop(columns = ['SaleL2M','SaleL3M','SaleL3M','SaleL4M', 'SaleL11M', 'SaleL12M', 'SaleL13M'])
    
    ### Categorical price
    data_lagdiff = data_lagdiff.assign(categVolume = np.where( data_lagdiff.SaleL1M > 100, 'A',
                                                np.where( data_lagdiff.SaleL1M > 70, 'B',
                                                np.where( data_lagdiff.SaleL1M > 30, 'C',
                                                np.where( data_lagdiff.SaleL1M > 10, 'D',
                                                np.where( data_lagdiff.SaleL1M > 5, 'F',
                                                np.where( data_lagdiff.SaleL1M >= 3, 'G','H'
                                                         )))))))
    
    ### Categorical completion
    data_lagdiff = data_lagdiff.assign( possCounts = np.where(data_lagdiff.Sale > 0, 1, 0) )
    data_lagdiff['possCounts'] = data_lagdiff.groupby(['shopitem']).possCounts.transform(lambda x: x.sum())
    data_lagdiff = data_lagdiff.assign(categSale = np.where( data_lagdiff.possCounts > 10, 'A',
                                                np.where( data_lagdiff.possCounts > 5, 'B',
                                                np.where( data_lagdiff.possCounts > 3, 'C',
                                                np.where( data_lagdiff.possCounts <= 3, 'D','D'
                                                        )))))
    
    #focussed max shop
    data_lagdiff['maxSection'] = data_lagdiff.groupby(['shop_id','categSale']).Sale.transform('max')
    
    ### furhter lags in function of diff
    data_lagdiff['L11M'] = data_lagdiff.sort_values('Date').groupby(['shopitem'])['L1M_L2M'].shift(11)
    data_lagdiff['L12M'] = data_lagdiff.sort_values('Date').groupby(['shopitem'])['L1M_L2M'].shift(12)
    data_lagdiff['L13M'] = data_lagdiff.sort_values('Date').groupby(['shopitem'])['L1M_L2M'].shift(13)
     
    ## Count zeros
    data_lagdiff['count0'] = np.where(data_lagdiff.SaleL1M == 0,1,0)
    data_lagdiff['Roll0count'] = data_lagdiff.sort_values('Date').groupby(['shopitem']).count0.transform(lambda x: x.rolling(4, 1).sum())
    data_lagdiff = data_lagdiff.drop(columns = ['count0'])
    ## Count no variations
    data_lagdiff['count0'] = np.where(data_lagdiff.L1M_L2M == 0,1,0)
    data_lagdiff['Roll0L1'] = data_lagdiff.sort_values('Date').groupby(['shopitem']).count0.transform(lambda x: x.rolling(4, 1).sum())
    #data_lagdiff['Roll0L1Mean'] = data_lagdiff.sort_values('Date').groupby(['shopitem']).Roll0L1.transform(lambda x: x.rolling(4, 1).mean())
    data_lagdiff = data_lagdiff.drop(columns = ['count0'])
    
    ## addition lags
    data_lagdiff['Roll0L1L6'] = data_lagdiff.sort_values('Date').groupby(['shopitem'])['Roll0L1'].shift(6)
    data_lagdiff['Roll0L1L12'] = data_lagdiff.sort_values('Date').groupby(['shopitem'])['Roll0L1'].shift(12)
    data_lagdiff['Roll0countL6'] = data_lagdiff.sort_values('Date').groupby(['shopitem'])['Roll0count'].shift(6)
    data_lagdiff['Roll0countL12'] = data_lagdiff.sort_values('Date').groupby(['shopitem'])['Roll0count'].shift(12)
    
    ## month
    data_lagdiff = data_lagdiff.assign(month = data_lagdiff.Date.dt.month)
    
    max_date = data_lagdiff.Date.max() + relativedelta(months=1)
    max_date_month = max_date.month
    data_lagdiff['monthToPredict'] = np.where(data_lagdiff['month'] == max_date_month,1,0)
        
    ## year
    data_lagdiff['Year'] = data_lagdiff.Date.dt.year
    ## quarter
    data_lagdiff['Quarter'] = np.where(data_lagdiff.month.isin([1,2,3]),1,
                            np.where(data_lagdiff.month.isin([4,5,6]),2,
                            np.where(data_lagdiff.month.isin([7,8,9]),3,
                            np.where(data_lagdiff.month.isin([10,11,12]),4,0))))
    

    
    data_lagdiff['maxSaleL1M'] = data_lagdiff.groupby(['shopitem']).SaleL1M.transform('max') ## To drop Later
    
    ####################################
    ### Features for explosion items ###
    ####################################
    data_lagdiff['maxVar'] = data_lagdiff.groupby(['shopitem']).L1M_L2M.transform('max')
    data_lagdiff['minVar'] = data_lagdiff.groupby(['shopitem']).L1M_L2M.transform('min')

    data_lagdiff = data_lagdiff.assign( noVar = np.where(data_lagdiff.L1M_L2M == 0, 1, 0) )
    data_lagdiff['countNoVar'] = data_lagdiff.groupby(['shopitem']).noVar.transform('sum')
    
    data_lagdiff['FlagShop'] = np.where((data_lagdiff.maxVar > 10) & (data_lagdiff.minVar < 2), data_lagdiff.shop_id, 0)
    data_lagdiff['FlagCategory'] = np.where((data_lagdiff.maxVar > 10) & (data_lagdiff.minVar < 2), data_lagdiff.item_category_id, 0)
    
    data_lagdiff = data_lagdiff.assign(possitionMax = np.where( data_lagdiff.L1M_L2M == data_lagdiff.maxVar, data_lagdiff.Order, 0))
    data_lagdiff['pivotx'] = data_lagdiff.groupby(['shopitem']).possitionMax.transform('max')
    data_lagdiff = data_lagdiff.assign(befExplotion = np.where(data_lagdiff.Order < data_lagdiff.pivotx , 1, 0))
    data_lagdiff['SumBefExplotion'] = data_lagdiff.groupby(['shopitem']).befExplotion.transform('sum')
    
    data_lagdiff = data_lagdiff.drop(columns = ['pivotx','possitionMax'])
    
    ## interaction
    data_lagdiff['countNoVar'] = data_lagdiff['noVar'] * data_lagdiff['countNoVar']
    data_lagdiff['FlagShop'] = data_lagdiff['noVar'] * data_lagdiff['FlagShop']
    data_lagdiff['FlagCategory'] = data_lagdiff['noVar'] * data_lagdiff['FlagCategory']
    data_lagdiff['befExplotion'] = data_lagdiff['noVar'] * data_lagdiff['befExplotion']
    data_lagdiff['SumBefExplotion'] = data_lagdiff['noVar'] * data_lagdiff['SumBefExplotion']
    
    ####################################
    ##### end ##########################
    ####################################
    
    ####################################
    ####### maps for feature gen #######
    ####################################
    
    data_lagdiff['maxVar'] = data_lagdiff.groupby(['shopitem']).L0M_L1M.transform('max')
    data_lagdiff['minVar'] = data_lagdiff.groupby(['shopitem']).L0M_L1M.transform('min')
    
    df_feature = data_lagdiff[(data_lagdiff.maxVar > 10) & (data_lagdiff.minVar < 2) ]\
    [['shop_id','item_id','shopitem','item_category_id','Date', 'Sale','Order', 'L0M_L1M','maxVar']].copy()

    df_feature['month'] = df_feature['Date'].dt.month
    df_feature['pivotDiff'] = np.where(df_feature.Sale == 0, 0,1)
    df_feature['reorder1'] = df_feature.sort_values(['Date'], ascending=[True]).groupby(['shopitem','pivotDiff']).cumcount() + 1
    df_feature['Starter'] = np.where((df_feature.pivotDiff == 1) & (df_feature.reorder1 == 1),df_feature.Order,0)

    df_feature['pivotMax'] = df_feature.groupby(['shopitem']).Starter.transform('max')
    df_feature['OrderBef'] = np.where(df_feature.Order < df_feature.pivotMax, df_feature.Order,0)
    df_feature['pivotSep'] = np.where(df_feature.OrderBef > 0 ,1,0)

    df_feature['reorder'] = df_feature.sort_values(['Date'], ascending=[True]).groupby(['shopitem','pivotSep']).cumcount() + 1
    df_feature['capture'] = np.where(df_feature.pivotSep == 1, 1,np.where((df_feature.pivotSep == 0) & (df_feature.reorder <= 3),1,0 ) )
    df_feature = df_feature[df_feature.capture == 1]

    feature_item = df_feature[(df_feature.pivotSep == 0) ].groupby(['item_id','reorder']).agg(meanVarExploItem = ('L0M_L1M','median')).reset_index()
    feature_item['reorder'] = feature_item['reorder'] - 1
    feature_shop = df_feature[(df_feature.pivotSep == 0) ].groupby(['shop_id','reorder']).agg(meanVarExploShop = ('L0M_L1M','median')).reset_index()
    feature_shop['reorder'] = feature_shop['reorder'] - 1
    feature_month = df_feature[df_feature.Starter != 0 ].groupby('month').agg(countMonth = ('L0M_L1M', 'count')).reset_index()
    
    features_dictionary = {'Item_feature': feature_item, 'Shop_feature': feature_shop, 'month_feature': feature_month}
    
    ####################################
    ##### end ##########################
    ####################################
    
    
    ###############################################
    ####### maps for feature highVariation #######
    ###############################################
    
    df_feature = data_lagdiff[['shop_id','item_id','shopitem','item_category_id','Date', 'Sale','Order', 'L0M_L1M','minVar','maxVar']].copy()
    df_feature['count0'] = np.where(df_feature.Sale == 0,1,0)
    df_feature['month'] = df_feature['Date'].dt.month
    df_feature['count0'] = df_feature.sort_values('Date').groupby(['shopitem']).count0.transform('sum')
    df_feature['SaleNew'] = np.where(df_feature.Sale == 0,np.nan, df_feature.Sale)
    df_feature['minSale'] = df_feature.sort_values('Date').groupby(['shopitem']).SaleNew.transform('min')
    
    df_feature = df_feature[(abs(df_feature.maxVar) > 3) &(df_feature.count0 < 6)]
    
    def q2(x):
        return x.quantile(0.50)

    def q3(x):
        return x.quantile(0.95)

    df_feature['Year'] = df_feature.Date.dt.year
    df_feature['Quarter'] = np.where(df_feature.month.isin([1,2,3]),1,
                            np.where(df_feature.month.isin([4,5,6]),2,
                            np.where(df_feature.month.isin([7,8,9]),3,
                            np.where(df_feature.month.isin([10,11,12]),4,0))))

    feature_med = df_feature.groupby(['item_id','Order','Year','Quarter','month']).agg(medianVar = ('L0M_L1M',q2),
                                                                 highVar = ('L0M_L1M',q3)).reset_index()
    feature_med['HighMeanQuart'] = feature_med.groupby(['Year','Quarter']).highVar.transform('mean')
    feature_med['medianVarL12'] = feature_med.sort_values('Order').groupby(['item_id'])['medianVar'].shift(12)
    feature_med['HighVarL12'] = feature_med.sort_values('Order').groupby(['item_id'])['highVar'].shift(12)

    feature_var_season = feature_med.groupby(['item_id','Year','Quarter'])[['HighMeanQuart']].max().reset_index()
    feature_var_season['HighMeanQuartL4'] = feature_var_season.sort_values(['Year','Quarter']).groupby(['item_id'])['HighMeanQuart'].shift(4)
    feature_var_season = feature_var_season.fillna(0)
    feature_var_season['SeasonalVariation'] = feature_var_season.HighMeanQuart - feature_var_season.HighMeanQuartL4
    feature_var_season = feature_var_season.drop(columns = 'HighMeanQuart')


    feature_med = feature_med.merge(feature_var_season, on = ['item_id','Year','Quarter'],how = 'left')
    feature_med = feature_med[['item_id','Order','Year','month','medianVarL12','SeasonalVariation','HighVarL12']]
    
    features_dictionary['HighVarSeason'] = feature_med
    ####################################
    ##### end ##########################
    ####################################
    
    data_lagdiff = data_lagdiff.drop(columns = ['maxVar','minVar'])
    data_result = data_lagdiff
    return data_result, features_dictionary

def integration_new_features_map(data, dict_feature):
    
    data_wow = data.copy()
    data_wow['maxVar'] = data_wow.groupby(['shopitem']).L1M_L2M.transform('max')
    data_wow['minVar'] = data_wow.groupby(['shopitem']).L1M_L2M.transform('min')
    
    ##### detector ######
    
    data_wow['labeling'] = np.where((data_wow.maxVar > 10) & (data_wow.minVar < 2),1,0)
    data_wow['labeling'] = data_wow.groupby(['shopitem']).labeling.transform('max')
    data_wow['pivotDiff'] = np.where(data_wow.Sale == 0, 0,1)
    data_wow['reorder'] = data_wow.sort_values(['Date'], ascending=[True]).groupby(['shopitem','pivotDiff']).cumcount() + 1
    data_wow['reorder'] = data_wow['reorder']*data_wow['pivotDiff']
    data_wow['reorder'] = data_wow.sort_values('Date').groupby(['shopitem'])['reorder'].shift(1)
    
    #### mergin
    
    data_wow = data_wow.merge(dict_feature['Item_feature'], on = ['item_id','reorder'], how = 'left')
    data_wow = data_wow.merge(dict_feature['Shop_feature'], on = ['shop_id','reorder'], how = 'left')
    data_wow = data_wow.merge(dict_feature['month_feature'], on = ['month'], how = 'left')
    data_wow = data_wow.merge(dict_feature['HighVarSeason'], on = ['item_id','Year','month','Order'], how = 'left')
    
    ### feature correction
    data_wow['meanVarExploItem'] = np.where(data_wow['labeling'] == 1, data_wow['meanVarExploItem'],0 )
    data_wow['meanVarExploShop'] = np.where(data_wow['labeling'] == 1, data_wow['meanVarExploShop'],0 )
    data_wow['countMonth'] = np.where(data_wow['labeling'] == 1, data_wow['countMonth'],0 )
    
    data_wow = data_wow.drop(columns = ['maxVar','minVar','pivotDiff','reorder','Year','Quarter','befExplotion'])
    
    return data_wow

def features_rows_train_silver(data, features, target ,sample_1 = 0.70, sample_2 = 0.20,validation = None, dates_back = 3):
    data = data.copy()
    if validation:
        data_result = data[ data.Date == validation][features + [target]].fillna(0)
        features_new = list(data_result.columns)
        features_new.remove(target)
        data_result = data_result[features_new + [target]]
        
    else:
        ### splits
        ### conventional for regular time series
    
        cutoff = 3
        
        ### Date cqpture ####################
        date_max = data.Date.max()
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
        ######################################
    
        data_comp1 = data[ (data.maxSaleL1M > cutoff) & (data.Date.isin(dates_result))][features + [target]].sample(frac = sample_1)
        ### until here we just covered a very little section
        data_comp2 = data[ (data.maxSaleL1M <= cutoff) & (data.Date >= min_date) & (data.Date <= date_max)][features + [target]].sample(frac = sample_2)
        data_comp3 = data[ (data.maxSaleL1M <= cutoff) & (data.Date.isin(dates_result))][features + [target]].sample(frac = sample_2)
        
        ## concat results 
        
        data_result = pd.concat([data_comp1,data_comp2,data_comp3], axis = 0).fillna(0)
        ### Result
        features_new = list(data_result.columns)
        features_new.remove(target)
        data_result = data_result[features_new + [target]]
    
    return data_result