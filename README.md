# Predict_Future_Sales
this repository corresponds to the kaggle competitione: https://www.kaggle.com/c/competitive-data-science-predict-future-sales/overview

### some charatestics of the challenge:
* the goal is to predict one month forward volume of sales for many shops and items combinations - multiple time series forecast
* the scope of the modeling is time series, multiple time series and multi dimensional time series

#### challenge
* predict more than 200K time series
* data available just for 30% of the test data
* time series are very irregular

### my approach
* extensive data exploration
* extensive feature engineering
  + sum encoding
  + count encoding
  + mean encoding
* Sampling and data reduction techniques
* adaptive boost model for prediction

### my Pipelines
* Exploratory Data Analysis 
* Feature enginering 1 and modeling 1
* Feature enginering 1 and modeling 1
* Exploratory Data Analysis 2 Feature enginering 2
* Feature enginering 2 and modeling 2
* Cross Validation
* Final prediction
