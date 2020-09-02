import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet

#extract data from JHU repository
#if those do links do not work
#replace the links with the directories that point to the similar dataset inside
#the data folder in code.zip package
dtf = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv", sep=",")
dtf2 = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv", sep=",")
dtf3 = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv", sep=",")

#clean unused columns and transpose the dataset to make date as row and country as column
dtf = dtf.drop(['Province/State','Lat','Long'], axis=1).groupby("Country/Region").sum().T
dtf.index = pd.to_datetime(dtf.index, infer_datetime_format=True)
dtf2 = dtf2.drop(['Province/State','Lat','Long'], axis=1).groupby("Country/Region").sum().T

## convert index to datetime
dtf2.index = pd.to_datetime(dtf2.index, infer_datetime_format=True)

#get the number of countries in the dataset
#and the amount of those countries in the dataset
country_name = dtf.iloc[:,1].name
length = len(dtf.iloc[:,1])

#remodelling datasets
#add confirmed cases, fatailities and country_region columns
#those will be used for remodelling by prediciton function
country_name = dtf.iloc[:,1].name
data = {'Date': dtf.iloc[:,1].index,
        'ConfirmedCases':dtf.iloc[:,1],
        'Fatalities':dtf2.iloc[:,1],
        'Country_Region': [country_name]*length
        }

#create a country and case model that will be used for prediction
dtf_real = df = pd.DataFrame(data, columns = ['Date','ConfirmedCases','Fatalities','Country_Region'])

#get the columns and data for all the countries
for k in range(2,188):
  
  country_name = dtf.iloc[:,k].name
  data = {'Date': dtf.iloc[:,k].index,
        'ConfirmedCases':dtf.iloc[:,k],
        'Fatalities':dtf2.iloc[:,k],
        'Country_Region': [country_name]*length
        }

  dtf_real2 = pd.DataFrame(data, columns = ['Date','ConfirmedCases','Fatalities','Country_Region'])
  dtf_real = dtf_real.append(dtf_real2, ignore_index=True)



# the function that runs the training and testing of datasets
def helper(df,countries):
    train_data,test_data = training(df,30)

    # print("train")
    # print(train_data)
    # print("test")
    # print(test_data)

    #this one is for training
    #pm = modeling(train_data)

    #this one is for doing actual stuff
    p = prophet(df)
    predictions = forecasting(p,30)
    result = predictions[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].merge(test_data)

    print(result)
    fig1 = p.plot(predictions)
    fig1.suptitle(country)
    return result

#Prophet algorithm helps to predict the time series data
def prophet(train_data):

  p = Prophet(changepoint_prior_scale=0.95,interval_width=1)
  p.fit(train_data)
  return p

#train data to predict the recent 30-day data that exist in the dataset
def training(df,split_days):
  train_data = df[:-split_days]
  
  test_data =  df[-split_days:]
  return train_data,test_data

#Make prediction of the countries' cases using the
#trained models
#Predict 30 days of the set
def forecasting(p,days):

  model = p.make_future_dataframe(days)
  p_forecast = p.predict(model)
  return p_forecast

#Preprocessing dataset for prediction model
#use the confirmed cases column of the dataset
def confirmedCasePredictions(df,country):
  
  #group country and get the fatalitie model
  country_data = dtf_real.groupby(['Country_Region','Date'])[['ConfirmedCases','Fatalities']].sum().reset_index()
  country_data = country_data[country_data.Country_Region == country]
  #Rename Data for plot; clean data that are not used for prediction
  country_confirmed = country_data.rename(columns={"Date":"ds","ConfirmedCases":"y"})
  country_confirmed = country_confirmed.reset_index().drop(["index","Fatalities","Country_Region"],axis= 1)

  return country_confirmed

#Preprocessing dataset for prediction model
#use the death cases column of the dataset
def deathPredictions(df,country):
  
  #group country and filter country data
  country_data = df.groupby(['Country_Region','Date'])[['ConfirmedCases','Fatalities']].sum().reset_index()
  country_data = country_data[country_data.Country_Region == country]

  #Rename Data for plot; clean data that are not used for prediction
  country_confirmed = country_data.rename(columns={"Date":"ds","Fatalities":"y"})
  country_confirmed = country_confirmed.reset_index().drop(["index","ConfirmedCases","Country_Region"],axis= 1)
  return country_confirmed



#Preprocessing dataset for prediction model
#use the 'new' column of the dataset
def newCasePredictions(df,country):
 
  #grouping country wise
  country_data = df.groupby(['Country_Region','Date'])[['new']].sum().reset_index()
 
  country_data = country_data[country_data.Country_Region == country]
  #Rename Data for plot; clean data that are not used for prediction
  country_daily = country_data.rename(columns={"Date":"ds","new":"y"})
  country_daily = country_daily.reset_index().drop(["index","Country_Region"],axis= 1)
  return country_daily

#Preprocessing dataset for prediction model
#use the 'active' column of the dataset
def activePredictions(df,country):

  #group country and filter country data
  country_data = df.groupby(['Country_Region','Date'])[['active']].sum().reset_index()
  country_data = country_data[country_data.Country_Region == country]

  #Rename Data for plot; clean data that are not used for prediction
  country_active = country_data.rename(columns={"Date":"ds","active":"y"})
  country_active = country_active.reset_index().drop(["index","Country_Region"],axis= 1)
  return country_active

#Confirmed Cases Prediction
countries = ['US','Brazil','India','Russia','South Africa', 'Mexico', 'Peru', 'Colombia', 'Chile', 'Iran']
for country in countries:
   df = confirmedCasePredictions(dtf_real,country)
   results = helper(df,countries)
   results.head()


#Deaths Prediction
countries = ['US','Brazil','India','Russia','South Africa', 'Mexico', 'Peru', 'Colombia', 'Chile', 'Iran']
for country in countries:
   df = deathPredictions(dtf_real,country)
   results = helper(df,countries)
   results.head()



#new case prediction
#data model is different; needs to extract datasets to create new data frame
dtf = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv", sep=",")
dtf2 = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv", sep=",")
dtf3 = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv", sep=",")

countries = ['US','Brazil','India','Russia','South Africa', 'Mexico', 'Peru', 'Colombia', 'Chile', 'Iran']

dtf = dtf.drop(['Province/State','Lat','Long'], axis=1).groupby("Country/Region").sum().T
dtf2 = dtf2.drop(['Province/State','Lat','Long'], axis=1).groupby("Country/Region").sum().T
dtf3 = dtf3.drop(['Province/State','Lat','Long'], axis=1).groupby("Country/Region").sum().T



#New Cases Prediction
countries = ['US','Brazil','India','Russia','South Africa', 'Mexico', 'Peru', 'Colombia', 'Chile', 'Iran']

# for each country, do re-model of the dataset to get the 'new' column, containing new reported cases daily
for country in countries:
  dtf = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv", sep=",")
  dtf2 = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv", sep=",")

  country_name = country
  #follow the same procedure as the first part
  #clean and remodel the data
  dtf = dtf.drop(['Province/State','Lat','Long'], axis=1).groupby("Country/Region").sum().T

  ## convert index to datetime
  dtf.index = pd.to_datetime(dtf.index, infer_datetime_format=True)

  ## create total cases column
  dtfcan = pd.DataFrame(index=dtf.index, data=dtf[country_name].values, columns=["total"])

  # get the new reported cases for a particular date
  # by subtracting today's cases by yesterday's cases  
  dtfcan["new"] = dtfcan["total"] - dtfcan["total"].shift(1)
  dtfcan["new"] = dtfcan["new"].fillna(method='bfill')
  dtfcan['Date'] = dtfcan.index
  dtfcan['Country_Region'] = country_name

  #do the new case prediction here for all ten countries
  df = newCasePredictions(dtfcan,country_name)
  results = helper(df,country_name)
  results.head()



#Active Case prediction
#extract dataset again to make new kinds of data to do active case prediction
dtf = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv", sep=",")
dtf2 = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv", sep=",")
dtf3 = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv", sep=",")


countries = ['US','Brazil','India','Russia','South Africa', 'Mexico', 'Peru', 'Colombia', 'Chile', 'Iran']


dtf = dtf.drop(['Province/State','Lat','Long'], axis=1).groupby("Country/Region").sum().T
dtf2 = dtf2.drop(['Province/State','Lat','Long'], axis=1).groupby("Country/Region").sum().T
dtf3 = dtf3.drop(['Province/State','Lat','Long'], axis=1).groupby("Country/Region").sum().T

# for each country, do re-model of the dataset to get the 'active' column, containing
# total number of remaining patients
#Active Cases
for country in countries:
  ## convert index to datetime
  dtf.index = pd.to_datetime(dtf.index, infer_datetime_format=True)
  dtf2.index = pd.to_datetime(dtf2.index, infer_datetime_format=True)
  dtf3.index = pd.to_datetime(dtf3.index, infer_datetime_format=True)

  country_name = country

  # get the total of confirmed cases, deaths and recoveries by date for all countries
  dtfcan = pd.DataFrame(index=dtf.index, data=dtf[country_name].values, columns=["total"])
  dtfcan2 = pd.DataFrame(index=dtf2.index, data=dtf2[country_name].values, columns=["total"])
  dtfcan3 = pd.DataFrame(index=dtf3.index, data=dtf3[country_name].values, columns=["total"])

  dtf_active = pd.DataFrame(index=dtf.index, data=dtf[country_name].values, columns=["total"])
  #use the newly create datasets to subtract confirmed cases by deaths and recoveries
  #to get the active cases by date
  dtf_active["active"] = dtfcan["total"] - dtfcan2["total"] - dtfcan3["total"]
  dtf_active["active"] = dtf_active["active"].fillna(method='bfill')
  dtf_active['Date'] = dtf_active.index
  dtf_active['Country_Region'] = country_name

  print(dtf_active)
  #do the active case prediction here for all ten countries
  df = activePredictions(dtf_active,country)
  results = helper(df,country)
  results.head()
