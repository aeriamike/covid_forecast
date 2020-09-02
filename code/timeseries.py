import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from fbprophet import Prophet


#extract data from JHU repository
#if those do links do not work
#replace the links with the directories that point to the similar dataset inside
#the data folder in code.zip package
dtf = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv", sep=",")
dtf2 = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv", sep=",")
dtf3 = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv", sep=",")

#Project Time Series Analysis
#Find the top 10 countries with the most confirmed cases

#Plot global case model

#clean unused columns and transpose the dataset to make date as row and country as column
dtf = dtf.drop(['Province/State','Lat','Long'], axis=1).groupby("Country/Region").sum().T
dtf.index = pd.to_datetime(dtf.index, infer_datetime_format=True)

dtf2 = dtf2.drop(['Province/State','Lat','Long'], axis=1).groupby("Country/Region").sum().T
dtf2.index = pd.to_datetime(dtf2.index, infer_datetime_format=True)

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

dtf_real = df = pd.DataFrame(data, columns = ['Date','ConfirmedCases','Fatalities','Country_Region'])

for k in range(2,188):
  
  country_name = dtf.iloc[:,k].name
  data = {'Date': dtf.iloc[:,k].index,
        'ConfirmedCases':dtf.iloc[:,k],
        'Fatalities':dtf2.iloc[:,k],
        'Country_Region': [country_name]*length
        }

  dtf_real2 = pd.DataFrame(data, columns = ['Date','ConfirmedCases','Fatalities','Country_Region'])
  dtf_real = dtf_real.append(dtf_real2, ignore_index=True)


dtf_total = dtf.loc[:,'ConfirmedCases'] = dtf.sum(axis=1)
dtf_total.index
dtf_total.values


dtf2_total = dtf2.loc[:,'Fatalities'] = dtf2.sum(axis=1)
dtf2_total.index
dtf2_total.values

#set up variables, labels and legends for the plot
plt.bar(dtf_total.index, dtf_total.values, label="Confirmed Cases")
plt.bar(dtf2_total.index, dtf2_total.values, color='orange', label="Deaths")

#plot the global pandemic cases histogram
plt.ylabel('Cases')
plt.xlabel('Date')
plt.title('Global Pandemic Cases')
plt.rcParams["figure.figsize"] = [16,9]
plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)
plt.legend()
plt.show()


#Find top 10 countries
dtf = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv", sep=",")
dtf_p = dtf.sort_values('8/8/20', ascending = False)

#get the rows of cases for the top ten countries
row = dtf_p.iloc[0][5:]
row2 = dtf_p.iloc[1][5:]
row3 = dtf_p.iloc[2][5:]
row4 = dtf_p.iloc[3][5:]
row5 = dtf_p.iloc[4][5:]

rowa = dtf_p.iloc[5][5:]
row2a = dtf_p.iloc[6][5:]
row3a = dtf_p.iloc[7][5:]
row4a = dtf_p.iloc[8][5:]
row5a = dtf_p.iloc[9][5:]

ap = []
for a in range(0,10):
  ap.append(dtf_p.iloc[a][1])

row.plot()
row2.plot()
row3.plot()
row4.plot()
row5.plot()

rowa.plot()
row2a.plot()
row3a.plot()
row4a.plot()
row5a.plot()

#plot the lines of cases for the top ten countries
plt.title('Top Countries with Covid-19 Confirmed Cases')


plt.ylabel('Date')
plt.xlabel('Cases')
plt.rcParams["figure.figsize"] = [16,9]
plt.legend(ap)
plt.show()

row.plot()
row2.plot()
row3.plot()
row4.plot()
row5.plot()

rowa.plot()
row2a.plot()
row3a.plot()
row4a.plot()
row5a.plot()

plt.title('Top Countries with Covid-19 Confirmed Cases (Logarithmic)')

# set logarithmic scale

plt.yscale('log')
# get the logarithmic version of the
# global pandemic cases data
plt.ylabel('Date')
plt.xlabel('Cases')
plt.legend(ap)
plt.rcParams["figure.figsize"] = [16,9]
plt.show()


#clean data for creating the pie chart
dtf = dtf.drop(['Province/State','Lat','Long'], axis=1).groupby("Country/Region").sum().T
dtf.index = pd.to_datetime(dtf.index, infer_datetime_format=True)

dtf_count = dtf.tail(1)
#get the required data to remodel
dtf_count_t = dtf_count.T
dtf_count_t.columns = ['Confirmed Cases']
dtf_sort = dtf_count_t.sort_values('Confirmed Cases', ascending = False)
result = dtf_sort.set_index([np.arange(len(dtf_sort)), dtf_sort.index])

#plot the top countries
top_count = 10
result[:top_count]
# pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = ['US', 'Brazil', 'India', 'Russia', 'South Africa', 'Mexico', 'Peru', 'Chile', 'Colombia', 'Iran', 'Others']
sizes = [4713540, 2750318, 1855745, 854641, 516862, 443813, 433100, 361493, 327850, 312035, 7360224]
explode = (0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle.
ax1.axis('equal')  
plt.rcParams["figure.figsize"] = [16,9]
plt.show()


#new cases plot
#extract data to do another remodelling
#section to promote day-by-day new case section
dtf = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv", sep=",")
dtfd = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv", sep=",")

#clean and transpose the data into date as rows and coutnries as columns
dtf = dtf.drop(['Province/State','Lat','Long'], axis=1).groupby("Country/Region").sum().T
dtf.index = pd.to_datetime(dtf.index, infer_datetime_format=True)
print(dtf)

# do some data arithmetic
#to produce the growth of new cases by date
dtf["Sum"] = dtf.sum(axis=1)
dtf2 = pd.DataFrame(index=dtf.index, data=dtf["Sum"].values, columns=["Sum"])
#create confirmed cases growth
dtf2["new"] = dtf2["Sum"] - dtf2["Sum"].shift(1)
dtf2["new"] = dtf2["new"].fillna(method='bfill')


#clean and remodel data for the death section of the chart
dtfd = dtfd.drop(['Province/State','Lat','Long'], axis=1).groupby("Country/Region").sum().T
dtfd.index = pd.to_datetime(dtfd.index, infer_datetime_format=True)

dtfd["Sum"] = dtfd.sum(axis=1)

## create daily changes column
dtf22 = pd.DataFrame(index=dtfd.index, data=dtfd["Sum"].values, columns=["Sum"])
#create death data growth
dtf22["new"] = dtf22["Sum"] - dtf22["Sum"].shift(1)
dtf22["new"] = dtf22["new"].fillna(method='bfill')

plt.bar(dtf2.index, dtf2['new'].values, label="New Cases")

plt.bar(dtf22.index, dtf22['new'].values, color='orange', label="New Deaths")

#plot the new confirmed cases and deaths by date
plt.ylabel('Cases')
plt.xlabel('Date')
plt.title('Global Pandemic Daily New Cases and Deaths')
plt.rcParams["figure.figsize"] = [16,9]
plt.legend()
plt.show()
