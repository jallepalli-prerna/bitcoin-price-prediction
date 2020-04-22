import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime;
import pygal;
#%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6


df = pd.read_csv("/home/jallepalli/Desktop/all/bitflyerJPY_1-min_data_2017-07-04_to_2018-01-08.csv")
#pf = df;
#print(df.head())

#df basic description
df_description = df.describe()
print(df_description)


Timestamp = df['Timestamp']

#Getting the values of Open columns in an array
Open1 = df['Open'].values
Open1.tolist()



Date = []; Time = []; DateTime = []; Year = []; Month = []; Day = []; Hour = []; Minutes = []; Year_Month = []
for time in Timestamp:
	value = datetime.datetime.fromtimestamp(time)
	Year.append(value.strftime('%Y'))
	Month.append(value.strftime('%m'))
	Day.append(value.strftime('%d'))
	Hour.append(value.strftime('%H'))
	Minutes.append(value.strftime('%M'))
	Date.append(value.strftime('%Y-%m-%d'))
	Year_Month.append(value.strftime('%Y-%m'))
	Time.append(value.strftime('%H:%M:%S'))
	DateTime.append(value.strftime('%Y-%m-%d %H:%M:%S'))


#print(DateTime)


#print(Year)
#print(DateTime)
#Setting Date as index of the DataFrame
df.set_index(pd.DatetimeIndex(Date), inplace=True)
df.index.name = 'Date'

'''
pf.set_index(pd.DatetimeIndex(Year), inplace=True)
pf.index.name = 'Year'
'''
#Dropping Timestamp column
df.drop(['Timestamp'], axis = 1, inplace = True)

#Setting the names of the column names
df.columns = ['Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price']

#pf.columns = ['Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price']

#pf['Month'] = Month


df['Year'] = Year
df['Month'] = Month
df['Day'] = Day
df['Year_Month'] = Year_Month
df['Time'] = Time

#print(df)

#grouped = df.groupby(['Year', 'Month', 'Day'])

#print(grouped['Open'])



'''
#Open values
Open = df['Open']
Open_values = Open.values
Open_values.tolist()
#print(Open_values[0:10])
'''
#print(Open['2018-01-08 05:30:00'])

#Groupby on date attribute
df_by_Year = df.groupby('Year')
'''
for name,group in df_by_Year:
    print name
    print group
'''
df_by_Year_desc = df_by_Year.describe()
#print(df_by_Year_desc).head()

df_by_Month = df.groupby('Month')
df_by_Month_desc = df_by_Month.describe()
#print(df_by_Month_desc).head()

df_by_Day = df.groupby('Day')
df_by_Day_desc = df_by_Day.describe()
#print(df_by_Day_desc)




df_by_Date = df.groupby('Date')
df_by_Date_desc = df_by_Date.describe()
open_mean = df_by_Date_desc['Open']['mean']


df_by_Year_Month = df.groupby('Year_Month')
df_by_Year_Month_desc = df_by_Year_Month.describe()


'''
for name, group['Open', 'Weighted_Price'] in df_by_Date['Open', 'Weighted_Price']:
    print name
    print group
'''
#print(df_by_Date['Open', 'Weighted_Price'])

#df_by_Month_desc.columns = ['Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price']


#By Date
open_max = df_by_Date_desc['Open']['max']
close_max = df_by_Date_desc['Close']['max']
high_max = df_by_Date_desc['High']['max']
low_max = df_by_Date_desc['Low']['max']

open_mean = df_by_Date_desc['Open']['mean']
close_mean = df_by_Date_desc['Close']['mean']
high_mean = df_by_Date_desc['High']['mean']
low_mean = df_by_Date_desc['Low']['mean']

open_min = df_by_Date_desc['Open']['min']
close_min = df_by_Date_desc['Close']['min']
high_min = df_by_Date_desc['High']['min']
low_min = df_by_Date_desc['Low']['min']

open_std = df_by_Date_desc['Open']['std']
close_std = df_by_Date_desc['Close']['std']

weighted_max = df_by_Date_desc['Weighted_Price']['max']
weighted_mean = df_by_Date_desc['Weighted_Price']['mean']
weighted_std = df_by_Date_desc['Weighted_Price']['std']
weighted_min = df_by_Date_desc['Weighted_Price']['min']

#Visualization, plot values of date corresponding to Open

#print(open_max)
x1 = open_max.index.tolist()
y1 = open_max.tolist()
x11 = weighted_max.index.tolist()
y11 = weighted_max.tolist()

x2 = open_mean.index.tolist()
y2 = open_mean.tolist()
x22 = weighted_mean.index.tolist()
y22 = weighted_mean.tolist()

x3 = open_min.index.tolist()
y3 = open_min.tolist()
x33 = weighted_min.index.tolist()
y33 = weighted_min.tolist()

x4 = open_std.index.tolist()
y4 = open_std.tolist()





plt.subplot(2, 2, 1)
plt.scatter(x1, y1, color='blue', label='Max Values of Open for that day')
plt.plot(x11, y11, color='red', label='Max Weighted_Price')
plt.xlabel('Date')
plt.ylabel('Open Max')
plt.legend()

plt.subplot(2, 2, 2)
plt.scatter(x2, y2, color= 'green', label='Mean Values of Open for that day')
plt.plot(x22, y22, color='red', label='Mean Weighted_Price')
plt.xlabel('Date')
plt.ylabel('Open Mean')
plt.legend()

plt.subplot(2, 2, 3)
plt.scatter(x3, y3, color= 'red', label='Min Values of Open for that day')
plt.plot(x33, y33, color='green', label='Minimum Weighted_Price')
plt.xlabel('Date')
plt.ylabel('Open Min')
plt.legend()

plt.subplot(2, 2, 4)
plt.scatter(x4, y4, color= 'yellow', label='std values of Open for that day')
plt.xlabel('Date')
plt.ylabel('Open std')
plt.legend()
plt.show()



x1 = close_max.index.tolist()
y1 = close_max.tolist()

x2 = close_mean.index.tolist()
y2 = close_mean.tolist()

x3 = close_min.index.tolist()
y3 = close_min.tolist()

x4 = close_std.index.tolist()
y4 = close_std.tolist()



plt.subplot(2, 2, 1)
plt.scatter(x1, y1, color='blue', label='Max Values of Close for that day')
plt.plot(x11, y11, color='red', label='Max Weighted_Price')
plt.xlabel('Date')
plt.ylabel('Close Max')
plt.legend()

plt.subplot(2, 2, 2)
plt.scatter(x2, y2, color= 'green', label='Mean Values of Close for that day')
plt.plot(x22, y22, color='red', label='Mean Weighted_Price')
plt.xlabel('Date')
plt.ylabel('Close Mean')
plt.legend()

plt.subplot(2, 2, 3)
plt.scatter(x3, y3, color= 'red', label='Min Values of Open for that day')
plt.plot(x33, y33, color='green', label='Minimum Weighted_Price')
plt.xlabel('Date')
plt.ylabel('Close Min')
plt.legend()

plt.subplot(2, 2, 4)
plt.scatter(x4, y4, color= 'yellow')
plt.xlabel('Date')
plt.ylabel('Close std')
plt.legend()
plt.show()



#By Year-Month
open_max = df_by_Year_Month_desc['Open']['max']
close_max = df_by_Year_Month_desc['Close']['max']
high_max = df_by_Year_Month_desc['High']['max']
low_max = df_by_Year_Month_desc['Low']['max']

open_mean = df_by_Year_Month_desc['Open']['mean']
close_mean = df_by_Year_Month_desc['Close']['mean']
high_mean = df_by_Year_Month_desc['High']['mean']
low_mean = df_by_Year_Month_desc['Low']['mean']

open_min = df_by_Year_Month_desc['Open']['min']
close_min = df_by_Year_Month_desc['Close']['min']
high_min = df_by_Year_Month_desc['High']['min']
low_min = df_by_Year_Month_desc['Low']['min']

weighted_max = df_by_Year_Month_desc['Weighted_Price']['max']
weighted_mean = df_by_Year_Month_desc['Weighted_Price']['mean']
weighted_std = df_by_Year_Month_desc['Weighted_Price']['std']
weighted_min = df_by_Year_Month_desc['Weighted_Price']['min']



x1 = open_max.index.tolist()
y1 = open_max.tolist()
x11 = weighted_max.index.tolist()
y11 = weighted_max.tolist()

x2 = open_mean.index.tolist()
y2 = open_mean.tolist()
x22 = weighted_mean.index.tolist()
y22 = weighted_mean.tolist()

x3 = open_min.index.tolist()
y3 = open_min.tolist()
x33 = weighted_min.index.tolist()
y33 = weighted_min.tolist()

x4 = open_std.index.tolist()
y4 = open_std.tolist()


plt.subplot(2, 2, 1)
plt.scatter(x1, y1, color='blue', label='Max Values of Open')
plt.plot(x11, y11, color='red', label='Max Weighted_Price')
plt.xlabel('Year-Month')
plt.ylabel('Open Max')
plt.legend()

plt.subplot(2, 2, 2)
plt.scatter(x2, y2, color= 'green', label='Mean Values of Open')
plt.plot(x22, y22, color='red', label='Mean Weighted_Price')
plt.xlabel('Year-Month')
plt.ylabel('Open Mean')
plt.legend()

plt.subplot(2, 2, 3)
plt.scatter(x3, y3, color= 'red', label='Min Values of Open')
plt.plot(x33, y33, color='green', label='Minimum Weighted_Price')
plt.xlabel('Year-Month')
plt.ylabel('Open Min')
plt.legend()

plt.subplot(2, 2, 4)
plt.scatter(x4, y4, color= 'yellow')
plt.xlabel('Year-Month')
plt.ylabel('Open std')
plt.legend()
plt.show()






x1 = close_max.index.tolist()
y1 = close_max.tolist()
x11 = weighted_max.index.tolist()
y11 = weighted_max.tolist()

x2 = close_mean.index.tolist()
y2 = close_mean.tolist()
x22 = weighted_mean.index.tolist()
y22 = weighted_mean.tolist()

x3 = close_min.index.tolist()
y3 = close_min.tolist()
x33 = weighted_min.index.tolist()
y33 = weighted_min.tolist()

x4 = close_std.index.tolist()
y4 = close_std.tolist()


plt.subplot(2, 2, 1)
plt.scatter(x1, y1, color='blue', label='Max Values of Close')
plt.plot(x11, y11, color='red', label='Max Weighted_Price')
plt.xlabel('Year-Month')
plt.ylabel('Close Max')
plt.legend()

plt.subplot(2, 2, 2)
plt.scatter(x2, y2, color= 'green', label='Mean Values of Close')
plt.plot(x22, y22, color='red', label='Mean Weighted_Price')
plt.xlabel('Year-Month')
plt.ylabel('Close Mean')
plt.legend()

plt.subplot(2, 2, 3)
plt.scatter(x3, y3, color= 'red', label='Min Values of Close')
plt.plot(x33, y33, color='green', label='Minimum Weighted_Price')
plt.xlabel('Year-Month')
plt.ylabel('Close Min')
plt.legend()

plt.subplot(2, 2, 4)
plt.scatter(x4, y4, color= 'yellow', label='std values of Close')
plt.xlabel('Year-Month')
plt.ylabel('Close Min')
plt.show()
plt.legend()



#For a Particular Date all Max, Mean, min, std




'''
print(pf)

pf_by_Date = pf.groupby('Year')
pf_by_Date_desc = pf_by_Date.describe()
print(pf_by_Date_desc).head()

trythis = pf_by_Date.groupby('Month')
trythis_desc = trythis.describe()
print(trythis_desc).head()
'''


#print(df)
df_by_particular_date = df[['Weighted_Price', 'Time', 'Open', 'Close', 'High', 'Low']]
grouppped = df_by_particular_date.groupby('Date')
#df_test = df_by_particular_date.iloc['2018-01-08']
ggp = grouppped.get_group('2018-01-08')
#print ggp

x = ggp['Open'].tolist()
y = ggp['Weighted_Price'].tolist()
z = ggp['Time'].tolist()

plt.plot(z[0:20], y[0:20], label='Weighted_Price')
plt.scatter(z[0:20], y[0:20])
plt.plot(z[0:20], x[0:20], label='Open')
plt.scatter(z[0:20], x[0:20])
plt.xlabel('Time')
plt.ylabel('Bitcoin Price in Dollars ($)')
plt.legend()
plt.show()


plt.plot(z, y, label='Weighted_Price')
plt.scatter(z, y)
plt.plot(z, x, label='Open')
plt.scatter(z, x)
plt.xlabel('Time')
plt.ylabel('Bitcoin Price in Dollars ($)')
plt.legend()
plt.show()




plt.plot(ggp['Time'].tolist()[10:20], ggp['Weighted_Price'].tolist()[10:20], 'r--', label='Weighted_Price')
#plt.scatter(ggp['Time'].tolist()[10:20], ggp['Weighted_Price'].tolist()[10:20])
plt.plot(ggp['Time'].tolist()[10:20], ggp['Open'].tolist()[10:20], 'bs', label='Open')
#plt.scatter(ggp['Time'].tolist()[10:20], ggp['Open'].tolist()[10:20])
plt.plot(ggp['Time'].tolist()[10:20], ggp['Close'].tolist()[10:20], 'g^', label='Close')
#plt.scatter(ggp['Time'].tolist()[10:20], ggp['Close'].tolist()[10:20])
plt.plot(ggp['Time'].tolist()[10:20], ggp['High'].tolist()[10:20], 'bs', label='High')
#plt.scatter(ggp['Time'].tolist()[10:20], ggp['High'].tolist()[10:20])
plt.plot(ggp['Time'].tolist()[10:20], ggp['Low'].tolist()[10:20], 'g^', label='Low')
#plt.scatter(ggp['Time'].tolist()[10:20], ggp['Low'].tolist()[10:20])
plt.xlabel('Time')
plt.ylabel('Bitcoin Price in Dollars ($)')
plt.legend()
plt.show()

'''
line_chart = pygal.Line()
line_chart.title = 'Bitcoin trend on 2018-01-08'
#line_chart.x_labels = map(str, range('00:01:00', '23:59:00'))
line_chart.add('Weighted Price', ggp['Weighted_Price'].tolist() )
line_chart.add('Close',  ggp['Close'].tolist())
line_chart.add('Open', ggp['Open'].tolist())
line_chart.add('High',  ggp['High'].tolist())
line_chart.add('Low',  ggp['Low'].tolist())
line_chart.render_to_file('/home/jallepalli/Desktop/all/photo1.svg')
'''






#Seasonal Graphs
#2015 to 2018 graphs
df_seasnal = pd.read_csv("/home/jallepalli/Desktop/all/coinbaseUSD_1-min_data_2014-12-01_to_2018-01-08.csv")

Timestamp = df_seasnal['Timestamp']

Date = []; Time = []; DateTime = []; Year = []; Month = []; Day = []; Hour = []; Minutes = []; Year_Month = []
for time in Timestamp:
	value = datetime.datetime.fromtimestamp(time)
	Year.append(value.strftime('%Y'))
	Month.append(value.strftime('%m'))
	Day.append(value.strftime('%d'))
	Hour.append(value.strftime('%H'))
	Minutes.append(value.strftime('%M'))
	Date.append(value.strftime('%Y-%m-%d'))
	Year_Month.append(value.strftime('%Y-%m'))
	Time.append(value.strftime('%H:%M:%S'))
	DateTime.append(value.strftime('%Y-%m-%d %H:%M:%S'))

df_seasnal.set_index(pd.DatetimeIndex(Date), inplace=True)
df_seasnal.index.name = 'Date'

'''
pf.set_index(pd.DatetimeIndex(Year), inplace=True)
pf.index.name = 'Year'
'''
#Dropping Timestamp column
df_seasnal.drop(['Timestamp'], axis = 1, inplace = True)

#Setting the names of the column names
df_seasnal.columns = ['Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price']

#pf.columns = ['Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price']

#pf['Month'] = Month


df_seasnal['Year'] = Year

df_by_Year = df_seasnal.groupby('Year')

df_by_Year_desc = df_by_Year.describe()

df_by_Year.plot(kind='line')




#Linear Regression Implementation




#Arima Model Implementation





#Random Forest Implementation




#Bayesian statistics using Markov Chains and Recurring Neural Networks




#Visualization on a particular day


#Visualization per Hour



