import pandas as pd
import numpy as np
from random import random, seed, SystemRandom
import matplotlib.pyplot as plt
from itertools import groupby
import time
import statsmodels
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller


''' Import the Tornado Data and prepare data'''
dataframe = pd.read_excel("Texas Tone Dataset.xlsx")
print(dataframe.shape)
year_list = dataframe['Year'].to_list()
number_tornado = []
torna_per_year = {}
yearly_set = set()
count = 0
year_increment = 1
start_year = 1994

for this_year in year_list:
    if this_year == start_year + year_increment:
        count += 1
    start_year += 1
    yearly_set.add(this_year)

number_tornado = [len(list(group)) for key, group in groupby(year_list)]
torna_per_year = {key: len(list(group)) for key, group in groupby(year_list)}
yearly_data = list(yearly_set)

print(f" Tornadoes/year = {torna_per_year}")
print(f"No. of Tornadoes in 1996 = {torna_per_year[1996]}")
print(f"Years(Range) = {yearly_data}")
print(f"No of tornadoes/year = {number_tornado}")

print(f"Mean= {np.mean(number_tornado)}")
print(np.average(number_tornado))
print(f"STD= {np.std(number_tornado)}")
y = number_tornado
x = yearly_data
plt.title(f"Tornado From 1995-2004")
plt.plot(x, y, linestyle='--', marker='o', color='b')
plt.xticks(yearly_data)
plt.show()

''' Simulate a simple random walk for the next 3 years '''
'''
SEED = int(time.time())
seed(SEED)
print(SEED)
start_point = 144.6206896551724
sd = 69.401964
sim_tornado_data = [start_point,]
for i in range(1, 4):
    rand_predict = -sd if random() < 0.5 else sd
    predicted_data = sim_tornado_data[i-1] + rand_predict
    sim_tornado_data.append(predicted_data)

print(f"{[sim_tornado_data]}")

# 1574216540'''

class StationarityTests:
    def __init__(self, significance=.05):
        self.SignificanceLevel = significance
        self.pValue = None
        self.isStationary = None

    def ADF_Stationarity_Test(self, timeseries, printResults=True):

        # Dickey-Fuller test:
        adfTest = adfuller(timeseries, autolag='AIC')

        self.pValue = adfTest[1]

        if self.pValue < self.SignificanceLevel:
            self.isStationary = True
        else:
            self.isStationary = False

        if printResults:
            dfResults = pd.Series(adfTest[0:4],
                                  index=['ADF Test Statistic', 'P-Value', '# Lags Used', '# Observations Used'])

            # Add Critical Values
            for key, value in adfTest[4].items():
                dfResults['Critical Value (%s)' % key] = value

            print('Augmented Dickey-Fuller Test Results:')
            print(dfResults)


null_test = StationarityTests()
null_test.ADF_Stationarity_Test(number_tornado, printResults=True)
print("Test shows stationary? {0}".format(null_test.isStationary))

autocorrelation_plot(number_tornado)
plt.show()