import pandas as pd
import numpy as np
from random import random
import matplotlib.pyplot as plt
from itertools import groupby

''' Import the Tornado Data and prepare data'''
dataframe = pd.read_excel("Torn Data All.xlsx")
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

print(f"Means = {np.mean(number_tornado)}")
print(np.average(number_tornado))
print(f"St Dev = {np.std(number_tornado)}")

data_mean = np.mean(number_tornado)
data_deviation = np.std(number_tornado)
y = number_tornado
x = yearly_data
plt.title(f"Tornado From 1995-2004")
plt.plot(x, y, linestyle='--', marker='o', color='b')
plt.show()