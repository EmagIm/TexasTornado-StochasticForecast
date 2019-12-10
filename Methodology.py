"""
This is the working code for the ARIMA tornado predict model.
"""

""" Library and tool #include """

import statistics
from itertools import groupby

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import autocorrelation_plot
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, \
    plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller

'''function prototype and def for use in predict()'''


def invert_diff(history_arg, yhat_arg, interval=1):
    return yhat_arg + history_arg[-interval]


''' Import the Tornado Data and prepare data'''
dataframe = pd.read_excel("Torn Data All.xlsx")
print(dataframe.shape)
year_list = dataframe['Year'].to_list()
number_tornado = []
torna_per_year = {}
yearly_set = set()
count = 0
year_increment = 1
start_year = 1949

for this_year in year_list:
    if this_year == start_year + year_increment:
        count += 1
    start_year += 1
    yearly_set.add(this_year)

number_tornado = [len(list(group)) for key, group in groupby(year_list)]
torna_per_year = {key: len(list(group)) for key, group in groupby(year_list)}
yearly_data = list(yearly_set)
print(number_tornado)
new_data = pd.DataFrame(number_tornado)
new_data.to_csv('NewTornData.csv', index=False)

'''Data visualisation plot'''
x = yearly_data
y = number_tornado
plt.title(f'Annual Number of Tornado 1950-2018')
plt.plot(x, y, linestyle='dotted', marker='o', color='b', antialiased=True)
plt.show()

'''Mean Calculated from t_0 to t_n in iteration of 1'''
x_count = 0
y_count = 0
count = len(number_tornado)
mean_series = []
while y_count < len(number_tornado):
    summed = np.mean(number_tornado[x_count:(y_count + 1)])
    mean_series.append(summed)
    y_count += 1
plt.title(f"Backward Shift E[x], t$_0$ -  t$_n$ ")
plt.plot(mean_series, linestyle='--', marker='o', color='b')
plt.show()

'''
Series of Moving Average computation time steps 6,4 and 3
'''
movx_count = 0
mov6_count = 6
mov4_count = 4
mov3_count = 3
mov_count = len(number_tornado)
mov_mean6_series = []
mov_mean4_series = []
mov_mean3_series = []
while movx_count < len(number_tornado):
    summed = np.mean(number_tornado[x_count:(movx_count + mov6_count)])
    mov_mean6_series.append(summed)
    movx_count += 3

movx_count = 0
while movx_count < len(number_tornado):
    summed = np.mean(number_tornado[x_count:(movx_count + mov4_count)])
    mov_mean4_series.append(summed)
    movx_count += 2

movx_count = 0
while movx_count < len(number_tornado):
    summed = np.mean(number_tornado[x_count:(movx_count + mov3_count)])
    mov_mean3_series.append(summed)
    movx_count += 1
plt.title(f"E[x], TimeStep t$_1$ -  t$_3$ ")
plt.plot(mov_mean3_series, linestyle='--', marker='*', color='b', label='3-step MA')
plt.show()

"Visualization of Moving Average"
plt.title(f"E[x], TimeSteps 6, 4 and 3 ")
plt.plot(mov_mean3_series, linestyle='--', marker='*', color='b', label='3-step MA')
plt.plot(mov_mean4_series, linestyle='dotted', marker='o', color='g', label='4-step MA')
plt.plot(mov_mean6_series, linestyle='-.', marker='D', color='y', label='6-step MA')
plt.legend()
plt.show()

"""
Using the Augment Dicke-Fuller package to check Null Hypothesis
"""


class StationarityTest:
    def __init__(self, significance=.05):
        self.SignificanceLevel = significance
        self.pValue = None
        self.isStationary = None

    def adfuller_test(self, timeseries, printResults=True):

        # Dickey-Fuller test:
        adf_test = adfuller(timeseries, maxlag=7, autolag='t-stat')

        self.pValue = adf_test[1]

        if self.pValue < self.SignificanceLevel:
            self.isStationary = True
        else:
            self.isStationary = False

        if printResults:
            df_results = pd.Series(adf_test[0:4],
                                   index=['ADF Test Statistic', 'P-Value', '# Lags Used', '# Observations Used'])

            # Add Critical Values
            for key, value in adf_test[4].items():
                df_results['Critical Value (%s)' % key] = value

            print('Augmented Dickey-Fuller Test Results:')
            print(df_results)


null_test = StationarityTest()
null_test.adfuller_test(number_tornado, printResults=True)
print("Test shows stationary? {0}".format(null_test.isStationary))
'''End'''

"""
AutoCorrelation and Partial correlation function Computation
"""
plot_acf(number_tornado, lags=65)
plt.show()

# Using the differencing method on the dataset and analysing residual
new_correlation = []
unbiased_statistic = list()

for i in range(1, len(number_tornado)):
    static = number_tornado[i] - number_tornado[i - 1]  # Swap around to eliminate the -ve val
    new_correlation.append(static)

plot_pacf(number_tornado, lags=65, method='ols-inefficient')
plt.show()
print(new_correlation)

dwn_list = []
dwn_count = 0
while len(dwn_list) < 68:
    dwn_new = number_tornado[dwn_count + 1] - mean_series[dwn_count]
    dwn_list.append(dwn_new)
    dwn_count += 1

dwn_list.insert(0, 0)
(print(dwn_list))
print(np.mean(dwn_list))
# plt.plot(x, dwn_list)
# plt.show()
corr, _ = pearsonr(number_tornado, dwn_list)
print(f"Correlation = %.3f" % corr)

'''Alternative comparison ACFunction check'''
'''
acf2 = pacf(new_correlation, nlags=65, method='ols')
plt.plot(acf2, marker='o')
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96 / np.sqrt(len(new_correlation)), linestyle='--', color='gray')
plt.axhline(y=1.96 / np.sqrt(len(new_correlation)), linestyle='--', color='gray')
plt.title('Partial Autocorrelation Function')
plt.xlabel('number of lags')
plt.ylabel('correlation')
plt.tight_layout()
plt.show()
print(new_correlation)
plot_acf(number_tornado, lags=65, unbiased=True)
'''

'''Calculating statistical features in Stationary dataset for Analysis'''

print(f"Variance = {np.var(new_correlation, ddof=1)}")
print(f"Std = {np.std(new_correlation, ddof=1)}")
print(f"Mean = {np.mean(new_correlation)}")
var = statistics.variance(new_correlation)
mean = statistics.stdev(new_correlation)
print(f"Var is %1, Std %2", var, mean)
autocorrelation_plot(new_correlation)
plt.title("Detrended Dataset Correlogram")
plt.show()
statistic_1 = np.std(new_correlation[:10], ddof=1)
unbiased_statistic.append(statistic_1)
statistic_2 = np.std(new_correlation[10:20], ddof=1)
unbiased_statistic.append(statistic_2)
statistic_3 = np.std(new_correlation[20:], ddof=1)
unbiased_statistic.append(statistic_3)
print(unbiased_statistic)

print("Stat deviation sample 1 is %f" % statistic_1,
      "\n Stat deviation sample 2 is %f" % statistic_2,
      "\n Stat deviation sample 3 is %f" % statistic_3)

estimator_stat = np.mean(unbiased_statistic)
print(estimator_stat)

"""
ARIMA model fit and Prediction
"""
split_point = len(number_tornado) - 6
train, test = number_tornado[0: split_point], number_tornado[split_point:]
history = [x for x in train]
print(history)
prediction = []
start_n = len(number_tornado) - 6
end_n = start_n + 18
model = ARIMA(number_tornado, order=(20, 1, 0))
model_fit = model.fit(method='mle', disp=0)
forecast = model_fit.predict(start_n, end_n)
step = 1
year_interval = 1
for yhat in forecast:
    inverted = invert_diff(history, yhat, year_interval)
    print('Year %d: %f' % (step, inverted))
    history.append(inverted)
    step += 1

error = mean_squared_error(number_tornado[63:], history[63:69])
print('MSE=%f' % error)

for x in history[63:]:
    add_year = yearly_data[-1] + 1
    yearly_data.append(add_year)
print(yearly_data)

plt.plot(yearly_data[63:len(number_tornado)], number_tornado[63:], linestyle=':', marker='D', color='b',
         label='Expected')
plt.plot(yearly_data[63:len(history)], history[63:], linestyle='--', marker='o', color='r', label='Predicted')
plt.xticks(yearly_data[63: len(history)])
plt.legend()
plt.show()

'''
Finding the Mean Square Error.
'''
'''
train_size, test_zize = train_test_split(number_tornado, train_size=0.8)
print(np.mean(new_correlation))
history = train_size[-1]
predictions = []
for i in range(len(test_zize)):
    y_hat = history
    predictions.append(y_hat)
    history = test_zize[i]
print(predictions)
print(train_size)
error = mean_squared_error(test_zize, predictions)
print('Error = %.3f' % error)
'''
