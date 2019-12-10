import nipy
from nipy.algorithms.statistics.models.model import Model, \
    LikelihoodModel, \
    LikelihoodModelResults
from random import seed
from random import random
from sklearn.metrics import mean_squared_error
import numpy as np

from numpy.random import standard_normal as stan
from nipy.algorithms.statistics.models.regression import OLSModel

x = np.hstack((stan((30, 1)), stan((30, 1)), stan((30, 1))))
beta = np.array([3.25, 1.5, 7.0])
y = np.dot(x, beta) + stan(30)
model = OLSModel(x).fit(y)
confidence_intervals = model.conf_int(cols=(1, 2))
print(confidence_intervals)
FIM = LikelihoodModel.information
rsults = LikelihoodModel.information(FIM, theta=x, nuisance=None)
'''seems this nipy doesn't work
    or I read the documentation wrong. Medical or Mathematical package?
'''


class FisherMatrix:
    def __init__(self):
        self.pvalue = False

    def info_matrix(self, theta_estimate):
        FIMatrix = LikelihoodModel.information(theta_estimate, nuisance=None)
        print(FIMatrix)
        self.pvalue = True


FIM_test = FisherMatrix()
FIM_test.info_matrix(y)

'''
# generate the random walk
seed(1)
random_walk = list()
random_walk.append(-1 if random() < 0.5 else 1)
for i in range(1, 1000):
    movement = -1 if random() < 0.5 else 1
    value = random_walk[i - 1] + movement
    random_walk.append(value)
# prepare dataset
train_size = int(len(random_walk) * 0.66)
train, test = random_walk[0:train_size], random_walk[train_size:]
# persistence
predictions = list()
history = train[-1]
for i in range(len(test)):
    yhat = history
    predictions.append(yhat)
    history = test[i]
error = mean_squared_error(test, predictions)
print('Persistence MSE: %.3f' % error)

theat = np.array(predictions)
theat2 = np.array(random_walk)

LikelihoodModel.logL(random_walk)

'''
