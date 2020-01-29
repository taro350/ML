import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

x_train = array([1,2,3,4,5,6,7,8,9])
y_train = array([0.2, 0.7, 1.2, 2.5, 1.8, 5.1, 1.9, 2.1, 3.2])

x_test = array([11, 12])
y_test = array([6.4, 8.7])

x_train=x_train.reshape(-1, 1)
y_train=y_train.reshape(-1, 1)

x_test=x_test.reshape(-1, 1)
y_test=y_test.reshape(-1, 1)

rg = linear_model.LinearRegression()
rg.fit(x_train, y_train)

Pre = rg.predict(x_test)

Sc = rg.score(x_test, y_test)
print(Sc)
print(mean_squared_error(y_test, Pre))
print(r2_score(y_test, Pre))

x = np.concatenate((x_train, x_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)
y_p = rg.predict(x)


plt.scatter(x,y)
plt.plot(x, y_p)
plt.show()
