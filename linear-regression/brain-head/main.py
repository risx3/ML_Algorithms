import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Reading Data
data = pd.read_csv('brainhead.csv')

#Collecting X and Y
X=data['headsize'].values
Y=data['brainwgt'].values

#Mean of X and Y
mean_x = np.mean(X)
mean_y = np.mean(Y)
print(data.head())

#Total number of Values
n = len(X)

#Using formula to calculate m and c
numer = 0
denom = 0
for i in range (n):
    numer += (X[i]-mean_x) * (Y[i]-mean_y)
    denom += (X[i]-mean_x)**2

m = numer/denom
c = mean_y - m * mean_x

#Plotting Graph
max_x = np.max(X) + 100
min_x = np.min(X) - 100

#Calculating linvalues of x and y
x = np.linspace(min_x,max_x,1000)
y = m*x + c

plt.plot(x,y,color = 'red' ,label='Regression Line')
plt.scatter(X,Y,label='Scatter Plot')
plt.xlabel('Head Size (cm^3)')
plt.ylabel('Brain Weight (gms)')
plt.show()

#Calculating R^2
snumer = 0
sdenom = 0
for i in range(n):
    yp = m*X[i] + c
    snumer += (yp-mean_y)**2
    sdenom += (Y[i] - mean_y)**2

r2 = (snumer/sdenom)
print('R^2 value: ',r2)

# Using sklearn ML
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = X.reshape((n,1))

#Creating model
reg = LinearRegression()

#Fitting Training data
reg = reg.fit(X,Y)

Y_pred = reg.predict(X)

#Calculating R2
r2 = reg.score(X,Y)
print('Using sklearn')
print('R^2 value: ',r2)
