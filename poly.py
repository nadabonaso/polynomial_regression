# This ML app shows the relationship between car engine sizes and the CO2 emissions related to them from year 200-2014.
# The data is called (Original Fuel Consumption Ratings 2000-2014) and can be found below.
# Dataset url: https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64

# Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Training set
x_train = [[1.6], [1.8], [1.8], [1.8], [3], [3.2], [1.8], [2.8], [2.8], [1.8], [2.8], [2.8], [4.2], [2.5]] #Engine size
y_train = [[186], [198], [189], [191], [267], [269], [193], [248], [225], [232], [255], [251], [269], [246]] #CO2 Emissions

# Testing set
x_test = [[1.6], [3.2], [3.5], [1.8], [1.8], [2.8]] #Engine size
y_test = [[175], [230], [264], [218], [214], [258]] #CO2 Emissions

# Train the Linear Regression model and plot a prediction
regressor = LinearRegression()
regressor.fit(x_train, y_train)
xx = np.linspace(0, 5.5, 100)

yy = regressor.predict(xx.reshape(xx.shape[0], 1))
plt.plot(xx, yy)

# Set the degree of the Polynomial Regression model
quadratic_featurizer = PolynomialFeatures(degree=2)

# This preprocessor transforms an input data matrix into a new data matrix of a given degree
X_train_quadratic = quadratic_featurizer.fit_transform(x_train)
X_test_quadratic = quadratic_featurizer.transform(x_test)

# Train and test the regressor_quadratic model
regressor_quadratic = LinearRegression()
regressor_quadratic.fit(X_train_quadratic, y_train)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))

# Plot the graph
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), c='r', linestyle='--')
plt.title('Engine size vs CO2 Emissions')
plt.xlabel('Engine Size (L)')
plt.ylabel('CO2 Emissions (g/km)')
plt.axis([0, 5.5, 0, 400])
plt.grid(True)
plt.scatter(x_train, y_train)
plt.show()

print(x_train)
print(X_train_quadratic)
print(x_test)
print(X_test_quadratic)

