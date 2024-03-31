import numpy as np
import matplotlib.pyplot as plt

import joblib

from scipy.io import loadmat

# Load the saved model
clf_model = joblib.load('svm_model.joblib')

# Extract model parameters
w = clf_model.coef_[0]
b = clf_model.intercept_[0]

#loading matlab
Data_2d = loadmat('ex6data1.mat')

#seperating x and y and the y that are 1 and 0
y_Data_2d = Data_2d['y']
X_Data_2d = Data_2d['X']
y_data_squeeze = np.squeeze(y_Data_2d)
indices_y_eq_1 = np.where(y_Data_2d==1)[0]
indices_y_eq_0 = np.where(y_Data_2d==0)[0]

#finding x that their y is 0 or 1
X_eq_1 = X_Data_2d[indices_y_eq_1]
X_eq_0 = X_Data_2d[indices_y_eq_0]

# Generate points for the decision boundary line
xp = np.linspace(np.min(X_Data_2d[:, 0]), np.max(X_Data_2d[:, 0]), 100)
yp = -(w[0] * xp + b) / w[1]

#ploting the result
plt.scatter(X_eq_1[:, 0], X_eq_1[:, 1], marker = 'o', label= "X that their y = 1")
plt.scatter(X_eq_0[:, 0], X_eq_0[:, 1], marker = 'x', label= "X that their y = 0")
plt.plot(xp, yp, '-b', label='Decision Boundary')
plt.title('SVM Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.legend()
plt.show()