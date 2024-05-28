# Required Libraries
import numpy as np
import matplotlib.pyplot as plt
import joblib
from scipy.io import loadmat

def model_Linear_classification(model_path: str, data_path: str, first_feature: str, second_feature: str):
    # Load the saved model
    clf_model = joblib.load(model_path)

    # Extract model parameters
    w = clf_model.coef_[0]
    b = clf_model.intercept_[0]

    # Load MATLAB data
    Data_2d = loadmat(data_path)

    # Separate features (X) and target (y)
    y_Data_2d = Data_2d[second_feature]
    X_Data_2d = Data_2d[first_feature]
    y_data_squeeze = np.squeeze(y_Data_2d)
    indices_y_eq_1 = np.where(y_Data_2d == 1)[0]
    indices_y_eq_0 = np.where(y_Data_2d == 0)[0]

    # Extract X values where y is 1 and 0
    X_eq_1 = X_Data_2d[indices_y_eq_1]
    X_eq_0 = X_Data_2d[indices_y_eq_0]

    # Generate points for the decision boundary line
    xp = np.linspace(np.min(X_Data_2d[:, 0]), np.max(X_Data_2d[:, 0]), 100)
    yp = -(w[0] * xp + b) / w[1]

    # Plot the result
    plt.scatter(X_eq_1[:, 0], X_eq_1[:, 1], marker='o', label='Class 1')
    plt.scatter(X_eq_0[:, 0], X_eq_0[:, 1], marker='x', label='Class 0')
    plt.plot(xp, yp, '-b', label='Decision Boundary')
    plt.title('SVM Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()