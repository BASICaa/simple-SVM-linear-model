# Required Libraries
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

def model_Linear_classification(model_name: str, data_path: str, first_feature: str, second_feature: str):
    # Load MATLAB data
    Data_2d = loadmat(data_path)

    # Separate features (X) and target (y)
    y_Data_2d = Data_2d[second_feature]
    y_data_squeeze = np.squeeze(y_Data_2d)
    X_Data_2d = Data_2d[first_feature]

    # Find indices where y is 1 and 0
    indices_y_eq_1 = np.where(y_Data_2d == 1)[0]
    indices_y_eq_0 = np.where(y_Data_2d == 0)[0]

    # Extract X values where y is 1 and 0
    X_eq_1 = X_Data_2d[indices_y_eq_1]
    X_eq_0 = X_Data_2d[indices_y_eq_0]

    # Plot X values based on y values
    plt.scatter(X_eq_1[:, 0], X_eq_1[:, 1], marker='o', label='Class 1')
    plt.scatter(X_eq_0[:, 0], X_eq_0[:, 1], marker='x', label='Class 0')
    plt.legend()
    plt.show()

    # Train the linear SVM
    X_train, X_test, y_train, y_test = train_test_split(X_Data_2d, y_data_squeeze, test_size=0.4, random_state=42)
    clf = SVC(kernel='linear', C=1)
    clf.fit(X_train, y_train)

    # Evaluate accuracy of the SVM
    y_predicted = clf.predict(X_test)
    accuracy = accuracy_score(y_predicted, y_test)
    print(f"Model Accuracy: {accuracy}")

    # Save the model
    joblib.dump(clf, f'{model_name}.joblib')