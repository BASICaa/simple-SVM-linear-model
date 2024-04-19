# Liberaries needed:
# from scipy.io import loadmat
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
# import joblib
def model_Linear_classification(model_name:str, data_path:str, first_feature:str, second_feature:str):
    #loading matlab
    Data_2d = loadmat(data_path)

    #seperating x and y
    y_Data_2d = Data_2d[second_feature]
    y_data_squeeze = np.squeeze(y_Data_2d)
    X_Data_2d = Data_2d[first_feature]

    #finding indices of y which  are 1 and 0
    indices_y_eq_1 = np.where(y_Data_2d==1)[0]
    indices_y_eq_0 = np.where(y_Data_2d==0)[0]

    #finding x that their y is 0 or 1
    X_eq_1 = X_Data_2d[indices_y_eq_1]
    X_eq_0 = X_Data_2d[indices_y_eq_0]

    #ploting the X which we seperated
    plt.scatter(X_eq_1[:, 0], X_eq_1[:, 1], marker = 'o')
    plt.scatter(X_eq_0[:, 0], X_eq_0[:, 1], marker = 'x')
    plt.show()

    #training the linear SVM
    X_train, X_test, y_train, y_test = train_test_split(X_Data_2d, y_data_squeeze, test_size=0.4, random_state=42)
    clf = SVC(kernel='linear', C=1)
    clf.fit(X_train,y_train)

    #accuracy of the SVM
    Y_predicted = clf.predict(X_test)
    accuracy = accuracy_score(Y_predicted,y_test)

    #saving the model
    joblib.dump(clf, '%s.joblib'%model_name)