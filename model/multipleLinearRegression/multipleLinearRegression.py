# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures

import joblib



def train(dataPath, modelSavePath):
    print('\ntraining: ', dataPath)

    # Importing the dataset
    dataset = pd.read_csv(dataPath)
    X = dataset.iloc[:, :-1]
    # X = dataset.iloc[:, [0,1,2,3,5,9,10,12]]
    y = dataset.iloc[:, dataset.shape[1] - 1]
    # y = dataset.iloc[:, dataset.shape[1] - 1: dataset.shape[1]]

    # normalize
    X_scaled = preprocessing.StandardScaler().fit_transform(X)
    # X_scaled = preprocessing.MinMaxScaler().fit_transform(X)
    # X_scaled = preprocessing.MaxAbsScaler().fit_transform(X)
    # X_scaled = preprocessing.Normalizer().fit_transform(X)

    # X_scaled = None
    # if dataPath=='../../data/ingotRate.csv':
    #     X_scaled = preprocessing.Normalizer().fit_transform(X)
    # elif dataPath=='../../data/yieldRate.csv':
    #     X_scaled = preprocessing.StandardScaler().fit_transform(X)


    # total_score=0
    # for ti in range(0,20):
    #     print('==============round: ',ti,'====================')

    # Splitting the dataset into the Training set and Test set
    # X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values, test_size=0.2, random_state=0)
    if dataPath == '../../data/ingotRate.csv':
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values, test_size=0.2, random_state=3)
    elif dataPath == '../../data/yieldRate.csv':
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values, test_size=0.2, random_state=0)

    # X_train, y_train=upsample(X_train,y_train)

    # quadratic_featurizer = PolynomialFeatures(degree=1)
    # X_train = quadratic_featurizer.fit_transform(X_train)

    # Fitting Multiple Linear Regression to the Training set
    regressor = LinearRegression()

    regressor.fit(X_train, y_train)

    y_train_pred = regressor.predict(X_train)

    # plt.plot(y_train, label='y_train')
    plt.plot(y_train, label='y_train')
    plt.plot(y_train_pred, label='y_train_pred')
    plt.title('LinearRegression', fontsize='large', fontweight='bold')
    plt.legend()
    plt.show()

    # print("y_train: ")
    # print("[")
    # for i in range(0, y_train.shape[0]):
    #     print(y_train[i], ',', end="")
    # print("\n]")
    # print("y_train_pred: ")
    # print("[")
    # for i in range(0, y_train_pred.shape[0]):
    #     print(y_train_pred[i], ',', end="")
    # print("\n]")

    print('\ntrain: ')
    MSE = mean_squared_error(y_train, y_train_pred)
    MAE = mean_absolute_error(y_train, y_train_pred)
    R2_Score = r2_score(y_train, y_train_pred)
    print("MSE: ", MSE)
    print("MAE: ", MAE)
    print("R2_Score: ", R2_Score)

    # save model
    joblib.dump(regressor, modelSavePath)


    # quadratic_featurizer = PolynomialFeatures(degree=1)
    # X_test = quadratic_featurizer.fit_transform(X_test)

    # Predicting the Test set results
    y_pred = regressor.predict(X_test)

    plt.plot(y_test, label='y_test', linewidth=2.0, linestyle='--')
    plt.plot(y_pred, label='y_pred', linewidth=2.0)
    plt.title('Multiple Linear regression', fontsize='large', fontweight = 'bold')
    plt.grid(linewidth=0.3, linestyle='--')
    plt.xlabel('Indexes of samples')
    if dataPath == '../../data/ingotRate.csv':
        plt.ylabel('Ingot rate')
        saveName='ingotFittingLinearRegression.png'
    elif dataPath == '../../data/yieldRate.csv':
        plt.ylabel('Yield rate')
        saveName='yieldFittingLinearRegression.png'
    plt.legend()
    # plt.show()
    plt.savefig('../../result/'+saveName, bbox_inches='tight')

    # print("y_test: ")
    # print("[")
    # for i in range(0, y_test.shape[0]):
    #     print(y_test[i], ',', end="")
    # print("\n]")
    # print("y_pred: ")
    # print("[")
    # for i in range(0, y_pred.shape[0]):
    #     print(y_pred[i], ',', end="")
    # print("\n]")

    print("\ntest: ")
    MSE = mean_squared_error(y_test, y_pred)
    MAE = mean_absolute_error(y_test, y_pred)
    R2_Score = r2_score(y_test, y_pred)
    print("MSE: ", MSE)
    print("MAE: ", MAE)
    print("R2_Score: ", R2_Score)


    #     total_score += R2_Score
    #
    # print('mean_score: ', total_score / 20)


def predict(predictionObject, data):
    regressor = None
    testData = None
    if predictionObject == 'ingotRate':
        regressor = joblib.load('./model/multipleLinearRegression/ingotRatePrediction.pkl')
        testData = pd.DataFrame(data,
                                columns=['WS_MM', 'CS_MM', 'FS_MM', 'Mn_MM', 'CL_SM', 'Out_TE', 'S_EL', 'SN_QM',
                                         'UD_QM', 'NI_QM', 'OE_QM', 'PO_QM', 'C_QM', 'SI_QM'], dtype=float)
    elif predictionObject == 'yieldRate':
        regressor = joblib.load('./model/multipleLinearRegression/yieldRatePrediction.pkl')
        testData = pd.DataFrame(data,
                                columns=['X31', 'X33', 'X34', 'X35', 'X36'], dtype=float)

    # testData = preprocessing.scale(testData)
    result=regressor.predict(testData.values)
    print('multipleLinearRegression result:\n', result[0])
    return result[0]


if __name__ == '__main__':
    # train('../../data/ingotRate.csv','./ingotRatePrediction.pkl')
    train('../../data/yieldRate.csv','./yieldRatePrediction.pkl')

    # data = [[3.12, 6.12, 8.61, 5.82, 2.44, 1.59, 1.24, 0.66, 0.143, 1.824, 1.126, 1.46, 0.38, 0.46]]  # ingotRate
    # predict('ingotRate', data)
    # data = [[6.684, 2.561, 1.654, 1.36, 0.679]]  # yieldRate
    # predict('yieldRate', data)
