import joblib
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor, \
    ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso, LassoCV
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR

import pandas as pd


def train(dataPath, modelSavePath):
    # X, y = load_diabetes(return_X_y=True)

    dataset = pd.read_csv(dataPath)
    X = dataset.iloc[:, :-1]
    # X = dataset.iloc[:, [0,1,2,3,5,9,10,12]]
    y = dataset.iloc[:, dataset.shape[1] - 1]

    # normalize
    scl = preprocessing.StandardScaler()
    X_scaled = scl.fit_transform(X)

    # y = preprocessing.MinMaxScaler().fit_transform(y.values.reshape(-1,1))

    # X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values, test_size=0.2, random_state=3)
    if dataPath == '../../data/ingotRate.csv':
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values, test_size=0.2, random_state=3)
    elif dataPath == '../../data/yieldRate.csv':
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values, test_size=0.2, random_state=0)

    if dataPath == '../../data/ingotRate.csv':
        estimators = [
            ('tree',DecisionTreeRegressor(max_depth=13, random_state=10)),
            ('knn',KNeighborsRegressor(n_neighbors=14)),
            ('lasso', LassoCV()),
            # ('mlp', MLPRegressor(hidden_layer_sizes=(14,8,4,2), activation='tanh', solver='lbfgs', alpha=1e-2,
            #                          max_iter=2000, random_state=0)),
            # ('linear', LinearRegression()),
            ('ridge', RidgeCV()),
            ('svr', SVR(kernel='sigmoid', C=0.1, gamma='scale', degree=2, epsilon=0.001)),
        ]
        regressor = StackingRegressor(
            estimators=estimators,
            final_estimator=RandomForestRegressor(n_estimators=10, random_state=42)
        )

    elif dataPath == '../../data/yieldRate.csv':
        estimators = [
            ('tree', DecisionTreeRegressor(max_depth=20, random_state=7)),
            ('tree1', DecisionTreeRegressor(max_depth=22, random_state=7)),
            ('tree2', DecisionTreeRegressor(max_depth=10, random_state=7)),
            ('knn', KNeighborsRegressor(n_neighbors=8)),
            # ('lasso', LassoCV()),
            # ('mlp', MLPRegressor(hidden_layer_sizes=(4,2), activation='tanh', solver='lbfgs', alpha=1e-2,
            #                          max_iter=2000, random_state=0)),
            ('linear', LinearRegression()),
            ('ridge', RidgeCV()),
            ('svr', SVR(kernel='linear', C=0.1, gamma='scale', degree=2, epsilon=0.08)),
        ]

        regressor = StackingRegressor(
            # cv=5,
            estimators=estimators,
            # final_estimator=AdaBoostRegressor(random_state=0, n_estimators=5)
            # final_estimator=GradientBoostingRegressor(random_state=0)
            final_estimator=LinearRegression()
            # final_estimator=RandomForestRegressor(n_estimators=10, random_state=42)
        )

    # scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
    # scores = cross_validate(regressor, X_train, y_train, cv=5, scoring=scoring, return_train_score=True)
    # print('scores: ', scores)
    #
    # train_losses = {'MSE': [], 'MAE': [], 'R2': []}
    # valid_losses = {'MSE': [], 'MAE': [], 'R2': []}
    # train_losses['MSE'] = scores['train_neg_mean_squared_error']
    # train_losses['MAE'] = scores['train_neg_mean_absolute_error']
    # train_losses['R2'] = scores['train_r2']
    # valid_losses['MSE'] = scores['test_neg_mean_squared_error']
    # valid_losses['MAE'] = scores['test_neg_mean_absolute_error']
    # valid_losses['R2'] = scores['test_r2']

    regressor.fit(X_train, y_train)

    y_train_pred = regressor.predict(X_train)

    # plt.plot(y_train, label='y_train')
    plt.plot(y_train, label='y_train')
    plt.plot(y_train_pred, label='y_train_pred')
    plt.title('EL_stacking', fontsize='large', fontweight='bold')
    plt.legend()
    plt.show()


    joblib.dump(regressor, modelSavePath)


    # print('\ntrain: ')
    # print("MSE: ", np.mean(-train_losses['MSE']))
    # print("MAE: ", np.mean(-train_losses['MAE']))
    # print("R2_Score: ", np.mean(train_losses['R2']))
    # print('\nvalid: ')
    # print("MSE: ", np.mean(-valid_losses['MSE']))
    # print("MAE: ", np.mean(-valid_losses['MAE']))
    # print("R2_Score: ", np.mean(valid_losses['R2']))

    print("y_train: ")
    print("[")
    for i in range(0, y_train.shape[0]):
        print(y_train[i], ',', end="")
    print("\n]")
    print("y_train_pred: ")
    print("[")
    for i in range(0, y_train_pred.shape[0]):
        print(y_train_pred[i], ',', end="")
    print("\n]")

    print('\ntrain: ')
    MSE = mean_squared_error(y_train, y_train_pred)
    MAE = mean_absolute_error(y_train, y_train_pred)
    R2_Score = r2_score(y_train, y_train_pred)
    print("MSE: ", MSE)
    print("MAE: ", MAE)
    print("R2_Score: ", R2_Score)

    y_pred = regressor.predict(X_test)

    # plt.plot(y_test, label='y_test')
    plt.plot(y_test, label='y_test', linewidth=2.0, linestyle='--')
    plt.plot(y_pred, label='y_pred', linewidth=2.0)
    plt.title('Ours', fontsize='large', fontweight='bold')
    plt.grid(linewidth=0.3, linestyle='--')
    plt.xlabel('Indexes of samples')
    if dataPath == '../../data/ingotRate.csv':
        plt.ylabel('Ingot rate')
        saveName='ingotFittingEL.png'
    elif dataPath == '../../data/yieldRate.csv':
        plt.ylabel('Yield rate')
        saveName='yieldFittingEL.png'
    plt.legend()
    # plt.show()
    plt.savefig('../../result/'+saveName, bbox_inches='tight')

    print("y_test: ")
    print("[")
    for i in range(0, y_test.shape[0]):
        print(y_test[i], ',', end="")
    print("\n]")
    print("y_pred: ")
    print("[")
    for i in range(0, y_pred.shape[0]):
        print(y_pred[i], ',', end="")
    print("\n]")


    print('\ntest: ')
    MSE = mean_squared_error(y_test, y_pred)
    MAE = mean_absolute_error(y_test, y_pred)
    R2_Score = r2_score(y_test, y_pred)
    print("MSE: ", MSE)
    print("MAE: ", MAE)
    print("R2_Score: ", R2_Score)

def predict(predictionObject, data):
    regressor = None
    testData = None
    if predictionObject == 'ingotRate':
        regressor = joblib.load('./model/MLPRegression/ingotRatePrediction.pkl')
        testData = pd.DataFrame(data,
                                columns=['WS_MM', 'CS_MM', 'FS_MM', 'Mn_MM', 'CL_SM', 'Out_TE', 'S_EL', 'SN_QM',
                                         'UD_QM', 'NI_QM', 'OE_QM', 'PO_QM', 'C_QM', 'SI_QM'], dtype=float)
    elif predictionObject == 'yieldRate':
        regressor = joblib.load('./model/MLPRegression/yieldRatePrediction.pkl')
        testData = pd.DataFrame(data,
                                columns=['X31', 'X33', 'X34', 'X35', 'X36'], dtype=float)

    # testData = preprocessing.scale(testData)
    result=regressor.predict(testData.values)
    print('EL_stacking result:\n', result[0])
    return result[0]

if __name__ == '__main__':
    train('../../data/ingotRate.csv','./ingotRatePrediction.pkl')
    # train('../../data/yieldRate.csv', './yieldRatePrediction.pkl')


    # data = [[3.12, 6.12, 8.61, 5.82, 2.44, 1.59, 1.24, 0.66, 0.143, 1.824, 1.126, 1.46, 0.38, 0.46]]  # ingotRate
    # predict('ingotRate', data)
    # data = [[6.684, 2.561, 1.654, 1.36, 0.679]]  # yieldRate
    # predict('yieldRate', data)

