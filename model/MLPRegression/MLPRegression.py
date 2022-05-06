# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, KFold, \
    cross_validate
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor

import joblib

def train(dataPath, modelSavePath):
    print('\ntraining: ', dataPath)

    # Importing the dataset
    dataset = pd.read_csv(dataPath)
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, dataset.shape[1] - 1]

    # normalize
    X_scaled = preprocessing.StandardScaler().fit_transform(X)
    # X_scaled = preprocessing.MinMaxScaler().fit_transform(X)
    # X_scaled = preprocessing.MaxAbsScaler().fit_transform(X)
    # X_scaled = preprocessing.Normalizer().fit_transform(X)

    # X_scaled = None
    # if dataPath=='../../data/ingotRate.csv':
    #     X_scaled = preprocessing.MaxAbsScaler().fit_transform(X)
    # elif dataPath=='../../data/yieldRate.csv':
    #     X_scaled = preprocessing.StandardScaler().fit_transform(X)



    # total_score=0
    # for ti in range(0,10):
    #     print('==============round: ',ti,'====================')

    # Splitting the dataset into the Training set and Test set
    # X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values, test_size=0.2, random_state=0)
    if dataPath == '../../data/ingotRate.csv' or dataPath == '../../data/ingotRate_average.csv':
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values, test_size=0.2, random_state=3)
        # X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=3)
    elif dataPath == '../../data/yieldRate.csv':
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values, test_size=0.2, random_state=0)
        # X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=0)

    # X_train, y_train=upsample(X_train,y_train)
    # Splitting the dataset into the Training set and Validation set
    # X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.25, random_state=0)

    train_losses = {'MSE': [], 'MAE': [], 'R2': []}
    valid_losses = {'MSE': [], 'MAE': [], 'R2': []}

    scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
    alpha_range = [1e-2, 1e-1, 1e+0, 1e+1, 1e+2, 1e+3]
    regressor=None
    if dataPath == '../../data/ingotRate.csv' or dataPath == '../../data/ingotRate_average.csv':
        regressor = MLPRegressor(hidden_layer_sizes=(14,8,4,2), activation='tanh', solver='lbfgs', alpha=1e-3,
                                 max_iter=2000, random_state=0)  # 惩罚系数为0.01，最大迭代次数为200
    # if dataPath == '../../data/newIngotRate.csv':
    #     regressor = MLPRegressor(hidden_layer_sizes=(14, 8, 4, 2), activation='tanh', solver='lbfgs', alpha=1e-3,
    #                              max_iter=2000, random_state=0)  # 惩罚系数为0.01，最大迭代次数为200
    elif dataPath == '../../data/yieldRate.csv':
        regressor = MLPRegressor(hidden_layer_sizes=(7,2), activation='tanh', solver='lbfgs', alpha=1e-3,
                                 max_iter=2000, random_state=0)  # 惩罚系数为0.01，最大迭代次数为200
    # maxScore = cross_val_score(regressor, X_train, y_train, cv=10, scoring='r2').mean()  # cv：选择每次测试折数
    scores = cross_validate(regressor, X_train, y_train, cv=10, scoring=scoring, return_train_score=True)

    train_losses['MSE'] = scores['train_neg_mean_squared_error']
    train_losses['MAE'] = scores['train_neg_mean_absolute_error']
    train_losses['R2'] = scores['train_r2']
    valid_losses['MSE'] = scores['test_neg_mean_squared_error']
    valid_losses['MAE'] = scores['test_neg_mean_absolute_error']
    valid_losses['R2'] = scores['test_r2']
    maxMSEScore = scores['test_neg_mean_squared_error'].mean()
    # maxMAEScore = scores['test_neg_mean_absolute_error'].mean()
    maxR2Score = scores['test_r2'].mean()
    alpha = 1e-3
    for i in alpha_range:
        if dataPath == '../../data/ingotRate.csv' or dataPath == '../../data/ingotRate_average.csv':
            regressor = MLPRegressor(hidden_layer_sizes=(14,8,4,2), activation='tanh', solver='lbfgs', alpha=i,
                                     max_iter=2000, random_state=0)
        # if dataPath == '../../data/newIngotRate.csv':
        #     regressor = MLPRegressor(hidden_layer_sizes=(14, 8, 4, 2), activation='tanh', solver='lbfgs', alpha=i,
        #                              max_iter=2000, random_state=0)
        elif dataPath == '../../data/yieldRate.csv':
            regressor = MLPRegressor(hidden_layer_sizes=(7,2), activation='tanh', solver='lbfgs', alpha=i,
                                     max_iter=2000, random_state=0)
        # scores = cross_val_score(regressor, X_train, y_train, cv=10, scoring='r2')  # cv：选择每次测试折数
        # score=scores.mean()
        # if maxScore < score:
        #     maxScore = score
        #     alpha = i
        scores = cross_validate(regressor, X_train, y_train, cv=10, scoring=scoring, return_train_score=True)
        MSEScore = scores['test_neg_mean_squared_error'].mean()
        R2Score = scores['test_r2'].mean()
        # if maxMSEScore > MSEScore:
        #     maxMSEScore = MSEScore
        if maxR2Score < R2Score:
            maxR2Score = R2Score
            alpha = i
            train_losses['MSE'] = scores['train_neg_mean_squared_error']
            train_losses['MAE'] = scores['train_neg_mean_absolute_error']
            train_losses['R2'] = scores['train_r2']
            valid_losses['MSE'] = scores['test_neg_mean_squared_error']
            valid_losses['MAE'] = scores['test_neg_mean_absolute_error']
            valid_losses['R2'] = scores['test_r2']

    print('best alpha: ', alpha)
    # plt.plot([1e-3] + alpha_range, cv_scores)
    # plt.xlabel('alpha_range')
    # plt.ylabel('r2')  # 通过图像选择最好的参数
    # plt.show()


    print('\ntrain: ')
    print("MSE: ", np.mean(-train_losses['MSE']))
    print("MAE: ", np.mean(-train_losses['MAE']))
    print("R2_Score: ", np.mean(train_losses['R2']))
    print('\nvalid: ')
    print("MSE: ", np.mean(-valid_losses['MSE']))
    print("MAE: ", np.mean(-valid_losses['MAE']))
    print("R2_Score: ", np.mean(valid_losses['R2']))

    if dataPath == '../../data/ingotRate.csv' or dataPath == '../../data/ingotRate_average.csv':
        regressor = MLPRegressor(hidden_layer_sizes=(14,8,4,2), activation='tanh', solver='lbfgs', alpha=alpha,
                                 max_iter=2000, random_state=0)  # 惩罚系数为0.01，最大迭代次数为200
    # if dataPath == '../../data/newIngotRate.csv':
    #     regressor = MLPRegressor(hidden_layer_sizes=(14, 8, 4, 2), activation='tanh', solver='lbfgs', alpha=alpha,
    #                              max_iter=2000, random_state=0)  # 惩罚系数为0.01，最大迭代次数为200
    elif dataPath == '../../data/yieldRate.csv':
        regressor = MLPRegressor(hidden_layer_sizes=(7,2), activation='tanh', solver='lbfgs', alpha=alpha,
                                 max_iter=2000, random_state=0)  # 惩罚系数为0.01，最大迭代次数为200
    regressor.fit(X_train, y_train)

    y_train_pred = regressor.predict(X_train)

    plt.plot(y_train, label='y_train')
    plt.plot(y_train_pred, label='y_train_pred')
    plt.title('MLPRegression', fontsize='large', fontweight='bold')
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

    # print('\ntrain: ')
    # MSE = mean_squared_error(y_train, y_train_pred)
    # MAE = mean_absolute_error(y_train, y_train_pred)
    # R2_Score = r2_score(y_train, y_train_pred)
    # print("MSE: ", MSE)
    # print("MAE: ", MAE)
    # print("R2_Score: ", R2_Score)

    # save model
    joblib.dump(regressor, modelSavePath)

    # Predicting the Test set results
    y_pred = regressor.predict(X_test)

    plt.plot(y_test, label='y_test', linewidth=2.0, linestyle='--')
    plt.plot(y_pred, label='y_pred', linewidth=2.0)
    plt.title('MLP regression', fontsize='large', fontweight = 'bold')
    plt.grid(linewidth=0.3, linestyle='--')
    plt.xlabel('Indexes of samples')
    if dataPath == '../../data/ingotRate.csv':
        plt.ylabel('Ingot rate')
        saveName='ingotFittingMLP.png'
    elif dataPath == '../../data/yieldRate.csv':
        plt.ylabel('Yield rate')
        saveName='yieldFittingMLP.png'
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

    print('\ntest: ')
    MSE = mean_squared_error(y_test, y_pred)
    MAE = mean_absolute_error(y_test, y_pred)
    R2_Score = r2_score(y_test, y_pred)
    print("MSE: ", MSE)
    print("MAE: ", MAE)
    print("R2_Score: ", R2_Score)

    #     total_score += R2_Score
    #
    # print('mean_score: ', total_score / 10)


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
    print('MLPRegresssion result:\n', result[0])
    return result[0]


if __name__ == '__main__':
    # train('../../data/ingotRate.csv','./ingotRatePrediction.pkl')
    # train('../../data/ingotRate_average.csv','./ingotRatePrediction_average.pkl')
    train('../../data/yieldRate.csv', './yieldRatePrediction.pkl')

    # data = [[3.12, 6.12, 8.61, 5.82, 2.44, 1.59, 1.24, 0.66, 0.143, 1.824, 1.126, 1.46, 0.38, 0.46]]  # ingotRate
    # predict('ingotRate', data)
    # data = [[6.684, 2.561, 1.654, 1.36, 0.679]]  # yieldRate
    # predict('yieldRate', data)
