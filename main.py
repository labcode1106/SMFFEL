from model.decisionTreeRegressor.decisionTreeRegressor import predict as dtrPredict
from model.KNNRegression.KNNRegression import predict as knnPredict
from model.lassoRegression.lassoRegression import predict as lassoPredict
from model.MLPRegression.MLPRegression import predict as mlpPredict
from model.multipleLinearRegression.multipleLinearRegression import predict as lrPredict
from model.ridgeRegression.ridgeRegression import predict as ridgePredict
from model.SVR.SVR import predict as svrPredict
from model.EL_stacking.EL_stacking import predict as elPredict

if __name__ == '__main__':
    data_ingotRate = [[3.12, 6.12, 8.61, 5.82, 2.44, 1.59, 1.24, 0.66, 0.143, 1.824, 1.126, 1.46, 0.38, 0.46]]  # ingotRate
    data_yieldRate = [[6.684, 2.561, 1.654, 1.36, 0.679]]  # yieldRate

    print('\ningotRate prediction:')
    dtrPredict('ingotRate', data_ingotRate)
    knnPredict('ingotRate', data_ingotRate)
    lassoPredict('ingotRate', data_ingotRate)
    mlpPredict('ingotRate', data_ingotRate)
    lrPredict('ingotRate', data_ingotRate)
    ridgePredict('ingotRate', data_ingotRate)
    svrPredict('ingotRate', data_ingotRate)
    elPredict('ingotRate', data_ingotRate)

    print('\nyieldRate prediction:')
    dtrPredict('yieldRate', data_yieldRate)
    knnPredict('yieldRate', data_yieldRate)
    lassoPredict('yieldRate', data_yieldRate)
    mlpPredict('yieldRate', data_yieldRate)
    lrPredict('yieldRate', data_yieldRate)
    ridgePredict('yieldRate', data_yieldRate)
    svrPredict('yieldRate', data_yieldRate)
    elPredict('yieldRate', data_yieldRate)