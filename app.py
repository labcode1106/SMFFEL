from flask import Flask, request
from flask_cors import *

import math

from model.EL_stacking import EL_stacking
from model.KNNRegression import KNNRegression
from model.MLPRegression import MLPRegression
from model.SVR import SVR
from model.decisionTreeRegressor import decisionTreeRegressor
from model.lassoRegression import lassoRegression
from model.multipleLinearRegression import multipleLinearRegression
from model.ridgeRegression import ridgeRegression

app = Flask(__name__)
CORS(app, supports_credentials=True)

@app.route('/')
def hello_world():
    return 'Hello World!'


# post: http://192.168.31.238:5000/predict
# param: modelName: [decisionTreeRegressor, KNNRegression, lassoRegression, MLPRegression, multipleLinearRegression, ridgeRegression, SVR]
#        predictionObject: [ingotRate, yieldRate]
#        data: list type, missing 'X32'
# example: {
#               "modelName": "decisionTreeRegressor",
#               "predictionObject": "yieldRate",
#               "data": [6.684, 2.561, 1.654, 1.36, 0.679]
#           }
# return: prediction value
@app.route('/predict', methods=['POST'])
def predict():
    # 传入要调用的 模型名称, 预测对象 和 数据参数
    print('request: ')
    print(request)
    modelName = request.json.get('modelName')
    predictionObject = request.json.get('predictionObject')
    data = request.json.get('data')

    ingot_mean=[5.03462,6.76084,6.95272,4.44687,5.04503,5.16038,1.28759,1.91933,0.37570,1.67074,1.07155,1.89887,0.41667,0.47000]
    ingot_var=[3.51204,5.33356,3.05886,1.21162,2.24650,4.33852,0.26669,1.39364,0.02152,0.43735,0.33963,0.22663,0.00448,0.00450]
    yield_mean=[6.19660, 2.62793, 1.23502, 1.37805, 0.69160]
    yield_var=[0.56427,0.09674,0.76684,0.05519,0.02216]

    if predictionObject=='ingotRate':
        data_son = list(map(lambda x: x[0] - x[1], zip(data, ingot_mean)))
        data = [list(map(lambda x: x[0] / math.sqrt(x[1]), zip(data_son, ingot_var)))]
    elif predictionObject=='yieldRate':
        data_son = list(map(lambda x: x[0] - x[1], zip(data, yield_mean)))
        data = [list(map(lambda x: x[0] / math.sqrt(x[1]), zip(data_son, yield_var)))]
    else:
        return 'error: predictionObject is wrong!'


    result=None
    if modelName=='decisionTreeRegressor':
        result=decisionTreeRegressor.predict(predictionObject,data)
    elif modelName=='KNNRegression':
        result=KNNRegression.predict(predictionObject,data)
    elif modelName=='lassoRegression':
        result=lassoRegression.predict(predictionObject,data)
    elif modelName=='MLPRegression':
        result=MLPRegression.predict(predictionObject,data)
    elif modelName=='multipleLinearRegression':
        result=multipleLinearRegression.predict(predictionObject,data)
    elif modelName=='ridgeRegression':
        result=ridgeRegression.predict(predictionObject,data)
    elif modelName=='SVR':
        result=SVR.predict(predictionObject,data)
    elif modelName=='EL_stacking':
        result=EL_stacking.predict(predictionObject,data)
    else:
        result='error: modelName is wrong!'

    return str(result)




if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
