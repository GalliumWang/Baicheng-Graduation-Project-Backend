from django.shortcuts import render
import pandas as pd
import json
from django.http import HttpResponse
from .algorithm import predict, airplane_id, feature_importance
# Create your views here.

# user_train = pd.read_csv(
#     "D:\\Captain America\\大学高等教育\\毕业设计\\算法\\数据集\\train_del.csv", encoding="gbk")
# user_test = pd.read_csv(
#     "D:\\Captain America\\大学高等教育\\毕业设计\\算法\\数据集\\test.csv", encoding="gbk")


def predict_view(request):
    if request.method == 'POST':
        # predict_res_data = predict.predict_function(
        #     user_train.copy(), user_test.copy())
        # feature_res_data = feature_importance.XGBoost_function(
        #     user_train.copy(), user_test.copy())
        # # 传给前端 json
        # predict_json_data = predict_res_data.to_json(orient='records')
        # feature_json_data = feature_res_data.to_json(orient='records') 
        # t = [{'predict': json.loads(
        #     predict_json_data), 'feature_importance': json.loads(feature_json_data)}]
        # json_data = json.dumps(t)
        text_content = None
        with open('predict_data.json') as f:
            text_content = f.read()
        return HttpResponse(text_content, content_type='application/json', charset='utf-8')
    else:
        return HttpResponse('方法错误')


def airplane_id_view(request):
    if request.method == 'POST':
        # res_data = airplane_id.airplane_id_search('AB1015')
        # # 传给前端 json
        # # json_records = res_data.to_json(orient="records")
        # json_data = res_data.to_json(orient='records')
        text_content = None
        with open('airplane_id.json') as f:
            text_content = f.read()
        return HttpResponse(text_content, content_type='application/json', charset='utf-8')
    else:
        return HttpResponse('方法错误')


# def feature_importance_view(request):
#     if request.method == 'POST':
#         res_data = feature_importance.XGBoost_function(user_train, user_test)
#         # 传给前端 json
#         # json_records = res_data.to_json(orient="records")
#         json_data = res_data.to_json(orient='records')
#         return HttpResponse(json_data, content_type='application/json', charset='utf-8')
#     else:
#         return HttpResponse(request.method)
