# -*- coding: utf-8 -*-
from pyhive import hive
import pandas as pd
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.estimator import DeepFMEstimator
from deepctr.estimator.inputs import input_fn_pandas
from deepctr.layers import custom_objects

pd.set_option('display.max_columns', None)
from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, DenseFeat,get_feature_names
import time

from flask import Flask, request, jsonify
import redis
from log import Logger
import time
import json
from tensorflow.python.keras.backend import set_session
import os


# define logging configuration
logger = Logger(filename='./logs/global_search_rank_ctr.log', level='debug', when='midnight', back_count=14).logger


"""
接口功能:
@param: user_id, resource_id_list, resource_type
@内部逻辑：根据用户和物料的特征打分，查询得到resource_id list
@output: resource_id list
"""

# env = 'dev'

sess = tf.Session()
# sess = tf.compat.v1.Session()
graph = tf.get_default_graph()


env = os.getenv('RECSYS_ENV')
logger.info("Current environment variable RECSYS_ENV: {env}".format(env=env))
if env not in ('dev', 'test', 'prod'):
    raise ValueError('The environment variable RECSYS_ENV should be dev/test/prod, got {env}.'.format(env=env))

# define redis configuration  47.97.207.88 root Oba.2247d
redis_conf = {
    'dev': {'host': '172.29.28.203', 'port': 6379, 'password': None},  # 应用算法开发机
    'test': {'host': '172.29.28.49', 'port': 6379, 'password': '123456'},  # 测试环境
    'prod': {'host': 'r-uf6ojxzvetp1r9m1id.redis.rds.aliyuncs.com', 'port': 6379, 'password': '!@Oba2247d'}  # 阿里云生产环境
}
# define redis connection
host = redis_conf[env]['host']
port = redis_conf[env]['port']
password = redis_conf[env]['password']
r = redis.StrictRedis(host=host, port=port, password=password, decode_responses=True)
logger.info("Connected to Redis Server {host}.".format(host=host))

# define flask app
app = Flask(__name__)


# 从Redis 批量的读取
def get_batch_features_redis(user_id, resource_id_list, resource_type_list):
    # redis_resource_key_list = ['applied_ag:resource_feature:resource_id:' + resource_type + '|' + str(i) for i in resource_id_list]
    redis_resource_key_list = []
    for i in range(len(resource_id_list)):
        redis_resource_key_list.append('applied_ag:resource_feature:resource_id:' + str(resource_type_list[i]) + '|' + str(resource_id_list[i]))
    redies_resource_feat = r.mget(redis_resource_key_list)

    resource_feature_list = []
    # 遍历redis数据，进行处理空值
    for idx, item in enumerate(redies_resource_feat):
        if item is None:
            resource_feature_list.append('{"resource_type":"'+ resource_type_list[idx] +'","resource_id":"'+ resource_id_list[idx] +'","total_show":0,"total_click":0,"total_vote":0,"total_collection":0,"total_comment":0,"total_share":0,"classification_label":"-1","keyword_label":"-1"}')
        else:
            resource_feature_list.append(item)
    resource_feature_list_join = '[' + ','.join(resource_feature_list) + ']'
    resource_df = pd.read_json(resource_feature_list_join, orient='records')
    user_default_feat = '{"user_id":"%s","sex":1,"role":"0","age":40}'%user_id
    redis_user_key = 'applied_ag:user_feature:user_id:'
    if user_id is not None:
        redis_user_key = 'applied_ag:user_feature:user_id:' + user_id

    redies_user_feat = r.get(redis_user_key)
    if redies_user_feat:
        redies_user_feat = '[' + redies_user_feat + ']'
    else:
        redies_user_feat = '[' + user_default_feat + ']'
    user_df = pd.read_json(redies_user_feat, orient='records')
    user_resource_feature_df = resource_df.assign(joinkey=1).merge(user_df.assign(joinkey=1), on='joinkey').drop('joinkey',axis=1)
    return user_resource_feature_df

# 在model加载前添加set_session
set_session(sess)
path = 'models_1221'
custom = custom_objects
# custom.update({'binary_focal_loss_fixed': None})
# Load the saved keras model back.
loaded_mode = tf.keras.experimental.load_from_saved_model(path, custom_objects=custom)

# 预测
def online_inference(predict_data):
    print("predict_data__________")
    print(predict_data.head(3))
    # predict_data.columns = [i.split('.')[1] for i in list(predict_data.columns)]
    sparse_features = ['sex', 'role']
    dense_features = ['total_show', 'total_click', 'total_vote', 'total_collection', 'total_comment', 'total_share', 'age']
    feature_names = sparse_features + dense_features
    # 给指定的字段填充默认值
    predict_data[dense_features] =  predict_data[dense_features].fillna(0)
    predict_data[sparse_features] = predict_data[sparse_features].fillna('-1')
    predict_data = predict_data.replace('None', '-1')
    predict_data = predict_data.replace('', '-1')

    # 使用LabelEncoder()，为类别特征的每一个item编号
    for feat in sparse_features:
        lbe = LabelEncoder()
        predict_data[feat] = lbe.fit_transform(predict_data[feat])

    # 数值特征 max-min 0-1归化
    mms = MinMaxScaler(feature_range=(0, 1))
    predict_data[dense_features] = mms.fit_transform(predict_data[dense_features])

    predict_data_input = {name: predict_data[name] for name in feature_names}

    # predictions = predict(predict_data_input, batch_size = 20)
    global sess
    global graph
    with graph.as_default():
        set_session(sess)
        predictions = loaded_mode.predict(predict_data_input, 20)
    return predictions


# 参数准备
# resource_id_list = [15827,15829,15833,20893,20885,22753,15827,16341,23655]
def rank(user_id, resource_ids, resource_types):
    # 获取特征
    # predict_data = get_features_by_resource_id()
    print(resource_ids)
    print(user_id)
    union_dict_list = []
    # 获取特征 Redis
    redis_start_time = time.time()
    predict_data = get_batch_features_redis(user_id, resource_ids, resource_types)
    redis_end_time = time.time()
    redis_total_time = redis_end_time - redis_start_time
    print("====================== redis total time = %s======================"%int((redis_total_time)*1000))

    # 预测
    online_inference_start_time = time.time()
    result = online_inference(predict_data)
    online_inference_end_time = time.time()
    print("====================== online_inference total time = %s======================"%int((online_inference_end_time - online_inference_start_time)*1000))

    wuliao_df = predict_data[['resource_id','resource_type']]
    trans_data = [x[0] for x in result]
    trans_set = pd.DataFrame({'score': trans_data})
    wuliao_score_df = pd.concat([wuliao_df, trans_set], axis=1)
    ## 拿到以后需要按照物料ID的打分进行排序,分数越高排在前面
    sort = wuliao_score_df.sort_values(by='score',ascending=False)

    resource_id_list = sort['resource_id'].to_list()

    # 构造返回体
    rank_http_response={}
    rank_http_response['data'] = sort[['resource_id','resource_type']].to_json(orient='records')
    # aa = wuliao_score_df.to_json(orient="records")

    return sort[['resource_id','resource_type']].to_json(orient='records')



# 健康检查，定期访问请求
@app.route("/health_check", methods=["GET"])
def health_check():
    return "This service is healthy!", 200


@app.route("/global_search_ranking", methods=["GET","POST"])
def search_rank_ctr():
    print("%s程序开始运行.....")

    total_start_time = time.time()
    # --------------------------解析参数--------------------------
    resource_id_list, resource_type_list = [], []
    user_id = ''
    if request.method == 'POST':
        print(request)
        res = request.json.get("data")
        print(res)
        user_id = request.json.get("userId")
        for item in res:
            resource_id_list.append(item.get("resourceId"))
            resource_type_list.append(item.get("resourceType"))
    # --------------------------解析参数--------------------------

    # --------------------------排序--------------------------
    rank_result = rank(user_id, resource_id_list, resource_type_list)
    #     aa = rank(user_id, resource_ids, resource_types)
    total_end_time = time.time()
    print("====================== total time = %s======================"%int((total_end_time-total_start_time)*1000))
    print(rank_result)
    return rank_result


if __name__ == '__main__':

    # 模型部署
    app.run(host='0.0.0.0', threaded=True, port=5013)