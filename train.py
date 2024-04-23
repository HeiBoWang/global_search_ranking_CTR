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
import datetime


def get_data_hive():
    print("正在从Hive读取数据......")
    conn = hive.Connection(host='172.30.2.60', port=10000, database='recsys')
    train_sql = """select * from recsys.search_full_link_30d"""
    train_data = pd.read_sql(train_sql, conn)
    conn.close()
    train_data.columns = [i.split('.')[1] for i in list(train_data.columns)]
    return train_data

# 切分时间的函数
def split_data_by_dt(dataframe):
    # 先获取最大的时间
    max_dt = dataframe['log_date'].max()
    # 得到分割时间
    dt = datetime.datetime.strptime(max_dt, "%Y-%m-%d")
    split_date = (dt + datetime.timedelta(days=-7)).strftime("%Y-%m-%d")
    # where条件区分训练测试集, 去空
    test = dataframe.where(dataframe['log_date'] > split_date)
    test = test.dropna(how='all')
    train = dataframe.where(dataframe['log_date'] <= split_date)
    train = train.dropna(how='all')
    # 返回训练测试集
    print("训练数据量 = " + str(train.shape))
    print("测试数据量 = " + str(test.shape))
    return train, test

"""
准备数据
@return: 训练集, 测试集 
"""
def processData():
    # 从hive查询特征
    data_set = get_data_hive()

    # 连续字段和分类型字段
    sparse_features = ['sex','role']
    dense_features =  ['total_show','total_click','total_vote','total_collection','total_comment','total_share','age']

    col_names_train = ['click_label'] + dense_features + sparse_features
    col_names_test = col_names_train[1:]
    
    print(data_set.head(3))

    data_set[dense_features] =  data_set[dense_features].fillna(0)
    data_set[sparse_features] = data_set[sparse_features].fillna('-1')

    #  Step 2: Simple preprocessing
    for feat in sparse_features:
        lbe = LabelEncoder()
        data_set[feat] = lbe.fit_transform(data_set[feat])

    # Step 3: Generate feature columns
    dense_feature_cols = []
    sparse_feature_cols = []
    # 处理数值类型的特征
    for feat in dense_features:
        dense_feature_cols.append(DenseFeat(feat, 1,))
    # 处理类别类型的特征
    for i,feat in enumerate(sparse_features):
        sparse_feature_cols.append( SparseFeat(feat, vocabulary_size=data_set[feat].max() + 1,embedding_dim=4) )

    fixlen_feature_columns = sparse_feature_cols + dense_feature_cols

    # generate feature columns
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)


    # Step 4: Generate the training samples and train the model
    # train, test = train_test_split(data_set, test_size=0.1)
    # ********将固定的一九分，改为7天的测试集，其他为训练集
    train, test = split_data_by_dt(data_set)

    train_model_input = {name:train[name].values for name in feature_names}
    test_model_input = {name:test[name].values for name in feature_names}

    # train_model_input,test_model_input = processData()

    return train_model_input, test_model_input, linear_feature_columns,dnn_feature_columns,train, test


''' focal loss '''
def binary_focal_loss(gamma=2, alpha=0.25):
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        return K.mean(focal_loss)

    return binary_focal_loss_fixed



# 模型训练
def train():
    # Generate the training samples and train the model
    train_model_input,test_model_input,linear_feature_columns,dnn_feature_columns,train, test = processData()

    model = DeepFM(linear_feature_columns,dnn_feature_columns,task='binary', dnn_hidden_units=(64, 32, 16))
    model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy','AUC'], )
    tf_callback = tf.keras.callbacks.TensorBoard(log_dir="./tensorboard_logs")

    history = model.fit(train_model_input, train['click_label'].values,
                        batch_size=512, epochs=20, verbose=1, validation_split=0.1, callbacks=[tf_callback])
    
    predictions = model.predict(test_model_input)
    print(predictions)

    # 模型保存
    path = 'models_2'
    tf.keras.experimental.export_saved_model(model, path)

# 加载模型预测
def predict(data_frame, batch_size):
    path = 'models_2'
    custom = custom_objects
    # custom.update({'binary_focal_loss_fixed': None})
    # Load the saved keras model back.
    loaded_mode = tf.keras.experimental.load_from_saved_model(path, custom_objects=custom)
    predictions = loaded_mode.predict(data_frame, batch_size)

    return predictions




if __name__ == "__main__":
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    # 训练且保存模型
    train()
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
   
    
    # 加载模型并预测
    # train_model_input,test_model_input,linear_feature_columns,dnn_feature_columns,train, test = processData()
    # # model.predict(test_model_input, batch_size=32)
    # predictions = predict(train_model_input, batch_size = len(test_model_input))
    # print(time.time())
    # with open('hh.txt', 'a') as f:
    #     for i in predictions: 
    #         f.write(str(i))
    # print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))