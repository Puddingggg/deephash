import numpy as np
import tensorflow as tf
import pandas as pd
import random
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import sys
STEPS = 2
BATCH_SIZE = 10
SAMPLE_NUM = 71721
LEARNNING_RATE_BASE = 0.0005  #最初的学习率
LEARNING_RATE_DECAY = 0.999  #学习率衰减率
MODEL_SAVE_PATH = "./hashcons-model/"
MODEL_NAME = "hash_model"
alpha = 4.0
MOVING_AVERAGE_DECAY = 0.99 #滑动平均衰减值


'''
#初始化theta
def theta_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)
    
#初始化beta 
def beta_variable(shape):
    initial = tf.constant(10,shape=shape, dtype=tf.float32)
    return tf.Variable(initial)
'''
 #定义卷积函数
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')


#定义一个2*2的最大池化层
def max_pool_2_2(x):
    return tf.nn.max_pool(x,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')
'''
#从数据文件中取batch
def getBatch(batch_size):
    randNum = random.randint(0,60000)
    fileRead = pd.read_csv('againdeal_seven_ple2_random_normalized.csv' ,header=None ,chunksize=10000, skiprows=randNum)
    #分块读取
    for chunk in fileRead:
        #从文件块中取n=batch_size个样本
        dfRandom = chunk.sample(n=batch_size, frac=None, replace=True,  weights=None, random_state=None, axis=None)
        #取值（无行列名）
        dfRandom = dfRandom.values
        break
    #将数据转化为tensor类型
    dfRandom = tf.cast(dfRandom, tf.float32)
    return dfRandom
    '''
    
def csvread(filelist):
    '''
    读取CSV文件
    :param filename:  路径+文件名的列表
    :return: 读取内容
    '''
    # 1. 构造文件的队列
    file_queue = tf.train.string_input_producer([filelist])

    # 2. 构造csv阅读器
    reader = tf.TextLineReader()
    #每次执行阅读器都从文件读取一行数据
    key, value = reader.read(file_queue)

    # 3.对每行内容解码
    # record_defaults:指定每一个样本的每一列的类型，指定默认值[['None'],[列数]]
    records = [[0.]] * 7168

    result = [[]] * 7168

    result = tf.decode_csv(value,record_defaults=records)

    # 4. 想要读取多个数据，就需要批处理
    features = tf.stack(result) 
    example_batch = tf.train.batch([features],batch_size=BATCH_SIZE,num_threads=4,capacity=32)
    return example_batch

'''
#参考结构化信息损失
def calculate_loss(theta, sim_list, beta):
    loss = 0
    for y in range(0,6):
        for t in range(1,5):
            logits = tf.nn.sigmoid(theta[t-1] * sim_list[5-y] + beta[t-1])
            if t <= y :
                loss = loss + tf.log(logits)
            else:
                loss = loss + tf.log(1-logits)
    return loss
'''
 
if __name__ == "__main__":
    #定义输入变量
    x = tf.placeholder("float",shape=[None,7168],name='x')
    col1=x[:,0:1024]
    col2=x[:,1024:2048]
    col3=x[:,2048:3072 ]
    col4=x[:,3072:4096]
    '''
    col5=x[:,4096:5120]
    col6=x[:,5120:6144]
    col7=x[:,6144:7168]
    '''
    col1_image = tf.reshape(col1,[-1,32,32,1])
    col2_image = tf.reshape(col2,[-1,32,32,1])
    col3_image = tf.reshape(col3,[-1,32,32,1])
    col4_image = tf.reshape(col4,[-1,32,32,1])
    '''
    col5_image = tf.reshape(col5,[-1,32,32,1])
    col6_image = tf.reshape(col6,[-1,32,32,1])
    col7_image = tf.reshape(col7,[-1,32,32,1])
    '''
    
    #初始化第一层权重
    w_conv1 = tf.Variable(tf.truncated_normal([11,11,1,96],stddev=0.1),name = 'wc1')
    #初始化第一层偏置项
    b_conv1 = tf.Variable(tf.constant(0.1, shape = [96]),name = 'bc1')

    # 第一层卷积并激活
    h_col1_conv1 = tf.nn.relu(conv2d(col1_image,w_conv1) + b_conv1)
    h_col2_conv1 = tf.nn.relu(conv2d(col2_image,w_conv1) + b_conv1)
    h_col3_conv1 = tf.nn.relu(conv2d(col3_image,w_conv1) + b_conv1)
    h_col4_conv1 = tf.nn.relu(conv2d(col4_image,w_conv1) + b_conv1)
    '''
    h_col5_conv1 = tf.nn.relu(conv2d(col5_image,w_conv1) + b_conv1)
    h_col6_conv1 = tf.nn.relu(conv2d(col6_image,w_conv1) + b_conv1)
    h_col7_conv1 = tf.nn.relu(conv2d(col7_image,w_conv1) + b_conv1)
    '''

    #第一层池化
    h_col1_pool1 = max_pool_2_2(h_col1_conv1)
    h_col2_pool1 = max_pool_2_2(h_col2_conv1)
    h_col3_pool1 = max_pool_2_2(h_col3_conv1)
    h_col4_pool1 = max_pool_2_2(h_col4_conv1)
    '''
    h_col5_pool1 = max_pool_2_2(h_col5_conv1)
    h_col6_pool1 = max_pool_2_2(h_col6_conv1)
    h_col7_pool1 = max_pool_2_2(h_col7_conv1)
    '''

    #初始第二层权重
    w_conv2 = tf.Variable(tf.truncated_normal([5,5,96,256],stddev=0.1),name = 'wc2')
    #初始化第二层偏置项
    b_conv2 = tf.Variable(tf.constant(0.1, shape = [256]),name = 'bc2')

    #第二层卷积
    h_col1_conv2 = tf.nn.relu(conv2d(h_col1_pool1,w_conv2) + b_conv2)
    h_col2_conv2 = tf.nn.relu(conv2d(h_col2_pool1,w_conv2) + b_conv2)
    h_col3_conv2 = tf.nn.relu(conv2d(h_col3_pool1,w_conv2) + b_conv2)
    h_col4_conv2 = tf.nn.relu(conv2d(h_col4_pool1,w_conv2) + b_conv2)
    '''
    h_col5_conv2 = tf.nn.relu(conv2d(h_col5_pool1,w_conv2) + b_conv2)
    h_col6_conv2 = tf.nn.relu(conv2d(h_col6_pool1,w_conv2) + b_conv2)
    h_col7_conv2 = tf.nn.relu(conv2d(h_col7_pool1,w_conv2) + b_conv2)
    '''

    #第二层池化
    h_col1_pool2 = max_pool_2_2(h_col1_conv2)
    h_col2_pool2 = max_pool_2_2(h_col2_conv2)
    h_col3_pool2 = max_pool_2_2(h_col3_conv2)
    h_col4_pool2 = max_pool_2_2(h_col4_conv2)
    '''
    h_col5_pool2 = max_pool_2_2(h_col5_conv2)
    h_col6_pool2 = max_pool_2_2(h_col6_conv2)
    h_col7_pool2 = max_pool_2_2(h_col7_conv2)
    '''

    #初始化第三层权重
    w_conv3 = tf.Variable(tf.truncated_normal([3,3,256,384],stddev=0.1),name = 'wc3')
    #初始化第三层偏置项
    b_conv3 = tf.Variable(tf.constant(0.1, shape = [384]),name = 'bc3')
    
    #第三层卷积
    h_col1_conv3 = tf.nn.relu(conv2d(h_col1_pool2,w_conv3) + b_conv3)
    h_col2_conv3 = tf.nn.relu(conv2d(h_col2_pool2,w_conv3) + b_conv3)
    h_col3_conv3 = tf.nn.relu(conv2d(h_col3_pool2,w_conv3) + b_conv3)
    h_col4_conv3 = tf.nn.relu(conv2d(h_col4_pool2,w_conv3) + b_conv3)
    '''
    h_col5_conv3 = tf.nn.relu(conv2d(h_col5_pool2,w_conv3) + b_conv3)
    h_col6_conv3 = tf.nn.relu(conv2d(h_col6_pool2,w_conv3) + b_conv3)
    h_col7_conv3 = tf.nn.relu(conv2d(h_col7_pool2,w_conv3) + b_conv3)
    '''

    #初始化第四层权重
    w_conv4 = tf.Variable(tf.truncated_normal([3,3,384,384],stddev=0.1),name = 'wc4')
    #初始化第四层偏置项
    b_conv4 = tf.Variable(tf.constant(0.1, shape = [384]),name = 'bc4')
    #第四层卷积
    h_col1_conv4 = tf.nn.relu(conv2d(h_col1_conv3,w_conv4) + b_conv4)
    h_col2_conv4 = tf.nn.relu(conv2d(h_col2_conv3,w_conv4) + b_conv4)
    h_col3_conv4 = tf.nn.relu(conv2d(h_col3_conv3,w_conv4) + b_conv4)
    h_col4_conv4 = tf.nn.relu(conv2d(h_col4_conv3,w_conv4) + b_conv4)
    '''
    h_col5_conv4 = tf.nn.relu(conv2d(h_col5_conv3,w_conv4) + b_conv4)
    h_col6_conv4 = tf.nn.relu(conv2d(h_col6_conv3,w_conv4) + b_conv4)
    h_col7_conv4 = tf.nn.relu(conv2d(h_col7_conv3,w_conv4) + b_conv4)
    '''
    
    #初始化第五层权重
    w_conv5 = tf.Variable(tf.truncated_normal([3,3,384,256],stddev=0.1),name = 'wc5')
    #初始化第五层偏置项
    b_conv5 = tf.Variable(tf.constant(0.1, shape = [256]),name = 'bc5')
    #第五层卷积
    h_col1_conv5 = tf.nn.relu(conv2d(h_col1_conv4,w_conv5) + b_conv5)
    h_col2_conv5 = tf.nn.relu(conv2d(h_col2_conv4,w_conv5) + b_conv5)
    h_col3_conv5 = tf.nn.relu(conv2d(h_col3_conv4,w_conv5) + b_conv5)
    h_col4_conv5 = tf.nn.relu(conv2d(h_col4_conv4,w_conv5) + b_conv5)
    '''
    h_col5_conv5 = tf.nn.relu(conv2d(h_col5_conv4,w_conv5) + b_conv5)
    h_col6_conv5 = tf.nn.relu(conv2d(h_col6_conv4,w_conv5) + b_conv5)
    h_col7_conv5 = tf.nn.relu(conv2d(h_col7_conv4,w_conv5) + b_conv5)
    '''

    #第五层池化
    h_col1_pool5 = max_pool_2_2(h_col1_conv5)
    h_col2_pool5 = max_pool_2_2(h_col2_conv5)
    h_col3_pool5 = max_pool_2_2(h_col3_conv5)
    h_col4_pool5 = max_pool_2_2(h_col4_conv5)
    '''
    h_col5_pool5 = max_pool_2_2(h_col5_conv5)
    h_col6_pool5 = max_pool_2_2(h_col6_conv5)
    h_col7_pool5 = max_pool_2_2(h_col7_conv5)
    '''

    # 将第五层卷积池化后的结果，转成一个3*3*256的数组
    h_col1_pool5_flat = tf.reshape(h_col1_pool5,[-1,3*3*256])
    h_col2_pool5_flat = tf.reshape(h_col2_pool5,[-1,3*3*256])
    h_col3_pool5_flat = tf.reshape(h_col3_pool5,[-1,3*3*256])
    h_col4_pool5_flat = tf.reshape(h_col4_pool5,[-1,3*3*256])
    '''
    h_col5_pool5_flat = tf.reshape(h_col5_pool5,[-1,3*3*256])
    h_col6_pool5_flat = tf.reshape(h_col6_pool5,[-1,3*3*256])
    h_col7_pool5_flat = tf.reshape(h_col7_pool5,[-1,3*3*256])
    '''

    # 设置第一层全连接层的权重
    w_fc1 = tf.Variable(tf.truncated_normal([3*3*256,4096],stddev=0.1),name = 'wf1')
    # 设置第一层全连接层的偏置
    b_fc1 = tf.Variable(tf.constant(0.1, shape = [4096]),name = 'bf1')

    # 第一层全连接
    
    h_col1_fc1_temp = tf.nn.l2_normalize(tf.matmul(h_col1_pool5_flat,w_fc1) + b_fc1)
    h_col2_fc1_temp = tf.nn.l2_normalize(tf.matmul(h_col2_pool5_flat,w_fc1) + b_fc1)
    h_col3_fc1_temp = tf.nn.l2_normalize(tf.matmul(h_col3_pool5_flat,w_fc1) + b_fc1)
    h_col4_fc1_temp = tf.nn.l2_normalize(tf.matmul(h_col4_pool5_flat,w_fc1) + b_fc1)
    
    h_col1_fc1 = tf.nn.relu(h_col1_fc1_temp)
    h_col2_fc1 = tf.nn.relu(h_col2_fc1_temp)
    h_col3_fc1 = tf.nn.relu(h_col3_fc1_temp)
    h_col4_fc1 = tf.nn.relu(h_col4_fc1_temp)
    '''
    h_col1_fc1 = tf.nn.relu(tf.matmul(h_col1_pool5_flat,w_fc1) + b_fc1)
    h_col2_fc1 = tf.nn.relu(tf.matmul(h_col2_pool5_flat,w_fc1) + b_fc1)
    h_col3_fc1 = tf.nn.relu(tf.matmul(h_col3_pool5_flat,w_fc1) + b_fc1)
    h_col4_fc1 = tf.nn.relu(tf.matmul(h_col4_pool5_flat,w_fc1) + b_fc1)
    
    h_col5_fc1 = tf.nn.relu(tf.matmul(h_col5_pool5_flat,w_fc1) + b_fc1)
    h_col6_fc1 = tf.nn.relu(tf.matmul(h_col6_pool5_flat,w_fc1) + b_fc1)
    h_col7_fc1 = tf.nn.relu(tf.matmul(h_col7_pool5_flat,w_fc1) + b_fc1)
    '''
    # 防止过拟合
    keep_prob = tf.placeholder(tf.float32,name='keep_prob')
    
    #修改h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
    h_col1_fc1_drop = tf.nn.dropout(h_col1_fc1,keep_prob)
    h_col2_fc1_drop = tf.nn.dropout(h_col2_fc1,keep_prob)
    h_col3_fc1_drop = tf.nn.dropout(h_col3_fc1,keep_prob)
    h_col4_fc1_drop = tf.nn.dropout(h_col4_fc1,keep_prob)
    '''
    h_col5_fc1_drop = tf.nn.dropout(h_col5_fc1,keep_prob)
    h_col6_fc1_drop = tf.nn.dropout(h_col6_fc1,keep_prob)
    h_col7_fc1_drop = tf.nn.dropout(h_col7_fc1,keep_prob)
    '''


    # 设置第二层全连接层的权重
    w_fc2 = tf.Variable(tf.truncated_normal([4096,4096],stddev=0.1),name = 'wf2')
    # 设置第二层全连接层的偏置
    b_fc2 = tf.Variable(tf.constant(0.1, shape = [4096]),name = 'bf2')

    # 第二层全连接
    
    h_col1_fc2_temp = tf.nn.l2_normalize(tf.matmul(h_col1_fc1_drop,w_fc2) + b_fc2)
    h_col2_fc2_temp = tf.nn.l2_normalize(tf.matmul(h_col2_fc1_drop,w_fc2) + b_fc2)
    h_col3_fc2_temp = tf.nn.l2_normalize(tf.matmul(h_col3_fc1_drop,w_fc2) + b_fc2)
    h_col4_fc2_temp = tf.nn.l2_normalize(tf.matmul(h_col4_fc1_drop,w_fc2) + b_fc2)
    
    h_col1_fc2 = tf.nn.relu(h_col1_fc2_temp)
    h_col2_fc2 = tf.nn.relu(h_col2_fc2_temp)
    h_col3_fc2 = tf.nn.relu(h_col3_fc2_temp)
    h_col4_fc2 = tf.nn.relu(h_col4_fc2_temp)
    '''
    h_col1_fc2 = tf.nn.relu(tf.matmul(h_col1_fc1_drop,w_fc2) + b_fc2)
    h_col2_fc2 = tf.nn.relu(tf.matmul(h_col2_fc1_drop,w_fc2) + b_fc2)
    h_col3_fc2 = tf.nn.relu(tf.matmul(h_col3_fc1_drop,w_fc2) + b_fc2)
    h_col4_fc2 = tf.nn.relu(tf.matmul(h_col4_fc1_drop,w_fc2) + b_fc2)
    
    h_col5_fc2 = tf.nn.relu(tf.matmul(h_col5_fc1_drop,w_fc2) + b_fc2)
    h_col6_fc2 = tf.nn.relu(tf.matmul(h_col6_fc1_drop,w_fc2) + b_fc2)
    h_col7_fc2 = tf.nn.relu(tf.matmul(h_col7_fc1_drop,w_fc2) + b_fc2)
    '''


    # 防止过拟合
    h_col1_fc2_drop = tf.nn.dropout(h_col1_fc2,keep_prob)
    h_col2_fc2_drop = tf.nn.dropout(h_col2_fc2,keep_prob)
    h_col3_fc2_drop = tf.nn.dropout(h_col3_fc2,keep_prob)
    h_col4_fc2_drop = tf.nn.dropout(h_col4_fc2,keep_prob)
    '''
    h_col5_fc2_drop = tf.nn.dropout(h_col5_fc2,keep_prob)
    h_col6_fc2_drop = tf.nn.dropout(h_col6_fc2,keep_prob)
    h_col7_fc2_drop = tf.nn.dropout(h_col7_fc2,keep_prob)
    '''


    # 设置第三层全连接层的权重
    #w_fc1 = weight_variable([3*3*256,1024])
    w_fc3 = tf.Variable(tf.truncated_normal([4096,1000],stddev=0.1),name = 'wf3')
    # 设置第三层全连接层的偏置
    b_fc3 = tf.Variable(tf.constant(0.1, shape = [1000]),name = 'bf3')

    # 第三层全连接
    
    h_col1_fc3_temp = tf.nn.l2_normalize(tf.matmul(h_col1_fc2_drop,w_fc3) + b_fc3)
    h_col2_fc3_temp = tf.nn.l2_normalize(tf.matmul(h_col2_fc2_drop,w_fc3) + b_fc3)
    h_col3_fc3_temp = tf.nn.l2_normalize(tf.matmul(h_col3_fc2_drop,w_fc3) + b_fc3)
    h_col4_fc3_temp = tf.nn.l2_normalize(tf.matmul(h_col4_fc2_drop,w_fc3) + b_fc3)
    
    h_col1_fc3 = tf.nn.relu(h_col1_fc3_temp)
    h_col2_fc3 = tf.nn.relu(h_col2_fc3_temp)
    h_col3_fc3 = tf.nn.relu(h_col3_fc3_temp)
    h_col4_fc3 = tf.nn.relu(h_col4_fc3_temp)

    '''
    h_col1_fc3 = tf.nn.relu(tf.matmul(h_col1_fc2_drop,w_fc3) + b_fc3)
    h_col2_fc3 = tf.nn.relu(tf.matmul(h_col2_fc2_drop,w_fc3) + b_fc3)
    h_col3_fc3 = tf.nn.relu(tf.matmul(h_col3_fc2_drop,w_fc3) + b_fc3)
    h_col4_fc3 = tf.nn.relu(tf.matmul(h_col4_fc2_drop,w_fc3) + b_fc3)
    h_col5_fc3 = tf.nn.relu(tf.matmul(h_col5_fc2_drop,w_fc3) + b_fc3)
    h_col6_fc3 = tf.nn.relu(tf.matmul(h_col6_fc2_drop,w_fc3) + b_fc3)
    h_col7_fc3 = tf.nn.relu(tf.matmul(h_col7_fc2_drop,w_fc3) + b_fc3)
    '''
    # 防止过拟合
    h_col1_fc3_drop = tf.nn.dropout(h_col1_fc3,keep_prob)
    h_col2_fc3_drop = tf.nn.dropout(h_col2_fc3,keep_prob)
    h_col3_fc3_drop = tf.nn.dropout(h_col3_fc3,keep_prob)
    h_col4_fc3_drop = tf.nn.dropout(h_col4_fc3,keep_prob)
    '''
    h_col5_fc3_drop = tf.nn.dropout(h_col5_fc3,keep_prob)
    h_col6_fc3_drop = tf.nn.dropout(h_col6_fc3,keep_prob)
    h_col7_fc3_drop = tf.nn.dropout(h_col7_fc3,keep_prob)
    '''

    #输出层
    w_fc4 = tf.Variable(tf.truncated_normal([1000,64],stddev=0.1),name = 'wf4')
    b_fc4 = tf.Variable(tf.constant(0.1, shape = [64]),name = 'bf4')

    '''
    y_col1_conv = tf.nn.leaky_relu(tf.matmul(h_col1_fc3_drop,w_fc4) + b_fc4,name='y_col1_conv')
    y_col2_conv = tf.nn.leaky_relu(tf.matmul(h_col2_fc3_drop,w_fc4) + b_fc4,name='y_col2_conv')
    y_col3_conv = tf.nn.leaky_relu(tf.matmul(h_col3_fc3_drop,w_fc4) + b_fc4,name='y_col3_conv')
    y_col4_conv = tf.nn.leaky_relu(tf.matmul(h_col4_fc3_drop,w_fc4) + b_fc4,name='y_col4_conv')
    y_col5_conv = tf.nn.leaky_relu(tf.matmul(h_col5_fc3_drop,w_fc4) + b_fc4,name='y_col5_conv')
    y_col6_conv = tf.nn.leaky_relu(tf.matmul(h_col6_fc3_drop,w_fc4) + b_fc4,name='y_col6_conv')
    y_col7_conv = tf.nn.leaky_relu(tf.matmul(h_col7_fc3_drop,w_fc4) + b_fc4,name='y_col7_conv')
    
    y_col1_conv_tmp = tf.nn.l2_normalize(tf.matmul(h_col1_fc3_drop,w_fc4) + b_fc4)
    y_col2_conv_tmp = tf.nn.l2_normalize(tf.matmul(h_col2_fc3_drop,w_fc4) + b_fc4)
    y_col3_conv_tmp = tf.nn.l2_normalize(tf.matmul(h_col3_fc3_drop,w_fc4) + b_fc4)
    y_col4_conv_tmp = tf.nn.l2_normalize(tf.matmul(h_col4_fc3_drop,w_fc4) + b_fc4)
    '''
    y_col1_conv_tmp = tf.matmul(h_col1_fc3_drop,w_fc4) + b_fc4
    y_col2_conv_tmp = tf.matmul(h_col2_fc3_drop,w_fc4) + b_fc4
    y_col3_conv_tmp = tf.matmul(h_col3_fc3_drop,w_fc4) + b_fc4
    y_col4_conv_tmp = tf.matmul(h_col4_fc3_drop,w_fc4) + b_fc4
    
    '''
    y_col5_conv_tmp = tf.nn.l2_normalize(tf.matmul(h_col5_fc3_drop,w_fc4) + b_fc4)
    y_col6_conv_tmp = tf.nn.l2_normalize(tf.matmul(h_col6_fc3_drop,w_fc4) + b_fc4)
    y_col7_conv_tmp = tf.nn.l2_normalize(tf.matmul(h_col7_fc3_drop,w_fc4) + b_fc4)
    '''
    
    y_col1_conv = tf.nn.sigmoid(y_col1_conv_tmp,name='y_col1_conv')
    y_col2_conv = tf.nn.sigmoid(y_col2_conv_tmp,name='y_col2_conv')
    y_col3_conv = tf.nn.sigmoid(y_col3_conv_tmp,name='y_col3_conv')
    y_col4_conv = tf.nn.sigmoid(y_col4_conv_tmp,name='y_col4_conv')
    '''
    y_col5_conv = tf.nn.sigmoid(y_col5_conv_tmp,name='y_col5_conv')
    y_col6_conv = tf.nn.sigmoid(y_col6_conv_tmp,name='y_col6_conv')
    y_col7_conv = tf.nn.sigmoid(y_col7_conv_tmp,name='y_col7_conv')
    
    y_col1_conv = tf.nn.sigmoid(tf.matmul(h_col1_fc3_drop,w_fc4) + b_fc4,name='y_col1_conv')
    y_col2_conv = tf.nn.sigmoid(tf.matmul(h_col2_fc3_drop,w_fc4) + b_fc4,name='y_col2_conv')
    y_col3_conv = tf.nn.sigmoid(tf.matmul(h_col3_fc3_drop,w_fc4) + b_fc4,name='y_col3_conv')
    y_col4_conv = tf.nn.sigmoid(tf.matmul(h_col4_fc3_drop,w_fc4) + b_fc4,name='y_col4_conv')
    y_col5_conv = tf.nn.sigmoid(tf.matmul(h_col5_fc3_drop,w_fc4) + b_fc4,name='y_col5_conv')
    y_col6_conv = tf.nn.sigmoid(tf.matmul(h_col6_fc3_drop,w_fc4) + b_fc4,name='y_col6_conv')
    y_col7_conv = tf.nn.sigmoid(tf.matmul(h_col7_fc3_drop,w_fc4) + b_fc4,name='y_col7_conv')
    '''
    '''
    sim1 = - tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(y_col1_conv, y_col2_conv)),1))
    sim2 = - tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(y_col1_conv, y_col3_conv)),1))
    sim3 = - tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(y_col1_conv, y_col4_conv)),1))
    sim4 = - tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(y_col1_conv, y_col5_conv)),1))
    sim5 = - tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(y_col1_conv, y_col6_conv)),1))
    sim6 = - tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(y_col1_conv, y_col7_conv)),1))
    sim_list = []
    sim_list.append(sim1)
    sim_list.append(sim2)
    sim_list.append(sim3)
    sim_list.append(sim4)
    sim_list.append(sim5)
    sim_list.append(sim6)
    ''' #loss=tf.maximum(dist1-dist2,0)*0.4+tf.maximum(dist2-dist3,0)*0.3+tf.maximum(dist3-dist4,0)*0.15+tf.maximum(dist4-dist5,0)*0.1+tf.maximum(dist5-dist6,0)*0.05 
    #loss=tf.maximum(dist1-dist2+1,0)+tf.maximum(dist2-dist3+1,0)+tf.maximum(dist3-dist4+1,0)+tf.maximum(dist4-dist5+1,0)+tf.maximum(dist5-dist6+1,0)
    
    #loss = tf.maximum(dist1-5,0)+tf.maximum(5-dist2,0)+tf.maximum(dist2-10,0)+tf.maximum(10-dist3,0)+tf.maximum(dist3-15,0)+tf.maximum(15-dist4,0)+tf.maximum(dist4-20,0)+tf.maximum(20-dist5,0)+tf.maximum(dist5-25,0)+tf.maximum(25-dist6,0)
    
    
    #参考facenet的损失,使用欧氏距离的平方
    dist1 = tf.reduce_sum(tf.abs(tf.subtract(y_col1_conv, y_col2_conv)),1)
    dist2 = tf.reduce_sum(tf.abs(tf.subtract(y_col1_conv, y_col3_conv)),1)
    dist3 = tf.reduce_sum(tf.abs(tf.subtract(y_col1_conv, y_col4_conv)),1)
 

    '''
    Adist1 = tf.sqrt(dist1)
    Adist2 = tf.sqrt(dist2)
    Adist3 = tf.sqrt(dist3)
    Adist4 = tf.sqrt(dist4)
    Adist5 = tf.sqrt(dist5)
    Adist6 = tf.sqrt(dist6)

    basic_loss1 = tf.add(tf.subtract(Adist1,Adist2), alpha)
    basic_loss2 = tf.add(tf.subtract(Adist1,Adist3), 2*alpha)
    basic_loss3 = tf.add(tf.subtract(Adist1,Adist4), 3*alpha)
    basic_loss4 = tf.add(tf.subtract(Adist1,Adist5), 4*alpha)
    basic_loss5 = tf.add(tf.subtract(Adist1,Adist6), 5*alpha)

    basic_loss1 = tf.add(tf.subtract(dist1,dist2), alpha)
    basic_loss2 = tf.add(tf.subtract(dist1,dist3), 2*alpha)
    basic_loss3 = tf.add(tf.subtract(dist1,dist4), 3*alpha)
    basic_loss4 = tf.add(tf.subtract(dist1,dist5), 4*alpha)
    basic_loss5 = tf.add(tf.subtract(dist1,dist6), 5*alpha)
    '''
    basic_loss1 = tf.add(tf.subtract(dist1,dist2), alpha)
    basic_loss2 = tf.add(tf.subtract(dist1,dist3), 2*alpha)
    basic_loss3 = tf.add(tf.subtract(dist2,dist3), alpha)
 

    loss1 = tf.maximum(basic_loss1, 0.0)
    loss2 = tf.maximum(basic_loss2, 0.0)
    loss3 = tf.maximum(basic_loss3, 0.0)

    loss_sum=tf.add(tf.add(loss1,loss2),loss3)
    
    #位平衡(mean(y)-0.5)^2
    bitloss1 = tf.square(tf.reduce_mean(y_col1_conv,1)- 0.5)
    bitloss2 = tf.square(tf.reduce_mean(y_col2_conv,1)- 0.5)
    bitloss3 = tf.square(tf.reduce_mean(y_col3_conv,1)- 0.5)
    bitloss4 = tf.square(tf.reduce_mean(y_col4_conv,1)- 0.5)
    bitloss_sum = tf.add(tf.add(tf.add(bitloss1,bitloss2),bitloss3),bitloss4)
    
    #哈希约束-(||y-0.5||)^2
    hashloss1 = -tf.reduce_sum(tf.square(tf.subtract(y_col1_conv,0.5)),1)
    hashloss2 = -tf.reduce_sum(tf.square(tf.subtract(y_col2_conv,0.5)),1)
    hashloss3 = -tf.reduce_sum(tf.square(tf.subtract(y_col3_conv,0.5)),1)
    hashloss4 = -tf.reduce_sum(tf.square(tf.subtract(y_col4_conv,0.5)),1)
    hashloss_sum = tf.add(tf.add(tf.add(hashloss1,hashloss2),hashloss3),hashloss4)
    '''
    #哈希约束min(||y||,||y-1||)
    hashloss1_1 = tf.reduce_sum(tf.square(tf.subtract(y_col1_conv,0)),1)
    hashloss1_2 = tf.reduce_sum(tf.square(tf.subtract(y_col1_conv,1)),1)
    hashloss1 = tf.minimum(hashloss1_1,hashloss1_2)
    hashloss2_1 = tf.reduce_sum(tf.square(tf.subtract(y_col2_conv,0)),1)
    hashloss2_2 = tf.reduce_sum(tf.square(tf.subtract(y_col2_conv,1)),1)
    hashloss2 = tf.minimum(hashloss2_1,hashloss2_2)
    hashloss3_1 = tf.reduce_sum(tf.square(tf.subtract(y_col3_conv,0)),1)
    hashloss3_2 = tf.reduce_sum(tf.square(tf.subtract(y_col3_conv,1)),1)
    hashloss3 = tf.minimum(hashloss3_1,hashloss3_2)
    hashloss4_1 = tf.reduce_sum(tf.square(tf.subtract(y_col4_conv,0)),1)
    hashloss4_2 = tf.reduce_sum(tf.square(tf.subtract(y_col4_conv,1)),1)
    hashloss4 = tf.minimum(hashloss4_1,hashloss4_2)
    hashloss_sum = tf.add(tf.add(tf.add(hashloss1,hashloss2),hashloss3),hashloss4)
    '''
    tuple_loss = tf.add(tf.add(loss_sum,bitloss_sum),hashloss_sum)
    loss = tf.reduce_mean(tuple_loss, 0)
    
    #tuple_loss = tf.reduce_mean(loss_sum, 0)
#    tuple_loss=loss_sum
#    loss=loss_sum
    #loss=tuple_loss
    #加入‘losses’集合中，和参数正则化损失一起
#    tf.add_to_collection('losses', tuple_loss)
#    loss = tf.add_n([tf.get_collection('losses')])

#    loss = tf.get_collection('losses')
    '''
    theta = theta_variable([5])
    beta = beta_variable([5])
    loss = -calculate_loss(theta, sim_list, beta)
    loss = tf.reduce_mean(loss, 0)
    '''
    #定义存储训练轮数的变量，该变量不需要计算滑动平均值
    global_step = tf.Variable(0, trainable=False)

    #学习率
    learning_rate = tf.train.exponential_decay(LEARNNING_RATE_BASE, global_step, SAMPLE_NUM/BATCH_SIZE, LEARNING_RATE_DECAY, staircase=True)
    '''
    #梯度
    opt = tf.train.AdagradOptimizer(learning_rate)
    grads = opt.compute_gradients(loss, tf.global_variables())
    train_step = opt.apply_gradients(grads, global_step=global_step)
    '''
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
        
    #给定滑动平均衰减率MOVING_AVERAGE_DECAY和训练轮数的变量global_step，初始化滑动平均类
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    #在所有参数的变量上使用滑动平均
    ema_op = ema.apply(tf.trainable_variables())
    #更新每个参数的滑动平均值
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    #保存模型，最多保存最近4个  keep_checkpoint_every_n_hours=2两小时保存一次
    saver=tf.train.Saver(max_to_keep = 4, keep_checkpoint_every_n_hours = 1)

    #data_batch = getBatch(BATCH_SIZE)
    data_batch = csvread("againdeal_seven_ple2_random_normalized.csv")
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        #断点续练，若模型已存在则加载模型参数，在已训练好的模型基础上继续训练
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        #开启线程协调器
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        for i in range(STEPS):
            #batch=getBatch(BATCH_SIZE)
            xs = sess.run([data_batch])   #将取数据的操作加入图的运行
            reshape_xs = np.reshape(xs, [BATCH_SIZE,7168])
            
            train_oper, loss_value, train_dist1, train_dist2, train_dist3,basic1,basic2,basic3,bit1,bit2,bit3,bit4,hash1,hash2,hash3,hash4,y_col1,y_col2,y_col3,y_col4= sess.run([train_op, loss, dist1, dist2,dist3,basic_loss1,basic_loss2,basic_loss3,bitloss1,bitloss2,bitloss3,bitloss4,hashloss1,hashloss2,hashloss3,hashloss4,y_col1_conv,y_col2_conv,y_col3_conv,y_col4_conv], feed_dict={x:reshape_xs, keep_prob:0.5})
            
            if i % 1 == 0:
                print("step %d"%(i) , "\nloss is:" , np.mean(loss_value) ,"\ndist1 is:" , train_dist1 ,"\ndist2 is:" ,train_dist2 ,"\ndist3 is:" , train_dist3,"\nbasic_loss1 is:" ,np.mean(basic1),"\nbasic_loss2 is:",np.mean(basic2),"\nbasic_loss3 is:",np.mean(basic3),"\nbitloss1:",np.mean(bit1),"\nbitloss2:",np.mean(bit2),"\nbitloss3:",np.mean(bit3),"\nbitloss4:",np.mean(bit4),"\nhashloss1:",np.mean(hash1),"\nhashloss2:",np.mean(hash2),"\nhashloss3:",np.mean(hash3),"\nhashloss4:",np.mean(hash4),"\ny_col1_conv:",y_col1,"\ny_col2_conv",y_col2,"\ny_col3_conv",y_col3,"\ny_col4_conv",y_col4,"\nx",xs)
                #print(xs)
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step) #保存模型
                sys.stdout.flush()
            
            #print(loss.shape)

        #关闭线程协调器
        coord.request_stop()
        coord.join(threads)



