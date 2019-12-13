import numpy as np
import tensorflow as tf
import pandas as pd
import random
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import sys
import csv

alpha = 4.0
MOVING_AVERAGE_DECAY = 0.99 #滑动平均衰减值
MODEL_SAVE_PATH = "./hashcons-model/"
MODEL_NAME = "hash_model"
BATCH_SIZE = 100 
OUTPUT_DATA_NUM = 10000 #输出文件的hash码条数，是BATCH_SIZE倍数


def threshold(net_output):
    for j in range(0,np.array(net_output).shape[0]):  #batch大小，即样本个数
        for i in range(0,np.array(net_output).shape[1]):   #哈希码位数
            if(net_output[j][i] > 0.5):
                net_output[j][i] = 1
            else:
                net_output[j][i] = 0
    return net_output
    
 #定义卷积函数
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

#定义一个2*2的最大池化层
def max_pool_2_2(x):
    return tf.nn.max_pool(x,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')
    
    
x = tf.placeholder("float",shape=[None,7168],name='x')
col1=x[:,0:1024]
col2=x[:,1024:2048]
col3=x[:,2048:3072 ]
col4=x[:,3072:4096]

col1_image = tf.reshape(col1,[-1,32,32,1])
col2_image = tf.reshape(col2,[-1,32,32,1])
col3_image = tf.reshape(col3,[-1,32,32,1])
col4_image = tf.reshape(col4,[-1,32,32,1])

#初始化第一层权重
w_conv1 = tf.Variable(tf.truncated_normal([11,11,1,96],stddev=0.1),name = 'wc1')
#初始化第一层偏置项
b_conv1 = tf.Variable(tf.constant(0.1, shape = [96]),name = 'bc1')

# 第一层卷积并激活
h_col1_conv1 = tf.nn.relu(conv2d(col1_image,w_conv1) + b_conv1)
h_col2_conv1 = tf.nn.relu(conv2d(col2_image,w_conv1) + b_conv1)
h_col3_conv1 = tf.nn.relu(conv2d(col3_image,w_conv1) + b_conv1)
h_col4_conv1 = tf.nn.relu(conv2d(col4_image,w_conv1) + b_conv1)


#第一层池化
h_col1_pool1 = max_pool_2_2(h_col1_conv1)
h_col2_pool1 = max_pool_2_2(h_col2_conv1)
h_col3_pool1 = max_pool_2_2(h_col3_conv1)
h_col4_pool1 = max_pool_2_2(h_col4_conv1)


#初始第二层权重
w_conv2 = tf.Variable(tf.truncated_normal([5,5,96,256],stddev=0.1),name = 'wc2')
#初始化第二层偏置项
b_conv2 = tf.Variable(tf.constant(0.1, shape = [256]),name = 'bc2')

#第二层卷积
h_col1_conv2 = tf.nn.relu(conv2d(h_col1_pool1,w_conv2) + b_conv2)
h_col2_conv2 = tf.nn.relu(conv2d(h_col2_pool1,w_conv2) + b_conv2)
h_col3_conv2 = tf.nn.relu(conv2d(h_col3_pool1,w_conv2) + b_conv2)
h_col4_conv2 = tf.nn.relu(conv2d(h_col4_pool1,w_conv2) + b_conv2)


#第二层池化
h_col1_pool2 = max_pool_2_2(h_col1_conv2)
h_col2_pool2 = max_pool_2_2(h_col2_conv2)
h_col3_pool2 = max_pool_2_2(h_col3_conv2)
h_col4_pool2 = max_pool_2_2(h_col4_conv2)


#初始化第三层权重
w_conv3 = tf.Variable(tf.truncated_normal([3,3,256,384],stddev=0.1),name = 'wc3')
#初始化第三层偏置项
b_conv3 = tf.Variable(tf.constant(0.1, shape = [384]),name = 'bc3')

#第三层卷积
h_col1_conv3 = tf.nn.relu(conv2d(h_col1_pool2,w_conv3) + b_conv3)
h_col2_conv3 = tf.nn.relu(conv2d(h_col2_pool2,w_conv3) + b_conv3)
h_col3_conv3 = tf.nn.relu(conv2d(h_col3_pool2,w_conv3) + b_conv3)
h_col4_conv3 = tf.nn.relu(conv2d(h_col4_pool2,w_conv3) + b_conv3)


#初始化第四层权重
w_conv4 = tf.Variable(tf.truncated_normal([3,3,384,384],stddev=0.1),name = 'wc4')
#初始化第四层偏置项
b_conv4 = tf.Variable(tf.constant(0.1, shape = [384]),name = 'bc4')
#第四层卷积
h_col1_conv4 = tf.nn.relu(conv2d(h_col1_conv3,w_conv4) + b_conv4)
h_col2_conv4 = tf.nn.relu(conv2d(h_col2_conv3,w_conv4) + b_conv4)
h_col3_conv4 = tf.nn.relu(conv2d(h_col3_conv3,w_conv4) + b_conv4)
h_col4_conv4 = tf.nn.relu(conv2d(h_col4_conv3,w_conv4) + b_conv4)


#初始化第五层权重
w_conv5 = tf.Variable(tf.truncated_normal([3,3,384,256],stddev=0.1),name = 'wc5')
#初始化第五层偏置项
b_conv5 = tf.Variable(tf.constant(0.1, shape = [256]),name = 'bc5')
#第五层卷积
h_col1_conv5 = tf.nn.relu(conv2d(h_col1_conv4,w_conv5) + b_conv5)
h_col2_conv5 = tf.nn.relu(conv2d(h_col2_conv4,w_conv5) + b_conv5)
h_col3_conv5 = tf.nn.relu(conv2d(h_col3_conv4,w_conv5) + b_conv5)
h_col4_conv5 = tf.nn.relu(conv2d(h_col4_conv4,w_conv5) + b_conv5)


#第五层池化
h_col1_pool5 = max_pool_2_2(h_col1_conv5)
h_col2_pool5 = max_pool_2_2(h_col2_conv5)
h_col3_pool5 = max_pool_2_2(h_col3_conv5)
h_col4_pool5 = max_pool_2_2(h_col4_conv5)


# 将第五层卷积池化后的结果，转成一个3*3*256的数组
h_col1_pool5_flat = tf.reshape(h_col1_pool5,[-1,3*3*256])
h_col2_pool5_flat = tf.reshape(h_col2_pool5,[-1,3*3*256])
h_col3_pool5_flat = tf.reshape(h_col3_pool5,[-1,3*3*256])
h_col4_pool5_flat = tf.reshape(h_col4_pool5,[-1,3*3*256])


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

# 防止过拟合
#keep_prob = tf.placeholder(tf.float32,name='keep_prob')

#修改h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
# h_col1_fc1_drop = tf.nn.dropout(h_col1_fc1,keep_prob)
# h_col2_fc1_drop = tf.nn.dropout(h_col2_fc1,keep_prob)
# h_col3_fc1_drop = tf.nn.dropout(h_col3_fc1,keep_prob)
# h_col4_fc1_drop = tf.nn.dropout(h_col4_fc1,keep_prob)


# 设置第二层全连接层的权重
w_fc2 = tf.Variable(tf.truncated_normal([4096,4096],stddev=0.1),name = 'wf2')
# 设置第二层全连接层的偏置
b_fc2 = tf.Variable(tf.constant(0.1, shape = [4096]),name = 'bf2')

# 第二层全连接
h_col1_fc2_temp = tf.nn.l2_normalize(tf.matmul(h_col1_fc1,w_fc2) + b_fc2)
h_col2_fc2_temp = tf.nn.l2_normalize(tf.matmul(h_col2_fc1,w_fc2) + b_fc2)
h_col3_fc2_temp = tf.nn.l2_normalize(tf.matmul(h_col3_fc1,w_fc2) + b_fc2)
h_col4_fc2_temp = tf.nn.l2_normalize(tf.matmul(h_col4_fc1,w_fc2) + b_fc2)

h_col1_fc2 = tf.nn.relu(h_col1_fc2_temp)
h_col2_fc2 = tf.nn.relu(h_col2_fc2_temp)
h_col3_fc2 = tf.nn.relu(h_col3_fc2_temp)
h_col4_fc2 = tf.nn.relu(h_col4_fc2_temp)


# 防止过拟合
# h_col1_fc2_drop = tf.nn.dropout(h_col1_fc2,keep_prob)
# h_col2_fc2_drop = tf.nn.dropout(h_col2_fc2,keep_prob)
# h_col3_fc2_drop = tf.nn.dropout(h_col3_fc2,keep_prob)
# h_col4_fc2_drop = tf.nn.dropout(h_col4_fc2,keep_prob)


# 设置第三层全连接层的权重
#w_fc1 = weight_variable([3*3*256,1024])
w_fc3 = tf.Variable(tf.truncated_normal([4096,1000],stddev=0.1),name = 'wf3')
# 设置第三层全连接层的偏置
b_fc3 = tf.Variable(tf.constant(0.1, shape = [1000]),name = 'bf3')

# 第三层全连接

h_col1_fc3_temp = tf.nn.l2_normalize(tf.matmul(h_col1_fc2,w_fc3) + b_fc3)
h_col2_fc3_temp = tf.nn.l2_normalize(tf.matmul(h_col2_fc2,w_fc3) + b_fc3)
h_col3_fc3_temp = tf.nn.l2_normalize(tf.matmul(h_col3_fc2,w_fc3) + b_fc3)
h_col4_fc3_temp = tf.nn.l2_normalize(tf.matmul(h_col4_fc2,w_fc3) + b_fc3)

h_col1_fc3 = tf.nn.relu(h_col1_fc3_temp)
h_col2_fc3 = tf.nn.relu(h_col2_fc3_temp)
h_col3_fc3 = tf.nn.relu(h_col3_fc3_temp)
h_col4_fc3 = tf.nn.relu(h_col4_fc3_temp)


# 防止过拟合
# h_col1_fc3_drop = tf.nn.dropout(h_col1_fc3,keep_prob)
# h_col2_fc3_drop = tf.nn.dropout(h_col2_fc3,keep_prob)
# h_col3_fc3_drop = tf.nn.dropout(h_col3_fc3,keep_prob)
# h_col4_fc3_drop = tf.nn.dropout(h_col4_fc3,keep_prob)


#输出层
w_fc4 = tf.Variable(tf.truncated_normal([1000,64],stddev=0.1),name = 'wf4')
b_fc4 = tf.Variable(tf.constant(0.1, shape = [64]),name = 'bf4')


y_col1_conv_tmp = tf.matmul(h_col1_fc3,w_fc4) + b_fc4
y_col2_conv_tmp = tf.matmul(h_col2_fc3,w_fc4) + b_fc4
y_col3_conv_tmp = tf.matmul(h_col3_fc3,w_fc4) + b_fc4
y_col4_conv_tmp = tf.matmul(h_col4_fc3,w_fc4) + b_fc4


y_col1_conv = tf.nn.sigmoid(y_col1_conv_tmp,name='y_col1_conv')
y_col2_conv = tf.nn.sigmoid(y_col2_conv_tmp,name='y_col2_conv')
y_col3_conv = tf.nn.sigmoid(y_col3_conv_tmp,name='y_col3_conv')
y_col4_conv = tf.nn.sigmoid(y_col4_conv_tmp,name='y_col4_conv')

ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
variable_to_restore = ema.variables_to_restore()
saver = tf.train.Saver(variable_to_restore)

#data_batch, randNum = getBatch(BATCH_SIZE)
#data_batch = csvread("data.csv")

fileRead = pd.read_csv('testdata.csv' ,header=None ,chunksize=BATCH_SIZE)

with open('hash.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            for chunk in fileRead:
                chunk = chunk.values
                reshape_xs = np.reshape(chunk, [BATCH_SIZE,7168])
                y_col1, y_col2, y_col3, y_col4 = sess.run([y_col1_conv, y_col2_conv, y_col3_conv, y_col4_conv], feed_dict = {x:reshape_xs})
                '''
                produce_hash_1=threshold(y_col1)
                produce_hash_2=threshold(y_col2)
                produce_hash_3=threshold(y_col3)
                produce_hash_4=threshold(y_col4)
                '''
                result = np.concatenate((y_col1, y_col2, y_col3, y_col4), axis=1)
                #print(result)
                writer.writerows(result)





