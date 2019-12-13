import numpy as np
import tensorflow as tf
import pandas as pd
import random
import os
import sys
#不使用GPU
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
STEPS = 1
BATCH_SIZE = 10
SAMPLE_NUM = 71721
LEARNNING_RATE_BASE = 0.0005  #最初的学习率
LEARNING_RATE_DECAY = 0.999  #学习率衰减率
MODEL_SAVE_PATH = "./test-model/"
MODEL_NAME = "hash_model"
alpha = 4.0
MOVING_AVERAGE_DECAY = 0.99 #滑动平均衰减值
keep_prob = 0.5

CONV1_SIZE = 11
NUM_CHANNELS = 1
CONV1_KERNEL_NUM = 96

CONV2_SIZE = 5
CONV2_KERNEL_NUM = 256

CONV3_SIZE = 3
CONV3_KERNEL_NUM = 384

CONV4_SIZE = 3
CONV4_KERNEL_NUM = 384

CONV5_SIZE = 3
CONV5_KERNEL_NUM = 256

FC1_SIZE = 4096
FC2_SIZE = 4096
FC3_SIZE = 1000
OUTPUT_SIZE = 64
'''
def threshold(net_output):
    for j in range(0,BATCH_SIZE):  #batch大小，即样本个数
        for i in range(0,64):   #哈希码位数
            if(net_output[j][i] > 0.5):
                net_output[j][i] = 1
            else:
                net_output[j][i] = 0
    return net_output
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

#定义卷积函数
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

#定义一个2*2的最大池化层
def max_pool_2_2(x):
    return tf.nn.max_pool(x,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')

#定义一个AlexNet
def cnn(sequence_reshape):
    #初始化第一层权重
    w_conv1 = tf.Variable(tf.truncated_normal([CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_KERNEL_NUM],stddev=0.1),name = 'wc1')
    #初始化第一层偏置项
    b_conv1 = tf.Variable(tf.constant(0.1, shape = [CONV1_KERNEL_NUM]),name = 'bc1')
    # 第一层卷积并激活
    conv1 = conv2d(sequence_reshape,w_conv1)
    relu1 =tf.nn.relu(tf.nn.bias_add(conv1, b_conv1))
    #第一层池化
    pool1 = max_pool_2_2(relu1)

    #初始第二层权重
    w_conv2 = tf.Variable(tf.truncated_normal([CONV2_SIZE,CONV2_SIZE,CONV1_KERNEL_NUM,CONV2_KERNEL_NUM],stddev=0.1),name = 'wc2')
    #初始化第二层偏置项
    b_conv2 = tf.Variable(tf.constant(0.1, shape = [CONV2_KERNEL_NUM]),name = 'bc2')
    #第二层卷积
    conv2 = conv2d(pool1,w_conv2)
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, b_conv2))
    #第二层池化
    pool2 = max_pool_2_2(relu2)

    #初始化第三层权重
    w_conv3 = tf.Variable(tf.truncated_normal([CONV3_SIZE,CONV3_SIZE,CONV2_KERNEL_NUM,CONV3_KERNEL_NUM],stddev=0.1),name = 'wc3')
    #初始化第三层偏置项
    b_conv3 = tf.Variable(tf.constant(0.1, shape = [CONV3_KERNEL_NUM]),name = 'bc3')
    #第三层卷积
    conv3 = conv2d(pool2,w_conv3)
    relu3 = tf.nn.relu(tf.nn.bias_add(conv3, b_conv3))

    #初始化第四层权重
    w_conv4 = tf.Variable(tf.truncated_normal([CONV4_SIZE,CONV4_SIZE,CONV3_KERNEL_NUM,CONV4_KERNEL_NUM],stddev=0.1),name = 'wc4')
    #初始化第四层偏置项
    b_conv4 = tf.Variable(tf.constant(0.1, shape = [CONV4_KERNEL_NUM]),name = 'bc4')
    #第四层卷积
    conv4 = conv2d(relu3,w_conv4)
    relu4 = tf.nn.relu(tf.nn.bias_add(conv4, b_conv4))

    #初始化第五层权重
    w_conv5 = tf.Variable(tf.truncated_normal([CONV5_SIZE,CONV5_SIZE,CONV4_KERNEL_NUM,CONV5_KERNEL_NUM],stddev=0.1),name = 'wc5')
    #初始化第五层偏置项
    b_conv5 = tf.Variable(tf.constant(0.1, shape = [CONV5_KERNEL_NUM]),name = 'bc5')
    #第五层卷积
    conv5 = conv2d(relu4,w_conv5)
    relu5 = tf.nn.relu(tf.nn.bias_add(conv5,b_conv5))
    #第五层池化
    pool5 = max_pool_2_2(relu5)

    #将第五层卷积池化后的结果，转成一个3*3*256的数组
    pool5_flat = tf.reshape(pool5,[-1,3*3*256])
    #pool_shape[1]:3 ;pool_shape[2]:3 ;pool_shape[3]:256
    #nodes:3*3*256
    #pool_shape = pool5.get_shape().as_list()
    #nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    #pool5_flat = tf.reshape(pool5, [pool_shape[0], nodes])

    #设置第一层全连接层的权重
    w_fc1 = tf.Variable(tf.truncated_normal([3*3*256,FC1_SIZE],stddev=0.1),name = 'wf1')
    # 设置第一层全连接层的偏置
    b_fc1 = tf.Variable(tf.constant(0.1, shape = [FC1_SIZE]),name = 'bf1')
    # 第一层全连接
    fc1_normalized = tf.nn.l2_normalize(tf.matmul(pool5_flat,w_fc1) + b_fc1)
    fc1 = tf.nn.relu(fc1_normalized)
    #防止过拟合
    fc1_drop = tf.nn.dropout(fc1,keep_prob)

    #设置第二层全连接层的权重
    w_fc2 = tf.Variable(tf.truncated_normal([FC1_SIZE,FC2_SIZE],stddev=0.1),name = 'wf2')
    # 设置第二层全连接层的偏置
    b_fc2 = tf.Variable(tf.constant(0.1, shape = [FC2_SIZE]),name = 'bf2')
    # 第二层全连接
    fc2_normalized = tf.nn.l2_normalize(tf.matmul(fc1_drop,w_fc2) + b_fc2)
    fc2 = tf.nn.relu(fc2_normalized)
    #防止过拟合
    fc2_drop = tf.nn.dropout(fc2,keep_prob)

    #设置第三层全连接层的权重
    w_fc3 = tf.Variable(tf.truncated_normal([FC2_SIZE,FC3_SIZE],stddev=0.1),name = 'wf3')
    # 设置第三层全连接层的偏置
    b_fc3 = tf.Variable(tf.constant(0.1, shape = [FC3_SIZE]),name = 'bf3')
    # 第三层全连接
    fc3_normalized = tf.nn.l2_normalize(tf.matmul(fc2_drop,w_fc3) + b_fc3)
    fc3 = tf.nn.relu(fc3_normalized)
    #防止过拟合
    fc3_drop = tf.nn.dropout(fc3,keep_prob)

    #输出层
    w_fc4 = tf.Variable(tf.truncated_normal([FC3_SIZE,OUTPUT_SIZE],stddev=0.1),name = 'wf4')
    b_fc4 = tf.Variable(tf.constant(0.1, shape = [OUTPUT_SIZE]),name = 'bf4')
    fc4_tmp = tf.matmul(fc3_drop,w_fc4) + b_fc4
    output = tf.nn.sigmoid(fc4_tmp)

    return output

if __name__ == '__main__':
    #定义输入变量
    x = tf.placeholder("float",shape=[None,7168],name='x')
    col1=x[:,0:1024]
    col2=x[:,1024:2048]
    col3=x[:,2048:3072 ]
    col4=x[:,3072:4096]
    col1_reshape = tf.reshape(col1,[-1,32,32,1])
    col2_reshape = tf.reshape(col2,[-1,32,32,1])
    col3_reshape = tf.reshape(col3,[-1,32,32,1])
    col4_reshape = tf.reshape(col4,[-1,32,32,1])

    y1 = cnn(col1_reshape)
    y2 = cnn(col2_reshape)
    y3 = cnn(col3_reshape)
    y4 = cnn(col4_reshape)

    #参考facenet的损失,使用欧氏距离的平方
    dist1 = tf.reduce_sum(tf.abs(tf.subtract(y1, y2)),1)
    dist2 = tf.reduce_sum(tf.abs(tf.subtract(y1, y3)),1)
    dist3 = tf.reduce_sum(tf.abs(tf.subtract(y1, y4)),1)

    basic_loss1 = tf.add(tf.subtract(dist1,dist2), alpha)
    basic_loss2 = tf.add(tf.subtract(dist1,dist3), 2*alpha)
    basic_loss3 = tf.add(tf.subtract(dist2,dist3), alpha)

    loss1 = tf.maximum(basic_loss1, 0.0)
    loss2 = tf.maximum(basic_loss2, 0.0)
    loss3 = tf.maximum(basic_loss3, 0.0)
    loss_sum=tf.add(tf.add(loss1,loss2),loss3)

    #位平衡(mean(y)-0.5)^2
    bitloss1 = tf.square(tf.reduce_mean(y1,1)- 0.5)
    bitloss2 = tf.square(tf.reduce_mean(y2,1)- 0.5)
    bitloss3 = tf.square(tf.reduce_mean(y3,1)- 0.5)
    bitloss4 = tf.square(tf.reduce_mean(y4,1)- 0.5)
    bitloss_sum = tf.add(tf.add(tf.add(bitloss1,bitloss2),bitloss3),bitloss4)

    #哈希约束-(||y-0.5||)^2
    hashloss1 = -tf.reduce_sum(tf.square(tf.subtract(y1,0.5)),1)
    hashloss2 = -tf.reduce_sum(tf.square(tf.subtract(y2,0.5)),1)
    hashloss3 = -tf.reduce_sum(tf.square(tf.subtract(y3,0.5)),1)
    hashloss4 = -tf.reduce_sum(tf.square(tf.subtract(y4,0.5)),1)
    hashloss_sum = tf.add(tf.add(tf.add(hashloss1,hashloss2),hashloss3),hashloss4)

    tuple_loss = tf.add(tf.add(loss_sum,bitloss_sum),hashloss_sum)
    loss = tf.reduce_mean(tuple_loss, 0)

    #定义存储训练轮数的变量，该变量不需要计算滑动平均值
    global_step = tf.Variable(0, trainable=False)

    #学习率
    learning_rate = tf.train.exponential_decay(LEARNNING_RATE_BASE, global_step, SAMPLE_NUM/BATCH_SIZE, LEARNING_RATE_DECAY, staircase=True)

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
            
            train_oper, loss_value, train_dist1, train_dist2, train_dist3,basic1,basic2,basic3,bit1,bit2,bit3,bit4,hash1,hash2,hash3,hash4,y_col1,y_col2,y_col3,y_col4= sess.run([train_op, loss, dist1, dist2,dist3,basic_loss1,basic_loss2,basic_loss3,bitloss1,bitloss2,bitloss3,bitloss4,hashloss1,hashloss2,hashloss3,hashloss4,y1,y2,y3,y4], feed_dict={x:reshape_xs})
            '''
            produce_hash_1=threshold(y_col1)
            produce_hash_2=threshold(y_col2)
            produce_hash_3=threshold(y_col3)
            produce_hash_4=threshold(y_col4)
            '''
            if i % 1 == 0:
                print("step %d"%(i) , "\nloss is:" , np.mean(loss_value) ,"\ndist1 is:" , train_dist1 ,"\ndist2 is:" ,train_dist2 ,"\ndist3 is:" , train_dist3,"\nbasic_loss1 is:" ,np.mean(basic1),"\nbasic_loss2 is:",np.mean(basic2),"\nbasic_loss3 is:",np.mean(basic3),"\nbitloss1:",np.mean(bit1),"\nbitloss2:",np.mean(bit2),"\nbitloss3:",np.mean(bit3),"\nbitloss4:",np.mean(bit4),"\nhashloss1:",np.mean(hash1),"\nhashloss2:",np.mean(hash2),"\nhashloss3:",np.mean(hash3),"\nhashloss4:",np.mean(hash4),"\ny_col1_conv:",y_col1,"\ny_col2_conv",y_col2,"\ny_col3_conv",y_col3,"\ny_col4_conv",y_col4,"\nx",x)
                #print(xs)
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step) #保存模型
                sys.stdout.flush()
            
            #print(loss.shape)

        #关闭线程协调器
        coord.request_stop()
        coord.join(threads)