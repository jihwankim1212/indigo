import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import datetime
# import tensorflow as tf
import os

def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        # txt 파일의 헤더(id document label)는 제외하기
        data = data[1:]
    return data

div = 50

# adaptor
def adaptorStatusMonitor():

    log_path = 'input/filestore/adaptorStatusMonitor'
    output_path = 'output/plt/adaptorStatusMonitor'
    dir_lst = os.listdir(log_path)

    for logs in range(len(dir_lst)):

        print(dir_lst[logs])
        adaptorStatusMonitor = read_data(log_path + '/' + dir_lst[logs])

        list_adaptorStatusMonitor = []
        for i in range(len(adaptorStatusMonitor)):
            list_adaptorStatusMonitor.append(adaptorStatusMonitor[i][0].split('#')[1])

        timestamp = []
        s1 = []
        s0 = []
        s_2 = []
        s_1 = []
        s_9 = []
        sNULL = []

        for i in range(len(list_adaptorStatusMonitor)):
            if i == 0 or i % div ==0:
                temp = json.loads(list_adaptorStatusMonitor[i])
                timestamp.append(datetime.fromtimestamp(temp['timestamp'] / 1000))
                try:
                    s1.append(float(temp['s1']))
                except:
                    s1.append(float(0))
                try:
                    s0.append(float(temp['s0']))
                except:
                    s0.append(float(0))
                try:
                    s_2.append(float(temp['s_2']))
                except:
                    s_2.append(float(0))
                try:
                    s_1.append(float(temp['s_1']))
                except:
                    s_1.append(float(0))
                try:
                    s_9.append(float(temp['s_9']))
                except:
                    s_9.append(float(0))
                try:
                    sNULL.append(float(temp['sNULL']))
                except:
                    sNULL.append(float(0))

        plt.figure(figsize=(12, 8))
        plt.plot(timestamp, s1, label='s1')
        plt.plot(timestamp, s0, label='s0')
        plt.plot(timestamp, s_2, label='s_2')
        plt.plot(timestamp, s_1, label='s_1')
        plt.plot(timestamp, s_9, label='s_9')
        plt.plot(timestamp, sNULL, label='sNULL')
        plt.xlabel('time')
        plt.yticks(np.arange(0, 100, 10.0))
        plt.legend(loc='right')
        plt.savefig(output_path + '/' + dir_lst[logs] + '.png')
        # plt.show()
        plt.close()

    return list_adaptorStatusMonitor

# broker
def brokerMonitor():

    log_path = 'input/filestore/brokerMonitor'
    output_path = 'output/plt/brokerMonitor'
    dir_lst = os.listdir(log_path)

    for logs in range(len(dir_lst)):

        print(dir_lst[logs])
        brokerMonitor = read_data(log_path + '/' + dir_lst[logs])

        list_brokerMonitor = []
        for i in range(len(brokerMonitor)):
            list_brokerMonitor.append(brokerMonitor[i][0].split('#')[1])

        timestamp = []
        address = []
        store = []
        mem = []
        size = []

        for i in range(len(list_brokerMonitor)):
            if i ==0 or i % div ==0:
                temp = json.loads(list_brokerMonitor[i])
                timestamp.append(datetime.fromtimestamp(temp['timestamp'] / 1000))
                # try:
                #     address.append(temp['address'])
                # except:
                #     address.append('')
                try:
                    store.append(float(temp['store']))
                except:
                    store.append(float(0))
                try:
                    mem.append(float(temp['mem']))
                except:
                    mem.append(float(0))
                try:
                    size.append(temp['size'])
                except:
                    size.append(float(0))

        plt.figure(figsize=(12, 8))
        # plt.plot(timestamp, address, label='address')
        plt.plot(timestamp, store, label='store')
        plt.plot(timestamp, mem, label='mem')
        plt.plot(timestamp, size, label='size')
        plt.xlabel('time')
        plt.yticks(np.arange(0, 100, 10.0))
        plt.legend(loc='right')
        plt.savefig(output_path + '/' + dir_lst[logs] + '.png')
        # plt.show()
        plt.close()

    return list_brokerMonitor

# cpu
def cpuResourceMonitor():

    csv_path = 'input/csv/data/cpuResourceMonitor'
    log_path = 'input/filestore/cpuResourceMonitor'
    output_path = 'output/plt/cpuResourceMonitor'
    dir_lst = os.listdir(log_path)

    output = []
    output2 = []

    for logs in range(len(dir_lst)):

        print(dir_lst[logs])
        cpuResourceMonitor = read_data(log_path + '/' + dir_lst[logs])

        list_cpuResourceMonitor = []
        for i in range(len(cpuResourceMonitor)):
            list_cpuResourceMonitor.append(cpuResourceMonitor[i][0].split('#')[1])

        sec = []
        timestamp = []
        cpu = []

        # print(len(list_cpuResourceMonitor))

        for i in range(len(list_cpuResourceMonitor)):
            # if i ==0 or i % div ==0:
            temp = json.loads(list_cpuResourceMonitor[i])
            sec.append(i/div)
            timestamp.append(datetime.fromtimestamp(temp['timestamp']/1000))
            cpu.append(float(temp['cpu']))
            TempStr = str(datetime.fromtimestamp(temp['timestamp']/1000)).split('.')[0][:-2]
            TempStr = TempStr + '00'
            print(TempStr)
            output.append(TempStr)
            output2.append(float(temp['cpu']))

        # regression

        # x_data = sec
        # y_data = cpu

        # version 1을 실행하게 해줌
        # tf.compat.v1.disable_eager_execution()
        #
        # X = tf.compat.v1.placeholder(tf.float32, shape=None, name='X')
        # Y = tf.compat.v1.placeholder(tf.float32, shape=None, name='Y')
        #
        # W = tf.Variable(tf.compat.v1.random_normal([1]), name='weight')
        # b = tf.Variable(tf.compat.v1.random_normal([1]), name='bias')
        #
        # # hyper parameter
        # learning_rate = 0.1
        # training_steps = 100
        #
        # # Our hypothesis Xw+b
        # hypothesis = W * X + b
        #
        # # cost/loss function
        # loss = tf.reduce_mean(tf.square(hypothesis - y_data))
        # # Minimize
        # optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)
        # train = optimizer.minimize(loss)
        #
        # # 세션을 생성하고 초기화합니다.
        # with tf.compat.v1.Session() as sess:
        #     sess.run(tf.compat.v1.global_variables_initializer())
        #
        #     # 최적화를 100번 수행합니다.
        #     for step in range(training_steps):
        #         # sess.run 을 통해 train_op 와 cost 그래프를 계산합니다.
        #         # 이 때, 가설 수식에 넣어야 할 실제값을 feed_dict 을 통해 전달합니다.
        #         cost_val, W_val, b_val, train_val = sess.run(
        #             [loss, W, b, train],
        #             feed_dict={
        #                 X: x_data,
        #                 Y: y_data
        #             }
        #         )
        #
        #         print(step, "Cost : ", cost_val, "\tW : ", W_val, "\tb : ", b_val)
        #
        #     # 최적화가 완료된 모델에 테스트 값을 넣고 결과가 잘 나오는지 확인해봅니다.
        #     print("\n=== Test ===")
        #     print("X: 5, Y:", sess.run(hypothesis, feed_dict={X: 5}))
        #     print("X: 2.5, Y:", sess.run(hypothesis, feed_dict={X: 2.5}))

        # plt.figure(figsize=(12, 8))
        # plt.plot(timestamp, cpu, label='cpu')
        # plt.xlabel('time')
        # plt.ylabel('cpu')
        # # plt.yticks(np.arange(min(cpu), max(cpu)+1, 10.0))
        # plt.yticks(np.arange(0, 99, 10.0))
        # plt.legend(loc='right')
        # plt.savefig(output_path + '/' + dir_lst[logs] + '.png')
        # plt.close()

        # res = [(a, b) for a, b in zip(output, output2)]
        # df = pd.DataFrame(data=res, columns=["timestamp", "value"])
        # df.to_csv(csv_path + '.csv', index=False)

    return list_cpuResourceMonitor

# disk Resource Monitor
def diskResourceMonitor():

    csv_path = 'input/csv/data/diskResourceMonitor'
    log_path = 'input/filestore/diskResourceMonitor'
    output_path = 'output/plt/diskResourceMonitor'
    dir_lst = os.listdir(log_path)

    output = []
    output2 = []

    for logs in range(len(dir_lst)):

        print(dir_lst[logs])
        diskResourceMonitor = read_data(log_path + '/' + dir_lst[logs])

        list_diskResourceMonitor = []
        for i in range(len(diskResourceMonitor)):
            list_diskResourceMonitor.append(diskResourceMonitor[i][0].split('#')[1])

        timestamp = []
        disk = []

        for i in range(len(list_diskResourceMonitor)):
            # if i ==0 or i % div ==0:
            temp = json.loads(list_diskResourceMonitor[i])
            timestamp.append(datetime.fromtimestamp(temp['timestamp'] / 1000))

            TempStr = datetime.fromtimestamp(temp['timestamp'] / 1000)
            TempStr = str(TempStr)
            # print(TempStr.split('.')[0])
            output.append(TempStr.split('.')[0])

            try:
                disk.append(float(temp['disk']))
                output2.append(float(temp['disk']))
            except:
                disk.append(float(0))
                output2.append(float(temp['disk']))

        # plt.figure(figsize=(12, 8))
        # plt.plot(timestamp, disk, label='disk')
        # plt.xlabel('time')
        # plt.yticks(np.arange(0, 100, 10.0))
        # plt.legend(loc='right')
        # plt.savefig(output_path + '/' + dir_lst[logs] + '.png')
        # # plt.show()
        # plt.close()

        res = [(a, b) for a, b in zip(output, output2)]
        df = pd.DataFrame(data=res, columns=["timestamp", "value"])
        df.to_csv(csv_path + '.csv', index=False)

    return list_diskResourceMonitor

# resource Monitor
def resourceMonitor():

    log_path = 'input/filestore/resourceMonitor'
    output_path = 'output/plt/resourceMonitor'
    dir_lst = os.listdir(log_path)

    for logs in range(len(dir_lst)):

        print(dir_lst[logs])
        resourceMonitor = read_data(log_path + '/' + dir_lst[logs])

        list = []
        for i in range(len(resourceMonitor)):
            list.append(resourceMonitor[i][0].split('#')[1])

        timestamp = []
        disk = []

        for i in range(len(list)):
            if i ==0 or i % div ==0:
                temp = json.loads(list[i])
                timestamp.append(datetime.fromtimestamp(temp['timestamp'] / 1000))
                try:
                    disk.append(float(temp['disk']))
                except:
                    disk.append(float(0))

        plt.figure(figsize=(12, 8))
        plt.plot(timestamp, disk, label='disk')
        plt.xlabel('time')
        plt.yticks(np.arange(0, 100, 10.0))
        plt.legend(loc='right')
        plt.savefig(output_path + '/' + dir_lst[logs] + '.png')
        # plt.show()
        plt.close()

    return list

# thread Resource Monitor
def threadResourceMonitor():

    log_path = 'input/filestore/threadResourceMonitor'
    output_path = 'output/plt/threadResourceMonitor'
    dir_lst = os.listdir(log_path)

    for logs in range(len(dir_lst)):

        print(dir_lst[logs])
        resourceMonitor = read_data(log_path + '/' + dir_lst[logs])

        list = []
        for i in range(len(resourceMonitor)):
            list.append(resourceMonitor[i][0].split('#')[1])

        timestamp = []
        disk = []

        for i in range(len(list)):
            if i ==0 or i % div ==0:
                temp = json.loads(list[i])
                timestamp.append(datetime.fromtimestamp(temp['timestamp'] / 1000))
                try:
                    disk.append(float(temp['threadCount']))
                except:
                    disk.append(float(0))

        plt.figure(figsize=(12, 8))
        plt.plot(timestamp, disk, label='threadCount')
        plt.xlabel('time')
        plt.yticks(np.arange(0, 5000, 500))
        plt.legend(loc='right')
        plt.savefig(output_path + '/' + dir_lst[logs] + '.png')
        # plt.show()
        plt.close()

    return list

# jvm Resource Monitor
def jvmResourceMonitor():

    log_path = 'input/filestore/jvmResourceMonitor'
    output_path = 'output/plt/jvmResourceMonitor'
    dir_lst = os.listdir(log_path)

    for logs in range(len(dir_lst)):

        print(dir_lst[logs])
        resourceMonitor = read_data(log_path + '/' + dir_lst[logs])

        list = []
        for i in range(len(resourceMonitor)):
            list.append(resourceMonitor[i][0].split('#')[1])

        timestamp = []
        disk = []

        for i in range(len(list)):
            if i ==0 or i % div ==0:
                temp = json.loads(list[i])
                timestamp.append(datetime.fromtimestamp(temp['timestamp'] / 1000))
                try:
                    disk.append(float(temp['jvm']))
                except:
                    disk.append(float(0))

        plt.figure(figsize=(12, 8))
        plt.plot(timestamp, disk, label='jvm')
        plt.xlabel('time')
        plt.yticks(np.arange(0, 100, 10))
        plt.legend(loc='right')
        plt.savefig(output_path + '/' + dir_lst[logs] + '.png')
        # plt.show()
        plt.close()

    return list

# sa Status Monitor
def saStatusMonitor():

    log_path = 'input/filestore/saStatusMonitor'
    output_path = 'output/plt/saStatusMonitor'
    dir_lst = os.listdir(log_path)

    for logs in range(len(dir_lst)):

        print(dir_lst[logs])
        resourceMonitor = read_data(log_path + '/' + dir_lst[logs])

        list = []
        for i in range(len(resourceMonitor)):
            list.append(resourceMonitor[i][0].split('#')[1])

        timestamp = []
        cnt = []
        err = []

        for i in range(len(list)):
            if i ==0 or i % div ==0:
                temp = json.loads(list[i])
                timestamp.append(datetime.fromtimestamp(temp['timestamp'] / 1000))
                try:
                    cnt.append(float(temp['cnt']))
                except:
                    cnt.append(float(0))
                try:
                    err.append(float(temp['err']))
                except:
                    err.append(float(0))

        plt.figure(figsize=(12, 8))
        plt.plot(timestamp, cnt, label='cnt')
        plt.plot(timestamp, err, label='err')
        plt.xlabel('time')
        max = 200
        plt.yticks(np.arange(0, max, max/10))
        plt.legend(loc='right')
        plt.savefig(output_path + '/' + dir_lst[logs] + '.png')
        # plt.show()
        plt.close()

    return list

def exam():
    log_path = 'input/csv/data'
    dir_lst = os.listdir(log_path)
    for logs in range(len(dir_lst)):
        print(dir_lst[logs])
        lst = read_data(log_path + '/' + dir_lst[logs])
        print(len(lst))  # 5,715,279
        print(type(lst))

def exam_len():

    log_path = 'input/csv/data'
    dir_lst = os.listdir(log_path)

    for logs in range(len(dir_lst)):
        print(dir_lst[logs])
        lst = read_data(log_path + '/' + dir_lst[logs])
        print(len(lst)) # 5,715,279

def corr():
    # exam()
    # exam_len()
    # adaptorStatusMonitor()
    # brokerMonitor()
    cpuResourceMonitor()
    # diskResourceMonitor()
    # threadResourceMonitor()
    # jvmResourceMonitor()
    # saStatusMonitor()

if __name__ == '__main__':
    corr()