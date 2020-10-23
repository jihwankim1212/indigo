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

# disk Resource Monitor
def sizeMonitor(name):

    csv_path = 'input/csv/size/' + name
    log_path = 'input/filestore/' + name
    output_path = 'output/size'
    dir_lst = os.listdir(log_path)

    timestamp = []
    size = []

    for logs in range(len(dir_lst)):

        date = dir_lst[logs].split('.')[1]
        time = date[0:4] + '-' + date[4:6] + '-' + date[6:] + ' 00:00:00'
        timestamp.append(time)

        # print(dir_lst[logs])
        monitor = read_data(log_path + '/' + dir_lst[logs])

        list_Monitor = []
        for i in range(len(monitor)):
            list_Monitor.append(monitor[i][0].split('#')[1])

        size.append(len(list_Monitor))

    res = [(a,b) for a,b in zip(timestamp, size)]
    df = pd.DataFrame(data=res, columns=["timestamp","value"])
    df.to_csv(csv_path + '.csv', index=False)

    plt.figure(figsize=(12, 8))
    plt.plot(size, label='size')
    plt.xlabel('time')
    # plt.yticks(np.arange(0, 100, 10.0))
    plt.legend(loc='right')
    plt.savefig(output_path + '/'+ name +'.png')
    # plt.show()
    plt.close()

    return list_Monitor

def corr():
    sizeMonitor('adaptorStatusMonitor')
    # sizeMonitor('brokerMonitor')
    # sizeMonitor('cpuResourceMonitor')
    # sizeMonitor('diskResourceMonitor')
    # sizeMonitor('esbServiceStatusMonitor')
    # sizeMonitor('esbTrafficMonitor')
    # sizeMonitor('imsAbuseMonitor')
    # sizeMonitor('imsDataMonitor')
    # sizeMonitor('imsWaitingDataMonitor')
    # sizeMonitor('jvmResourceMonitor')
    # sizeMonitor('networkResourceMonitor')
    # sizeMonitor('resourceMonitor')
    # sizeMonitor('saStatusMonitor')
    # sizeMonitor('threadResourceMonitor')

if __name__ == '__main__':
    corr()