import json

# import konlpy
# from konlpy.tag import Okt
# okt = Okt()
# print(okt.pos('이 밤 그날의 그날의반딧불을 당신의 창 가까이 보낼게요'))

import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import word_tokenize
# print(WordPunctTokenizer().tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))

from nltk.corpus import gutenberg   # Docs from project gutenberg.org
from nltk import regexp_tokenize

import re

def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        # txt 파일의 헤더(id document label)는 제외하기
        data = data[1:]
    return data

dict_adaptorStatusMonitor = {}
dict_brokerMonitor = {}
dict_cpuResourceMonitor = {}

def adaptorStatusMonitor():
    log_path = 'input/log/'
    adaptorStatusMonitor = read_data(log_path + 'adaptorStatusMonitor/adaptorStatusMonitor.20200904')
    # print(adaptorStatusMonitor)
    # print(type(adaptorStatusMonitor))
    # print(adaptorStatusMonitor[0])
    # print(type(adaptorStatusMonitor[0][0]))
    # print(adaptorStatusMonitor[0][0])

    # tmp_list = adaptorStatusMonitor[0][0].split('#')
    # print(tmp_list[0])
    # print(tmp_list[1])

    tmp_list = []
    for i in range(len(adaptorStatusMonitor)):
        tmp_list.append(adaptorStatusMonitor[i][0].split('#')[1])

    # print(type(tmp_list[1]))
    # data_dict = json.loads(tmp_list[1])
    # print(type(data_dict))
    # print(data_dict)
    # print(data_dict['timestamp'])
    # print(type(data_dict['timestamp']))

    return tmp_list

def brokerMonitor():
    log_path = 'input/log/'
    brokerMonitor = read_data(log_path + 'brokerMonitor/brokerMonitor.20200904')
    # print(brokerMonitor)
    # print(type(brokerMonitor))
    # print(brokerMonitor[0])
    # print(type(brokerMonitor[0][0]))
    # print(brokerMonitor[0][0])
    tmp_list = brokerMonitor[0][0].split('#')
    # print(tmp_list[0])
    # print(tmp_list[1])
    # print(type(tmp_list[1]))
    data_dict = json.loads(tmp_list[1])
    # print(type(data_dict))
    # print(data_dict)
    # print(data_dict['timestamp'])
    # print(type(data_dict['timestamp']))
    return data_dict

def cpuResourceMonitor():
    log_path = 'input/log/'
    cpuResourceMonitor = read_data(log_path + 'cpuResourceMonitor/cpuResourceMonitor.20200904')
    # print(cpuResourceMonitor)
    # print(type(cpuResourceMonitor))
    # print(cpuResourceMonitor[0])
    # print(type(cpuResourceMonitor[0][0]))
    # print(cpuResourceMonitor[0][0])
    tmp_list = cpuResourceMonitor[0][0].split('#')
    # print(tmp_list[0])
    # print(tmp_list[1])
    # print(type(tmp_list[1]))
    data_dict = json.loads(tmp_list[1])
    # print(type(data_dict))
    # print(data_dict)
    # print(data_dict['timestamp'])
    # print(type(data_dict['timestamp']))
    # print(data_dict['cpu'])
    return data_dict

def cluster():
    # read
    dict_adaptorStatusMonitor = adaptorStatusMonitor()
    dict_brokerMonitor = brokerMonitor()
    dict_cpuResourceMonitor = cpuResourceMonitor()

    # print(dict_adaptorStatusMonitor)
    # print(dict_brokerMonitor)
    # print(dict_cpuResourceMonitor)

    # pattern = r'''(?x) ([A-Z]\.)+ | \w+(-\w+)* | \$?\d+(\.\d+)?%? | \.\.\. | [][.,;"'?():-_`]'''
    # tokens_en = regexp_tokenize(dict_adaptorStatusMonitor, pattern)
    # print(tokens_en)

    # str = "123answkduf456"
    str = "{'timestamp': 1599192360013, 's1': 0, 's0': 0, 's_2': 0, 's_1': 0, 's2': 0, 's_9': 0, 'sNULL': 4}"
    # p = re.compile("[^0-9-=.#/?:${}' ]")
    # print("".join(p.findall(str)))
    print(WordPunctTokenizer().tokenize(str))
    print(type(WordPunctTokenizer().tokenize(str)))
    token = WordPunctTokenizer().tokenize(str)
    # print(type(token))
    text = nltk.Text(token)
    # print(len(text.tokens))
    # print(len(set(text.tokens)))
    # print(text.vocab().most_common(10))

    # print(type(dict_adaptorStatusMonitor))
    # print(len(dict_adaptorStatusMonitor))

    token = []
    # for i in range(len(dict_adaptorStatusMonitor)):
    #     token.append(WordPunctTokenizer().tokenize(dict_adaptorStatusMonitor[i]))

    token = [t for d in dict_adaptorStatusMonitor for t in d]

    # print(type(token))
    # print(token)
    text = nltk.Text(token)

    # total token
    print(len(text.tokens))
    print(len(set(text.tokens)))
    print(text.vocab().most_common(30))

if __name__ == '__main__':
    cluster()