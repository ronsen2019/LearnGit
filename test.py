import datetime
import os
import pandas as pd
import csv
from dateutil.parser import parse
import time as te
import numpy as np
import math
import glob
import psutil as p
from PIL import Image
import warnings
import time
from multiprocessing import Process, Queue, Pool, Lock

q = Queue()  # 参数代表对列的长度，如果不传默认对列可以无限长


def write(q, ):
    print("我是主进程,我的id是:", os.getpid())
    inpath = "C:/Users/issuser/Desktop/New/data_file/data/"
    outpath = "C:/Users/issuser/Desktop/New/data_file/"
    while 1:
        filenames = glob.glob(inpath + "/*.txt")
        print(filenames)
        print('总共发现%s个csv文件' % len(filenames))
        if filenames != None:
            for file_name in filenames:
                str1 = file_name.split("C:/Users/issuser/Desktop/New/data_file/data\\")[1].split(".txt")[0]
                txt_name = file_name  # txt文件
                new_name3 = outpath + str1 + '.jpg'
                print(new_name3)
                dataMat = []
                f = open(txt_name)
                for line in f.readlines():
                    lineArr = line.strip().split('\t')
                    dataMat.append([lineArr[0], float(lineArr[1]), float(lineArr[2])])
                txtDF = pd.DataFrame(dataMat)
                f.close()
                txtDF.groupby(0).agg(lambda x: x.value_counts().index[0]).reset_index()
                data1 = txtDF.groupby(0).agg(lambda x: x.value_counts().index[0]).reset_index()
                data1.dropna(axis=0, how='all', inplace=True)# 去掉空行
                label = data1
                label[0] = pd.to_datetime(label[0])
                df2 = pd.DataFrame()
                index_num = []
                for index, row in label.iterrows():
                    index_num.append(index)
                    if len(index_num) == 1:  # 补首行
                        hour = row[0].hour
                        minute = row[0].minute
                        second = row[0].second
                        frist_num = hour * 60 * 60 + minute * 60 + second + 1
                        first_row = label.loc[0]
                        first_row1 = pd.DataFrame(first_row).T
                        df2 = df2.append([first_row1] * int(frist_num))
                    if len(index_num) == len(label[0]):  # 补最后一行
                        last_loc = len(index_num) - 1
                        hour = row[0].hour
                        minute = row[0].minute
                        second = row[0].second
                        last_num = (23 - hour) * 60 * 60 + (59 - minute) * 60 + (60 - second)
                        last_row = label.loc[last_loc]
                        last_row1 = pd.DataFrame(last_row).T
                        df2 = df2.append([last_row1] * int(last_num))
                data_list = data1[0]
                column = []
                for i in data_list:
                    column.append(str(i))
                for j in range(len(column) - 1):
                    b = parse(column[j + 1])
                    c = parse(column[j])
                    num = (b - c).total_seconds()
                    n = label.loc[j]
                    nn = pd.DataFrame(n).T
                    df2 = df2.append([nn] * int(num))
                dict1 = {}
                dict1['df2'] = df2
                dict1['outpath'] = new_name3
                q.put(dict1)
                print(q.qsize())
                print(u'cpu个数：', p.cpu_count())
                print('CPU占用率:', p.cpu_percent())
                print(u'内存使用：', p.virtual_memory().used)
                print(u'当前进程内存使用：', p.Process(os.getpid()).memory_info().rss)
                print(u'当前进程内存使用：', p.Process(os.getpid()).memory_info().rss)
                os.remove(txt_name)
        else:
            continue


def read(q, lic):
    print('我是子进程,我的id是: ', os.getpid())
    while 1:
        if not q.empty():
            lic.acquire()
            print("============", q.qsize())
            data_dict = q.get(True)
            lic.release()
            df2 = data_dict.get('df2')
            new_name3 = data_dict.get('outpath')
            m = []
            for ii in ["jing", "wei", 0]:
                df_1 = df2.values.tolist()
                df_li = []
                for p1 in df_1:
                    df_li.append(p1[1:])
                a = np.array(df_li)
                df_li1 = a[:, 0]
                df_li2 = a[:, 1]
                test_data = pd.DataFrame({'jing': df_li1, 'wei': df_li2})
                p_new = test_data.groupby(['jing', 'wei']).size()
                p_new1 = p_new.reset_index()
                df_li = p_new1[ii]
                print('pp1', df_li)

                df_li = df_li.values.tolist()
                df_li1 = []
                for i in range(len(df_li)):
                    df_li1.append([df_li[i]])

                result = []
                for s_li in df_li1:
                    if type(s_li[0]) == float and s_li[0] > 120:
                        a = (s_li[0] - 120.850) / 0.001
                        b = math.ceil(a)
                        result.append(b)
                    elif type(s_li[0]) == float and s_li[0] < 32:
                        a = (s_li[0] - 30.667) / 0.001
                        b = math.ceil(a)
                        result.append(b)
                    else:
                        result.append(s_li[0])
                m.append(result)
            z1, z2, z3 = m
            array = np.ones((1449, 1216)) * 0
            for i in range(len(z1)):
                array[z1[i - 1]][z2[i - 1]] = z3[i - 1]
            max = np.max(array)
            img = Image.fromarray(array * 65525 / max)
            img = img.convert('L')
            img.save(new_name3)
            # img.show()
        else:
            continue


if __name__ == '__main__':
    # 创建队列
    q = Queue()
    # 创建锁
    lic = Lock()
    pw = Process(target=write, args=(q,))
    pr = Process(target=read, args=(q, lic))
    pr1 = Process(target=read, args=(q, lic))
    pr2 = Process(target=read, args=(q, lic))
    # 启动主进程pw,写入:
    pw.start()
    # 启动子进程pr,读取:
    pr.start()
    pr1.start()
    pr2.start()
    # 将主进程和子进程加入进程池
    pw.join()
    pr.join()
    pr1.join()
    pr2.join()
