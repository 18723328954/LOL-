import sys
import pymysql
import numpy as np
import joblib
import pandas as pd
import torch
from sqlalchemy import create_engine
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def csv2sql(db, path,table_name):
    '''将csv文件转化sql文件'''
    mysql_setting = {
        'host': 'localhost',
        'port': 3306,
        'user': 'root',
        'passwd': '616131',
        # 数据库名称
        'db': db,
        'charset': 'utf8'
    }
    path = path  # csv路径
    table_name = table_name  # mysql表
    df = pd.read_csv(path, encoding='utf-8')
    engine = create_engine("mysql+pymysql://{user}:{passwd}@{host}:{port}/{db}".format(**mysql_setting), max_overflow=5)
    df.to_sql(table_name, engine, index=False, if_exists='replace')
    print("ok")


def sql2df(db):
    '''从mysql中读取数据存入DataFrame'''
    mysql_setting = {
        'host': 'localhost',
        'port': 3306,
        'user': 'root',
        'passwd': '616131',
        # 数据库名称
        'db': db,
        'charset': 'utf8'
    }
    engine = create_engine("mysql+pymysql://{user}:{passwd}@{host}:{port}/{db}".format(**mysql_setting), max_overflow=5)
    sql = '''select * from matches'''  # sql查询语句
    df = pd.read_sql(sql, engine)
    return df

def load_ml(path):
    '''读取机器学习模型'''
    clf = joblib.load(path)
    return clf

def load_dl(clf,path):
    '''读取机器学习模型'''
    clf.load_state_dict(torch.load(path))
    return clf

def save_ml(clf,name):
    '''保存机器学习模型'''
    joblib.dump(clf,"./src/"+name+'.m')

def save_dl(clf,name):
    '''保存深度学习模型'''
    torch.save(clf.state_dict(), './src/'+name+'.pth')

def split_train_test_data(x,y,test_rate=0.2):
    '''划分训练集和数据集'''
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=test_rate)
    return x_train,x_test,y_train,y_test

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    '''训练一轮模型'''
    model.to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    '''模型预测'''
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num