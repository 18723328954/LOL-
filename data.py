import numpy as np

import torch

import utils

df = utils.sql2df('swim')


def countGameDuration(db='swim'):
    '''统计各个时间段的比赛数量'''
    ## 各个时间段
    zero2ten = []
    ten2twenty = []
    twenty2thirty = []
    thirty2forty = []
    other = []
    durationList = df['gameDuration'].values  # 游戏时长列表
    ## 时段统计
    for dur in durationList:
        if 0 <= dur < 60 * 10:
            zero2ten.append(dur)
        elif 60 * 10 <= dur < 60 * 20:
            ten2twenty.append(dur)
        elif 20 * 60 <= dur < 30 * 60:
            twenty2thirty.append(dur)
        elif 30 * 60 <= dur < 40 * 60:
            thirty2forty.append(dur)
        else:
            other.append(dur)
    return [['0~10', len(zero2ten)], ['10~20', len(ten2twenty)], ['20~30', len(twenty2thirty)],
            ['30~40', len(thirty2forty)], ['40+', len(other)]]


def countWinner(db='swim'):
    '''统计红蓝队胜利的次数'''
    winer = (df['winner'].value_counts().to_numpy())
    winer = [ele for ele in winer]
    a = dict(df['t1_champ3_sum1'].value_counts())
    b = dict(df['t2_champ3_sum1'].value_counts())
    return winer


def countEacPosChampSum(db='swim'):
    '''统计每个位置召唤师技能使用数量'''
    topName = ['t' + str(i) + "_champ1" + "_sum" + str(k) for i in range(1, 3) for k in range(1, 3)]
    jungleName = ['t' + str(i) + "_champ2" + "_sum" + str(k) for i in range(1, 3) for k in range(1, 3)]
    midName = ['t' + str(i) + "_champ3" + "_sum" + str(k) for i in range(1, 3) for k in range(1, 3)]
    adcName = ['t' + str(i) + "_champ4" + "_sum" + str(k) for i in range(1, 3) for k in range(1, 3)]
    supportName = ['t' + str(i) + "_champ5" + "_sum" + str(k) for i in range(1, 3) for k in range(1, 3)]
    ## 统计召唤师技能数量
    topSum = {}
    for name in topName:
        for k, v in dict(df[name].value_counts()).items():
            if k not in topSum.keys():
                topSum[k] = v
            else:
                topSum[k] += v
    jungleSum = {}
    for name in jungleName:
        for k, v in dict(df[name].value_counts()).items():
            if k not in jungleSum.keys():
                jungleSum[k] = v
            else:
                jungleSum[k] += v
    midSum = {}
    for name in midName:
        for k, v in dict(df[name].value_counts()).items():
            if k not in midSum.keys():
                midSum[k] = v
            else:
                midSum[k] += v
    adcSum = {}
    for name in adcName:
        for k, v in dict(df[name].value_counts()).items():
            if k not in adcSum.keys():
                adcSum[k] = v
            else:
                adcSum[k] += v
    supportSum = {}
    for name in supportName:
        for k, v in dict(df[name].value_counts()).items():
            if k not in supportSum.keys():
                supportSum[k] = v,
            else:
                supportSum[k] += v
    # print(sorted(topSum.items(),key=lambda x:x[1],reverse=True))
    # print(sorted(jungleSum.items(),key=lambda x:x[1],reverse=True))
    # print(sorted(midSum.items(),key=lambda x:x[1],reverse=True))
    # print(sorted(adcSum.items(),key=lambda x:x[1],reverse=True))
    # print(sorted(supportSum.items(),key=lambda x:x[1],reverse=True))
    return [topSum, jungleSum, midSum, adcSum, supportSum]


def countChampSum(db='swim'):
    '''统计召唤师技能使用数量'''
    champSumDict = {}
    ## 遍历所有队伍所有选手的召唤师技能使用次数
    summorNameList = ['t' + str(i) + "_champ" + str(j) + "_sum" + str(k) for i in range(1, 3) for j in range(1, 6) for k
                      in
                      range(1, 3)]
    ## 召唤师技能使用字典
    for summorName in summorNameList:
        summorDict = dict(df[summorName].value_counts())
        for k, v in summorDict.items():
            if k not in champSumDict.keys():
                champSumDict[k] = v
            else:
                champSumDict[k] += v
    return champSumDict


def countResource(db='swim'):
    '''获取野区资源'''
    blueTeam = df[['t1_towerKills', 't1_inhibitorKills', 't1_baronKills', 't1_dragonKills', 't1_riftHeraldKills']]
    redTeam = df[['t2_towerKills', 't2_inhibitorKills', 't2_baronKills', 't2_dragonKills', 't2_riftHeraldKills']]
    blueTeamResource = blueTeam.sum().to_list()
    redTeamResource = redTeam.sum().to_list()
    return blueTeamResource,redTeamResource

def countHero(db='swim'):
    '''统计英雄出场次数'''
    dic = {}
    name = ['t'+str(i)+'_champ'+str(j)+'id' for i in range(1,3) for j in range(1,6)]
    for na in name:
        dic_temp = dict(df[na].value_counts())
        for k,v in dic_temp.items():
            if k in dic.keys():
                dic[k] += v
            else:
                dic[k] = v
    return dic


print(sorted(countHero().items(),key=lambda x:x[1],reverse=True))
def getCorr(db='swim'):
    '''获取属性间的相关性'''
    df['bluewin'] = 2 - df['winner']
    df['redwin'] = abs(1 - df['winner'])
    df_attr = df[
        ['bluewin', 'redwin', 'firstBlood', 'firstTower', 'firstInhibitor', 'firstBaron', 'firstDragon',
         'firstRiftHerald',
         't1_towerKills', 't1_inhibitorKills', 't1_baronKills', 't1_dragonKills', 't1_riftHeraldKills', 't2_towerKills',
         't2_inhibitorKills', 't2_baronKills', 't2_dragonKills', 't2_riftHeraldKills']]
    df_corr = df_attr.corr().values
    return df_corr


def getData(db='swim'):
    '''获取训练和测试的数据'''
    df_model = df[['winner', 'firstBlood', 'firstTower', 'firstInhibitor', 'firstBaron',
                   'firstDragon', 'firstRiftHerald', 't1_towerKills', 't1_inhibitorKills', 't1_baronKills',
                   't1_dragonKills', 't2_towerKills', 't2_inhibitorKills', 't2_baronKills', 't2_dragonKills'
                   ]]
    x = df_model.drop('winner', axis=1)
    y = df_model['winner'] - 1  # 值控制在0，1
    return x.to_numpy(), y.to_numpy()

