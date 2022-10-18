import torch
from torch import nn
import torch.nn.functional as F
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report


class SVM():
    '''支持向量机'''

    def __init__(self, C=1.0,
                 kernel="rbf",
                 degree=3,
                 gamma="scale",
                 coef0=0.0,
                 max_iter=-1, ):
        self.C = C
        self.kernal = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.max_iter = max_iter
        self.clf = SVC(C=self.C, kernel=self.kernal, degree=self.degree, gamma=self.gamma, coef0=self.coef0,
                       max_iter=self.max_iter)

    def train(self, x, y):
        self.clf.fit(x, y)

    def evalue(self, x, y):
        eval_score = self.clf.score(x, y)
        print('valid acc: ' + str(eval_score))

    def confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        print(cm)


class DecisionTree():
    '''决策树'''

    def __init__(self,
                 criterion="gini",
                 splitter="best",
                 max_depth=None, ):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.clf = DecisionTreeClassifier(criterion=self.criterion,splitter=self.splitter,max_depth=self.max_depth)

    def train(self, x, y):
        self.clf.fit(x, y)

    def evalue(self, x, y):
        eval_score = self.clf.score(x, y)
        print('valid acc: ' + eval_score)

    def confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        print(cm)

class RandomForest():
    '''随机森林'''

    def __init__(self,
                 n_estimators = 10,
                 criterion="gini",
                 max_depth=None, ):
        self.criterion = criterion
        self.n_estimators =n_estimators
        self.max_depth = max_depth
        self.clf = RandomForestClassifier(n_estimators=self.n_estimators,criterion=self.criterion,max_depth=self.max_depth)

    def train(self, x, y):
        self.clf.fit(x, y)

    def evalue(self, x, y):
        eval_score = self.clf.score(x, y)
        print('valid acc: ' + eval_score)

    def confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        print(cm)

class MLP(nn.Module):
    '''多层感知机模型'''

    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2)

    def confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        print(cm)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        return F.softmax(x)


class Transformer(nn.Module):
    '''Transformer模型'''

    def __init__(self, dim, encoder_layers=6, encoder_heads=7, decoder_layers=6, decoder_heads=7):
        super(Transformer, self).__init__()
        self.encoder_list = []
        self.decoder_list = []
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(14, 2)
        for i in range(encoder_layers):
            self.encoder_list.append(nn.TransformerEncoderLayer(d_model=dim, nhead=encoder_heads))
        for i in range(decoder_layers):
            self.decoder_list.append(nn.TransformerDecoderLayer(d_model=dim, nhead=decoder_heads))
        self.encoder = nn.ModuleList(self.encoder_list)
        self.decoder = nn.ModuleList(self.decoder_list)

    def confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        print(cm)

    def forward(self, x):
        x = x[:, None, :]  # 先扩展维度
        for encoder in self.encoder:
            x = encoder(x)
        # for decoder in self.decoder:
        #     x = decoder(x)
        x = self.flatten(x)
        x = F.relu(self.fc(x))

        return F.softmax(x)
