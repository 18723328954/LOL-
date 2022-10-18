import utils
import torch
import data
from dataset import LOLDataSet
from models import MLP, Transformer,SVM,DecisionTree,RandomForest
from torch.utils.data import DataLoader

## 超参数设置
epoches = 20
batch_size = 256
learning_rate = 7e-5
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # 训练设备

## 模型定义
#model = MLP(input_dim=14)
#model = Transformer(dim=14)
model = SVM()

## 数据
img, label = data.getData('swim')
x_train,x_test,y_train,y_test = utils.split_train_test_data(img,label)

def train_ml(model=model):
    model.train(x_train,y_train)
    model.evalue(x_test,y_test)

def train_dl(model=model):
    ## 数据集导入
    train_dataset = LOLDataSet(x_train,y_train)
    test_dataset = LOLDataSet(x_test,y_test)
    train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

    ## 优化器和损失值
    optimizer = torch.optim.Adam(lr=learning_rate,params=model.parameters())

    ## 开始训练
    for i in range(epoches):
        utils.train_one_epoch(model=model,optimizer=optimizer,data_loader=train_dataloader,device=device,epoch=i+1)
        utils.evaluate(model=model,data_loader=test_dataloader,device=device,epoch=i+1)

    torch.save(model.state_dict(),'./src/mlp.pth')

