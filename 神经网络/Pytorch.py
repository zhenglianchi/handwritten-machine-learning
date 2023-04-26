from sklearn.datasets import load_iris
import torch
from torch import nn,optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

class mynet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Sequential(nn.Linear(4,4),nn.ReLU(),nn.Linear(4,3))
    def forward(self,x):
        out=self.layer1(x)
        return out

def train(model,traindata,l,epoch,device):
    cross=nn.CrossEntropyLoss()
    opt=optim.SGD(model.parameters(),lr=l)
    model.train()
    for i in range(epoch):
        for feautre,label in traindata:
            feautre=Variable(feautre.type(torch.FloatTensor)).to(device)
            label=Variable(label.type(torch.LongTensor)).to(device)
            print(feautre.is_cuda)
            out=model(feautre)
            loss=cross(out,label)
            print_loss=loss.data.item()
            opt.zero_grad()
            loss.backward()
            opt.step()

def test(model,testdata,m,device):
    model.eval()
    acc=0
    for feautre,label in testdata:
        feautre=Variable(feautre.type(torch.FloatTensor)).to(device)
        label=Variable(label.type(torch.LongTensor)).to(device)
        out=model(feautre)
        _ , pred = torch.max(out, 1)
        if pred==label:
            acc+=1
    print("acc:",acc/m)

def save(model):
    torch.save(model.state_dict(),"test.pt")

if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
    model=mynet().to(device)
    l=0.1
    batch_size=16
    epoch=100
    iris=load_iris()
    m=iris.data.shape[0]
    traindata=TensorDataset(torch.tensor(iris.data),torch.tensor(iris.target))
    testdata=TensorDataset(torch.tensor(iris.data),torch.tensor(iris.target))
    traindata=DataLoader(traindata,batch_size=batch_size,shuffle=True)
    testdata=DataLoader(testdata,batch_size=1,shuffle=True)
    train(model,traindata,l,epoch,device)
    test(model,testdata,m,device)
    save(model)