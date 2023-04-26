from Pytorch import mynet
import torch
from sklearn.datasets import load_iris
import torch
from torch.utils.data import DataLoader, TensorDataset
from Pytorch import test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
model=mynet()
model.load_state_dict(torch.load('test.pt',map_location=device))
model.to(device)
iris=load_iris()
m=iris.data.shape[0]
testdata=TensorDataset(torch.tensor(iris.data),torch.tensor(iris.target))
testdata=DataLoader(testdata,batch_size=1,shuffle=True)
test(testdata=testdata,model=model,m=m,device=device)
