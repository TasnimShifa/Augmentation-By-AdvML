import torch
import torch.optim as optim
import torch.nn as nn
import PIL
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision import models
from torchsummary import summary
from src.U_Net import *
from src.U_Net_attn import *
from src.U_Net_2 import *
from src.loss import FocalLoss
from config import config
from src.utils import *
from src.Colon import *
from src.IoU import *
from src.dice import *
from src.trainSF import *
from src.FGSM import *

#from src.evaluation import *

from src.eval import *
from tqdm import tqdm

CONFIG = config()

path = CONFIG.path
batch = CONFIG.batch
lr = CONFIG.lr
epochs = CONFIG.epochs
device = CONFIG.device
print(f"The device being used is: {device}\n")
id2code = CONFIG.id2code
input_size = CONFIG.input_size
model_sv_pth = CONFIG.model_path
load_model_pth = CONFIG.load_model

#f = open("Colon_Standard_FGSM.txt", "w+")
f = open("kidney_Standard_FGSM.txt", "w+")
f.close()




 
        
if __name__ == "__main__":
   
    train_transforms = transforms.Compose([transforms.Resize(input_size, 0)])
    valid_transforms = transforms.Compose([transforms.Resize(input_size, 0)])
    

    #pass transform here-in
    train_data = Colon(img_pth = path + 'train/', mask_pth = path + 'train_labels/', transform = train_transforms)
    valid_data = Colon(img_pth = path + 'val/', mask_pth = path + 'val_labels/', transform = valid_transforms)

    #data loaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch, shuffle=True)

    model = UNet(3, 2, True).to(device)
    summary(model, (3,128,128))
    #model = nn.DataParallel(model, device_ids=[3,4])
    criterion = nn.CrossEntropyLoss()# FocalLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.99)
    f = open("kidney_Standard_FGSM.txt","a+")
    f.write("\n\n ------------------------------------Standard Training by UNet------------------------------------------------")
    f.close()
    train(model, trainloader, validloader, criterion, optimizer, epochs, device, load_model_pth, model_sv_pth, plot=True, visualize=True, load_model=False)
    ''' 
    #Define transforms for the training data and validation data
    #augmentation
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(input_size,0),
        #torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(20, resample=PIL.Image.BILINEAR)
        ])
    valid_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(input_size,0),
        #torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(20, resample=PIL.Image.BILINEAR)
        ])
    '''
f = open("kidney_Standard_FGSM.txt","a+")
f.write("\n\n ------------------------------------- Standard testing------------------------------------------") 
f.close()
test(model, trainloader, validloader, criterion, optimizer, device, load_model_pth, model_sv_pth, plot=True, visualize=True, load_model=False)
    
    

f = open("kidney_Standard_FGSM.txt","a+")
f.write("\n\n -------------------------------------Adversarial testing by FGSM------------------------------------------")
f.close()
'''
max = 55
min = 0
e = (max-min)*torch.rand((1,10))+min
sorted, indices = torch.sort(e)
epsilon = sorted.numpy()
'''
epsilon = np.linspace(0.0, 1.0, num=100)
eps_stats =[]
eps_values = []
for ep in epsilon: 
    #ep = i/255
    print('Eplison:',ep)
    eps, mean_iu = fgsm_test(model, trainloader, validloader, criterion,ep, optimizer, device, load_model_pth, model_sv_pth, plot=True, visualize=True, load_model=False)
    eps_stats.append(mean_iu)
    #print(eps_stats)
    #eps_stat = pd.DataFrame(eps_stats, columns=['eps','mean_iu'])
    eps_values.append(eps)
plotCurves_eps(eps_stats,eps_values)

f = open("kidney_Standard_FGSM.txt","a+")
f.write("\n\n --------------------------------------Adversarial training by FGSM--------------------------------------------")
f.close()    
eps= 0.1
fgsm_train(model, trainloader, validloader, criterion,eps, optimizer, epochs, device, load_model_pth, model_sv_pth, plot=True, visualize=True, load_model=False)

f = open("kidney_Standard_FGSM.txt","a+")
f.write("\n\n ----------------------------------Adversarial testing after adversarial training----------------------------------------")
f.close()
epsilon = np.linspace(0.0, 1.0, num=100)
eps_stats =[]
eps_values = []
for ep in epsilon: 
    print('Eplison:',ep)
    eps, mean_iu = fgsm_test(model, trainloader, validloader, criterion,ep, optimizer, device, load_model_pth, model_sv_pth, plot=True, visualize=True, load_model=False)
    eps_stats.append(mean_iu)
    #print(eps_stats)
    #eps_stat = pd.DataFrame(eps_stats, columns=['eps','mean_iu'])
    eps_values.append(eps)
plotCurves_eps_test(eps_stats,eps_values)

f = open("kidney_Standard_FGSM.txt","a+")
f.write("\n\n ---------------------------------- Standard testing after adversarial training(clean image)------------------------------------------") 
f.close()
test(model, trainloader, validloader, criterion, optimizer, device, load_model_pth, model_sv_pth, plot=True, visualize=True, load_model=False)