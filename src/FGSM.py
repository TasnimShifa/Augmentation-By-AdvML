import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import torchvision
import torch.optim as optim
import torch.nn as nn
import PIL
from torchvision import datasets, transforms
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
#from src.evaluation import *

from src.eval import *
from tqdm import tqdm

CONFIG = config()

path = CONFIG.path
batch = CONFIG.batch
lr = CONFIG.lr
epochs = CONFIG.epochs
device = CONFIG.device
id2code = CONFIG.id2code
input_size = CONFIG.input_size
model_sv_pth = CONFIG.model_path
load_model_pth = CONFIG.load_model

    
    
def FGSM(net,inputs,labels,device,eps,criterion):
    
    #net,inputs,labels = net.to(device),inputs.to(device),labels.to(device)
       
    x = inputs.clone().detach().requires_grad_(True).to(device)
    alpha = eps 
    #print(x)
    pred = net(x)
    target = labels.argmax(1)
    loss = criterion(pred,target.long())
    loss.backward()
    noise = x.grad.data
    '''
    per = alpha * torch.sign(noise)
    #per.data.clamp_(min=0.0, max=1.0)
    
    th = alpha
    for i in range(per.shape[0]):
        for j in range(per.shape[1]):
            for k in range(per.shape[2]):
                for l in range(per.shape[3]):
                    if per[i,j,k,l]<th:
                        per[i,j,k,l]=0
                    elif per[i,j,k,l]==th:
                        per[i,j,k,l]=th
                    elif per[i,j,k,l]>th:
                        per[i,j,k,l]=1
    
    #print(per)
    '''
    x.data = x.data + alpha * torch.sign(noise)
    x.data.clamp_(min=0.0, max=1.0)
              
    x.grad.zero_()
    #print(x)
    return x







def fgsm_test(model, trainloader, validloader, criterion, eps, optimizer, device, load_pth, model_sv_pth, plot=True, visualize=False, load_model=False):
    if load_model: model.load_state_dict(torch.load(load_pth))
    model.eval()
    fgsm_stats = []
    #valid_loss_min = np.Inf
    eps_stats =[]
    epochs=10
    print('FGSM Testing Started.....')
    
    for epoch in range(epochs):
        fgsm_test_loss = 0
        fgsm_test_iou = []
        
        iterator = tqdm(validloader)
        for i, data in enumerate(iterator):
            inputs, mask, rgb = data
            inputs, mask = inputs.to(device), mask.to(device)
            #optimizer.zero_grad()
            
            adv_inputs = FGSM(model,inputs,mask,device,eps,criterion)
            output_adv = model(adv_inputs.float())
            #print(adv_inputs.detach().cpu())
            target = mask.argmax(1)
            
            loss = criterion(output_adv, target.long())
            #loss.backward()
            #optimizer.step()
            fgsm_test_loss += loss.item() * adv_inputs.size(0) 
            iou = iou_pytorch(output_adv.argmax(1), target)
            fgsm_test_iou.extend(iou)     
            
            if visualize and  i == 0:
                print('The testing images')
                show_databatch_adv_testing(str(round(eps,4)),inputs.detach().cpu(), size=(8,8))
                print('The testing adversarial images')
                show_databatch_adv_testing_adversarial(str(round(eps,4)),adv_inputs.detach().cpu(), size=(8,8))#problem here
                print('The original masks')
                show_databatch_adv_testing_original(str(round(eps,4)),rgb.detach().cpu(), size=(8,8))
                RGB_mask =  mask_to_rgb(output_adv.detach().cpu(), id2code)
                print('Predicted masks')
                show_databatch_adv_testing_predicted(str(round(eps,4)),torch.tensor(RGB_mask).permute(0,3,1,2), size=(8,8))
            break
        fgsm_miou = torch.FloatTensor(fgsm_test_iou).mean()
        fgsm_test_loss = fgsm_test_loss / len(trainloader.dataset)
        print(f'\n\t\t FGSM testing Loss: {fgsm_test_loss:.4f},',f' FGSM testing IoU: {fgsm_miou:.3f},')
        f = open("Colon_Standard_FGSM.txt","a+")
        #f.write("\n\n\t\t\t\t\t FGSM testing")
        f.write("\n\n\t\t Epsilon : %.4f \t\t FGSM testing Loss: : %.4f \t\t FGSM testing IoU: %.3f%%" %(eps,fgsm_test_loss,fgsm_miou))
        f.close()
        mean_iu=get_scores(np.asarray(target.cpu().detach().numpy()), np.asarray((output_adv.argmax(1)).cpu().detach().numpy()), n_classes=2)
        
        
        #with torch.no_grad():
            #valid_loss, valid_loss_min = Validate(model, validloader, criterion, valid_loss_min, device, model_sv_pth)
            
        #fgsm_stats.append([fgsm_test_loss])
        #fgsm_stat = pd.DataFrame(fgsm_stats, columns=['fgsm_test_loss'])
        
        #eps_stats.append([eps, mean_iu])
        #print(eps_stats)
        #eps_stat = pd.DataFrame(eps_stats, columns=['eps','mean_iu'])
        break
        
    #valid_loss, valid_loss_min = Validate(model, validloader, criterion, valid_loss_min, device, model_sv_pth)
    print('Finished FGSM Testing')
    #if plot: plotCurves_eps(eps_stat)
    return eps, mean_iu

    

    
def fgsm_train(model, trainloader, validloader, criterion,eps, optimizer, epochs, device, load_pth, model_sv_pth, plot=True, visualize=False, load_model=False):
    if load_model: model.load_state_dict(torch.load(load_pth))
    model.train()
    stats = []
    valid_loss_min = np.Inf
    print('FGSM Training Started.....')
    for epoch in range(epochs):
        fgsm_train_loss = 0
        fgsm_train_iou = []
        
        iterator = tqdm(trainloader)
        for i, data in enumerate(iterator):
            inputs, mask, rgb = data
            inputs, mask = inputs.to(device), mask.to(device)
            optimizer.zero_grad()
            adv_input= FGSM(model,inputs,mask,device,eps,criterion)
            output_adv = model(adv_input.float())
            target = mask.argmax(1)
            loss = criterion(output_adv, target.long())
            loss.backward()
            optimizer.step()
            fgsm_train_loss += loss.item() * adv_input.size(0) 
            fgsm_iou = iou_pytorch(output_adv.argmax(1), target)
            fgsm_train_iou.extend(fgsm_iou)     
            if visualize and i == 0:
                print('The training images')
                show_databatch_adv_training(epoch,inputs.detach().cpu(), size=(8,8))
                print('The adversarial training images')
                show_databatch_adv_training_adversarial(epoch,adv_input.detach().cpu(), size=(8,8))
                print('The original masks')
                show_databatch_adv_training_original(epoch,rgb.detach().cpu(), size=(8,8))
                RGB_mask =  mask_to_rgb(output_adv.detach().cpu(), id2code)
                print('Predicted masks')
                show_databatch_adv_training_predicted(epoch,torch.tensor(RGB_mask).permute(0,3,1,2), size=(8,8))
        fgsm_miou = torch.FloatTensor(fgsm_train_iou).mean()
        fgsm_train_loss = fgsm_train_loss / len(trainloader.dataset)
        print('Epoch',epoch,':',f'Lr ({optimizer.param_groups[0]["lr"]})',f'\n\t\t Training Loss: {fgsm_train_loss:.4f},',f' Training IoU: {fgsm_miou:.3f},')
        f = open("Colon_Standard_FGSM.txt","a+")
        f.write("\n\n\t\t Epoch: %d \t\t Epsilon : %.4f \t\t FGSM training Loss: : %.4f \t\t FGSM training IoU: %.3f%%" %(epoch,eps,fgsm_train_loss,fgsm_miou))
        f.close()
        get_scores(np.asarray(target.cpu().detach().numpy()), np.asarray((output_adv.argmax(1)).cpu().detach().numpy()), n_classes=2)
        
        with torch.no_grad():
            valid_loss, valid_loss_min = Validate(model, validloader, criterion, valid_loss_min, device, model_sv_pth)
            
        stats.append([fgsm_train_loss, valid_loss])
        stat = pd.DataFrame(stats, columns=['train_loss','valid_loss'])
    
    #valid_loss, valid_loss_min = Validate(model, validloader, criterion, valid_loss_min, device, model_sv_pth)
    print('Finished FGSM Training')
    if plot: plotCurves_adv(stat)
 

def fgsm_test_new(model, trainloader, validloader, criterion, eps, optimizer, device, load_pth, model_sv_pth, plot=True, visualize=False, load_model=False):
    if load_model: model.load_state_dict(torch.load(load_pth))
    model.eval()
    fgsm_stats = []
    #valid_loss_min = np.Inf
    epochs=1
    print('FGSM Testing Started.....')
    
    for epoch in range(epochs):
        fgsm_test_loss = 0
        fgsm_test_iou = []
        
        iterator = tqdm(trainloader)
        for i, data in enumerate(iterator):
            inputs, mask, rgb = data
            inputs, mask = inputs.to(device), mask.to(device)
            #optimizer.zero_grad()
            
            adv_inputs = FGSM(model,inputs,mask,device,eps,criterion)
            output_adv = model(adv_inputs.float())
            #print(adv_inputs.detach().cpu())
            target = mask.argmax(1)
            
            loss = criterion(output_adv, target.long())
            #loss.backward()
            #optimizer.step()
            fgsm_test_loss += loss.item() * adv_inputs.size(0) 
            iou = iou_pytorch(output_adv.argmax(1), target)
            fgsm_test_iou.extend(iou)     
            
            if visualize and  i == 0:
                print('The testing images')
                show_databatch_adv_testing_new(str(round(eps,4)),inputs.detach().cpu(), size=(8,8))
                print('The testing adversarial images')
                show_databatch_adv_testing_adversarial_new(str(round(eps,4)),adv_inputs.detach().cpu(), size=(8,8))#problem here
                print('The original masks')
                show_databatch_adv_testing_original_new(str(round(eps,4)),rgb.detach().cpu(), size=(8,8))
                RGB_mask =  mask_to_rgb(output_adv.detach().cpu(), id2code)
                print('Predicted masks')
                show_databatch_adv_testing_predicted_new(str(round(eps,4)),torch.tensor(RGB_mask).permute(0,3,1,2), size=(8,8))
            break
        fgsm_miou = torch.FloatTensor(fgsm_test_iou).mean()
        fgsm_test_loss = fgsm_test_loss / len(trainloader.dataset)
        print(f'\n\t\t FGSM testing Loss: {fgsm_test_loss:.4f},',f' FGSM testing IoU: {fgsm_miou:.3f},')
        f = open("Colon_Standard_FGSM.txt","a+")
        #f.write("\n\n\t\t\t\t\t FGSM testing")
        f.write("\n\n\t\t Epsilon : %.4f \t\t FGSM testing Loss: : %.4f \t\t FGSM testing IoU: %.3f%%" %(eps,fgsm_test_loss,fgsm_miou))
        f.close()
        get_scores(np.asarray(target.cpu().detach().numpy()), np.asarray((output_adv.argmax(1)).cpu().detach().numpy()), n_classes=2)
        
        break
        #with torch.no_grad():
            #valid_loss, valid_loss_min = Validate(model, validloader, criterion, valid_loss_min, device, model_sv_pth)
            
        #fgsm_stats.append([fgsm_test_loss])
        #fgsm_stat = pd.DataFrame(fgsm_stats, columns=['fgsm_test_loss'])
    
    #valid_loss, valid_loss_min = Validate(model, validloader, criterion, valid_loss_min, device, model_sv_pth)
    print('Finished FGSM Testing')
    #if plot: plotCurves(fgsm_stat)