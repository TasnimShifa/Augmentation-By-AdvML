import torch
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







def fast_hist(label_true, label_pred, n_class):
        #threshold = 0
        mask = (label_true >= 0) & (label_true < n_class)
        #label_pred[mask] = label_pred[mask] > threshold
        #print(label_pred)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask].astype(int), minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

def update(label_trues, label_preds, n_classes):
        confusion_matrix = np.zeros((n_classes, n_classes))
        #print(label_preds)
        for lt, lp in zip(label_trues, label_preds):
            confusion_matrix += fast_hist(lt.flatten(), lp.flatten(), n_classes)
        return confusion_matrix

def get_scores(label_trues, label_preds, n_classes):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = update(label_trues, label_preds, n_classes)
        #print(hist)
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(n_classes), iu))
        
        f = open("Colon_Standard_FGSM.txt","a+")
        f.write("\n \t\t Overall Acc: %.4f  \t\t Mean Acc: %.4f \t\t Freqw Acc: %.4f \t\t Mean IOU: %.4f %%" %(acc, acc_cls, fwavacc,mean_iu))
        f.close()

        return mean_iu
        
    
def train(model, trainloader, validloader, criterion, optimizer, epochs, device, load_pth, model_sv_pth, plot=True, visualize=False, load_model=False):
    if load_model: model.load_state_dict(torch.load(load_pth))
    model.train()
    stats = []
    valid_loss_min = np.Inf
    print('Training Started.....')
    for epoch in range(epochs):
        train_loss = 0
        train_iou = []
        train_dice = []
        running_loss = 0
        total_train = 0
        correct_train = 0
        acc = 0. # Accuracy
       
        iterator = tqdm(trainloader)
        for i, data in enumerate(iterator):
            inputs, mask, rgb = data
            inputs, mask = inputs.to(device), mask.to(device)
            optimizer.zero_grad()
            output = model(inputs.float())
            target = mask.argmax(1)
            #print(mask.size())
            #print(target.size())
            #print(mask.nelement())

            
            loss = criterion(output, target.long())
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0) 
            
            _, predicted = torch.max(output.data, 1)
            total_train += target.nelement()
            correct_train += predicted.eq(target.data).sum().item()
            train_accuracy = 100 * correct_train / total_train
            #print(train_accuracy)
            
            
            #------------------------------------------------------

            
            #-----------------------------------------------------==
            
            
            iou = iou_pytorch(output.argmax(1), target)
            train_iou.extend(iou)
            dice = dice_pytorch(output.argmax(1), target)
            train_dice.extend(dice)
            if visualize  and i == 0:
                print('The training images')
                show_databatch_training(epoch,inputs.detach().cpu(), size=(8,8))
                print('The original masks')
                show_databatch_original(epoch,rgb.detach().cpu(), size=(8,8))
                RGB_mask =  mask_to_rgb(output.detach().cpu(), id2code)
                print('Predicted masks')
                show_databatch_predicted(epoch,torch.tensor(RGB_mask).permute(0,3,1,2), size=(8,8))
        miou = torch.FloatTensor(train_iou).mean()
        mdice = torch.FloatTensor(train_dice).mean()
        train_loss = train_loss / len(trainloader.dataset)
        print('Epoch',epoch,':',f'Lr ({optimizer.param_groups[0]["lr"]})',f'\n\t\t Training Loss: {train_loss:.4f},',f' Training IoU: {miou:.3f},',f' Training ACC: {train_accuracy:.3f},')
        f = open("Colon_Standard_FGSM.txt","a+")
        f.write("\nEpoch %d :  \n\t\t Training Loss: : %.4f \t\t Training IoU: %.3f \t\t Training ACC: : %.3f%%" %(epoch,train_loss,miou,train_accuracy))
        f.close()
        get_scores(np.asarray(target.cpu().detach().numpy()), np.asarray((output.argmax(1)).cpu().detach().numpy()), n_classes=2)
       # f = open("result.txt","a+"),
        #f.write("Overall Acc: %.4f  \t\t Mean Acc: %.4f \t\t Freqw Acc: %.4f \t\t Mean IOU: %.4f %%" %(acc, acc_cls, fwavacc,mean_iou)),
        
        
        
        
        with torch.no_grad():
            valid_loss, valid_loss_min = Validate(model, validloader, criterion, valid_loss_min, device, model_sv_pth)
            
        stats.append([train_loss, valid_loss])
        stat = pd.DataFrame(stats, columns=['train_loss','valid_loss'])
    
    valid_loss, valid_loss_min = Validate(model, validloader, criterion, valid_loss_min, device, model_sv_pth)
    
    print('Finished Training')
    if plot: plotCurves(stat)

        
def test(model, trainloader, validloader, criterion, optimizer, device, load_pth, model_sv_pth, plot=True, visualize=False, load_model=False):
    if load_model: model.load_state_dict(torch.load(load_pth))
    model.eval()
    stats = []
    #valid_loss_min = np.Inf
    epochs=30
    print('Testing Started.....')
    
    for epoch in range(epochs):
        test_loss = 0
        test_iou = []
        
        iterator = tqdm(validloader)
        for i, data in enumerate(iterator):
            inputs, mask, rgb = data
            inputs, mask = inputs.to(device), mask.to(device)
            #optimizer.zero_grad()
            
            #adv_inputs = FGSM(model,inputs,mask,device,eps,criterion)
            #output_adv = model(adv_inputs.float())
            output = model(inputs.float())
            target = mask.argmax(1)
            #print(adv_inputs.detach().cpu())
            
            
            loss = criterion(output, target.long())
            #loss.backward()
            #optimizer.step()
            test_loss += loss.item() * inputs.size(0) 
            iou = iou_pytorch(output.argmax(1), target)
            test_iou.extend(iou)     
            
            if visualize and  i == 0:
                print('The training images')
                show_databatch_training(epoch,inputs.detach().cpu(), size=(8,8))
                print('The original masks')
                show_databatch_original(epoch,rgb.detach().cpu(), size=(8,8))
                RGB_mask =  mask_to_rgb(output.detach().cpu(), id2code)
                print('Predicted masks')
                show_databatch_predicted(epoch,torch.tensor(RGB_mask).permute(0,3,1,2), size=(8,8))
            break
        miou = torch.FloatTensor(test_iou).mean()
        test_loss = test_loss / len(validloader.dataset)
        print(f'\n\t\t testing Loss: {test_loss:.4f},',f' testing IoU: {miou:.3f},')
        f = open("Colon_Standard_FGSM.txt","a+")
        #f.write("\n\n\t\t\t\t\t FGSM testing")
        f.write("\n\nEpoch : %d \n \t\t testing Loss: : %.4f \t\t testing IoU: %.3f%%" %(epoch ,test_loss,miou))
        f.close()
        get_scores(np.asarray(target.cpu().detach().numpy()), np.asarray((output.argmax(1)).cpu().detach().numpy()), n_classes=2)
        
        #break
        #with torch.no_grad():
            #valid_loss, valid_loss_min = Validate(model, validloader, criterion, valid_loss_min, device, model_sv_pth)
            
        #fgsm_stats.append([fgsm_test_loss])
        #fgsm_stat = pd.DataFrame(fgsm_stats, columns=['fgsm_test_loss'])
    
    #valid_loss, valid_loss_min = Validate(model, validloader, criterion, valid_loss_min, device, model_sv_pth)
    print('Finished Testing')
    #if plot: plotCurves(fgsm_stat)
       