import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import torchvision
import seaborn as sns

def imshow(inp, size, title=None):
    '''
        Shows images
        Parameters:
            inp: images
            title: A title for image
    '''
    inp = inp.numpy().transpose((1, 2, 0))
    plt.figure(figsize=size)
    plt.axis('off')
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


#training
def show_databatch_training(epoch,inputs, size=(8,8), batch=16):
    out = torchvision.utils.make_grid(inputs[:batch])
    imshow(out, size)
    inp = out.numpy().transpose((1, 2, 0))
    result_impath = '/home/aminul/data3/tasnim/UNet/kidney_result_inverse/training/'
    plt.imsave(result_impath+str(epoch)+'_training.jpg',inp )    
def show_databatch_original(epoch,inputs, size=(8,8), batch=16):
    out = torchvision.utils.make_grid(inputs[:batch])
    imshow(out, size)
    inp = out.numpy().transpose((1, 2, 0))
    result_impath = '/home/aminul/data3/tasnim/UNet/kidney_result_inverse/training/'
    plt.imsave(result_impath+str(epoch)+'_original.jpg',inp )
def show_databatch_predicted(epoch,inputs, size=(8,8), batch=16):
    out = torchvision.utils.make_grid(inputs[:batch])
    imshow(out, size)
    inp = out.numpy().transpose((1, 2, 0))
    result_impath = '/home/aminul/data3/tasnim/UNet/kidney_result_inverse/training/'
    plt.imsave(result_impath+str(epoch)+'_predicted.jpg',inp )

#adv training
def show_databatch_adv_training(epoch,inputs, size=(8,8), batch=16):
    out = torchvision.utils.make_grid(inputs[:batch])
    imshow(out, size)
    inp = out.numpy().transpose((1, 2, 0))
    result_impath = '/home/aminul/data3/tasnim/UNet/kidney_result_inverse/adv_training/'
    plt.imsave(result_impath+str(epoch)+'_training.jpg',inp ) 
    
def show_databatch_adv_training_original(epoch,inputs, size=(8,8), batch=16):
    out = torchvision.utils.make_grid(inputs[:batch])
    imshow(out, size)
    inp = out.numpy().transpose((1, 2, 0))
    result_impath = '/home/aminul/data3/tasnim/UNet/kidney_result_inverse/adv_training/'
    plt.imsave(result_impath+str(epoch)+'_original.jpg',inp )
    
def show_databatch_adv_training_adversarial(epoch,inputs, size=(8,8), batch=16):
    out = torchvision.utils.make_grid(inputs[:batch])
    imshow(out, size)
    inp = out.numpy().transpose((1, 2, 0))
    result_impath = '/home/aminul/data3/tasnim/UNet/kidney_result_inverse/adv_training/'
    plt.imsave(result_impath+str(epoch)+'_adversarial.jpg',inp )

def show_databatch_adv_training_predicted(epoch,inputs, size=(8,8), batch=16):
    out = torchvision.utils.make_grid(inputs[:batch])
    imshow(out, size)
    inp = out.numpy().transpose((1, 2, 0))
    result_impath = '/home/aminul/data3/tasnim/UNet/kidney_result_inverse/adv_training/'
    plt.imsave(result_impath+str(epoch)+'_predicted.jpg',inp )
    
#adv_testing    
def show_databatch_adv_testing(epsilon,inputs, size=(8,8),  batch=16):
    out = torchvision.utils.make_grid(inputs[:batch])
    imshow(out, size)
    inp = out.numpy().transpose((1, 2, 0))
    result_impath = '/home/aminul/data3/tasnim/UNet/kidney_result_inverse/adv_testing_1/'
    plt.imsave(result_impath+epsilon+'_testing.jpg',inp ) 
    
def show_databatch_adv_testing_original(epsilon,inputs, size=(8,8), batch=16):
    out = torchvision.utils.make_grid(inputs[:batch])
    imshow(out, size)
    inp = out.numpy().transpose((1, 2, 0))
    result_impath = '/home/aminul/data3/tasnim/UNet/kidney_result_inverse/adv_testing_1/'
    plt.imsave(result_impath+epsilon+'_original.jpg',inp )
    
def show_databatch_adv_testing_adversarial(epsilon,inputs, size=(8,8),  batch=16):
    out = torchvision.utils.make_grid(inputs[:batch])
    imshow(out, size)
    inp = out.numpy().transpose((1, 2, 0))
    result_impath = '/home/aminul/data3/tasnim/UNet/kidney_result_inverse/adv_testing_1/'
    plt.imsave(result_impath+epsilon+'_adversarial.jpg',inp)

def show_databatch_adv_testing_predicted(epsilon,inputs, size=(8,8),  batch=16):
    out = torchvision.utils.make_grid(inputs[:batch])
    imshow(out, size)
    inp = out.numpy().transpose((1, 2, 0))
    result_impath = '/home/aminul/data3/tasnim/UNet/kidney_result_inverse/adv_testing_1/'
    plt.imsave(result_impath+epsilon+'_predicted.jpg',inp )  
    
#adv_testing_new   
def show_databatch_adv_testing_new(epsilon,inputs, size=(8,8),  batch=16):
    out = torchvision.utils.make_grid(inputs[:batch])
    imshow(out, size)
    inp = out.numpy().transpose((1, 2, 0))
    result_impath = '/home/aminul/data3/tasnim/UNet/kidney_result_inverse/adv_testing_2/'
    plt.imsave(result_impath+epsilon+'_testing.jpg',inp ) 
    
def show_databatch_adv_testing_original_new(epsilon,inputs, size=(8,8), batch=16):
    out = torchvision.utils.make_grid(inputs[:batch])
    imshow(out, size)
    inp = out.numpy().transpose((1, 2, 0))
    result_impath = '/home/aminul/data3/tasnim/UNet/kidney_result_inverse/adv_testing_2/'
    plt.imsave(result_impath+epsilon+'_original.jpg',inp )
    
def show_databatch_adv_testing_adversarial_new(epsilon,inputs, size=(8,8),  batch=16):
    out = torchvision.utils.make_grid(inputs[:batch])
    imshow(out, size)
    inp = out.numpy().transpose((1, 2, 0))
    result_impath = '/home/aminul/data3/tasnim/UNet/kidney_result_inverse/adv_testing_2/'
    plt.imsave(result_impath+epsilon+'_adversarial.jpg',inp)

def show_databatch_adv_testing_predicted_new(epsilon,inputs, size=(8,8),  batch=16):
    out = torchvision.utils.make_grid(inputs[:batch])
    imshow(out, size)
    inp = out.numpy().transpose((1, 2, 0))
    result_impath = '/home/aminul/data3/tasnim/UNet/kidney_result_inverse/adv_testing_2/'
    plt.imsave(result_impath+epsilon+'_predicted.jpg',inp )    
    
    
def Color_map(dataframe):
  '''
    Returns the reversed String.
    Parameters:
        dataframe: A Dataframe with rgb values with class maps.
    Returns:
        code2id: A dictionary with color as keys and class id as values.   
        id2code: A dictionary with class id as keys and color as values.
        name2id: A dictionary with class name as keys and class id as values.
        id2name: A dictionary with class id as keys and class name as values.
  '''
  cls = pd.read_csv(dataframe)
  color_code = [tuple(cls.drop("name",axis=1).loc[idx]) for idx in range(len(cls.name))]
  code2id = {v: k for k, v in enumerate(list(color_code))}
  id2code = {k: v for k, v in enumerate(list(color_code))}

  color_name = [cls['name'][idx] for idx in range(len(cls.name))]
  name2id = {v: k for k, v in enumerate(list(color_name))}
  id2name = {k: v for k, v in enumerate(list(color_name))}  
  return code2id, id2code, name2id, id2name

def rgb_to_mask(img, color_map):
    ''' 
        Converts a RGB image mask of shape to Binary Mask of shape [batch_size, classes, h, w]
        Parameters:
            img: A RGB img mask
            color_map: Dictionary representing color mappings
        returns:
            out: A Binary Mask of shape [batch_size, classes, h, w]
    '''
    num_classes = len(color_map)
    shape = img.shape[:2]+(num_classes,)
    out = np.zeros(shape, dtype=np.float64)
    for i, cls in enumerate(color_map):
        out[:,:,i] = np.all(np.array(img).reshape( (-1,3) ) == color_map[i], axis=1).reshape(shape[:2])
    return out.transpose(2,0,1)

def mask_to_rgb(mask, color_map):
    ''' 
        Converts a Binary Mask of shape to RGB image mask of shape [batch_size, h, w, 3]
        Parameters:
            img: A Binary mask
            color_map: Dictionary representing color mappings
        returns:
            out: A RGB mask of shape [batch_size, h, w, 3]
    '''
    single_layer = np.argmax(mask, axis=1)
    output = np.zeros((mask.shape[0],mask.shape[2],mask.shape[3],3))
    for k in color_map.keys():
        output[single_layer==k] = color_map[k]
    return np.uint8(output)

def plotCurves(stats):
    sns.set_theme()
    plt.figure(figsize=(8, 6))
    for c in ['train_loss']:
        plt.plot(stats[c], label=c)
    plt.legend()
    plt.xlabel('Epochs', fontsize = 20)
    plt.ylabel('Error', fontsize = 20)
    #plt.rcParams["font.size"] = "50"
    plt.title('Standard Training Loss Curve')
    plt.show()
    plt.savefig('/home/aminul/data3/tasnim/UNet/kidney_result_inverse/Standard_training.jpg' )
    
def plotCurves_adv(stats):
    sns.set_theme()
    plt.figure(figsize=(8, 6))
    for c in ['train_loss']:
        plt.plot(stats[c], label=c)
    plt.legend()
    plt.xlabel('Epochs', fontsize = 20)
    plt.ylabel('Error', fontsize = 20)
    plt.rcParams["font.size"] = "50"
    plt.title('FGSM Training Loss Curve')
    plt.show()
    plt.savefig('/home/aminul/data3/tasnim/UNet/kidney_result_inverse/FGSM_training.jpg' )
    
def plotCurves_I_adv(stats):
    sns.set_theme()
    plt.figure(figsize=(8, 6))
    for c in ['train_loss']:
        plt.plot(stats[c], label=c)
    plt.legend()
    plt.xlabel('Epochs', fontsize = 20)
    plt.ylabel('Error', fontsize = 20)
    plt.rcParams["font.size"] = "50"
    plt.title('Inverse FGSM Training Loss Curve')
    plt.show()
    plt.savefig('/home/aminul/data3/tasnim/UNet/kidney_result_inverse/I_FGSM_training.jpg' )
    
def plotCurves_eps(eps_stats,eps_values ):
    sns.set_theme()
    sns.set_theme()
    plt.figure(figsize=(8, 6))
    #for c in ['mean_iu']:
    plt.plot(eps_values,eps_stats)
    #plt.legend()
    plt.xlabel('Epsilon', fontsize = 20)
    plt.ylabel('Mean_IoU', fontsize = 20)
    plt.rcParams["font.size"] = "50"
    #plt.title('Effect of attack on model with incresing perturbation')
    plt.show()
    plt.savefig('/home/aminul/data3/tasnim/UNet/kidney_result_inverse/eps_InvFGSM.jpg', dpi = 300 )    
def plotCurves_eps_test(eps_stats,eps_values ):
    sns.set_theme()
    sns.set_theme()
    plt.figure(figsize=(8, 6))
    #for c in ['mean_iu']:
    plt.plot(eps_values,eps_stats)
    #plt.legend()
    plt.xlabel('Epsilon', fontsize = 20)
    plt.ylabel('Mean_IoU', fontsize = 20)
    plt.rcParams["font.size"] = "50"
    #plt.title('Effect of attack on model with increasing perturbation after adversarial training')
    plt.show()
    plt.savefig('/home/aminul/data3/tasnim/UNet/kidney_result_inverse/eps_InvFGSM_test.jpg',dpi= 300 )     
    

def Visualize(imgs, title='Original', cols=6, rows=1, plot_size=(16, 16), change_dim=False):
    fig=plt.figure(figsize=plot_size)
    columns = cols
    rows = rows
    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        plt.title(title+str(i))
        if change_dim: plt.imshow(imgs.transpose(0,2,3,1)[i])
        else: plt.imshow(imgs[i])
    plt.show()