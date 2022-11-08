import torch
import os
from torch import cuda
from src.utils import Color_map
#from src.utils_I import Color_map
class config(object):

  model_path = "/home/aminul/data1/new_CamVid"
  path = "/home/aminul/data3/tasnim/Data/Colon/"
  
  #path = "/home/aminul/data3/tasnim/Data/Colon_small/"
  load_model = "./Unet/model/state_dict.pt"
  batch = 4
  lr = 0.0001
  epochs = 100
  input_size = (128,128)
  if cuda.is_available(): device = torch.device("cuda:0")
  else: device = torch.device('cpu')
  code2id, id2code, name2id, id2name = Color_map(path+'class_dict.csv')