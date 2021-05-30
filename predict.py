
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from PIL import Image
import json
import argparse

parser = argparse.ArgumentParser(description='predict.py')
   
parser.add_argument('--save_directory' , dest='save_directory' , action ='store' , default = './checkpoint.pth') 
parser.add_argument('--top_k' , dest='top_k' , action ='store' , type = int, default= 5)
parser.add_argument('--gpu' , dest='gpu', action='store', default='gpu')
parser.add_argument('--image_path' , dest='path of the image', action='store', default='image_path')
parser.add_argument('--saved_model' , action='store',type = str ,default='checkpoint.pth')
parser.add_argument('--mapper' , action='store', type = str ,default='cat_to_name.json')
                    
parser = parser.parse_args() 
power = parser.gpu
input = parser.image_path
topk = parser.top_k
saved_model = parser.saved_model

with open (parser.mapper , 'r') as f:
    cat_to_name = json.load(f) 
  

functions.predict(input , saved_model , topk , power)
