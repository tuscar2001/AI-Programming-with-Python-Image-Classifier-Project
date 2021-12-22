import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
from torchvision import datasets, transforms, models
from collections import OrderedDict
import time
from PIL import Image
import numpy as np
import argparse
import json
import pandas as pd




# Create Parse using ArgumentParser
parser = argparse.ArgumentParser()
# Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
parser.add_argument ('--top_k', help = 'Top K most likely classes. Default: 5', type = int)
parser.add_argument ('path_to_image', help = 'Path to the image to classify', type = str)
parser.add_argument ('checkpoint', help = 'Path to the model', type = str)
parser.add_argument ('--category_names', help = 'mapping of categories to real names', type = str)
parser.add_argument ('--GPU', type = str, help = "Device: GPU or CPU. Default: GPU")
args = parser.parse_args ()                    


# TODO: Write a function that loads a checkpoint and rebuilds the model
def loading_checkpoint(path_to_model):
    checkpoint = torch.load(path_to_model)
    if checkpoint['model_type'] == 'resnet50':
        model = models.resnet50 (pretrained = True)
    else: 
        model = models.densenet121 (pretrained = True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict (checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

pretrained_model = loading_checkpoint(args.checkpoint)


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

   # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = {'valid': valid_transforms}
    with Image.open(image) as image:  
        image = data_transforms['valid'](image)
        
    return image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 # TODO: Implement the code to predict the class from an image file
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device)
    model.eval()
    image = process_image(image_path).numpy()
    image = torch.from_numpy(np.array([image])).float()
    image = image.to(device)
    with torch.no_grad():
            logps = model.forward(image)
    ps = torch.exp(logps).data     
    top_p, top_class = ps.topk(5, dim=1)
    idx_to_class = {model.class_to_idx[k]: k for k in model.class_to_idx}
    top_p2 = top_p.cpu().data.numpy()
    top_class2 = top_class.cpu().data.numpy()
    top_pred_labels = [idx_to_class[i] for i in top_class2[0]]
    return top_p2[0].tolist(), top_pred_labels


args = parser.parse_args ()
image_path = args.path_to_image
top_k = args.top_k if args.top_k else 5


if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
else:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
        
pretrained_model = loading_checkpoint(args.checkpoint)

prob, classes = predict(image_path, pretrained_model, top_k)

print({"Top Classes":classes,"Top Probabilities": prob})
