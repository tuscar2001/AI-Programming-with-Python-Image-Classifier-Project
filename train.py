import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from collections import OrderedDict
import time
from PIL import Image
import numpy as np
import argparse
import json



data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# Create Parse using ArgumentParser
parser = argparse.ArgumentParser()
# Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
parser.add_argument('data_directory', type=str, help='Data Directory')
parser.add_argument('--save_directory', type=str, help='Saving Directory')
parser.add_argument ('--arch', type = str, default = 'densenet121', help = 'Please choose architecture: resnet50, or densenet121. Default:\ densenet121')
parser.add_argument ('--hidden_units', type = int, help = 'Hidden units. Default: 500')
parser.add_argument ('--epochs',type = int, help = 'Number of epochs. Default: 5')
parser.add_argument ('--learning_rate',type = float, help = 'Learning Rate, Default: 0.001')
parser.add_argument ('--GPU', type = str, help = "Device: GPU or CPU. Default: GPU")
args = parser.parse_args ()                    
  
                         
resnet50 = models.resnet50(pretrained=True)
densenet121 = models.densenet121(pretrained = True)

modelsDict = {'resnet': resnet50, 'densenet121':densenet121}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_directory = args.data_directory
if data_directory:
    # data proprocessing
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 
    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    data_transforms = {'train': train_transforms,'test': test_transforms,'valid': valid_transforms}
    
    image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform=train_transforms),
    'test': datasets.ImageFolder(test_dir, transform=test_transforms),
    'valid': datasets.ImageFolder(valid_dir, transform=valid_transforms)}

# TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], 128, shuffle=True),
        'test': torch.utils.data.DataLoader(image_datasets['test'], 32, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], 32, shuffle=True)}
    
    
    
#mapping from category label to category name
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

    
def classifiers (arch, hidden_units):
    if args.arch == 'resnet50':
        model = modelsDict['resnet50']
        for param in model.parameters():
            param.requires_grad = False
        

        if hidden_units: #in case hidden_units were given
            classifier = nn.Sequential  (
                                         nn.Linear (2048, args.hidden_units),
                                         nn.ReLU (),
                                         nn.Dropout (p = 0.2),
                                         nn.Linear (args.hidden_units, 102),
                                         nn.LogSoftmax (dim =1))
                       
        else: #if hidden_units not given
                     
                     
                     
                     
            classifier = nn.Sequential  (
                                         nn.Linear (2048, 500),
                                         nn.ReLU (),
                                         nn.Dropout (p = 0.2),
                                         nn.Linear (500, 102),
                                         nn.LogSoftmax (dim =1))
                            
    else: #setting model based on default Alexnet ModuleList
        args.arch = 'densenet121'
        model = modelsDict['densenet121']
        for param in model.parameters():
            param.requires_grad = False
        if hidden_units: #in case hidden_units were given
            classifier = nn.Sequential  ( 
                                         nn.Linear (1024, args.hidden_units),
                                         nn.ReLU (),
                                         nn.Dropout (p = 0.2),
                                         nn.Linear (args.hidden_units, 102),
                                         nn.LogSoftmax (dim =1))
                         
        else: #if hidden_units not given
            classifier = nn.Sequential  ( 
                                          nn.Linear (1024, 500),
                                          nn.ReLU (),
                                          nn.Dropout (p = 0.2),
                                          nn.Linear (500, 102),
                                          nn.LogSoftmax (dim =1))
                           
    model.classifier = classifier #we can set classifier only once as cluasses self excluding (if/else)
    return model, arch
                     
                     
model, arch = classifiers (args.arch, args.hidden_units)
                     
                     
# Model Training
criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen

optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate if args.learning_rate else 0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device);
                     
epochs = args.epochs if args.epochs else 5
steps = 0
running_loss = 0
for epoch in range(epochs):
    for inputs, labels in dataloaders['train']:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        

        test_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in dataloaders['test']:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)

                test_loss += batch_loss.item()

                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {running_loss/len(dataloaders['train']):.3f}.. "
              f"Test loss: {test_loss/len(dataloaders['test']):.3f}.. "
              f"Test accuracy: {accuracy/len(dataloaders['test']):.3f}")
        running_loss = 0
        model.train()
                     
# Model Validation               
num = 0
denom = 0
with torch.no_grad():
    for inputs, labels in dataloaders['test']:
        inputs, labels = inputs.to(device), labels.to(device)
        out = model(inputs)
        _, predicted = torch.max(out.data, 1)
        denom += labels.size(0)
        num += (predicted == labels).sum().item()
print('Validation Accuracy: %d %%' % (100 * num / denom))
                     
                      
# Saving the checkpoint                 
model.class_to_idx = image_datasets['train'].class_to_idx
# checkpoint = {'model_type': args.arch,
#               'input_size': 2048 if args.arch == 'resnet50' else 1024,
#               'hidden_layers': args.hidden_units,
#               'output_size': 102,
#               'class_to_idx': model.class_to_idx,
#               'state_dict': model.state_dict()}




#torch.save(checkpoint, args.save_directory + '/checkpoint.pth' if args.save_directory else '/checkpoint.pth')
model.to(device)
torch.save({'model_type': args.arch,
            'classifier': model.classifier,
            'input_size': 2048 if args.arch == 'resnet50' else 1024,
            'hidden_layers': args.hidden_units,
            'output_size': 102,
            'class_to_idx': model.class_to_idx,
            'state_dict': model.state_dict()}, args.save_directory + '/checkpoint.pth' if args.save_directory else '/checkpoint.pth')