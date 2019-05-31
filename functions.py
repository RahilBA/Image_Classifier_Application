### Import Packages 
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from PIL import Image
%matplot inline
%config InlineBackend.figure_format = 'retina'


#### function to read data

def read_data(data_dir):
  
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
    
    train_data = datasets.ImageFolder(train_dir , transform = train_transforms)
    valid_data = datasets.Imagefolder(valid_dir , transform = valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)
    
     # TODO:  define the dataloaders
    
    train_loaders = torch.utils.data.DataLoader(datasets['train'], batch_size=64, shuffle=True)
    valid_loaders = torch.utils.data.DataLoader(datasets['valid'], batchsize=64, shuffle=True)
    test_loaders = torch.utils.data.DataLoader(datasets['test'], batchsize=64, shuffle=True)
    
    return train_data , test_data , valid_data , train_loaders , test_loaders , valid_loaders 

#####  
#### function to train and validate the two models
def model_setup (structure ,epochs = 15 ,dropout=0.5, hidden_layer1=500 , hidden_layer2=250 , lr=0.001, power = 'gpu'):
    if structure =='densenet121':
        model = models.densenet121(pretrained = True)
    if structure =='vgg16':
        model = models.vgg16(pretrained = True)
        
    for param in model.parameters():
        param.requires_grad = False
        
    input = {"vgg16" : 25088 , "densenet121" : 1024}
    
    classifier = nn.Sequential(OrderedDict([
                            ('fc1' ,nn.Linear(input[structure],hidden_layer1)),
                            ('relu',nn.ReLU()),
                            ('dropout' , nn.Dropout(dropout)),
                            ('fc2' , nn.Linear(hidden_layer1,hidden_layer2)),
                            ('relu' , nn.ReLU()),
                            ('fc3' , nn.Linear(hidden_layer2, 102)),
                            ('output' , nn.LogSoftmax(dim=1))
                            ]))
         
    model.classifier = classifier 
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
         
    if torch.cuda.is_available() and power == 'gpu':
        model.cuda()
      
    return model, criterion , optimizer

         

#### train the model 
def train (model  , criterion , optimizer, train_loaders , valid_loaders, power = 'gpu' , epochs = 15):        
    
    print_every = 30
    steps = 0 

    for  epoch in range(epochs):
        running_loss = 0

    for inputs , labels in train_loaders:
        steps += 1
            
        if torch.cuda.is_available() and power =='gpu':
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            
        optimizer.zero_grad()
            
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
            
        running_loss +=loss.item()
            
        if steps % print_every == 0:
            model.eval()
            valid_loss = 0 
            accuracy = 0
                
            for ii, (input_valid, label_valid) in enumerate(valid_loaders):
                optimizer.zero_grad()
                    
                if torch.cuda.is_available():
                        input_valid, label_valid = inputs_valid.to('cuda:0') , label_valid.to('cuda:0')
                        model.to ('cuda:0')    
                
                with torch.no_grad():
                    logps = model.forward(input_valid)
                    batch_loss = criterion(logps , label_valid)
                
                    valid_loss += batch_loss.item()
                    
                    #claculate accuracy 
                    ps = torch.exp(logps)
                    top_p , top_class = ps.topk(1, dim=1)
                    
                    
                    equals = top_class == label_valid.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                    
            valid_loss = valid_loss/len(validloaders)
                   
            print("Epoch: {}/{} | ".format(epoch+1 , epochs),
                  "Loss: {:.4f}".format(running_loss/print_every),
                  "Validation Loss {:.4f}".format(valid_loss),
                  "validation accuracy: {:.4f}".format(accuracy/len(valid_loaders)))       
         
            running_loss = 0        
            model.train() 
### function that saves the checkpoint
def save_checkpoint (path='checkpoint.pth', hidden_layer1=500, hidden_layer2= 250,dropout=0.5,lr=0.001,epochs=15):
    model.class_to_idx = train_data.class_to_idx
    model.cpu
    checkpoint = {'class_to_idx' :model.class_to_idx,
                  'hidden_layer1':hidden_layer1,
                  'hidden_layer2':hidden_layer2,
                  'dropout':dropout,
                  'lr':lr,
                  'epochs':epochs,
                  'state_dict':model.state_dict()}
    return torch.save(checkpoint , 'checkpoint.pth')     

###### function that loads the model
def load_checkpoint(path='checkpoint.pth'):
    checkpoint = torch.load(path)
    hidden_layer1 = checkpoint['hidden_layer1']
    hidden_layer2 = checkpoint['hidden_layer2']
    dropout = checkpoint['dropout']
    lr=checkpoint['lr']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return(model)
 
 ##### function that process the image 
def process_image(image):
    image_pil = Image.open(f'{image}' +'.jpg') 
    transformed = transforms.Compose([ 
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_to_tensor_image = transformed (image_pil)
    return image_to_tensor_image 
                      
 #### function that predicts
def predict(image_path , model , topk = 5, power = 'gpu'):
    image = process_image(image_path)
    image = image.unsqueeze_(0)
    image = image.cuda().float()
    model.eval()
        
    if power == 'gpu':
        with torch.no_grad():
            output = model.forward(image.cuda())
    else:
        with torch.no_grad():
            output = model.forward(image)
                    
    probability , index = torch.topk (output , topk)
    # load index
    class_to_idx = model.class_to_idx          
        
    return probability , index              
        
                
            return probability , index     
                      
                          
                     
                      