import torch
import numpy as np
import matplotlib.pyplot as plt
import torch
import flwr as fl
import time
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision
from collections import OrderedDict
from torch.autograd import Variable
from PIL import Image
from torch.optim import lr_scheduler
import copy
import json
import os
import shutil
from os.path import exists
import random
from numpy import random
from multiprocessing import Process, freeze_support
from tqdm import tqdm
import glob
import shutil
from torch.utils.data import Subset
from torch.utils.data import Sampler
import sys

from torch.utils.data import SubsetRandomSampler

CUDA_VISIBLE_DEVICES=0,1,2,3

# print(tqdm.__version__)

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

print("hello");

# check if CUDA is available


#startcomment: for avoidimg creating datasets repeatedly
"""
original_dataset_dir = '/home/mihir/Fed/PlantVillage'       # Change this to the actual path of your dataset
custom_dataset_dir = '/home/mihir/Fed/CustomDataset'        # Change this to the desired destination for the custom dataset

os.makedirs(os.path.join(custom_dataset_dir, 'train', 'healthy'), exist_ok=True)
os.makedirs(os.path.join(custom_dataset_dir, 'train', 'diseased'), exist_ok=True)
os.makedirs(os.path.join(custom_dataset_dir, 'val', 'healthy'), exist_ok=True)
os.makedirs(os.path.join(custom_dataset_dir, 'val', 'diseased'), exist_ok=True)

healthy_source_dir_train = os.path.join(original_dataset_dir, 'train', 'Tomato_healthy')
healthy_source_dir_test = os.path.join(original_dataset_dir, 'test', 'Tomato_healthy')

healthy_dest_dir_train = os.path.join(custom_dataset_dir, 'train', 'healthy')
healthy_dest_dir_val = os.path.join(custom_dataset_dir, 'val', 'healthy')

# Copy a fraction of healthy images to the validation set (adjust the ratio as needed)
healthy_images = os.listdir(healthy_source_dir_train)
random.shuffle(healthy_images)
split_index = int(0.2 * len(healthy_images))

for image in healthy_images[:split_index]:
     src = os.path.join(healthy_source_dir_train, image)
     dst = os.path.join(healthy_dest_dir_val, image)
     shutil.copy(src, dst)

for image in healthy_images[split_index:]:
     src = os.path.join(healthy_source_dir_train, image)
     dst = os.path.join(healthy_dest_dir_train, image)
     shutil.copy(src, dst)

# Copy healthy test images to the 'train' and 'val' 'healthy' directories
for image in os.listdir(healthy_source_dir_test):
    src = os.path.join(healthy_source_dir_test, image)
    dst = os.path.join(healthy_dest_dir_train, image)
    shutil.copy(src, dst)

    dst = os.path.join(healthy_dest_dir_val, image)
    shutil.copy(src, dst)


for root, _, files in os.walk(original_dataset_dir):
    if 'Tomato_' in root and root != healthy_source_dir_train and root != healthy_source_dir_test:
        for file in files:
            src = os.path.join(root, file)
            dst_train = os.path.join(custom_dataset_dir, 'train', 'diseased', file)
            dst_val = os.path.join(custom_dataset_dir, 'val', 'diseased', file)

            if random.random() < 0.5:
                shutil.copy(src, dst_val)
            else:
                shutil.copy(src, dst_train)
"""
#endcomment: for avoidimg creating datasets repeatedly



"""
 #for apple dataset
 #startcomment: to avoid repeated creation of dataset

original_dataset_dir = '/home/mihir/Fed/PlantVillage'       # Change this to the actual path of your dataset
custom_dataset_dir = '/home/mihir/Fed/CustomDatasetApple'        # Change this to the desired destination for the custom dataset

os.makedirs(os.path.join(custom_dataset_dir, 'train', 'healthy'), exist_ok=True)
os.makedirs(os.path.join(custom_dataset_dir, 'train', 'diseased'), exist_ok=True)
os.makedirs(os.path.join(custom_dataset_dir, 'val', 'healthy'), exist_ok=True)
os.makedirs(os.path.join(custom_dataset_dir, 'val', 'diseased'), exist_ok=True)

#manage healthy
healthy_source_dir_train = os.path.join(original_dataset_dir, 'train', 'Apple___healthy')
healthy_source_dir_test =  os.path.join(original_dataset_dir, 'test', 'Apple___healthy')

healthy_dest_dir_train = os.path.join(custom_dataset_dir, 'train', 'healthy')
healthy_dest_dir_test = os.path.join(custom_dataset_dir, 'val', 'healthy')


# coppy 'pv/train/apple_healthy' to 'custom/train/healthy'
healthy_images = os.listdir(healthy_source_dir_train)


for image in healthy_images:
     src = os.path.join(healthy_source_dir_train, image)
     dst = os.path.join(healthy_dest_dir_train, image)
     shutil.copy(src, dst)

# coppy 'pv/test/apple_healthy' to 'custom/test/healthy'
healthy_images_test = os.listdir(healthy_source_dir_test)


for image in healthy_images_test:
     src = os.path.join(healthy_source_dir_test, image)
     dst = os.path.join(healthy_dest_dir_test, image)
     shutil.copy(src, dst)

#manage unhealthy

#copy pv/train/**diseased** to custom/train/diseased

for root, _, files in os.walk(os.path.join(original_dataset_dir,'train')):
    if 'Apple_' in root and root != healthy_source_dir_train and root != healthy_source_dir_test:
        for file in files:
            src = os.path.join(root, file)
            dst_train = os.path.join(custom_dataset_dir, 'train', 'diseased', file)
           
            shutil.copy(src, dst_train)


#copy pv/test/**diseased** to custom/test/diseased

for root, _, files in os.walk(os.path.join(original_dataset_dir,'test')):
    if 'Apple_' in root and root != healthy_source_dir_train and root != healthy_source_dir_test:
        for file in files:
            src = os.path.join(root, file)
            dst_test = os.path.join(custom_dataset_dir, 'val', 'diseased', file)
           
            shutil.copy(src, dst_test)






#endcomment: for avoidimg creating datasets repeatedly
"""




def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

# Number of classes in the dataset
num_classes = 10

# Number of epochs to train for
num_epochs = 10

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True


def final_classifier(num_ftrs,num_classes):
  return nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(num_ftrs, 512)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(512, num_classes)),
                          ('output', nn.Softmax(dim=1))   #changed LogSoftmax to softmax
                          ]))

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet152":
        """ Resnet152
        """
        model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        print(num_ftrs)
        model_ft.fc = final_classifier(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "resnet50":
        """ Resnet18
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        print(num_ftrs)
        model_ft.fc = final_classifier(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        print(num_ftrs)
        model_ft.fc = final_classifier(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        print(num_ftrs)
        model_ft.classifier[6] = final_classifier(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        print(num_ftrs)
        model_ft.classifier[6] = final_classifier(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = 512
        print(num_ftrs)
        model_ft.classifier[1] = final_classifier(num_ftrs,num_classes)
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        print(num_ftrs)
        model_ft.classifier = final_classifier(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        print(num_ftrs)
        model_ft.AuxLogits.fc = final_classifier(num_ftrs,num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        print(num_ftrs)
        model_ft.fc = final_classifier(num_ftrs,num_classes)
        input_size = 299

    elif model_name == 'mobilenet':
        model_ft = models.mobilenet_v2(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_features
        print(num_ftrs)
        model_ft.classifier[1] = final_classifier(num_ftrs,num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def load_checkpoint(filepath,model_to_load):
    checkpoint = torch.load(filepath,map_location='cpu')
    model,initial_size = initialize_model(model_to_load,num_classes,feature_extract,use_pretrained=True)

    # model.load_state_dict(checkpoint)
    model.to(device)

    return model, checkpoint


# Load model and get index to class mapping
model_name = "densenet"
model, class_to_idx = load_checkpoint(r'onion_binary_densenet_pv.pt',model_name)
idx_to_class = { v : k for k,v in class_to_idx.items()}

model = model.to(device=device)

print("line 235")

print("\n")



# This line is necessary for Windows


#MJ: Third argument is for choosing the dataset plant. Maake sure you have them in your directory.
plant=sys.argv[3]

if(plant=="Apple"):
   data_dir = r'CustomApple'
   num_classes=4 

if(plant=="Tomato"):
    data_dir = r'CustomTomato'
    num_classes=10
if(plant=="Onion"):
    data_dir = r'tih_onion'
    num_classes=3


train_dir = data_dir + '/train'
valid_dir = data_dir + '/val'
nThreads = 1
batch_size = 16
use_gpu = torch.cuda.is_available()

# Define your transforms for the training and validation sets
# Data augmentation and normalization for training
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

transform_test = transforms.Compose([
    transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
])



def cutmix_criterion(preds, targets):
    targets1, targets2, lam = targets[0], targets[1], targets[2]
    criterion = nn.NLLLoss()
    return lam * criterion(preds, targets1) + (1 - lam) * criterion(preds, targets2)

def cutmix(data, targets, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    new_targets = [targets, shuffled_targets, lam]
    return data, new_targets


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2



print("line 327")
# from torchsampler import ImbalancedDatasetSampler
from torchsampler import ImbalancedDatasetSampler
#Function to train the model
def train_model(model, criterion, optimizer, scheduler, num_epochs=20,is_inception = False):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            # for inputs, labels in tqdm(dataloaders[phase]):
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                # if (phase == 'train'):
                #     p = np.random.rand()
                #     if p < p_cutmix:
                #         inputs, labels = cutmix(inputs, labels, 0.8)


                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    if (is_inception and phase == 'train'):
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else :
                        outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss = criterion(outputs, labels)
                        # if p < p_cutmix:
                        #     loss = cutmix_criterion(outputs, labels)

                        loss.backward()
                        optimizer.step()

                    elif phase == 'val':
                        loss = criterion(outputs,labels)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                if (phase == 'train'):
                    running_corrects += torch.sum(preds == labels.data)
                    # if (p < p_cutmix):
                    #     running_corrects += torch.sum(preds == labels[0].data)

                elif (phase == 'val'):
                    running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best valid accuracy: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# Train a model with a pre-trained network
num_epochs = 10
if use_gpu:
    print ("Using GPU: "+ str(use_gpu))
    model = model.cuda()

# NLLLoss because our output is LogSoftmax
criterion = nn.NLLLoss()

params_to_update = model.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Adam optimizer with a learning rate
optimizer = optim.Adam(params_to_update, lr=0.001)
# Decay LR by a factor of 0.1 every 5 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
print("line 486")

"""
torch.save(model.state_dict(), 'appleDenseNetFull.pth')

print("model saved \n")

while(True):
    continue
"""
np.random.seed(2)
torch.manual_seed(1)
if(sys.argv[2]=="even"):
    torch.manual_seed(100)


# model_ft = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=num_epochs,is_inception=False)

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

# Using the image datasets and the trainforms, define the dataloaders
train_len = len(image_datasets['train'])

random_mask = [random.choice([0, 1]) for _ in range(train_len)]

random_mask=torch.tensor(random_mask)
random_mask=random_mask.nonzero().reshape(-1)

#image_datasets['train']=Subset(image_datasets['train'], random_mask)


subset_size = 16000

# Create a subset of the CIFAR10 dataset

#torch.manual_seed(0)

subset_indices = torch.randperm(len(image_datasets['train']))[:subset_size]
subset_indices=list(range(1, len(image_datasets['train']), 2))
subset_indices=list(range(1, len(image_datasets['train'])))



class EvenIndexSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        if(sys.argv[2]=="even"):
            self.indices = list(range(0, len(data_source),2))
        else:
            self.indices = list(range(1, len(data_source),2))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
        

# even_index_sampler = EvenIndexSampler(image_datasets['train'])



train_subset_1 = Subset(image_datasets['train'], subset_indices)



#trainloader_subset_1=torch.utils.data.DataLoader(train_subset_1, sampler=ImbalancedDatasetSampler(train_subset_1),batch_size=64)


# One can use the Imbalaniced Dataset Smampler to sampleunevenly distributed classes equally during training
dataloaders = {
    x: torch.utils.data.DataLoader(image_datasets[x], sampler=ImbalancedDatasetSampler(image_datasets[x]),
                                   batch_size=batch_size, num_workers=0) for x in ['train', 'val']}
# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],batch_size=batch_size, num_workers=8)for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

class_names = image_datasets['train'].classes


dataset_size = len(image_datasets['train'])
dataset_indices = list(range(dataset_size))

#MJ: This logic is to shuffle and create data subset of 0.5 length of total data. The resultant data loader is trainloader_subset_1 Note that the 0.5 fraction is because there were 2 clients. if there are more, you might want to change this fraction.
np.random.shuffle(dataset_indices)


split_index = int(np.floor(0.5 * dataset_size))

train_idx, train2_idx = dataset_indices[:split_index], dataset_indices[split_index:]

sampler1 = SubsetRandomSampler(train_idx)
sampler2 = SubsetRandomSampler(train2_idx)

labled_subset=[]

if(sys.argv[2]=="even"):
    labled_subset=  [img for i, img in enumerate(image_datasets['train'])if i%2==0]
   
else:
    labled_subset=  [img for i, img in enumerate(image_datasets['train'])if i%2!=0]




# while(True):
#     continue
batch_size=16
if(sys.argv[2]=="even"):
    trainloader_subset_1=  torch.utils.data.DataLoader(dataset=image_datasets['train'],sampler=sampler1, shuffle=False, batch_size=batch_size,num_workers=0)
else:
    trainloader_subset_1=  torch.utils.data.DataLoader(dataset=image_datasets['train'],sampler=sampler2, shuffle=False, batch_size=batch_size,num_workers=0)





class_distribution = {}

for _, labels in trainloader_subset_1:
    for label in labels.numpy():  # Convert labels to a numpy array for easier handling
        if label in class_distribution:
            class_distribution[label] += 1
        else:
            class_distribution[label] = 1

print("Class Distribution:", class_distribution)




standalone_accuracy=0.5
eval_accuracy=0.5



class FlowerClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net2.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net2.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net2.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_mod(net2,epochs=10)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        orig_param=self.get_parameters(config={})
        self.set_parameters(parameters)
        loss, accuracy = test(net2, testloader)

        # if(standalone_accuracy>accuracy):
        #     print("standalone accuracy is greater than federated accuracy \n")
        #     print("federated accuracy:"+str(accuracy)+"\n")
        #     print("standalone accuracy:"+str(standalone_accuracy)+"\n")
        #     self.set_parameters(orig_param)
        #     accuracy=standalone_accuracy


        return loss, len(testloader.dataset), {"accuracy": accuracy}
    
    def set_standalone_accuracy(self,acc):
        self.set_standalone_accuracy=acc
    
    def get_standalone_accuracy(self):
        return self.standalone_accuracy




def train_mod(net, epochs):
    # """Train the model on the training set."""
    # net.train()
    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # for _ in range(epochs):
    #   for images, labels in tqdm(trainloader_subset_1):
    #   #for images, labels in tqdm(dataloaders['train']):
    #     # for images, labels in dataloaders['train']:
    #         optimizer.zero_grad()
    #         criterion(net(images.to(device)), labels.to(device)).backward()
    #         optimizer.step()

        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum = 0.9)
        optimizer.zero_grad()
        step_count = 0
        loss_func = nn.CrossEntropyLoss()
        
        net.to(device)
        dataset_local=[]
        for epoch in range(epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(trainloader_subset_1):
            # for batch_idx, (images, labels) in enumerate(dataloaders['train']):
                images, labels = images.to(device), labels.to(device)
                # if(self.args['augmentation']==True):
                #     images = self.transform_train(images)
                optimizer.zero_grad()
                
                log_probs = net(images)
                loss = loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
                # dataset_local.append(tuple([torch.tensor(feats).squeeze(),labels]))

                # print(f"length of local dataset is {len(dataset_local)}")
                # print(f"shape of local dataset[0][0] is {dataset_local[0][0].shape}")
                # print(f"dataset_local[0][0] is {dataset_local[0][0]}")

            print ("Epoch No. ", epoch, "Loss " , sum(batch_loss)/len(batch_loss))


def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            outputs = net(images.to(device))
            labels = labels.to(device)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy

import sys
import trace

if (__name__ == '__main__'):

    for inputs, labels in trainloader_subset_1:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = transform_test(inputs)
        print(labels)
        inputs, labels = cutmix(inputs, labels, 0.8)

        # plt.figure(figsize=(10, 10))
        # plt.imshow(inputs[0].permute(1, 2, 0).cpu().numpy())
        print(labels)
        break

    model.classifier[1].fc2 = nn.Linear(512, 4)

    print("line 350")
    train_path = '/home/mihir/Fed/CustomDataset/train'
    val_path = '/home/mihir/Fed/CustomDataset/val'

    # len_disease = len(os.listdir(train_path + "/diseased")) + len(os.listdir(val_path + "/diseased"))
    # len_healthy = len(os.listdir(train_path + "/healthy")) + len(os.listdir(val_path + "/healthy"))

    # print("Number of training healthy images = ", len(os.listdir(train_path + "/healthy")))
    # print("Number of training diseased images = ", len(os.listdir(train_path + "/diseased")))
    # print("Number of validation healthy images = ", len(os.listdir(val_path + "/healthy")))
    # print("Number of validation diseased images = ", len(os.listdir(val_path + "/diseased")))
    # plt.bar(['diseased', 'healthy'], [len_disease, len_healthy])
    # plt.show()

    import torchvision.models

    model=torchvision.models.densenet121(pretrained=False)
    print(model)
    model.classifier=nn.Linear(model.classifier.in_features, num_classes)
    # model= nn.DataParallel(model,device_ids = [0,1,2,3])

    model.to(device)

    net1 = model # use for standalone 
    trainloader = dataloaders['train']
    testloader = dataloaders['val']

    p_cutmix = 0.5
    
    
  

    client_pv=FlowerClient()
    # net1.compile()
    
    print(net1)

    # first do standalone training
    print("doing standalone training first \n")
    train_mod(net1, 20)
    loss,standalone_accuracy=test(net1,testloader)
    print("loss="+str(loss)+"/n")
    print("standalone_accuracy="+str(standalone_accuracy)+"/n")

    # torch.onnx.export(net1,torch.randn(1, 3, 224, 224).to(device),"onion_tih_mj.onnx",export_params=True,opset_version=13) 

    net2 = model # use for federated


    
    
    if (sys.argv[1]=="train"):
 
   
    	train_mod(net, 30)
    	loss,accuracy=test(net,testloader)
    	print("loss="+str(loss)+"/n")
    	print("accuracy="+str(accuracy)+"/n")
    
    
  
    else:
      fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=client_pv
    )
      
   
