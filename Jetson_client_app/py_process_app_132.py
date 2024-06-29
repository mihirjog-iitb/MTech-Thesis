import subprocess
import torch
import threading
CUDA_VISIBLE_DEVICES=0
device='cuda:0'
device='cpu'
import socket
import os
from time import sleep
import numpy as np

import sys
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support
import torch.optim as optim

BUFFER_SIZE=17
inf_flag=False


PRECISION = np.float32
BATCH_SIZE=32
from onnx_helper import ONNXClassifierWrapper


SERVER_HOST = '10.107.47.175'
SERVER_PORT = 5002
# receive 4096 bytes each time
BUFFER_SIZE = 4096
SEPARATOR = "<SEPARATOR>"
state=0
setup_done=False
is_fed=False
n_c=4

update_classifier_flag=False

data_lable_flag=False
train_flag=False
pre_test=False

image_datasets={}
unlabled_subset=[]
labled_subset=[]
unlabled_subset_lables=[]
labled_subset_lables=[]

x1=[]
x2=[]

backbone="densenet"

state_b_update=False

if(backbone=="densenet"):
    op_shape=1024
if(backbone=="mobnet_truncated"):
    op_shape=864



def receive_file(client_socket,BUFFER_SIZE,SEPARATOR):
        # receive the file infos
    # receive using client socket, not server socket
    received = client_socket.recv(BUFFER_SIZE).decode()
    filename, filesize = received.split(SEPARATOR)
    # remove absolute path if there is
    filename = 'rx_3.pkl'
    # convert to integer
    filesize = int(filesize)
    # start receiving the file from the socket
    # and writing to the file stream
    # progress = tqdm.tqdm(range(filesize), f"Receiving {filename}", unit="B", unit_scale=True, unit_divisor=1024)
    with open(filename, "wb") as f:
        while True:
            # read 1024 bytes from the socket (receive)
            bytes_read = client_socket.recv(BUFFER_SIZE)
            if not bytes_read:    
                # nothing is received
                # file transmitting is done
                printf(f"received file{filename}")
                break
            # write to the file the bytes we just received
            f.write(bytes_read)
            # update the progress bar
            # progress.update(len(bytes_read))


#pytorch neural network.

# simple neural network

import torch
import torch.nn as nn
import torch.nn.functional as F
hidden_layer=16

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.fc1 = nn.Linear(1024, hidden_layer)  # 5*5 from image dimension
        self.fc2 = nn.Linear(hidden_layer, 4)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, input):
        # Convolution layer C1: 1 input image channel, 6 output channels,
        # 5x5 square convolution, it uses RELU activation function, and
        # outputs a Tensor with size (N, 6, 28, 28), where N is the size of the batch
        # c1 = F.relu(self.conv1(input))
        # # Subsampling layer S2: 2x2 grid, purely functional,
        # # this layer does not have any parameter, and outputs a (N, 16, 14, 14) Tensor
        # s2 = F.max_pool2d(c1, (2, 2))
        # # Convolution layer C3: 6 input channels, 16 output channels,
        # # 5x5 square convolution, it uses RELU activation function, and
        # # outputs a (N, 16, 10, 10) Tensor
        # c3 = F.relu(self.conv2(s2))
        # # Subsampling layer S4: 2x2 grid, purely functional,
        # # this layer does not have any parameter, and outputs a (N, 16, 5, 5) Tensor
        # s4 = F.max_pool2d(c3, 2)
        # # Flatten operation: purely functional, outputs a (N, 400) Tensor
        # s4 = torch.flatten(s4, 1)
        # # Fully connected layer F5: (N, 400) Tensor input,
        # and outputs a (N, 120) Tensor, it uses RELU activation function
        f5 = F.relu(self.fc1(input))
        # Fully connected layer F6: (N, 120) Tensor input,
        # and outputs a (N, 84) Tensor, it uses RELU activation function
        # f6 = F.relu(self.fc2(f5))
        # Gaussian layer OUTPUT: (N, 84) Tensor input, and
        # outputs a (N, 10) Tensor
        output = self.fc2(f5)
        return output




def train_pt_model(net,dl,eps):
        
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum = 0.9)
        step_count = 0
        loss_func = nn.CrossEntropyLoss()
        for epoch in range(eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(dl):
                images, labels = images.to(device), labels.to(device)
                # if(self.args['augmentation']==True):
                #     images = self.transform_train(images)
                optimizer.zero_grad()
                log_probs = net(images)
                loss = loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

            print ("Epoch No. ", epoch, "Loss " , sum(batch_loss)/len(batch_loss))

def test_pt_model(net,dl):
    net.eval()
    test_loss = 0
    correct = 0
    
    
    l = len(dl)
    for idx, (data, target) in enumerate(dl):
        data, target = data.to(device), target.to(device)
        log_probs = net(data)
        test_loss += F.cross_entropy(log_probs, target, reduction = 'sum').item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        probabs= F.softmax(log_probs,dim=1)
        
    test_loss /= len(dl.dataset)
    accuracy = 100.00 * correct / len(dl.dataset)
    return accuracy.numpy(), test_loss, y_pred,probabs




def trt_inference(a1):
    global op_shape
    model_densenet_pv = ONNXClassifierWrapper("engine123.trt", [1, op_shape], target_dtype = PRECISION)
    pred_list=[]
    np.random.seed(8)
    prediction = model_densenet_pv.predict(a1)
    return prediction

# def data_prep():

def get_densenet_op(ds):
   
    import time
    start_time=time.time()
    apple_features=[]
    i=0
    for img in ds:
        output = trt_inference(img[0].unsqueeze(0).numpy())
        output=np.squeeze(output)
        apple_features.append(np.ndarray.flatten(output))
        i=i+1
      # if(i==4):
      #   break;
    end_time=time.time()

    print(end_time-start_time)
   
    return np.array(apple_features)
    

def predict_model_op_probab(ds,clf):       #this is like test() function and it returns probabilities
    dense_op=get_densenet_op(ds)
    # y_true= true_lables
    y_pred= clf.predict(dense_op)
    print("precision,recall,accuracy,f1 score:")
    # precision_recall_fscore_support(y_true, y_pred, average='macro')
    return clf.predict_proba(dense_op)

# def predict_model_op(ds,clf):
    
# 	true_lables= [lbl for (img,lbl) in ds]
# 	dense_op=get_densenet_op(ds)
# 	y_true= true_lables
# 	y_pred= clf.predict(dense_op)
# 	print("precision,recall,accuracy,f1 score:")
# 	print(precision_recall_fscore_support(y_true, y_pred, average='macro'))
# 	return y_pred
# 	# return clf.predict_proba(dense_op)



def predict_model_op(ds,clf):
    
    true_lables= [lbl for (img,lbl) in ds]
    dense_op=get_densenet_op(ds)
    y_true= true_lables
    test_lables= [lbls for (_,lbls) in image_datasets['val'] ]
    test_feature_dataset=[(torch.tensor(img),lbl) for img,lbl in zip(dense_op,true_lables)]
    test_feature_dataset_dataloader =torch.utils.data.DataLoader(dataset=test_feature_dataset, shuffle=False, batch_size=32)

    acc,loss,y_pred=test_pt_model(net,dl)
    print(f"accuracy is :{acc},loss is{loss}")
    return y_pred
    # return clf.predict_proba(dense_op)

import pickle
import math

def train_model_classifier(ds,true_lables,clf):
    dense_op=get_densenet_op(ds)
    clf.fit(dense_op, true_lables)
    

    with open("clf_params.pkl", "wb") as f:
        pickle.dump(clf.params(), f)
    # print("precision,recall,accuracy,f1 score:")
    # precision_recall_fscore_support(y_true, y_pred, average='macro')
    # return clf.predict_proba(dense_op)





def selfLearn(labled_subset,labled_subset_lables,unlabled_subset,image_datasets,net):
    
#   labled_subset_dataloader= torch.utils.data.DataLoader(dataset=labled_subset, shuffle=False, batch_size=32,num_workers=2)
#   dl=labled_subset_dataloader

  print(f"started self learn round")

#   testloader=dataloaders['val']
#   train_model_classifier(labled_subset,labled_subset_lables,clf)
#   predict_model_op(image_datasets['val'],clf)

  #for pytorch classifier

  x=get_densenet_op(labled_subset)

  print(f"now training pytorch classifier")
  feature_dataset=[(torch.tensor(img),lbl) for img,lbl in zip(x,labled_subset_lables)]
  feature_dataset_dataloader =torch.utils.data.DataLoader(dataset=feature_dataset, shuffle=False, batch_size=32)
  train_pt_model(net,feature_dataset_dataloader,20)



  # unlabled_test_dataloader=torch.utils.data.DataLoader(dataset=unlabled_subset, shuffle=False, batch_size=32)
  # rx_dataset=unlabled_subset

  """   commented by MJ 11/04/24

  unlabled_test_dataloader=torch.utils.data.DataLoader(dataset=rx_dataset, shuffle=False, batch_size=32)


  unlabled_loss,unlabled_accuracy,logits=test_logit_output(model_densenet_pv,unlabled_test_dataloader)
  # print(f"logits is :{logits}")

  logits=np.concatenate(logits,axis=0)
  logits.shape ,logits

  # sm= np.exp(logits)/np.sum(np.exp(logits),axis=-1,keepdims=True)
  sm=logits
  print(f"shape of sm is{sm.shape}")
  sm
  pseudo_lables=np.argmax(sm,axis=1)

  """
  x=get_densenet_op(unlabled_subset)
  test_lables= [lbls for (_,lbls) in unlabled_subset ]
  test_feature_dataset=[(torch.tensor(img),lbl) for img,lbl in zip(x,test_lables)]
  test_feature_dataset_dataloader =torch.utils.data.DataLoader(dataset=test_feature_dataset, shuffle=False, batch_size=32)
  acc,loss,pseudo_lables,prb=test_pt_model(net,test_feature_dataset_dataloader)



  print(f"generating psuedo lables")
#   pseudo_lables=predict_model_op(unlabled_subset,clf)

  # selection criterias:
  #1

  # selected_for_training_indices= np.where(abs(np.diff(sm,axis=1))>0.4)

  #2

  # selected_for_training_indices= np.any(sm>0.73, axis=1)
  # selected_for_training_indices= np.where(selected_for_training_indices)[0]

  #3
  # diff_array= abs(np.diff(sm,axis=1))
  # diff_array= np.ndarray.flatten(diff_array)
  # print(f"diff array shape is: {diff_array.shape}")
  # selected_for_training_indices = np.argsort(diff_array)[-100:]


  #4

  entropy_list=[]
  print(f"calculating probabilities")
#   prb=predict_model_op_probab(unlabled_subset,clf)
  for arr in prb:
          entropy = -sum(p * math.log2(p) for p in arr if p != 0)
          entropy_list.append(entropy)        
  entropy_list=np.array(entropy_list)

  lowest_entropy_indices=np.argsort(entropy_list)[:10]
  selected_for_training_indices=lowest_entropy_indices

  new_train_set=[unlabled_subset[i] for i in list(selected_for_training_indices)]
  
  selected_psuedo_lables= [pseudo_lables[i] for i in selected_for_training_indices]
  selected_psuedo_lables=[lbl.item() for lbl in selected_psuedo_lables]
  print(f"selected_psuedo_lables:{selected_psuedo_lables}")
  # print(f"new train set is {new_train_set}")

  # print(f"new train set length is {len(new_train_set)}, and shape is {new_train_set[0][0].shape}")  #, selected_psuedo_lables


  print(f"extending labled subset")
  for i,(first,second) in enumerate(new_train_set):
      new_train_set[i]=(first,selected_psuedo_lables[i])
  new_train_set[1]
  labled_subset.extend(new_train_set)
  labled_subset_lables.extend(selected_psuedo_lables)
#   print(f"labled_subset shape:{len(labled_subset)}")
  print(f"labled_subset_lables shape:{len(labled_subset_lables)}")
  print(labled_subset_lables)

  print(f"extended the original array by length {len(new_train_set)}")

#   labled_subset_dataloader= torch.utils.data.DataLoader(dataset=labled_subset, shuffle=False, batch_size=32)
#   dl=labled_subset_dataloader

#   train_model_classifier(labled_subset,labled_subset_lables,clf)
  print(f"now training pytorch classifier after extending labled set")
  x=get_densenet_op(labled_subset)
  feature_dataset=[(torch.tensor(img),lbl) for img,lbl in zip(x,labled_subset_lables)]
#   print(f"feature_dataset[0][0].shape:{feature_dataset[0][0].shape}")
#   print(f"feature_dataset[0][1]:{feature_dataset[0][1]}")
  feature_dataset_dataloader =torch.utils.data.DataLoader(dataset=feature_dataset, shuffle=False, batch_size=32)
  eps=20
  train_pt_model(net,feature_dataset_dataloader,20)
  
  

#   predict_model_op(image_datasets['val'],clf)

  print(f"doing prediction after extending model")
  
  x=get_densenet_op(image_datasets['val'])
  test_lables= [lbls for (_,lbls) in image_datasets['val'] ]
  test_feature_dataset=[(torch.tensor(img),lbl) for img,lbl in zip(x,test_lables)]
  test_feature_dataset_dataloader =torch.utils.data.DataLoader(dataset=test_feature_dataset, shuffle=False, batch_size=32)
  acc,loss,y_preds,probabs=test_pt_model(net,test_feature_dataset_dataloader)
  print(f"accuracy is:{acc} loss is :{loss}")
#    print(f"probabs are{probabs}")




  #delete the newly included training elements from unlabled dataset element list
  selected_for_training_indices= sorted(selected_for_training_indices, reverse=True)

  for i in selected_for_training_indices:
      del unlabled_subset[i]
  # net









def conv_thread_fun():
    global state
    global state_b_update

    while(1):
        if(state_b_update):
            # state_mutex.acquire()
            sleep(5)
            model_mutex.acquire()
            print("conv_thread:started the converter thread\n")
            # model=torch.load('modified_model.pt', map_location=device)
            # print("conv_thread:model loaded\n");
            # torch.onnx.export(model,                    
            # 			torch.randn(1, 3, 224, 224).to(device),            
            # 			"model123.onnx",        
            # 			export_params=True,     
            # 			opset_version=13) 
            print("conv_thread:onnx export complete\n")
            command = "/usr/src/tensorrt/bin/trtexec --onnx=model123.onnx --saveEngine=engine123.trt"
            output = subprocess.run(command, shell=True, check=True)
            print("Command Output:\n")
            print(output.stdout)
            # state_mutex.release()
            model_mutex.release()
            state_b_update=False
        sleep(5)

def other_thread_fun():
    print("other_thread:started thread 2, waiting for conv_thread to finish\n")
    
    print("other_thread:conv thread finished\n")


def msg_thread(s):
    global state
    global state_b_update
    BUFFER_SIZE = 4096
    msg="rqst_file" 
    global inf_flag
    global train_flag
    global data_lable_flag
    
    param_msg= "params"
    while (True):
        print(f"msg socket running")
        try:
            msg= s.recv(BUFFER_SIZE).decode()
            if True:
                print(f"received msg{msg}")
                recpt="sending_b"
                if(msg==recpt):
                    print(f"now receiving Backbone model")
                    state_b_update=True

                recpt="sending_c"
                if(msg==recpt):
                    print(f"now receiving classifier")
                    state=7

                recpt="run_inf"
                if(msg==recpt):
                    print(f"now running inference")
                    state=2

                if(True):  		#if only we are in federated state
                    BUFFER_SIZE=4096
                    # s = socket.socket()
                    # s.connect((host, port))
                    # s.setblocking(False)
                    filename="net1.pt"
                    # msg=s.recv(BUFFER_SIZE).decode()
                    if(msg=="fed_rqst"):
                        print(f"model requested for federated learning")
                        # send_file(s,filename)
                        send_file_scp()
                        print("file sending is done")
                        msg1="send_done"
                        s.send(msg1.encode())


                    if(msg=="sending_global"):
                        if(s.recv(BUFFER_SIZE).decode()=="model_sent"):
                            print("received global model")
                        # net.load_state

                    if(msg=="infer_train_test"):
                        inf_flag=True
                        train_flag=True
                        pre_test=True
                        # inference_thread()

                    if(msg=="infer_only_test"):
                        train_flag=False
                        inf_flag=True
                        # inference_thread()
                    if(msg="data_label"):
                        data_lable_flag=True
        except Exception as e:
                print(f"exception occured while reading the socket")

                print("Exception:", e)
                print("Exception type:", type(e))
                pass
        sleep(2)

    
        




def inference_thread():
    global is_fed
    global setup_done
    global n_c
    global state
    global image_datasets
    global unlabled_subset
    global labled_subset
    global unlabled_subset_lables
    global labled_subset_lables
   
    global x1
    global x2
    global pre_test
    import torch
    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
            print('CUDA is not available.  Training on CPU ...')
    else:
            print('CUDA is available!  Training on GPU ...')
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(device)

    print("packages imported and device set")

    
    print("hello vip_1")
    import torch
    import numpy as np
    # import matplotlib.pyplot as plt
    import torch
    # import flwr as fl
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
    # from tqdm import tqdm
    import glob
    import shutil
    from torch.utils.data import Subset
    from torch.utils.data import Sampler
    import sys
    import pickle
    
    from torch.utils.data import SubsetRandomSampler
    from torch.utils.data import DataLoader, Dataset
    
    from torch.nn.utils import parameters_to_vector, vector_to_parameters
        

        # print(tqdm.__version__)

    if(setup_done==False):

        

        #Organizing the dataset
        data_dir = r'CustomApple'
        data_dir = r'tih_onion'
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

        print("now creating dataset from folder")
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train','val']}
        print("created dataset from folder")
        # dataloaders = {
        # x: torch.utils.data.DataLoader(image_datasets[x], 
        #                                batch_size=batch_size, num_workers=0) for x in ['train', 'val']}


        


        print("now creating labeled and unlabled datasets")
        unlabled_subset=[img for i, img in enumerate(image_datasets['train'])if i%49==0 ]
        labled_subset=  [img for i, img in enumerate(image_datasets['train'])if i%50==0 ]
        unlabled_subset_lables=  [img[1] for img in unlabled_subset]
        labled_subset_lables  =  [img[1] for img in   labled_subset]

        unlabled_images=[img.numpy() for (img,_) in unlabled_subset]
        labled_images=  [img.numpy() for (img,_) in labled_subset]
        
    print("created labeled and unlabled datasets")
    # unlabled_images=np.array(unlabled_images)
    
    # model_densenet_pv=torch.load('/home/mihir/Downloads/densenet121_pv_10.pt')
    # a1=unlabled_images[0:32,:,:,:]
    # print(f"a1 shape is {a1.shape}")
    print("now running inference of ")
    # a1=unlabled_images[0].reshape((1,3,224,224))
    # pred=trt_inference(a1)

    # print(f"pred shape is {pred.shape}")
    # print(pred)
    # print(f"unlabled_images shape is {unlabled_images.shape}")
    # print(f"unlabled_images[0] shape is {unlabled_images[0].shape}")

    

    #group and save npz files for unlabled data. This can be done once and then commented.

    # num_batches = len(labled_images) // 32 + (1 if len(labled_images) % 32 != 0 else 0)


    # for i in range(num_batches):
    # 	start_idx = i * 32
    # 	end_idx = min((i + 1) * 32, len(labled_images))
    # 	batch_arrays = labled_images[start_idx:end_idx]
    # 	batch_name = f'labled_data/batch_{i}.npz'
    # 	np.savez(batch_name, *batch_arrays)
    # 	print(f'Saved {len(batch_arrays)} arrays to {batch_name}')
    # model_mutex.acquire()
    if(setup_done==False):
        x1=get_densenet_op(labled_subset)
    print(f"shape of x is {x1.shape}")

    print(f"faeture extraction done")
    #train the classifier

    #train pytorch classifier
    net = Net()
    print(net)
    #take the net from .pt file
    net=torch.load('net.pt')
    # net.compile()
    print(net)
    setPowMode("HIGH")
    #pre-testing for federated learning
    if(pre_test==True):
        test_lables= [lbls for (_,lbls) in image_datasets['val'] ]
        test_feature_dataset=[(torch.tensor(img),lbl) for img,lbl in zip(x2,test_lables)]
        test_feature_dataset_dataloader =torch.utils.data.DataLoader(dataset=test_feature_dataset, shuffle=False, batch_size=1)
        acc,loss,y_preds,probabs=test_pt_model(net,test_feature_dataset_dataloader)
        print(f"pre test accuracy is:{acc} loss is :{loss}")
    
    

   
    print(f"now training pytorch classifier")
    feature_dataset=[(torch.tensor(img),lbl) for img,lbl in zip(x1,labled_subset_lables)]
    feature_dataset_dataloader =torch.utils.data.DataLoader(dataset=feature_dataset, shuffle=False, batch_size=1)

    #save the dataset as pkl file
    print(f"saving the dataset as pkl file")
    with open('feat_dataset.pkl', 'wb') as f:
        pickle.dump(feature_dataset, f)

    if(is_fed):
        eps=20
    else:
        eps=20
    if(train_flag or is_fed):
        train_pt_model(net,feature_dataset_dataloader,eps)
    torch.save(net,'net.pt')
    """
    #extract fisher info
    if is_fed:
        print(f"now calculating fisher params")
        # F_kfac = FIM(model=net,
        #                   loader=feature_dataset_dataloader,
        #                   representation=PMatKFAC,
        #                   device='cuda',
        #                   n_output=n_c)
        
        F_diag = FIM(model=net,
                          loader=feature_dataset_dataloader,
                          representation=PMatDiag,
                          device='cuda',
                          n_output=n_c)

        F_diag = F_diag.get_diag()

        print(f"extracted fisher info")
    """

    #end train pytorch classifier

    #begin test pytorch classifier
    if(setup_done==False):
        x2=get_densenet_op(image_datasets['val'])
    setup_done=True
    test_lables= [lbls for (_,lbls) in image_datasets['val'] ]
    test_feature_dataset=[(torch.tensor(img),lbl) for img,lbl in zip(x2,test_lables)]
    test_feature_dataset_dataloader =torch.utils.data.DataLoader(dataset=test_feature_dataset, shuffle=False, batch_size=1)
    acc,loss,y_preds,probabs=test_pt_model(net,test_feature_dataset_dataloader)
    print(f"accuracy is:{acc} loss is :{loss}")
    setPowMode("LOW")
    if is_fed:
        return
    print(f"probabs are{probabs}")
    #end test pyorch classifier
    print(f"shape of first layer weights is{net.fc1.weight.shape}")
    print(f"shape of first layer bias is{net.fc1.bias.shape}")
    print(f"shape of second layer weights is{net.fc1.weight.shape}")
    print(f"shape of second layer bias is{net.fc1.bias.shape}")
    # model_mutex.release()
    import pickle

    with open("net.pkl", "wb") as f:
        pickle.dump(net.state_dict(), f)

    """
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(256, 4), random_state=1,max_iter=1000);
    print(f"now training the classifier")
    clf.fit(x, labled_subset_lables)
    print(f"classifier fit done")
    #test the model on unlabled data and see the performance metrices.

    apple_features_unlabled=get_densenet_op(unlabled_subset)
    print(f"unlabled features extracted")
    y_true= unlabled_subset_lables
    y_pred= clf.predict(apple_features_unlabled)

    
    print(precision_recall_fscore_support(y_true, y_pred, average='macro'))

    #doing predictions for validation dataset

    print(f"now doig prediction for validation dataset.")

    """

    # predict_model_op(image_datasets['val'],clf)

    
    # selfLearn(labled_subset,labled_subset_lables,unlabled_subset,image_datasets,clf)
   
    # selfLearn(labled_subset,labled_subset_lables,unlabled_subset,image_datasets,net)		#for pytorch classifier





    # print(f"thread5 ended")


def info_thread():
    global state
    while(1):
        print(f"currently in state:{state}")
        sleep(5)


import socket


def send_file(s,filename):
    # get the file size
    filesize = os.path.getsize(filename)
    # create the client socket
    # s = socket.socket()
    print(f"[+] Connecting to {host}:{port}")
    # s.connect((host, port))
    print("[+] Connected.")

    # send the filename and filesize
    s.send(f"{filename}{SEPARATOR}{filesize}".encode('utf-8').strip())

    # start sending the file
    # progress = tqdm.tqdm(range(filesize), f"Sending {filename}", unit="B", unit_scale=True, unit_divisor=1024)
    with open(filename, "rb") as f:
        while True:
            # read the bytes from the file
            bytes_read = f.read(BUFFER_SIZE)
            if not bytes_read:
                print("file transmitting is done")
                break
            # we use sendall to assure transimission in 
            # busy networks
            s.sendall(bytes_read)
            # update the progress bar
            # progress.update(len(bytes_read))
            sleep(0.01)
    # close the socket
    # s.close()


def receive_file(client_socket,filename):
    received = client_socket.recv(BUFFER_SIZE).decode()
    filename, filesize = received.split(SEPARATOR)
    # remove absolute path if there is
    filename = 'net.pkl'
    # convert to integer
    filesize = int(filesize)
    # start receiving the file from the socket
    # and writing to the file stream
    progress = tqdm.tqdm(range(filesize), f"Receiving {filename}", unit="B", unit_scale=True, unit_divisor=1024)
    with open(filename, "wb") as f:
        while True:
            # read 1024 bytes from the socket (receive)
            bytes_read = client_socket.recv(BUFFER_SIZE)
            if not bytes_read:    
                # nothing is received
                print("file transmitting is done")
                break
            # write to the file the bytes we just received
            f.write(bytes_read)
            # update the progress bar
            progress.update(len(bytes_read))

    # close the client socket
    client_socket.close()
    # close the server socket
    s.close()

def send_file_scp():
    fp="talk.txt"
    with open(fp,'w') as fd:
        rqst='r'
        fd.write(rqst)

def federated_thread(s):
    global train_flag
    global pre_test
    print("now in federated state:")
    while(True):		#run the thread always
        if(state==7):  		#if only we are in federated state
            BUFFER_SIZE=4096
            # s = socket.socket()
            # s.connect((host, port))
            # s.setblocking(False)
            filename="net1.pt"
            msg=s.recv(BUFFER_SIZE).decode()
            if(msg=="fed_rqst"):
                print(f"model requested for federated learning")
                # send_file(s,filename)
                send_file_scp()
                print("file sending is done")
                msg1="send_done"
                s.send(msg1.encode())


            if(msg=="sending_global"):
                if(s.recv(BUFFER_SIZE).decode()=="model_sent"):
                    print("received global model")
                # net.load_state

            if(msg=="infer_train_test"):
                train_flag=True
                pre_test=True
                inference_thread()

            if(msg=="infer_only_test"):
                train_flag=False
                inference_thread()
            

        sleep(1)
    # print("[+] Connected.")
    # continue


def train_process(labled_subset,labled_subset_lables):
    x=get_densenet_op(labled_subset)
    print(f"now training pytorch classifier")
    feature_dataset=[(torch.tensor(img),lbl) for img,lbl in zip(x,labled_subset_lables)]
    #save the dataset as pkl file
    print(f"saving the dataset as pkl file")
    with open('feat_dataset.pkl', 'wb') as f:
        pickle.dump(feature_dataset, f)

    feature_dataset_dataloader =torch.utils.data.DataLoader(dataset=feature_dataset, shuffle=False, batch_size=32)
    train_pt_model(net,feature_dataset_dataloader,20)
    torch.save(net, 'net.pt')

def test_process(ds):
    x=get_densenet_op(ds)
    test_lables= [lbls for (_,lbls) in ds]
    test_feature_dataset=[(torch.tensor(img),lbl) for img,lbl in zip(x,test_lables)]
    test_feature_dataset_dataloader =torch.utils.data.DataLoader(dataset=test_feature_dataset, shuffle=False, batch_size=32)
    acc,loss,y_preds,probabs=test_pt_model(net,test_feature_dataset_dataloader)
    print(f"accuracy is:{acc}")


def setPowMode(mode):
    if mode=="LOW":
        char='L'
    if mode =="HIGH":
        char='H'
    fp="powtalk.txt"
    with open(fp,'w') as fd:
        
        fd.write(char)


model_mutex = threading.Lock()

if __name__ =="__main__":
    host="10.107.47.157"
    port=5002
    BUFFER_SIZE=4096
    setPowMode("LOW")
    #initial data setup
    # is_fed=True
    # train_flag=True
    # inference_thread()
    fp="../powtalk.txt"
    with open(fp,'w') as fd:
        
        fd.write("S")





    s = socket.socket()
    s.connect((host, port))
    s.setblocking(True)




   

   


    
    t1 = threading.Thread(target=conv_thread_fun)
    # # t2 = threading.Thread(target=other_thread_fun)
    t3 = threading.Thread(target=msg_thread,args=(s,))
    # # t4 = threading.Thread(target=info_thread)
    # t5 = threading.Thread(target=inference_thread)

    
    # # inference_thread()
    # # state=1
    # state_mutex=threading.Lock()
    # # s.setblocking(False)
    # # state=7
    # # federated_thread(s)

    # state=0
    # # msg_thread(s)

    t1.start()
    # # t2.start()
    t3.start()
    # # t4.start()
    # t5.start()
    train_flag=True
    inf_flag=True
    while(True):
        # print("inf_flag is {inf_flag}")
        if(inf_flag==True):
            inference_thread()
            inf_flag=False
        if(data_lable_flag==True):
            selfLearn();
            data_lable_flag=False
        

    # # state=2
    # while(True):
    #     if(state==2):
    #         inference_thread()
    #         break;
    
    
    # print(f"data setup done done, now ready for federated learning")
    # print(f"doing one training round before starting fed")
    # train_process(labled_subset,labled_subset_lables)
    # test_process(image_datasets['val'])
    # while(True):
    #     # state=7
    #     if(state==7):
    #         is_fed=True
    #         train_flag=True
    #         inference_thread()
    #         print(f"entering fed")
            
    #         while(state==7):
    #             federated_thread(s)

    # t4.start()
    # while(True):
    # 	if(state==2):
    # 		state_mutex.acquire()
    # 		inference_thread()
    # 		state_mutex.release()
    # t1.join()
    # t2.join()
    # t3.join()
    # t4.join()



    # t5.start()
    # t5.join()
    # x=get_densenet_op(labled_subset)
    # feature_dataset=[(torch.tensor(img),lbl) for img,lbl in zip(x,labled_subset_lables)]
    # feature_dataset_dataloader =torch.utils.data.DataLoader(dataset=feature_dataset, shuffle=False, batch_size=32)


    print("Done!")
