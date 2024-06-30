import numpy as np
import numpy as np
import time
import torch
import cv2
PRECISION = np.float32
modelname='mobilenet'	#MJ:change as per trt file of the model that you want to use
BATCH_SIZE=16		# MJ:change as per the batch size
from onnx_helper import ONNXClassifierWrapper
N_CLASSES = 1000 # Our ResNet-50 is trained on a 1000 class ImageNet task

#dummy_input_batch = np.zeros((BATCH_SIZE, 224, 224, 3), dtype = PRECISION)

dummy_input_batch= np.random.random(size=(BATCH_SIZE,224, 224, 3))


import torchvision.datasets as datasets

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader




# Define the path to the root folder of your downloaded ImageNet dataset
data_path = 'tiny-imagenet-200'

# Define the transformation to apply to the images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the ImageNet dataset
dataset = ImageFolder(root=data_path, transform=transform)


sample_im= torch.permute(dataset[1][0],(0,1,2))
img_ten= torch.empty((3,224,224), dtype=torch.float32)
img_ten=sample_im.unsqueeze(0)
for i in range (0,BATCH_SIZE-1):
	img_ten=torch.vstack((img_ten,torch.permute(dataset[i][0],(0,1,2)).unsqueeze(0)))
	print("original label is "+str(dataset[i][1]))




print(img_ten.size())

batch_size = BATCH_SIZE
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)




img_ten_np=img_ten.numpy()
img_ten_np=img_ten_np.astype(PRECISION)


"""
trt_model = ONNXClassifierWrapper("resnet18_engine_32.trt", [BATCH_SIZE, N_CLASSES], target_dtype = PRECISION)
"""
trt_model = ONNXClassifierWrapper(str(modelname)+str(BATCH_SIZE)+"_engine.trt", [BATCH_SIZE, N_CLASSES], target_dtype = PRECISION)

file_path = 'talk.txt'

def writeTofile(x):
	with open(file_path, 'w') as file:
   	 # Write data to the file
		#file.truncate(0)
    		file.write(x)



start_time=time.time()
print("inference started")
writeTofile('S')
predictions = trt_model.predict(img_ten_np)
writeTofile('E')
stop_time=time.time()
time_taken= stop_time-start_time



print("inference completed")
print( "time taken is")
print(time_taken)

lable=[]

for i in range (0,BATCH_SIZE-1):
	l=np.argmax(predictions[i])
	lable.append(l)



file = open('results.txt','w')
for i in lable:
	file.write(str(i) +"\n")
file.close()

inverse_transform = transforms.Compose([
    
   # transforms.ToPILImage(),
    transforms.Normalize(mean=[1/0.485, 1/0.456, 1/0.406], std=[1/0.229, 1/0.224, 1/0.225])
])

"""
img_to_show=inverse_transform(img_ten[0])
cv2.imshow(img_to_show.numpy())
"""





