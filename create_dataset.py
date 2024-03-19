import sys
import os
import cv2
import numpy as np
import random
import torchvision
import torchvision.transforms as transforms
from PIL import Image

# Assumes that full_dataset is always in a folder named 'full_dataset'

folder = sys.argv[1]
task = sys.argv[2]

# new code to create the folder and save images
if folder == "CIFAR-10":
	datasets_path = "./datasets"
	os.makedirs(datasets_path, exist_ok=True)
	folder_path = os.path.join(datasets_path, folder)
	os.makedirs(folder_path, exist_ok=True)
	full_dataset_path = os.path.join(folder_path, "full_dataset")
	os.makedirs(full_dataset_path, exist_ok=True)

	# Define the CIFAR10 download and transformation process
	transform = transforms.Compose([
		transforms.ToTensor()  # Convert images to PyTorch tensors
	])

	# Load CIFAR10 dataset
	train_set = torchvision.datasets.CIFAR10(root=folder_path, train=True, download=True, transform=transform)
	test_set = torchvision.datasets.CIFAR10(root=folder_path, train=False, download=True, transform=transform)

	# Function to save images
	def save_images(dataset, prefix):
		for idx, (image, label) in enumerate(dataset):
			img_path = os.path.join(full_dataset_path, f"{prefix}_{idx}_{label}.jpg")
			image = transforms.ToPILImage()(image)
			image.save(img_path)

	# Save train and test set images
	save_images(train_set, 'train')
	save_images(test_set, 'test')


files = os.listdir(full_dataset_path)

files = [file for file in files if file.endswith(".jpg")]
total = len(files)

# Fraction of dataset to be trained and tested on can be changed here 
train_frac = int(0.9*total)
test_frac = int(0.05*total)
eval_frac = total-train_frac-test_frac

random.shuffle(files)

os.mkdir(folder_path+"/train_"+task)
os.mkdir(folder_path+"/test_"+task)
os.mkdir(folder_path+"/eval_"+task)
os.mkdir(folder_path+"/train_"+task+"/input/")
os.mkdir(folder_path+"/train_"+task+"/target/")
os.mkdir(folder_path+"/test_"+task+"/input/")
os.mkdir(folder_path+"/test_"+task+"/target/")
os.mkdir(folder_path+"/test_"+task+"/output/")
os.mkdir(folder_path+"/eval_"+task+"/input/")
os.mkdir(folder_path+"/eval_"+task+"/target/")
os.mkdir(folder_path+"/eval_"+task+"/output/")

index = 1
for file in files:
	img = cv2.imread(folder_path+"/full_dataset/"+file)

	if task == "bnw2color":
		out = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		out = cv2.cvtColor(out,cv2.COLOR_GRAY2RGB)
	elif task == "deblur":
		kernel = np.ones((4,4),np.float32)/16
		out = cv2.filter2D(img,-1,kernel)
	elif task == "impaint_fix":
		out = img
		out[100:140,100:140,0:3] = 0

	if index <= train_frac:
		cv2.imwrite(folder_path+"/train_"+task+"/input/"+str(index)+".jpg", cv2.resize(out, (256, 256)))
		cv2.imwrite(folder_path+"/train_"+task+"/target/"+str(index)+".jpg", cv2.resize(img, (256, 256)))
	elif index <= train_frac+test_frac:
		cv2.imwrite(folder_path+"/test_"+task+"/input/"+str(index-train_frac)+".jpg", cv2.resize(out, (256, 256)))
		cv2.imwrite(folder_path+"/test_"+task+"/target/"+str(index-train_frac)+".jpg", cv2.resize(img, (256, 256)))	
	else:
		cv2.imwrite(folder_path+"/eval_"+task+"/input/"+str(index-train_frac-test_frac)+".jpg", cv2.resize(out, (256, 256)))
		cv2.imwrite(folder_path+"/eval_"+task+"/target/"+str(index-train_frac-test_frac)+".jpg",cv2.resize(img, (256, 256)))
	index += 1