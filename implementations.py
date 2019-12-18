# import PyTorch, Torchvision packages
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from torchvision import transforms

# import Numpy
import numpy as np

# import Maplotlib
%matplotlib inline
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# import Python standard library
import time
import os,sys
import random


#seed for reproducible results
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Helper functions

def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

# Function that implements random cropping

def uniform(a,b):
    return(a+(b-a)*random())

def img_rnd_crop(im, w, h, i = -1, j = -1):
    is_2d = len(im.shape) < 3
    imgwidth = im.shape[len(im.shape)-2]
    imgheight = im.shape[len(im.shape)-1]
    if (i == -1 and j == -1):
        i = int(uniform(0, imgwidth-w-1))
        j = int(uniform(0, imgheight-h-1))
    if is_2d:
        im_patch = im[i:i+w, j:j+h]
    else:
        im_patch = im[:, i:i+w, j:j+h]
    return im_patch, i, j


	def rotated_expansion(imgs):
    shape = [imgs.shape[i] for i in range(len(imgs.shape))]
    shape[0] = shape[0]*4 # there will be 4 times as many images after we rotate in each direction
    shape = tuple(shape)
    rotated_imgs = np.empty(shape)
    
    for index in range(int(shape[0]/4)):
        img = imgs[index]
        if(len(np.shape(img))>2):
            img90 = np.rot90(img.swapaxes(1,2).swapaxes(0,2)).swapaxes(0,2).swapaxes(1,2)
            img180 = np.rot90(img90.swapaxes(1,2).swapaxes(0,2)).swapaxes(0,2).swapaxes(1,2)
            img270 = np.rot90(img180.swapaxes(1,2).swapaxes(0,2)).swapaxes(0,2).swapaxes(1,2)
        else:
            img90 = np.rot90(img)
            img180 = np.rot90(img90)
            img270 = np.rot90(img180)
        
        rotated_imgs[index*4] = img
        rotated_imgs[index*4+1] = img90
        rotated_imgs[index*4+2] = img180
        rotated_imgs[index*4+3] = img270
    
    return rotated_imgs

def flipped_expansion(imgs):
    shape = [imgs.shape[i] for i in range(len(imgs.shape))]
    shape[0] = shape[0]*3 # there will be 4 times as many images after we rotate in each direction
    shape = tuple(shape)
    flipped_imgs = np.empty(shape)
    
    for index in range(int(shape[0]/3)):
        img = imgs[index]
        if(len(np.shape(img))>2):
            imgup = np.flipud(img.swapaxes(1,2).swapaxes(0,2)).swapaxes(0,2).swapaxes(1,2)
            imglr = np.fliplr(img.swapaxes(1,2).swapaxes(0,2)).swapaxes(0,2).swapaxes(1,2)
        else:
            imgup = np.flipud(img)
            imglr = np.fliplr(img)
        
        flipped_imgs[index*3] = img
        flipped_imgs[index*3+1] = imgup
        flipped_imgs[index*3+2] = imglr
    
    return flipped_imgs



def train():
	end_train = 500 # normally = 90
	end_validation = 550 # normally = 100

	# Loading a set of 100 training images
	root_dir = "training/"

	image_dir = root_dir + "images/"
	files = os.listdir(image_dir)
	n = min(100, len(files)) # Load maximum 100 images
	print("Loading " + str(n) + " images")
	imgs = np.array([load_image(image_dir + files[i]) for i in range(n)]).swapaxes(1,3).swapaxes(2,3)
	print(np.shape(imgs))

	# train_input = imgs[0:end_train] #normally = 0:90
	# validation_input = imgs[end_train:end_validation] #normally = 90:100

	image_dir = root_dir + "groundtruth/"
	files = os.listdir(image_dir)
	n = min(100, len(files)) # Load maximum 100 images
	print("Loading " + str(n) + " images")
	grounds = [load_image(image_dir + files[i]) for i in range(n)]
	print(np.shape(grounds))

	imgs = np.array(imgs)
	grounds = np.array(grounds)

	# train_target = grounds[0:end_train] #normally = 0:90
	# validation_target = grounds[end_train:end_validation] #normally = 90:100
	rotated_grounds = rotated_expansion(grounds)
	rotated_imgs = rotated_expansion(imgs)


	flipped_rotated_grounds = flipped_expansion(rotated_grounds)
	flipped_rotated_imgs = flipped_expansion(rotated_imgs)

	display(np.shape(flipped_rotated_grounds))
	display(np.shape(flipped_rotated_imgs))

	# crop images to their 256*256 counterparts
	cropped_imgs = []
	cropped_targets = []

	for i in range(1200):
	    cropped_img, k, l = img_rnd_crop(flipped_rotated_imgs[i], 256, 256)
	    cropped_target, _, _ = img_rnd_crop(flipped_rotated_grounds[i], 256, 256, k, l)
	    cropped_imgs.append(cropped_img)
	    cropped_targets.append(cropped_target)

	x = list(range(1200))
	random.shuffle(x)

	train_input = [cropped_imgs[i] for i in x[:end_train]] #normally = 0:1080
	validation_input = [cropped_imgs[i] for i in x[end_train:end_validation]] #normally = 1080:1200



	train_target = [cropped_targets[i] for i in x[:end_train]] #normally = 0:1080
	validation_target = [cropped_targets[i] for i in x[end_train:end_validation]] #normally = 1080:1200

	display(np.shape(train_input))
	display(np.shape(validation_input))
	display(np.shape(train_target))
	display(np.shape(validation_target))

	# This class takes our input of size 400*400 and enlarges it to size 512*512

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



	res50_conv = torch.nn.Sequential(*list(
	        torchvision.models.resnet50(pretrained=True).children())[:-2])  # get all layers except avg-pool & fc


	# for p in res50_conv.parameters():
	#     p.requires_grad=False

	model = torch.nn.Sequential(
	    res50_conv,  # encoder
	    torch.nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1),  # 2x upsample
	    torch.nn.BatchNorm2d(1024),
	    torch.nn.ReLU(),
	    torch.nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # 2x upsample
	    torch.nn.BatchNorm2d(512),
	    torch.nn.ReLU(),
	    torch.nn.ConvTranspose2d(512, 256, kernel_size=6, stride=4, padding=1),  # 4x upsample
	    torch.nn.BatchNorm2d(256),
	    torch.nn.ReLU(),
	    torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 2x upsample
	    torch.nn.BatchNorm2d(128),
	    torch.nn.ReLU(),
	    torch.nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0),  # logits per pixel
	    torch.nn.Sigmoid()  # predictions per pixel  # could remove and use BCEWithLogitsLoss instead of BCELoss.
	)
	model.to(device)



	for p in model.parameters():
	    try:
	        torch.nn.init.xavier_normal_(p)
	    except ValueError:
	        pass
	    
	    
	model = torch.load('best_modelgood150.pth')
	model.eval()

	from PIL import Image


	class GrayscaleAndThreshold:
	    """ Reduce image to a single binary channel """
	    def __init__(self, level=0.1):
	        self.level = level

	    def __call__(self, img):
	        img = img.convert('L')  # 0..255, single channel

	        np_img = np.array(img, dtype=np.uint8)
	        np_img[np_img > self.level*255] = 255
	        np_img[np_img <= self.level*255] = 0

	        img = Image.fromarray(np_img, 'L')

	        return img

	class WeightedBCELoss(torch.nn.BCELoss):
	    def __init__(self, class_weights=None):  # does not support weight, size_average, reduce, reduction
	        super().__init__(reduction='none')
	        if class_weights is None:
	            class_weights = torch.ones(2)
	        self.class_weights = torch.as_tensor(class_weights)

	    def forward(self, input, target):
	        raw_loss = super().forward(input, target)
	        class_weights = self.class_weights.to(input.device)
	        weight_matrix = class_weights[0]*(1-target) + class_weights[1]*target
	        loss = weight_matrix * raw_loss
	        loss = loss.mean()  # reduction='elementwise_mean'
	        return loss


	def compute_class_weights(imgs):
	    mask_transform = transforms.Compose([
	        GrayscaleAndThreshold(),
	        transforms.ToTensor()
	        ])

	    road_pxs = 0
	    bg_pxs = 0
	    for img in imgs:
	        img = Image.fromarray(np.uint8(img*255))
	        mask_tr = torch.squeeze(mask_transform(img)).numpy().astype(int)
	        road_pxs += mask_tr.sum()
	        bg_pxs += (1 - mask_tr).sum()

	    bg_px_weight = (road_pxs + bg_pxs) / (2 * bg_pxs)  # "class 0"
	    road_px_weight = (road_pxs + bg_pxs) / (2 * road_pxs)  # "class 1"

	    return bg_px_weight, road_px_weight

	    # Data statistics
	class_weights = compute_class_weights(train_target)

	criterion = WeightedBCELoss(class_weights)

	# We will optimize the cross-entropy loss using adam algorithm
	loss_function = torch.nn.CrossEntropyLoss()
	# optimizer = optim.Adam(model.parameters(), lr=0.001)
	optimizer = optim.SGD(model.parameters(), lr=3.75e-2, momentum=0.9)


	def trainNet(model, n_epochs):
	    loss_train_epoch =[]
	    loss_validation_epoch =[]
	    
	    #Time for printing
	    training_start_time = time.time()
	    
	    #Loop for n_epochs
	    for epoch in range(n_epochs):
	        
	        total_loss = 0.0
	        
	        for index in range(np.shape(train_input)[0]):
	            model.train()
	            
	            input_image = torch.tensor(train_input[index]).unsqueeze(0).to(device)
	            target_image = torch.tensor(train_target[index]).to(device)
	            
	            #Forward pass, backward pass, optimize
	            outputs = model(input_image.float())
	            loss = criterion(outputs, target_image.float())
	            
	            #Set the parameter gradients to zero
	            optimizer.zero_grad()
	            loss.backward()
	            optimizer.step()
	            
	            #Print statistics
	            total_loss += loss.item() * input_image.size(0)
	            
	            print("Epoch", epoch, ", image", index, ", image loss:", loss.item(), ", time elapsed:", time.time() - training_start_time)
	            
	        #At the end of the epoch, do a pass on the validation set
	        total_val_loss = 0
	        for index in range(np.shape(validation_input)[0]):
	            model.eval()
	                        
	            
	            input_image = torch.tensor(validation_input[index]).unsqueeze(0).to(device)
	            target_image = torch.tensor(validation_target[index]).to(device)
	            
	            #Forward pass
	            val_outputs = model(input_image.float())
	            val_loss = criterion(val_outputs, target_image.float())
	            total_val_loss += val_loss.item() * input_image.size(0)
	            
	        print("Validation loss for epoch", epoch, ":", total_val_loss/np.shape(validation_input)[0])
	        
	        loss_train_epoch.append(total_loss/np.shape(train_input)[0])
	        loss_validation_epoch.append(total_val_loss/np.shape(validation_input)[0])
	        
	    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
	    
	    torch.save(model, 'best_model.pth')
	    return loss_train_epoch, loss_validation_epoch

	    losstrain, loss_validation = trainNet(model, 1)

	    input_image = torch.tensor(validation_input[5]).unsqueeze(0)
		target_image = torch.tensor(validation_target[5]).unsqueeze(0)
		           
		#Forward pass
		val_output = model(input_image.to(device).float())
		output_image = val_output[0,0]

		np.savetxt('losstrain.csv', losstrain, delimiter=',')
		np.savetxt('loss_validation.csv', loss_validation, delimiter=',')


	root_dir = "test_set_images/"
	test_images=[]
	for i in range(1, 51):
	    image_filename = root_dir + "test_" + str(i) + "/test_" + str(i) + '.png'
	    test_images.append(np.array(load_image(image_filename)).swapaxes(0,2).swapaxes(1,2))
	    test_images
	print(np.shape(test_images))7

		
	crop = 256
	masks = []
	for test_image in test_images:
	    _, im_height, im_width = np.shape(test_image)
	    imgheight = test_image.shape[1]
	    imgwidth = test_image.shape[2]
	    mask = torch.zeros(1, imgheight, imgwidth)
	    for i in range(0, imgheight, crop):
	        for j in range(0, imgwidth, crop):
	            # when the crop is bigger than the image size, we increase the temporary image with 0
	            if(i+crop>imgheight and j+crop>imgwidth):
	                im_patch = np.zeros([3,crop,crop],dtype = np.float32)
	                im_patch[:, :imgheight-i, :imgwidth-j] = test_image[:, i:imgheight, j:imgwidth]
	                im_patch = torch.tensor(im_patch).unsqueeze(0).to(device)
	                mask[:, i:imgheight, j:imgwidth] = model(im_patch.float()).detach()[0,0,:imgheight-i,:imgwidth-j]
	            
	            elif(i+crop>imgheight):
	                im_patch = np.zeros([3,crop,crop],dtype = np.float32)
	                im_patch[:, :imgheight-i, :] = test_image[:, i:imgheight, j:j+crop]
	                im_patch = torch.tensor(im_patch).unsqueeze(0).to(device)
	                mask[:, i:imgheight, j:j+crop] = model(im_patch.float()).detach()[0,0,:imgheight-i,:]
	            
	            elif(j+crop>imgwidth):
	                im_patch = np.zeros([3,crop,crop])
	                im_patch[:, :, :imgwidth-j] = test_image[:, i:i+crop, j:imgwidth]
	                im_patch = torch.tensor(im_patch).unsqueeze(0).to(device)
	                mask[:, i:i+crop, j:imgwidth] = model(im_patch.float()).detach()[0,0,:,:imgwidth-j]
	            
	            else: # cas normal
	                im_patch = test_image[:, i:i+crop, j:j+crop]
	                im_patch = torch.tensor(im_patch).unsqueeze(0).to(device)
	                mask[:, i:i+crop, j:j+crop] = model(im_patch).detach()[0,0,:,:]
	    masks.append(mask.numpy())