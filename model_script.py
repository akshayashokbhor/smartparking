'''
*********************************************************************************************************************
                                  Smart Parking detection Project

                                                  By, Akshay Ashok Bhor (Deep Learning Engineer)

*********************************************************************************************************************

   Import packages
*********************************************************************************************************************
'''

from __future__ import division
import matplotlib.pyplot as plt
import cv2
import os, glob
import numpy as np
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from moviepy.editor import VideoFileClip
cwd = os.getcwd()
import numpy
import os
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.applications.resnet50 import ResNet50 


print("Loaded All")
'''**********************
Get sample parking image

*************************'''

org_image=cv2.imread("C:/Users/Akshay-pc/Desktop/Parking Image/16.jpe")

print("The Type of original Image is",type(org_image))


#**************************************define Coordinates*********************************************************
coordinates=[15,205,1272,474]


#******************code to generate All parking slots for the parking**********************************************

def bound_rectangle(image,spot):

	rect_ver=[]
	for i in range(1,10):
		gap=180

		X1=spot[0]
		x_1=X1-gap
		
		x1=x_1+gap*i
		x_1=int(x1)
		y1=spot[1]
		y1=int(y1)
		x2=x1+gap+i
		x2=int(x2)
		print("diff",x2-x1)
		print("x2",x2)
		y2=spot[3]
		y2=int(y2)
		if x2<=1300:
			cv2.rectangle(image,(x1,y1),(x2,y2),color=[255, 0, 0],thickness=3)
			rect_ver.append([x1,y1,x2,y2])
		
	return image,rect_ver


parking_spot,vertices=bound_rectangle(org_image,coordinates)
print("Vertices of Parking area are:",vertices)

plt.imshow(parking_spot)
plt.show()

'''
#*************Taking All parking images for an interval of time to generate data********************* 

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        img
        if img is not None:
            images.append(img)
    return images

folder_="C:/Users/Akshay-pc/Desktop/Parking Image"
All_parking_images=load_images_from_folder(folder_)

#*****************Crop All the individual images in all the parking images and store this in local pc for future use*******


def train_data(folder,vertices):
	train_data_image=[]
	for image in folder:
		for i in vertices:
			[x1,y1,x2,y2]=i
			spot_img = image[y1:y2, x1:x2]
			train_data_image.append(spot_img)
		
		
        	

	return train_data_image

train_cropped_images=train_data(All_parking_images,vertices)
print("Number of total data images generated are:",len(train_cropped_images))


def image_data(tr_image): 
	for i in range(len(tr_image)):
		filename = 'Parking' + str(i) +'.jpg'
		folder_name='C:\\Users\\Akshay-pc\\Desktop\\SMART_PARKING_PROJECT\\Cropped_images_data'
    
        
		cv2.imwrite(os.path.join(folder_name, filename), tr_image[i]) 


#image_data(train_cropped_images)

#**************************************Create model for future prediction****************************************

resnet_model= ResNet50(weights = None, include_top=False, input_shape = (227, 170, 3))

x = resnet_model.output
x = Flatten()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(2, activation="softmax")(x)


# creating the final model
resnet_model_final = Model(input = resnet_model.input, output = predictions)

print("Model Created")


#*****************************load best model weights to model we created above********************************* 

top_model_weights_path="Resnet_weights.best.hdf5"
resnet_model_final.load_weights(top_model_weights_path)
print("Loaded model from disk")

#******************************vgg16 model loaded**************************************************************
model = applications.VGG16(weights = None, include_top=False, input_shape = (227, 170, 3))


num_classes=2
x = model.output
x = Flatten()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation="softmax")(x)


# creating the final model
model_final = Model(input = model.input, output = predictions)

#**************************************************************************************************************
top_model_weights_path="weights.best.vgg16.hdf5"
model_final.load_weights(top_model_weights_path)
print("Loaded model from disk")


#****************************predict on parking image***********************************************************

class_dictionary = {}
class_dictionary[0] = 'empty'
class_dictionary[1] = 'occupied'
def make_prediction(image11):
   
	#img = image/255.

	image = np.expand_dims(image11, axis=0)
   

    # make predictions on the preloaded model
	class_predicted = model_final.predict(image)
    #print("Predicted class is:",class_predicted)
	inID = np.argmax(class_predicted[0])
    #print("Index is:",inID)

	label = class_dictionary[inID]
	return label

def predict_on_image(image, vertices, make_copy=True, color = [0, 255, 0], alpha=0.5):
	cnt_empty = 0
	all_spots = 0

	if make_copy:
		new_image = np.copy(image)
		overlay = np.copy(image)
    

	

	for i in vertices:
		print(i)
		[x1,y1,x2,y2]=i
		spot_img = image[y1:y2, x1:x2]
		spot_img=cv2.resize(spot_img,(170,227))
		print("Afetr resize shape is:",spot_img.shape)

		label = make_prediction(spot_img)
		print(label)
		if label == 'empty':
			plt.imshow(spot_img)
			cv2.rectangle(overlay, (int(x1),int(y1)), (int(x2),int(y2)), color, -1)
			cnt_empty += 1
		plt.show()
            
	cv2.addWeighted(overlay, alpha, new_image, 1 - alpha, 0, new_image)
            
	cv2.putText(new_image, "Available: %d spots" %cnt_empty, (30, 95),
	cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 255, 255), 2)
    
	cv2.putText(new_image, "Total: %d spots" %all_spots, (30, 125),
	cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 255, 255), 2)
   
    

    
	return new_image


final_result=predict_on_image(org_image, vertices)
plt.imshow(parking_spot)
plt.show()

#**********************************Predict With Video***********************************************************************

video_name = 'ezgif.com-crop.mp4'
cap = cv2.VideoCapture(video_name)

ret = True
count = 0

while ret:
	ret, image = cap.read()
	count += 1
    
	if count == 5:
		count = 0
            
		new_image = np.copy(image)
		overlay = np.copy(image)
		cnt_empty = 0
		all_spots = 0
		color = [0, 255, 0] 
		alpha=0.5

		for i in vertices:
			#print(i)
			[x1,y1,x2,y2]=i
			spot_img = image[y1:y2, x1:x2]
			spot_img=cv2.resize(spot_img,(170,227))

			#plt.imshow(spot_img)
		#	plt.show()
			
			label = make_prediction(spot_img)
			print(label)
			if label == 'empty':
				plt.imshow(spot_img)
				cv2.rectangle(overlay, (int(x1),int(y1)), (int(x2),int(y2)), color, -1)
				cnt_empty += 1
			plt.show()

		cv2.addWeighted(overlay, alpha, new_image, 1 - alpha, 0, new_image)

		cv2.putText(new_image, "Available: %d spots" %cnt_empty, (30, 95),
		cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 255, 255), 2)

		cv2.putText(new_image, "Total: %d spots" %all_spots, (30, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (255, 255, 255), 2)
		cv2.imshow('frame', new_image)
		
		if cv2.waitKey(10) & 0xFF == ord('q'):
			break
     

cv2.destroyAllWindows()
cap.release()
'''