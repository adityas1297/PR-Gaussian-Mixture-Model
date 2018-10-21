import cv2 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from pandas import DataFrame 
from sklearn import datasets 
from sklearn.mixture import GaussianMixture 
import os
from PIL import Image
import glob
from statistics import mean 
from statistics import variance


data = []
# path = '/home/aditya/5th sem/CS669PR/ass2/group09_2b/train/auditorium_bovw/'
path = '/home/aditya/5th sem/CS669PR/ass2/group09_2c/Train_Data/'
data_list = os.listdir(path)
for i in data_list:
	temp = np.loadtxt(path+i)
	for line in temp:
	# temp1= line
		data.append(line)
# print(data)
gmm = GaussianMixture(n_components = 3)
gmm.fit(data)	



print("Fit data")


def get_data(image,stepSize,w_width,w_height,filename,image1,centers):
	all_data_points=[]
	for x in range(0,image.shape[1],stepSize):
		for y in range(0,image.shape[0],stepSize):
			# m = [0]*1
			# v = [0]*1 
			window = []
			data_point = []
			
			for j in range(x,x+w_width):
				for i in range(y,y+w_height):
					window.append((image[(i%image.shape[0])][(j%image.shape[1])]))

			m = sum(window)/len(window)
			v = np.var(window)

			data_point.append(m)
			data_point.append(v)

			label = gmm.predict([data_point])

			# dist = np.zeros(3)
			# count =0 
			# for center in centers:
			# 	dist[count] = np.linalg.norm(data_point-center)
			# 	count = count+1

			min_index = label
			# print(min_index)


			# for j in range(x,x+w_width):
			# 	for i in range(y,y+w_height):
			# 		image1[(i%image.shape[0])][(j%image.shape[1])]=(3**(min_index+2))


			for j in range(x,x+w_width):
				for i in range(y,y+w_height):
					# window.append((image[(i%image.shape[0])][(j%image.shape[1])]))
					if min_index==0:
						image1[(i%image.shape[0])][(j%image.shape[1])][0]=255
						image1[(i%image.shape[0])][(j%image.shape[1])][1]=0
						image1[(i%image.shape[0])][(j%image.shape[1])][2]=0
					elif min_index==1:
						image1[(i%image.shape[0])][(j%image.shape[1])][0]=0
						image1[(i%image.shape[0])][(j%image.shape[1])][1]=255
						image1[(i%image.shape[0])][(j%image.shape[1])][2]=0
					elif min_index==2:
						image1[(i%image.shape[0])][(j%image.shape[1])][0]=0
						image1[(i%image.shape[0])][(j%image.shape[1])][1]=0
						image1[(i%image.shape[0])][(j%image.shape[1])][2]=255

			# 		elif min_index==3:
			# 			image1[(i%image.shape[0])][(j%image.shape[1])][0]=0
			# 			image1[(i%image.shape[0])][(j%image.shape[1])][1]=165
			# 			image1[(i%image.shape[0])][(j%image.shape[1])][2]=255
			# 		elif min_index==4:
			# 			image1[(i%image.shape[0])][(j%image.shape[1])][0]=0
			# 			image1[(i%image.shape[0])][(j%image.shape[1])][1]=255/2
			# 			image1[(i%image.shape[0])][(j%image.shape[1])][2]=255/2
			# 		elif min_index==5:
			# 			image1[(i%image.shape[0])][(j%image.shape[1])][0]=255/2
			# 			image1[(i%image.shape[0])][(j%image.shape[1])][1]=0
			# 			image1[(i%image.shape[0])][(j%image.shape[1])][2]=255/2


			# for j in range(x,x+w_width):
			# 	for i in range(y,y+w_height):
			# 		# window.append((image[(i%image.shape[0])][(j%image.shape[1])]))
					
			# 		image1[(i%image.shape[0])][(j%image.shape[1])][0]=(3**(min_index+2))
			# 		image1[(i%image.shape[0])][(j%image.shape[1])][1]=(3**(min_index+2))
			# 		image1[(i%image.shape[0])][(j%image.shape[1])][2]=(3**(min_index+2))
				

	cv2.imwrite(img[:len(filename)-4]+"_gmm_s1_1.jpg",image1)
	print(img,"done")

  


# data1 = []
# path1 = '/home/aditya/5th sem/CS669PR/ass2/group09_2c/Test_Data/94.txt'
# data_list1 = os.listdir(path1)
# for i in data_list1:
# temp = np.loadtxt(path1)
# print(temp)
# for line in temp:
	# temp1= line
	# data1.append(line)
	# data1.append(temp)


# labels = gmm.predict(data1)
	# print(labels) 
	# for x in labels:
	# 	if x==0:
	# 		print("loool")
	# print(len(data))

centers = gmm.means_





print(gmm.lower_bound_) 
print("mu",centers)
print("pi",gmm.weights_)
print("cov",gmm.covariances_)
  
# print the number of iterations needed 
# for the log-likelihood value to converge 
print("no of iterations =",gmm.n_iter_)

print(gmm.means_)
# plt.show()


centers=np.loadtxt('cell_centers1.txt')


path = '/home/aditya/5th sem/CS669PR/ass2/group09_2c/Test/'
# print(path)
image_list = os.listdir(path)
l = len(image_list)
print("No of pics = ",l)
# print(image_list)
stepSize = 1
(w_width,w_height)=(7,7)

# image = cv2.imread('test2.png',cv2.IMREAD_UNCHANGED)
# image1 = cv2.imread('test2.png',cv2.IMREAD_UNCHANGED)
# get_data(image,stepSize,w_width,w_height,'test2.png',image1,centers)
count=0
for img in image_list:
	image = cv2.imread(path+img,cv2.IMREAD_UNCHANGED)
	# image = cv2.imread(path+img,cv2.IMREAD_UNCHANGED)
	# print(image)
	image1 = cv2.imread(path+img)
	# print(image1)
	get_data(image,stepSize,w_width,w_height,img,image1,centers)
	count = count+1
	print(count,"-",img+" Done.")

