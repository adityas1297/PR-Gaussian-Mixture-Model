import cv2 
import matplotlib.pyplot as plt
import numpy as np 
import os
from PIL import Image
import glob
from statistics import mean 
from statistics import variance


def get_data(image,stepSize,w_width,w_height,filename):
	count=0
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

			# m = mean(window)
			# np.int8(window)
			print(window)
			m = sum(window)/len(window)
			# v = variance(window)
			v = np.var(window)

			
			# print("count = ",count)

			data_point.append(m)
			data_point.append(v)
			# print(data_point)
			all_data_points.append(data_point)
			# print("sum ",sum(v))


	# print(all_data_points)
	new_file = open("./testbovw/"+filename[:len(img)-4]+".txt",'w')
	# new_file = open("./group09_2c/Train_Data/"+filename[:len(img)-4]+".txt",'w')

	for i in all_data_points:
		for j in range(0,2):
			# print(i[j],end=" ")
			new_file.write(str(i[j])+str(" "))
		# print("")
		new_file.write('\n')
	# print("Size of all = ",len(all_data_points))
	# print(count)



path ='./testimg/'
# path = '/home/aditya/5th sem/CS669PR/ass2/group09_2c/Train/'
# print(path)
image_list = os.listdir(path)
l = len(image_list)
print("No of pics = ",l)
# print(image_list)
stepSize = 7
(w_width,w_height)=(7,7)

# image = cv2.imread(path+'test.jpg')
count=0
for img in image_list:
	image = cv2.imread("test2.png",cv2.IMREAD_UNCHANGED)
	# print(image)
	# image = cv2.imread(path+img)
	get_data(image,stepSize,w_width,w_height,img)
	count = count+1
	print(count,"-",img+" Done.")