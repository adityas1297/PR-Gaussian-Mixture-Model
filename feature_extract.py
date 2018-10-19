import cv2 
import matplotlib.pyplot as plt
import numpy as np 
import os
from PIL import Image
import glob


def get_data(image,stepSize,w_width,w_height,filename):
	count=0
	all_data_points=[]
	for x in range(0,image.shape[1],stepSize):
		for y in range(0,image.shape[0],stepSize):
			
			# window = image[x:x+w_width,y:y+w_height,:]
			# cv2.rectangle(tmp,(x,y),(x+w_width,y+w_height),(0,255,0),2)
			# plt.imshow(np.array(tmp).astype('uint8'))
			count=count+1
			r = [0]*8
			g = [0]*8
			b = [0]*8
			data_point = []
			
			for j in range(x,x+w_width):
				for i in range(y,y+w_height):
					b[((image[i%image.shape[0],j%image.shape[1],0])//32)] = b[((image[i%image.shape[0],j%image.shape[1],0])//32)]+1
					g[((image[i%image.shape[0],j%image.shape[1],1])//32)] = g[((image[i%image.shape[0],j%image.shape[1],1])//32)]+1
					r[((image[i%image.shape[0],j%image.shape[1],2])//32)] = r[((image[i%image.shape[0],j%image.shape[1],2])//32)]+1
			# print("r = ",r)
			# print("b = ",b)
			# print("g = ",g)
			# print("count = ",count)

			data_point=b+g+r
			all_data_points.append(data_point)

	# print(all_data_points)
	# new_path = './new_days.txt'
	# new_file = open("./testimg/"+filename[:len(img)-4]+".txt",'w')
	new_file = open("./group09_2b/train/auditorium_data/"+filename[:len(img)-4]+".txt",'w')
	# new_days.write(title)
	# print(tit
	for i in all_data_points:
		for j in range(0,24):
			# print(i[j],end=" ")
			new_file.write(str(i[j])+str(" "))
		# print("")
		new_file.write('\n')
	# print("Size of all = ",len(all_data_points))
	# print(count)



# path ='./testimg/'
path = '/home/aditya/5th sem/CS669PR/ass2/group09_2b/train/auditorium/'
# print(path)
image_list = os.listdir(path)
l = len(image_list)
print("No of pics = ",l)
# print(image_list)
stepSize = 32
(w_width,w_height)=(32,32)

# image = cv2.imread(path+'test.jpg')
count=0
for img in image_list:
	image = cv2.imread(path+img)
	# print("image = ",image)
	# print(image.shape)
	# print(image.dtype)
	# image = cv2.imread("test1.jpg")
	get_data(image,stepSize,w_width,w_height,img)
	count = count+1
	print(count,"-",img+" Done.")