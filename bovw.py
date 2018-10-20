import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os
from PIL import Image
import glob

# path1 = './testimg/'
path1 = '/home/aditya/5th sem/CS669PR/ass2/group09_2b/test/auditorium_data/'
# path1 = '/home/aditya/5th sem/CS669PR/ass2/group09_2b/test/desert_vegetation_data/'
# path1 = '/home/aditya/5th sem/CS669PR/ass2/group09_2b/test/synagogue_outdoor_data/'

total_data =[]
data_list1 = os.listdir(path1)	
# data_list2 = os.listdir(path2)
# data_list3 = os.listdir(path3)

centers=np.loadtxt('bovw.txt')

# print(centers)

def bag_of_visual_words(centers,filename,path):
	# print("filename = " ,filename)
	data = np.loadtxt(path+filename)
	bovw = [0]*32
	for line in data:
		dist = np.zeros(32)
		count =0 
		for center in centers:
			dist[count] = np.linalg.norm(line-center)
			count = count+1
		bovw[np.argmin(dist)] = bovw[np.argmin(dist)] +1

	# print(bovw)
	# new_file = open("./testbovw/"+filename[:len(filename)-4]+"_bovw.txt",'w')
	new_file = open("/home/aditya/5th sem/CS669PR/ass2/group09_2b/test/auditorium_bovw/"+filename[:len(filename)-4]+"_bovw.txt",'w')
	for i in bovw:
		new_file.write(str(i)+str(" "))
	new_file.write('\n')

# filename = '/home/aditya/5th sem/CS669PR/ass2/test.txt'

count=0 
for file in data_list1:
	bag_of_visual_words(centers,file,path1)
	count = count +1
	print(count," - ",file,"done." )