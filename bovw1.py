import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os
from PIL import Image
import glob

# trainlsc1=np.loadtxt('./nls/class1_train.txt')
# test1=np.loadtxt('./nls/class1_test.txt')
# trainlsc2=np.loadtxt('./nls/class2_train.txt')
# test2=np.loadtxt('./nls/class2_test.txt')
# trainlsc3=np.loadtxt('./nls/class3_test.txt')
# test3=np.loadtxt('./nls/class3_test.txt')


# xs1 = [x[0] for x in trainlsc1]
# ys1 = [x[1] for x in trainlsc1]
# plt.scatter(xs1, ys1)

# xs2 = [x[0] for x in trainlsc2]
# ys2 = [x[1] for x in trainlsc2]
# plt.scatter(xs2, ys2)

# xs3 = [x[0] for x in trainlsc3]
# ys3 = [x[1] for x in trainlsc3]
# plt.scatter(xs3, ys3)

# data = np.concatenate((trainlsc1,trainlsc2),axis=0)
# data = np.concatenate((trainlsc3,data),axis=0)


from sklearn.cluster import KMeans

# df = pd.
path1 = '/home/aditya/5th sem/CS669PR/ass2/group09_2b/train/auditorium_data/'
path2 = '/home/aditya/5th sem/CS669PR/ass2/group09_2b/train/desert_vegetation_data/'
path3 = '/home/aditya/5th sem/CS669PR/ass2/group09_2b/train/synagogue_outdoor_data/'

total_data =[]
data_list1 = os.listdir(path1)	
data_list2 = os.listdir(path2)
data_list3 = os.listdir(path3)


# print(data_list1)
for data in data_list1:
	temp=np.loadtxt(path1+"/"+data)
	# total_data = total_data + [temp]
	# temp.tolist()
	for i in temp:
		total_data.append(i.tolist())

for data in data_list2:
	temp=np.loadtxt(path2+"/"+data)
	# total_data = total_data + [temp]
	# temp.tolist()
	for i in temp:
		total_data.append(i.tolist())

for data in data_list3:
	temp=np.loadtxt(path3+"/"+data)
	# total_data = total_data + [temp]
	# temp.tolist()
	for i in temp:
		total_data.append(i.tolist())

print(total_data)


kmeans = KMeans(n_clusters=32)
kmeans.fit(total_data)
print(kmeans)

labels = kmeans.predict(total_data)

centroids = kmeans.cluster_centers_

print(centroids)

# cx =[x[0] for x in centroids]
# cy =[x[1] for x in centroids]
# plt.scatter(cx,cy)
# plt.show()