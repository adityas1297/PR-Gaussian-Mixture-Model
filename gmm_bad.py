import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import random
import pandas as pd
import os
from PIL import Image
import glob


#---------------------------------Reading data------------------------------------
# if sys.argv[1]=='1':
# trainlsc1=np.loadtxt('./LS_Group09/Class1_train.txt')
# trainall=np.loadtxt('test.txt')
# test1=np.loadtxt('./LS_Group09/Class1_test.txt')
# trainlsc2=np.loadtxt('./LS_Group09/Class2_train.txt')
# test2=np.loadtxt('./LS_Group09/Class2_test.txt')
# trainlsc3=np.loadtxt('./LS_Group09/Class3_train.txt')
# test3=np.loadtxt('./LS_Group09/Class3_test.txt')
# elif sys.argv[1]=='2':
# trainlsc1=np.loadtxt('./rd_group9/class1_train.txt')
# 	test1=np.loadtxt('./rd_group9/class1_test.txt')
# trainlsc1=np.loadtxt('./rd_group9/class2_train.txt')
# 	test2=np.loadtxt('./rd_group9/class2_test.txt')
# trainlsc1=np.loadtxt('./rd_group9/class3_train.txt')
# 	test3=np.loadtxt('./rd_group9/class3_test.txt')
# elif sys.argv[1]=='3':
# trainlsc1=np.loadtxt('rwclass1_train.txt')
# 	test1=np.loadtxt('rwclass1_test.txt')
# trainlsc1=np.loadtxt('rwclass2_train.txt')
# 	test2=np.loadtxt('rwclass2_test.txt')
# trainlsc1=np.loadtxt('rwclass3_train.txt')
	# test3=np.loadtxt('rwclass3_test.txt')

# trainall=np.concatenate((trainlsc1,trainlsc2,trainlsc3),axis=0)

trainall = []
path = '/home/aditya/5th sem/CS669PR/ass2/group09_2b/train/auditorium_bovw/'
data_list = os.listdir(path)
for data in data_list:
	temp = np.loadtxt(path+data)
	trainall.append(temp)
# print(len(trainall))
# print(trainall)

# trainall=trainlsc1

k= int(input("Clusters :"))
clusters=np.zeros((k,trainall[0].size))
clustercount=np.zeros(k)
cost=0

for i in range(k):
	clusters[i]=trainall[random.randint(0,np.size(trainall,0)-1)]

# --------Random initialisation--------------------------------------------------------------
zlabel=np.zeros((np.size(trainall,0),1),dtype=int)
# for x in zlabel:
# 	x[0]=x[0]+random.randint(1,k)

# ----------------------------First time clustering----------------------------------------------

costf=0
cost=0
# for i in range(np.size(trainall,0)):
# 		clusters[zlabel[i]-1]= clusters[zlabel[i]-1] +trainall[i]
		# clustercount[zlabel[i]-1] = clustercount[zlabel[i]-1] +1

# for i in range(np.size(clusters,0)):
# 	clusters[i]= clusters[i]/clustercount[i]



# for i in range(np.size(trainall,0)):
# 	norms=np.zeros(k)
# 	for j in range(k):
# 		norms[j] = np.linalg.norm(trainall[i] - clusters[j])
# 	# print(norms)
# 	zlabel[i]=np.argmin(norms)+1
# 	cost=cost+norms[np.argmin(norms)]
# costf=cost



# print(clusters)

# ---------------------------------------------------------------------------------------------

# ----------------------------------------------functions--------------------------------------------------
def em(clusters,costf):
	cost=0

	while(1):
		# costf=cost
		cost=0
		# clusters=np.zeros((k,trainall[0].size))
		clustercount=np.zeros(k)

		for i in range(np.size(trainall,0)):
			norms=np.zeros(k)
			for j in range(k):
				norms[j] = np.linalg.norm(trainall[i] - clusters[j])
			# print(norms)
			zlabel[i]=np.argmin(norms)+1
			cost=cost+norms[np.argmin(norms)]

		for i in range(np.size(trainall,0)):
			clusters[zlabel[i]-1]= clusters[zlabel[i]-1] +trainall[i]
			clustercount[zlabel[i]-1] = clustercount[zlabel[i]-1] +1

		for i in range(k):
			if(clustercount[i]!=0):
				clusters[i]= clusters[i]/clustercount[i]


		# print(clusters)
		if(costf-cost>=0.0001):
			costf=cost
		else:
			return clusters,clustercount,zlabel

def density(x,mean_i,covariance_mat):
	return np.exp(-1*np.matmul((np.transpose(x-mean_i)),np.matmul((np.linalg.inv(covariance_mat)),(x-mean_i)))/2)/(math.sqrt(2*3.14*np.linalg.det(covariance_mat)))


def znk(gamma,clusters,pik,epsilon):
	gamma=np.zeros((np.size(trainall,0),k))
	for i in range(np.size(trainall,0)):
		for j in range(k):
			gamma[i][j]=pik[j]*density(trainall[i],clusters[j],epsilon[j])
			divide=0
			for z in range(k):
				divide=divide + pik[z]*density(trainall[i],clusters[z],epsilon[z])
			gamma[i][j]=gamma[i][j]/divide
	return gamma
    


def covariance(gamma,clusters,pik,epsilon):
	# np.delete(epsilon)
	epsilon=[]
	for i in range(k):
		numerator=np.zeros((trainall[0].size,trainall[0].size))
		# numerator=[]
		# print(numerator)
		for j in range(np.size(trainall,0)):
			# print(clusters[i])
			# print(trainall[j])
			# print(train)
			# matmul=0
			mat1=np.zeros((trainall[0].size,trainall[0].size))
			mat2=np.zeros((trainall[0].size,trainall[0].size))
			for q in range(trainall[0].size):
				mat1[0][q]=trainall[j][q]-clusters[i][q]
			for q in range(trainall[0].size):
				mat2[q][0]=trainall[j][q]-clusters[i][q]

			numerator=numerator+  gamma[j][i]*np.matmul(mat2,mat1)
			
			# for q in range(trainall[0].size):
			# 	numerator[q][q]=numerator[q][q]+ gamma[j][i]*(trainall[j][q]-clusters[i][q])**2
			# numerator = numerator*gamma[j][i]
			# print(((trainall[j]-clusters[i]),np.transpose(trainall[j]-clusters[i])))
			# numerator = numerator + [gamma[j][i]*np.matmul((trainall[j]-clusters[i]),np.transpose(trainall[j]-clusters[i]))]
			# print(gamma[j][i]*np.matmul((trainall[j]-clusters[i]),np.transpose(trainall[j]-clusters[i])))
		denominator=0
		for q in range(trainall[0].size):
			if(numerator[q][q]==0):
				numerator[q][q]=0.0001
		numerator=np.diag(np.diag(numerator))
		print(numerator)
		# print("\n\n\n\n")
		for j in range(np.size(trainall,0)):
			denominator= denominator + gamma[j][i]
		
		# print(denominator)	
		epsilon = epsilon + [numerator/denominator]
		

	return epsilon
	

def pi(gamma,clusters,pik,epsilon):
	# np.delete(pi)
	pik=np.zeros(k)
	for j in range(k):
		for i in range(np.size(trainall,0)):
			pik[j] = pik[j] + gamma[i][j]
		pik[j] = pik[j]/(np.size(trainall,0))
	return pik
	

def clustercenters(gamma,clusters,pik,epsilon):
	# np.delete(clusters)
	clusters=np.zeros((k,trainall[0].size))
	for j in range(k):
		for i in range(np.size(trainall,0)):
			clusters[j] = clusters[j]+gamma[i][j]*trainall[i]
		denominator=0
		for i in range(np.size(trainall,0)):
			denominator = denominator + gamma[i][j]
		if(denominator!=0):
			clusters[j] = clusters[j]/denominator
	return clusters
	
    
def l(gamma,clusters,pik,epsilon):
	cost=0
	for i in range(np.size(trainall,0)):
		cost1=0
		for j in range(k):
			likely= density(trainall[i],clusters[j],epsilon[j])
			cost1= cost1+ pik[j]*likely
		cost=cost+math.log(cost1)

	return cost

def gmm(gamma,clusters,pik,epsilon):
	costf=l(gamma,clusters,pik,epsilon)
	# print("Cost chahiye:",costf)
	gamma=znk(gamma,clusters,pik,epsilon)

	while(1):
		



		gamma=znk(gamma,clusters,pik,epsilon)
		# print(gamma,"gamma le")
		clusters= clustercenters(gamma,clusters,pik,epsilon)
		# print(clusters)
		epsilon= covariance(gamma,clusters,pik,epsilon)
		# print(epsilon)
		pik=pi(gamma,clusters,pik,epsilon)
		# print(pik)


		# print(gamma,"gamma le")
		
		
		

		cost=l(gamma,clusters,pik,epsilon)
		# print(clusters)
		if(costf-cost>=0.000001):
			costf=cost
		else:
			return gamma,clusters,pik,epsilon

# -------------------------------code-----------------------------------------
clusters,clustercount,zlabel=em(clusters,costf)
pik=np.zeros(k)
gamma=np.zeros((np.size(trainall,0),k))
epsilon=[]


for i in range(k):
	pik[i]=clustercount[i]/np.sum(clustercount)

for i in range(k):
	bin=[]
	for j in range(np.size(trainall,0)):
		if zlabel[j]==i+1:
			bin = bin + [np.array(trainall[j])]
	if(bin!=[]):
		df = pd.DataFrame(bin)
		matdf=np.diag(np.diag(df.cov()))
		for q in range(trainall[0].size):
			if(matdf[q][q]==0):
				matdf[q][q]=0.003
			

		epsilon = epsilon + [matdf]
		# print(matdf)
		# print(np.diag(np.diag(df.cov())))

		# print("\n\n\n\n")
		# print(df.cov())
	else :
		print("Caught again!!",i)
		mat1=np.zeros((trainall[0].size,trainall[0].size))
		for q in range(trainall[0].size):
			mat1[q][q]=0.5
		epsilon=epsilon + [mat1]
	

for i in range(np.size(trainall,0)):
	for j in range(k):
		gamma[i][j]=pik[j]*density(trainall[i],clusters[j],epsilon[j])
		divide=0
		for z in range(k):
			divide=divide + pik[z]*density(trainall[i],clusters[z],epsilon[z])
		gamma[i][j]=gamma[i][j]/divide

# print(pik)
# print(gamma)
gamma,clusters,pik,epsilon=gmm(gamma,clusters,pik,epsilon)
print(gamma)
print(clusters)
print(epsilon)
print(pik)
print(sum(pik))


# xc1 = [x[0] for x in trainlsc1] 
# yc1 = [x[1] for x in trainlsc1]
# # xc2 = [x[0] for x in trainlsc2] 
# # yc2 = [x[1] for x in trainlsc2]
# # xc3 = [x[0] for x in trainlsc3] 
# # yc3 = [x[1] for x in trainlsc3]


# xc4 = [x[0] for x in clusters] 
# yc4 = [x[1] for x in clusters]

# plt.scatter(xc1,yc1,label='Class 2- decision surface')
# # plt.scatter(xc2,yc2,label='Class 2- decision surface')
# # plt.scatter(xc3,yc3,label='Class 2- decision surface')
# plt.scatter(xc4,yc4,label='Class 2- decision surface')
# plt.show()

np.savetxt("./junk/realworld_1geight.txt",gamma,delimiter=" ")
np.savetxt("./junk/realworld_1ceight.txt",clusters,delimiter=" ")
np.savetxt("./junk/realworld_1peight.txt",pik,delimiter=" ")
np.save("./junk/realworld_1seight.npy",epsilon,allow_pickle=True)
# np.savetxt("nonlinear_1proir.txt",epsilon,allow_pickle=True)


# np.savetxt("nonlinear_1.txt",clusters,delimiter=" ")
# np.savetxt("nonlinear_1.txt",pik,delimiter=" ")
# np.savetxt("nonlinear_1.txt",epsilon,delimiter=" ")
