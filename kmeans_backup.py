import numpy as np
import matplotlib.pyplot as plt
import math
import sys

trainlsc1=np.loadtxt('./ls/class1_train.txt')
test1=np.loadtxt('./ls/class1_test.txt')
trainlsc2=np.loadtxt('./ls/class2_train.txt')
test2=np.loadtxt('./ls/class2_test.txt')
trainlsc3=np.loadtxt('./ls/class3_test.txt')
test3=np.loadtxt('./ls/class3_test.txt')


xs1 = [x[0] for x in trainlsc1]
ys1 = [x[1] for x in trainlsc1]
plt.scatter(xs1, ys1)

xs2 = [x[0] for x in trainlsc2]
ys2 = [x[1] for x in trainlsc2]
plt.scatter(xs2, ys2)

xs3 = [x[0] for x in trainlsc3]
ys3 = [x[1] for x in trainlsc3]
plt.scatter(xs3, ys3)

data = np.concatenate((trainlsc1,trainlsc2),axis=0)
data = np.concatenate((trainlsc3,data),axis=0)
from sklearn.cluster import KMeans

# df = pd.

kmeans = KMeans(n_clusters=3)
kmeans.fit(data)
print(kmeans)

labels = kmeans.predict(data)

centroids = kmeans.cluster_centers_

print(centroids)

cx =[x[0] for x in centroids]
cy =[x[1] for x in centroids]
plt.scatter(cx,cy)
plt.show()