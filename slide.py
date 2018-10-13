import cv2 
import matplotlib.pyplot as plt
import numpy as np 


image = cv2.imread("test.jpg")
tmp = image
stepSize = 32
(w_width,w_height)=(32,32)
print(image.shape)
print(image.size)
print(image.dtype)
count=0
all_data_points=[]
for x in range(0,image.shape[1],stepSize):
	for y in range(0,image.shape[0],stepSize):
		
		# window = image[x:x+w_width,y:y+w_height,:]
		cv2.rectangle(tmp,(x,y),(x+w_width,y+w_height),(0,255,0),2)
		# plt.imshow(np.array(tmp).astype('uint8'))
		count=count+1
		r = [0]*8
		g = [0]*8
		b = [0]*8
		data_point = []
		# if x == image.shape[1]-1 or y==image.shape[0]-1 
		for j in range(x,x+w_width):
			for i in range(y,y+w_height):
				b[((image[i,j,0])//32)] = b[((image[i,j,0])//32)]+1
				g[((image[i,j,1])//32)] = g[((image[i,j,1])//32)]+1
				r[((image[i,j,2])//32)] = r[((image[i,j,2])//32)]+1
		print("r = ",r)
		print("b = ",b)
		print("g = ",g)
		print("count = ",count)

		data_point=b+g+r
		all_data_points.append(data_point)

print(all_data_points)
print("Size of all = ",len(all_data_points))
print(count)
# plt.show()
cv2.imshow('tmp',tmp)
cv2.waitKey(0)
cv2.destroyAllWindows()