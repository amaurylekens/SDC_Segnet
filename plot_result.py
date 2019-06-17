import numpy as np
import cv2 
from prepare_label import prepare_label

def plot_result(result, size, n_labels):
	
	h = size[0]
	w = size[1]

	result = result.reshape((h, w, n_labels))
	print(result)
	label = np.zeros([h, w], dtype=np.uint8) 

	for i in range(h):
		for j in range(w):
			if result[i,j,0] < 0.5:
				label[i,j] = 255
			else:
				label[i,j] = 0 

	cv2.imwrite("img_result.png", label)


#img_label = cv2.imread('img.jpg', cv2.IMREAD_GRAYSCALE)
#label = prepare_label(img_label, 2)
#h = label.shape[0]
#w = label.shape[1]
#label = np.array(label).reshape((h * w, 2))
plot_result(label, (h,w), 2)
