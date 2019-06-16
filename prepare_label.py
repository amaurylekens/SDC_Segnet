import numpy as np
import cv2
import glob
import sys

def construct_image_label(): 
	
	path = './training/'

	files = [f for f in glob.glob(path + 'label1/*.jpg')]

	for f in files:
		file_name = f.replace('/training/label1/','')
		file_name = file_name.replace('.','')
		file_name = file_name.replace('jpg','.jpg')
		print(file_name)
		image = cv2.imread(f)
		h = image.shape[0]
		w = image.shape[1]
		label = np.zeros((h,w))

		for i in range(h):
			for j in range(w):
				if image[i,j][0] > 150 and image[i,j][1] < 25:
					label[i,j] = 0
				else:
					label[i,j] = 255

		cv2.imwrite("training/label/" + str(file_name), label)


def prepare_label(image_label, n_labels):

	h = image_label.shape[0]
	w = image_label.shape[1]
	
	label = np.zeros([h, w, n_labels], dtype=np.uint8)    
	for i in range(h):
		for j in range(w):
			if np.sum(image_label[i,j]) < 25:
				label[i, j, 0] = 1
			else:
				label[i, j, 1] = 1

	return label

def prep_data(size, n,n_labels):

	data = []
	label = []

	h = size[0]
	w = size[1]

	files = [f for f in glob.glob('./training/img/*.jpg')]
	files = sorted(files)

	for i in range(n):
		f = files[i]
		file_name = f.replace('/training/img/','').replace('.','').replace('jpg','.jpg')
		#print(f)
		img = cv2.imread(f)
		img = cv2.resize(img, (h, w))
		#print('./training/label/' + file_name)
		img_label = cv2.imread('./training/label/' + file_name, cv2.IMREAD_GRAYSCALE)
		img_label = cv2.resize(img_label, (h, w))
		
		data.append(img)
		label.append(prepare_label(img_label, n_labels))
		
		sys.stdout.write('\r')
		sys.stdout.write("data loading: {} %".format(str(i*100/n)))
		sys.stdout.flush()

	sys.stdout.write('\r')
	sys.stdout.flush()
	data, label = np.array(data), np.array(label).reshape((n, h * w, n_labels))

	print('Data loaded')
	print('\tshapes: {}, {}'.format(data.shape, label.shape))
	print('\ttypes:  {}, {}'.format(data.dtype, label.dtype))
	print('\tmemory: {}, {} MB'.format(data.nbytes / 1048576, label.nbytes / 1048576))

	return data, label




