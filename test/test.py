import sys
import cv2
import os
import glob
sys.path.append("..")
from segnet import Segnet
from compute_output_img import compute_output_img
import numpy as np


files = [f for f in glob.glob("IMG/*.jpg")]
files = sorted(files)

print(files)
segnet = Segnet()
segnet.load_weight()

#image_ids = [919]
image_id = 0
for file in files:
	image = cv2.imread(file)
	image = cv2.resize(image,(256,256))
	cv2.imwrite("image/" + str(image_id) + ".jpg", image)

	images = np.array([image])

	output = segnet.predict(images)

	print(output)

	image_output = compute_output_img(output, (256,256), 2)

	cv2.imwrite("predict/" + str(image_id) + ".jpg", image_output)
	image_id += 1
