import sys
import cv2
sys.path.append("..")
from segnet import Segnet
from compute_output_img import compute_output_img
import numpy as np

segnet = Segnet()

image_ids = [412]

for image_id in image_ids:
	image = cv2.imread(str(image_id) + ".jpg")
	image = cv2.resize(image,(256,256))
	cv2.imwrite(str(image_id) + ".jpg", image)

	images = np.array([image])

	output = segnet.predict(images)

	print(output)

	image_output = compute_output_img(output, (256,256), 2)

	cv2.imwrite("predict/" + str(image_id) + ".jpg", image_output)
