import sys
import cv2
sys.path.append("..")
from segnet import Segnet
from compute_output_img import compute_output_img

segnet = Segnet()

image_ids = [1,45]

for image_id in image_ids:
	image = cv2.imread(str(image_id) + ".jpg")

	output = segnet.predict(image)

	image_output = compute_output_img(output)

	cv2.imwrite("predict/" + str(image_id) + ".jpg")
