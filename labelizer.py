from sklearn.linear_model import LinearRegression
import numpy as np
import cv2
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

file = "1.jpg"


def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged

def plot_regression_line(x,y,img, n_round):

	triggers = [35,15,10,20,35,6,10]


	#plt.scatter(x_selected, y_selected,  color='green')

	n_points = int(len(x)/n_round)
	colors = ["green", "red", "yellow"]

	y_previous_selection = []
	previous_model = None

	x = np.array(x).reshape((-1,1))
	y = np.array(y)

	for round_ in range(0,n_round):
		if round_ == n_round:
			rest = n_points = len(x)%n_round
		else:
			rest = 0

		# take the point of the interval
		y_intervale = y[round_*n_points:((round_+1)*n_points)+rest]
		x_intervale = x[round_*n_points:((round_+1)*n_points)+rest]

		# initialize the selected point
		y_selected = y_intervale
		x_selected = x_intervale

		# make multiple regressions to find the best
		for i in range(0,7):
			# make the regression with the selected points
			x_selected = np.array(x_selected).reshape((-1,1))
			y_selected = np.array(y_selected)
			model_lr = LinearRegression()
			model_lr.fit(x_selected, y_selected)

			# compute the prediction error of the points in the interval
			y_predict = model_lr.predict(x_intervale)
			errors = abs(y_predict - y_intervale)

			# remove the point with a to big error
			y_selected = []
			x_selected = []
			for error, elem_x, elem_y in zip(errors,x_intervale,y_intervale):
				if error < triggers[i]:
					x_selected.append(elem_x[0])
					y_selected.append(elem_y)

		# compute predictions wiht all the points
		y_predict = model_lr.predict(x)

		# compute the intersection between the actual and the previous best regression
		if round_ > 0:
			idx = np.argwhere(np.diff(np.sign(y_predict - y_previous_selection))).flatten()
			if len(idx) != 0:
				index = idx[0]

			else:
				index = len(x_intervale)

			x_round = np.array([65,x[index]]).reshape((-1,1))
			y_round = previous_model.predict(x_round)
			point_a_y = int(y_round[0])
			point_a_x = int(x_round[0][0])
			point_b_y = int(y_round[1])
			point_b_x = int(x_round[1][0])
			cv2.line(img,(point_a_y, point_a_x), (point_b_y, point_b_x),(255,0,0),1)
			#plt.plot(x_round, y_round, color="blue", linewidth=3)

		if round_ == n_round - 1:
			if round_ == 0:
				index = 0

			x_round = np.array([x[index],140]).reshape((-1,1))
			y_round = model_lr.predict(x_round)
			point_a_y = int(y_round[0])
			point_a_x = int(x_round[0][0])
			point_b_y = int(y_round[1])
			point_b_x = int(x_round[1][0])
			cv2.line(img,(point_a_y, point_a_x), (point_b_y, point_b_x),(255,0,0),1)
			#plt.plot(x_round, y_round, color="blue", linewidth=3)


		y_previous_selection = y_predict
		previous_model = model_lr

	return img


def test(file, 
	     middle_ajust,
	     error_max,
	     degree_left,
	     degree_right,
	     top_left_padding,
	     bottom_left_padding, 
	     top_right_padding,
	     bottom_right_padding, 
	     ratio,
	     cells_number):

	image = cv2.imread(str(file) + ".jpg")
	img_final = cv2.imread(str(file) + ".jpg")
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (3, 3), 0)
	cv2.imwrite("training/img/" + str(file) + ".jpg", image)

	# height and width of the image
	h = image.shape[0]
	w = image.shape[1]

	# compute canny edges with a tight and auto treshold
	tight = cv2.Canny(blurred, 225, 250)
	auto = auto_canny(blurred)

	cut_height = int(h*ratio)
	img_top = auto[0:cut_height,:]
	img_bottom = tight[cut_height:h,:]
	img = np.concatenate((img_top, img_bottom), axis=0)	


	# remove the horizon
	for x in range(0, int(2*(h)/5)):
		for y in range(0, w-1):
			img[x,y] = 0

	# remove the party in front of the car
	for x in range(int(5*(h)/6),h):
		for y in range(0, w-1):
			img[x,y] = 0

    # imporove quality
	img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
	img = cv2.fastNlMeansDenoisingColored(img,None,40,40,7,21)
	img = cv2.Canny(img, 225, 250)
	img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)


	# find the best middle
	cv2.line(img,(int(w/2)+middle_ajust, 0),(int(w/2)+middle_ajust, h-1),(0,255,0),1)

	cv2.line(img,(0, int(5*(h)/6) - bottom_left_padding),(int(w/2)+middle_ajust, int(5*(h)/6) - bottom_left_padding),(0,255,255),1)
	cv2.line(img,(int(w/2)+middle_ajust, int(5*(h)/6) - bottom_right_padding),(w-1, int(5*(h)/6) - bottom_right_padding),(0,255,255),1)		
	cv2.line(img,(0, int(2*(h)/5) + top_left_padding),(int(w/2)+middle_ajust, int(2*(h)/5) + top_left_padding),(0,255,0),1)
	cv2.line(img,(int(w/2)+middle_ajust, int(2*(h)/5) + top_right_padding),(w-1, int(2*(h)/5) + top_right_padding),(0,255,0),1)


	# find the first white points in the rigth direction on the lines
	x = []
	y = []	
	differences = []
	i = int(5*(h)/6)
	for i in range(int(2*(h)/5)+top_right_padding,int(5*(h)/6)-bottom_right_padding):
		for j in range(int(w/2)+middle_ajust, w-cells_number):
			white_level_sum = 0
			for number in range(cells_number):
				white_level_sum += np.sum(img[i,j+number])
			white_level_mean = white_level_sum/cells_number
			if white_level_mean > 400:
				img[i,j] = [0,255,255]
				if len(y) > 0:			
					differences.append(abs(y[len(y)-1]-j))
				x.append(i)
				y.append(j)
				break


	#cv2.imwrite("top.jpg", img)
	#plt.show()


	plot_regression_line(x,y,img,1)

	# find the first white points in the left direction on the lines
	x = []
	y = []	
	differences = []
	i = int(5*(h)/6)
	for i in range(int(2*(h)/5)+top_left_padding,int(5*(h)/6)-bottom_left_padding):
		for j in range(int(w/2)+middle_ajust,cells_number,-1):
			white_level_sum = 0
			for number in range(cells_number):
				white_level_sum += np.sum(img[i,j+number])
			white_level_mean = white_level_sum/cells_number
			if white_level_mean > 400:
				img[i,j] = [0,0,255]
				if len(y) > 0:			
					differences.append(abs(y[len(y)-1]-j))
				x.append(i)
				y.append(j)
				break

	plot_regression_line(x,y,img,2)


	# complete the lines

	founded = False
	founded_position = 0
	while not founded:
		if img[founded_position,0][0] > 230 and img[founded_position,0][1] < 10:
			founded = True
			print("found")
		founded_position += 1
		if founded_position == h:
			break

	if founded:
		i = int(5*(h)/6)-5
		while i >= founded_position:
			img[i,0] = [255,0,0]
			i -= 1
	

    # fill the road in blue 
	for i in range(int(2*(h)/5), int(5*(h)/6)):
		color = False
		j = 0
		while j < (w):
			if img[i,j][0] > 120 and img[i,j][1] == 0 and color:
				break
			if ( np.array_equal(img[i,j],[255,0,0]) and not color):
				while np.array_equal(img[i,j],[255,0,0]) and j < w-1:
					#image[i,j] = [255,0,0]
					j += 1
				color = True
			if (j > int(w/2) + middle_ajust and not color):
				break
			if color :
				image[i,j] = [255,0,0]
			j += 1


	cv2.addWeighted(image, 0.5, img_final, 1 - 0.5, 0, img_final)

	# show the images	
	tight = cv2.cvtColor(tight,cv2.COLOR_GRAY2RGB)
	auto = cv2.cvtColor(auto,cv2.COLOR_GRAY2RGB)
	cv2.imwrite("Edges" + str(file) + ".png", np.hstack([tight, auto, img, img_final, image, ]))

	cv2.imwrite("training/label1/" + str(file) + ".jpg", image)
	cv2.imwrite("training/label2/" + str(file) + ".jpg", img_final)




for i in range(1456, 1461):
	test(file = i,
		 middle_ajust = -26,
		 error_max = 5,
		 degree_left = 2,
		 degree_right =2,
		 top_left_padding = 5,
		 bottom_left_padding =0,
		 top_right_padding = 0,
		 bottom_right_padding = 45,
		 ratio = 0.63,
		 cells_number = 3)

"""
def test(file, 
	     middle_ajust,
	     error_max,
	     degree_left,
	     degree_right,
	     top_left_padding,
	     bottom_left_padding, 
	     top_right_padding,
	     bottom_right_padding, 
	     ratio,
	     cells_number):
"""

