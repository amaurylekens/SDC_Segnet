from segnet import Segnet
from compute_output_img import compute_output_img
import cv2
import socketio
import eventlet
import base64
from PIL import Image
from io import BytesIO
import numpy as np
import scipy



def live_segmentation():

	while True:
		image = None

		h = image.shape[0]
		w = image.shape[1]

		prediction = segnet.predict()
		image = plot_result(prediction, (h,w), 2)

		cv2.imwrite("stream.jpg", image)


# event sent by the simulator
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])
        # The current throttle of the car, how hard to push peddle
        throttle = float(data["throttle"])
        # The current speed of the car
        speed = float(data["speed"])
        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image = np.asarray(image)
        image = cv2.resize(image, (256, 256))

		h = image.shape[0]
		w = image.shape[1]

		prediction = segnet.predict()
		image_output = compute_output_img(prediction, (h,w), 2)

	    cv2.imshow('frame',image_output)
    	
    	if cv2.waitKey(1) & 0xFF == ord('q'):
        	break


# event fired when simulator connect
@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send(0, 0)


# wrap with a WSGI application
app = socketio.WSGIApp(sio)

# create segnet
segnet = Segnet()


if __name__ == '__main__':
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

