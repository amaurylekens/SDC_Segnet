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


sio = socketio.Server()

# event sent by the simulator
@sio.on('telemetry')
def telemetry(sid, data):
	if data:
		print("round")
		# The current image from the center camera of the car
		image = Image.open(BytesIO(base64.b64decode(data["image"])))
		image = np.asarray(image)
		image = cv2.resize(image, (256, 256))

		h = image.shape[0]
		w = image.shape[1]

		images = np.array([image])
		print("begin prediction")
		prediction = segnet.predict(images)
		image_output = compute_output_img(prediction, (h,w), 2)
		print("end prediction")
		cv2.imwrite('frame.png',image_output)

		send(0,0)

	else:
		# Edge case
		sio.emit('manual', data={}, skip_sid=True)


# event fired when simulator connect
@sio.on('connect')
def connect(sid, environ):
	print("connect ", sid)
	send(0,0)

def send(steer, throttle):
	print("steer : {0}".format(steer*25))
	sio.emit("steer", data={'steering_angle': str(steer), 'throttle': str(throttle)}, skip_sid=True)


# wrap with a WSGI application
app = socketio.WSGIApp(sio)

# create segnet
segnet = Segnet()
segnet.load_weight()


if __name__ == '__main__':
	eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

