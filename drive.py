# parsing command line arguments
# decoding camera images
import base64
# reading and writing files
# high level file operations
# for frame timestamp saving
# input output
from io import BytesIO

# concurrent networking
# web server gateway interface
import eventlet.wsgi
# matrix math
import numpy as np
# real-time server
import socketio
# image manipulation
from PIL import Image
# web framework
from flask import Flask
# load our saved model
from tensorflow.keras.models import load_model

# helper class
import utils.utils as ut

# initialize our server
sio = socketio.Server()
# our flask (web) app
app = Flask(__name__)
# init our model and image array as empty
model = None
prev_image_array = None

# set min/max speed for our autonomous car
MAX_SPEED = 25
MIN_SPEED = 10

# and a speed limit
speed_limit = MAX_SPEED


# registering event handler for the server
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
        try:
            image = np.asarray(image)  # from PIL image to numpy array
            image = ut.pre_process(image)  # apply the preprocessing
            image = np.array([image])  # the model expects 4D array

            # predict the steering angle for the image
            steering_angle = float(model.predict(image, batch_size=1))
            # lower the throttle as the speed increases
            # if the speed is above the current speed limit, we are on a downhill.
            # make sure we slow down first and then go back to the original max speed.
            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED  # slow down
            else:
                speed_limit = MAX_SPEED
            throttle = 1.0 - steering_angle ** 2 - (speed / speed_limit) ** 2
            # throttle = 10
            print('{} {} {}'.format(steering_angle, throttle, speed))
            send_control(steering_angle, throttle)
        except Exception as e:
            print(e)
    else:
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':

    model = load_model("./model/model.h5")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
