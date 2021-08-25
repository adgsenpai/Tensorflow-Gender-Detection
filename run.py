from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv
from flask import Flask
from flask import render_template
from flask import Response
app = Flask(__name__)
model = load_model('gender_detection.model')
classes = ['Male', 'Female']

# for demo rtsp camera = cv2.VideoCapture('rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov') 
# for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)

def gen_frames():  # generate frame by frame from camera
    camera = cv2.VideoCapture('jpnwalk.mp4')  
    try:
        while True:
            # Capture frame-by-frame
            success, frame = camera.read()  # read the camera frame

            face, confidence = cv.detect_face(frame)
        

            # loop through detected faces
            for idx, f in enumerate(face):
                # get corner points of face rectangle
                (startX, startY) = f[0], f[1]
                (endX, endY) = f[2], f[3]

                # draw rectangle over face
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

                # crop the detected face region
                face_crop = np.copy(frame[startY:endY, startX:endX])

                if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
                    continue

                # preprocessing for gender detection model
                face_crop = cv2.resize(face_crop, (96, 96))
                face_crop = face_crop.astype("float") / 255.0
                face_crop = img_to_array(face_crop)
                face_crop = np.expand_dims(face_crop, axis=0)

                # apply gender detection on face
                # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]
                conf = model.predict(face_crop)[0]

                # get label with max accuracy
                idx = np.argmax(conf)
                label = classes[idx]

                label = "{}: {:.2f}%".format(label, conf[idx] * 100)

                Y = startY - 10 if startY - 10 > 10 else startY + 10

                # write label and confidence above face rectangle
                cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

            if not success:
                break
            else:
                
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
    except:
     gen_frames()  

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
  
    return render_template('index.html')

@app.route('/malefemaledetector')
def malefemaledetector():
    return render_template('app.html')

if __name__ == '__main__':
  app.run('localhost',8000)