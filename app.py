import os
import face_recognition
import numpy as np
import cv2
from datetime import datetime
from playsound import playsound
from flask import Flask, render_template, Response

# Loading images, splitting name and their extension, encode images is done first
# let's provide a path for a file we are going to fetch from the images path

path = 'ImageFiles'
img = []
Names = []
ListOfImageNames = os.listdir(path)
for datas in ListOfImageNames:
    CurrentFrame = cv2.imread(f'{path}/{datas}')
    img.append(CurrentFrame)
    Names.append(os.path.splitext(datas)[0])  # this will split the name of image and take the first element

# the next step is to encode our image to make it understandable for our compiler using def
BGRtoRGB = cv2.COLOR_BGR2RGB


def EncodeImages(img):
    encodeList = []

    for im in img:
        im = cv2.cvtColor(im, BGRtoRGB)  # this is because facerecognition uses RGB rather than BGR
        EncodeFace = face_recognition.face_encodings(im)[0]
        encodeList.append(EncodeFace)
    return encodeList


def SaveAttendance(name):
    with open('AttendanceSheet.csv', 'r+') as sheet:
        DataList = sheet.readlines()
        NameLst = []
        for Line in DataList:
            currentEntry = Line.split(',')
            NameLst.append(currentEntry[0])
        if name not in NameLst:
            Now = datetime.now()
            TimeFormat = Now.strftime('%H:%M:%S')
            sheet.writelines(f'\n{name},{TimeFormat}')


print("encode started...")
EncList_Known = EncodeImages(img)
print("Encoded successfully")
# Let's Capture video frames from our webcam
Vediocap = cv2.VideoCapture(0)  # default camera is 0

app = Flask(__name__)  # Flask application instance

def generate_frames():
    while True:
        suc, imag = Vediocap.read()
        if not suc:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', imag)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # speed the process we need to scale down the image size
        ImageSize = cv2.resize(imag, (0, 0), None, 0.25, 0.25)
        ImageSize = cv2.cvtColor(ImageSize, BGRtoRGB)
        FacesOnCurrentFrame = face_recognition.face_locations(ImageSize)
        EncodeCurrentFace = face_recognition.face_encodings(ImageSize, FacesOnCurrentFrame)
        for EncodeFace, FaceLocation in zip(EncodeCurrentFace, FacesOnCurrentFrame):
            Compute = face_recognition.compare_faces(EncList_Known, EncodeFace)
            faceDistance = face_recognition.face_distance(EncList_Known, EncodeFace)
            print(FaceLocation)
            ComputeIndex = np.argmin(faceDistance)
            if Compute[ComputeIndex]:
                name = Names[ComputeIndex].upper()
                print(name)
                y1, x2, y2, x1 = FaceLocation

                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

                print(y1, x2, y2, x1)
                # cv2.rectangle(imag, (x1, y1), (x2, y2), (255, 255, 50), 2)
                # cv2.putText(imag, name, (x1 + 6, y2 + 23), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 100), 3)
                SaveAttendance(name)
            else:

                print("Unknown ")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == "__main__":
    app.run(debug=True)
