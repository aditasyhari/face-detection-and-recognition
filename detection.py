import efficientnet.tfkeras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
import cv2
import os
import numpy as np
from PIL import Image
from math import *

input_size = (224, 224)
channel = (3,)
input_shape = input_size + channel

labels = ["Aditya R", "Cristiano Ronaldo", "Lionel Messi", "Paulo Dybala", "Sergio Aguero"]

def preprocess(img, input_size):
    image = Image.fromarray(img)
    nimg = image.convert('RGB').resize(input_size, resample = 0)
    img_arr = (np.array(nimg)) / 255
    return img_arr

def reshape(imgs_arr):
    return np.stack(imgs_arr, axis = 0)

model_path = 'MobileNetV2_m3_e50.h5'
model = load_model(model_path, compile = False)


face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade.xml')

key = cv2.waitKey(1)
webcam = cv2.VideoCapture(0)

while True:
    try:
        check, frame = webcam.read()
        frame = cv2.flip(frame, 1)
        # print(check)
        print(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,     
            scaleFactor=1.1,
            minNeighbors=5,     
            minSize=(30, 30)
        )

        color = (0, 255, 0)

        if len(faces) != 0:
            for coords in faces:
                x, y, w, h = coords

                H, W, _ = frame.shape

                # mencari koordinate wajah
                X_1, X_2 = (max(0, x - int(w * 0.35)), min(y + int(1.35 * w), W))
                Y_1, Y_2 = (max(0, y - int(0.35 * h)), min(y + int(1.35 * h), H))

                # ========================================================
                #                       Predict
                # ========================================================

                img_cp = frame[y:y+h, x:x+w].copy()
                # img_cp = frame[Y_1:Y_2, X_1:X_2].copy()
                

                img_cp1 = cv2.cvtColor(img_cp, cv2.COLOR_BGR2RGB)

                # data = cv2.resize(img_cp1, input_size)
                # data = np.array(data, dtype=np.float32)
                # data = np.expand_dims(data, axis=0)
                # data = data/255

                data = preprocess(img_cp1, input_size)
                data = reshape([data])
                L = model.predict(data)

                predictedLabel = labels[np.argmax(L)]
                predictedAcc = np.max(L)*100
                acc = floor(predictedAcc)

                # =======================================================

                cv2.rectangle(frame,(x,y),(x+w, y+h),(255,0,0),2)
                cv2.putText(frame, predictedLabel+" "+str(acc)+"%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
                # roi_gray = gray[y:y+h, x:x+w]
                # roi_color = frame[y:y+h, x:x+w]

        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            print("Mematikan Kamera...")
            webcam.release()
            print("Kamera sudah mati. Program Selesai !")
            cv2.destroyAllWindows()
            break

    except(KeyboardInterrupt):
        print("Mematikan Kamera...")
        webcam.release()
        print("Kamera sudah mati. Program Selesai !")
        cv2.destroyAllWindows()
        break