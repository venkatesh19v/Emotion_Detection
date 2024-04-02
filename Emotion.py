import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

new_model = tf.keras.models.load_model('emo_model_35e.h5')

frame = cv2.imread("./Emotions_Img_DS/Disgust/Disgust_23.jpg")
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = faceCascade.detectMultiScale(gray, 1.1, 4)

for x, y, w, h in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = frame[y:y+h, x:x+w]
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    facess = faceCascade.detectMultiScale(roi_gray)
    
if len(facess) == 0:
    print("Face not detected")
else:
    for (ex, ey, ew, eh) in facess:
        face_roi = roi_color[ey: ey+eh, ex:ex + ew]
if 'face_roi' in locals():
    final_image = cv2.resize(face_roi, (224, 224))
    final_image = np.expand_dims(final_image, axis=0)
    final_image = final_image / 255.0

    Predictions = new_model.predict(final_image)
    emotion_label = np.argmax(Predictions)
    labels = ["angry","disgust","fear","happy","neutral","sad","surprise"]
    print(emotion_label)
    print(labels[emotion_label])

else:
    print("face_roi not detected")
