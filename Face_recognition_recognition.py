import os
import cv2
import numpy as np
import tensorflow
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt


def rlsb():
    model = tensorflow.keras.models.load_model(r"./model/face_recognition_model.h5")

    label = os.listdir("./photo/data")


    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    model.class_indices = {i: label[i] for i in range(len(label))}

    face_cascade = cv2.CascadeClassifier( './haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)


    y_true = []
    y_scores = []

    while True:

        ret, frame = cap.read()

        gray_face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))


        for (x, y, w, h) in faces:

            face_roi = frame[y:y + h, x:x + w]


            resized_face = cv2.resize(face_roi, (224,224))


            normalized_face = resized_face / 255.0


            input_face = np.expand_dims(normalized_face, axis=0)


            predictions = model.predict(input_face)


            predicted_label_index = np.argmax(predictions)
            confidence = predictions[0][predicted_label_index]


            y_true.append(predicted_label_index)
            y_scores.append(confidence)


            confidence_threshold = 0.5
            if predicted_label_index < 0 or predicted_label_index >= len(label) or confidence < confidence_threshold:
                predicted_label = "Unknown"
            else:
                predicted_label = label[predicted_label_index]


            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            text = f"{predicted_label}: {confidence:.2f}"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)


        cv2.imshow('Face Recognition', frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



    evaluate_model(y_true, y_scores)



def evaluate_model(y_true, y_scores):

    fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)


    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)


    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")


    plt.subplot(1, 2, 2)
    plt.step(recall, precision, color='red', alpha=0.8, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='red')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve: AP={0:0.2f}'.format(avg_precision))


    plt.tight_layout()


    plt.savefig('curves.png')


    plt.show()

# rlsb()



