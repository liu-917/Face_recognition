
import time
import cv2
import os
from PIL import Image
import random
import numpy as np
import shutil


def entry():
    def split_data(source_dir, train_dir, val_dir, test_dir, split_ratio=(0.7, 0.2, 0.1)):

        for directory in [train_dir, val_dir, test_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)


        files = os.listdir(source_dir)
        random.shuffle(files)


        num_files = len(files)
        train_split = int(num_files * split_ratio[0])
        val_split = int(num_files * (split_ratio[0] + split_ratio[1]))


        train_files = files[:train_split]
        val_files = files[train_split:val_split]
        test_files = files[val_split:]


        for file in train_files:
            shutil.copy(os.path.join(source_dir, file), os.path.join(train_dir, file))

        for file in val_files:
            shutil.copy(os.path.join(source_dir, file), os.path.join(val_dir, file))

        for file in test_files:
            shutil.copy(os.path.join(source_dir, file), os.path.join(test_dir, file))

        name = os.listdir('./photo/data')
        for i in name:
            repath=os.path.dirname(os.path.dirname(train_dir))+'\\'+i
            if repath:
                shutil.rmtree(repath)

    def yuchuli(dataset_dir,output_dir):

        input_size = (224, 224)


        rotation_range = 20
        width_shift_range = 0.1
        height_shift_range = 0.1


        os.makedirs(output_dir, exist_ok=True)


        for label in os.listdir(dataset_dir):
            label_dir = os.path.join(dataset_dir, label)
            if os.path.isdir(label_dir):

                output_label_dir = os.path.join(output_dir, label)
                os.makedirs(output_label_dir, exist_ok=True)


                for file_name in os.listdir(label_dir):

                    image_path = os.path.join(label_dir, file_name)
                    image = Image.open(image_path)


                    output_path = os.path.join(output_label_dir, file_name)

                    image = image.resize(input_size)
                    Image.fromarray((np.array(image) / 255.0 * 255).astype(np.uint8)).save(output_path)


                    angle = random.uniform(-rotation_range, rotation_range)
                    image = image.rotate(angle)


                    width_shift = random.uniform(-width_shift_range, width_shift_range) * input_size[0]
                    height_shift = random.uniform(-height_shift_range, height_shift_range) * input_size[1]
                    image = image.transform(input_size, Image.AFFINE, (1, 0, width_shift, 0, 1, height_shift))


                    image = np.array(image) / 255.0

                    file_name1 = '_' + file_name
                    output_path1 = os.path.join(output_label_dir, file_name1)
                    Image.fromarray((image * 255).astype(np.uint8)).save(output_path1)





    def collect_faces(name, num_samples):

        face_cascade = cv2.CascadeClassifier( './haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')
        ear_cascade = cv2.CascadeClassifier('./haarcascade_lefteye_2splits.xml')
        mouth_cascade = cv2.CascadeClassifier('./haarcascade_smile.xml')
        cap = cv2.VideoCapture(0)

        print(f"Start the collection {name} Images of human faces...")

        path=fr"./photo/data/{name}"

        file_list = os.listdir(path)

        image_files = [file for file in file_list]

        count = len(image_files)
        number=count
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi_gray = frame[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]
                if count < number + int(num_samples*(1/4)):
                    face_img = gray[y:y + h, x:x + w]
                    cv2.imwrite(fr"./photo/data/{name}/" + f'{name}_{count}.jpg', face_img)
                    count += 1

                eyes = eye_cascade.detectMultiScale(roi_gray)
                if number + int(num_samples*(1/4))<=count<number+int(num_samples*(2/4)):
                    for (ex, ey, ew, eh) in eyes:

                        text = f"please to blink your's eyes"
                        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                        if count == number + int(num_samples * (1 / 4)):
                            text = f"please to blink your's eyes"
                            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                        if count < number+num_samples:

                            face_img = gray[y:y + h, x:x + w]

                            cv2.imwrite(fr"./photo/data/{name}/" + f'{name}_{count}.jpg', face_img)
                            count += 1



                ear = ear_cascade.detectMultiScale(roi_gray)
                if number+int(num_samples*(2/4))<=count < number + int(num_samples * (3 / 4)):
                    for (nx, ny, nw, nh) in ear:

                        text = f"please Shake your head"
                        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                        if count == number + int(num_samples * (2 / 4)):
                            text = f"please Shake your head"
                            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                        if count < number+num_samples:
                            face_img = gray[y:y + h, x:x + w]
                            cv2.imwrite(fr"./photo/data/{name}/" + f'{name}_{count}.jpg', face_img)
                            count += 1

                mouths = mouth_cascade.detectMultiScale(roi_gray)
                if number + int(num_samples * (3 / 4))<=count < number + num_samples:
                    for (mx, my, mw, mh) in mouths:

                        text = f"please to open your mouth"
                        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                        if count == number + int(num_samples * (3 / 4)):

                            text = f"please to open your mouth"
                            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)


                        if count < number+num_samples:
                            face_img = gray[y:y + h, x:x + w]
                            cv2.imwrite(fr"./photo/data/{name}/" + f'{name}_{count}.jpg', face_img)
                            count += 1


            cv2.imshow('frame', frame)

            if count >= number+num_samples:
                print(f"{name} The face image acquisition is completed！")
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        time.sleep(1)
        cv2.destroyAllWindows()


    def main():
        name = input("Please enter a name：")
        num_samples = int(input("Please enter the number of faces you want to capture："))

        directories = [
            fr"./photo/data/{name}",
            r"./photo/new_data",
            fr"./photo/new_data/{name}",
            fr"./photo/new_data/train_data_dir/{name}",
            fr"./photo/new_data/val_data_dir/{name}",
            fr"./photo/new_data/test_data_dir/{name}"
        ]

        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)

        collect_faces(name, num_samples)

        dataset_dir = r"./photo/data"
        output_dir = r"./photo/new_data"
        yuchuli(dataset_dir, output_dir)

        source_dir = fr"./photo/new_data/{name}"
        train_data_dir = fr"./photo/new_data/train_data_dir/{name}"
        val_data_dir = fr"./photo/new_data/val_data_dir/{name}"
        test_data_dir = fr"./photo/new_data/test_data_dir/{name}"

        split_data(source_dir, train_data_dir, val_data_dir, test_data_dir)

    main()
    print("Face photos are collected！！！")
# entry()





