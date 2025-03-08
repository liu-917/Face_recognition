import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import os
import matplotlib.pyplot as plt

def train():

    def create_model(input_shape, num_classes):
        model = tf.keras.Sequential([

            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),


            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),


            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),


            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),


            layers.Conv2D(512, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),


            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        return model


    loss_fn = SparseCategoricalCrossentropy()
    optimizer = Adam(learning_rate=0.001)


    train_data_dir = r"./photo/new_data/train_data_dir"
    val_data_dir = r"./photo/new_data/val_data_dir"
    test_data_dir = r"./photo/new_data/test_data_dir"


    input_shape = (224, 224, 3)
    num_classes = len(os.listdir(train_data_dir))


    model = create_model(input_shape, num_classes)


    model.compile(optimizer=optimizer,
                  loss=loss_fn,
                  metrics=['accuracy'])


    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(input_shape[0], input_shape[1]),
        batch_size=32,
        class_mode='sparse')

    val_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(input_shape[0], input_shape[1]),
        batch_size=32,
        class_mode='sparse')

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(input_shape[0], input_shape[1]),
        batch_size=32,
        class_mode='sparse')

    epochs = 10
    min_test_accuracy = 0.90


    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator)


    test_loss, test_acc = model.evaluate(test_generator)
    print("Test accuracy:", test_acc)


    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()


    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()


    plt.savefig(r"./result/training_validation_metrics.png")
    plt.show()


    if test_acc >= min_test_accuracy:
        model_dir = r"./model"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model.save(os.path.join(model_dir, "face_recognition_model.h5"))
        print('The model is saved！')
    else:
        print('The accuracy of the model did not reach 90%。')


# train()
