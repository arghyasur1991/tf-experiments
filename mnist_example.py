from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
# tf.config.set_visible_devices([], 'GPU')


def train_mnist_model(data, labels):
    # tf.debugging.set_log_device_placement(True)
    model = keras.Sequential([
        layers.Dense(512, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(optimizer="rmsprop",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.fit(data, labels, epochs=5, batch_size=128)
    return model


def load_and_process_mnist_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype("float32") / 255
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype("float32") / 255

    return (train_images, train_labels), (test_images, test_labels)


def load_and_train_mnist():
    (train_images, train_labels), (test_images, test_labels) = load_and_process_mnist_data()
    model = train_mnist_model(train_images, train_labels)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"test_acc: {test_acc}")
