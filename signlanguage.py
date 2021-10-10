import sys
import tensorflow as tf
import pandas as pandas
import numpy as np

from sklearn.model_selection import train_test_split


def main():

    x_train, x_test, y_train, y_test = load_data()

    model = get_model()

    model.fit(x_train, y_train, epochs=15)

    model.evaluate(x_test,  y_test, verbose=2)

    if len(sys.argv) == 2:
        filename = sys.argv[1]
        model.save(filename)


def load_data():

    data = pandas.read_csv("sign_mnist.csv", header=0)

    set(data['label'].unique())

    x_train = data.drop(['label'], axis=1).values
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_train = x_train.astype('float32')
    x_train /= 255

    y_train = data['label'].values
    y_train = tf.keras.utils.to_categorical(y_train)

    return train_test_split(x_train, y_train, test_size=.4)


def get_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            32, 3, 3, activation="relu", input_shape=(28, 28, 1)
        ),

        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(25, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return(model)


main()