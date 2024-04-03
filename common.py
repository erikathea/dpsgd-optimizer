import numpy as np
import tensorflow as tf
import collections
import time
import matplotlib.pyplot as plt
from dpsgd.accountant import *
from dpsgd.sanitizer import *
from pathlib import Path

ROOT_DIR = Path(__file__).absolute().parent.parent
MODELS_DIR = Path.joinpath(ROOT_DIR, 'models')
RESULTS_DIR = Path.joinpath(ROOT_DIR, 'results')

EpsDelta = collections.namedtuple("EpsDelta", ["spent_eps", "spent_delta"])
BATCH_SIZE = 64
LEARNING_RATE = 0.01
L2NORM_BOUND = 4.0
SIGMA = 4.0
DATASET = 'mnist'
MODEL_TYPE = 'dense'
USE_PRIVACY = True
PLOT_RESULTS = True
N_EPOCHS = 200

def load_mnist(image_size=28):
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], image_size, image_size, 1)
    X_test = X_test.reshape(X_test.shape[0], image_size, image_size, 1)
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    return X_train, y_train, X_test, y_test
        
def load_cifar10(image_size=32):
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    X_train = X_train.reshape(X_train.shape[0], image_size, image_size, 3)
    X_test = X_test.reshape(X_test.shape[0], image_size, image_size, 3)
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    return X_train, y_train, X_test, y_test

def shuffle_split_data(X, y):
    arr_rand = np.random.rand(X.shape[0])
    split = arr_rand < np.percentile(arr_rand, 70)
    X_train = X[split]
    y_train = y[split]
    X_test =  X[~split]
    y_test = y[~split]
    return X_train, y_train, X_test, y_test
 
def make_dense_model(input_shape, units, num_classes):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=input_shape))
    model.add(tf.keras.layers.Dense(units, activation='relu'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    return model

def make_cnn_model(input_shape, num_classes):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform',
                                     input_shape=input_shape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    return model

def random_batch(X, y, batch_size=64):
    idx = np.random.randint(len(X), size=batch_size)
    return X[idx], y[idx]

def print_status_bar(iteration, total, loss, time_taken, metrics=None, spent_eps_delta=None):
    metrics = " - ".join(["{}: {:.4f}".format(m.name, m.result())
                          for m in [loss] + (metrics or [])])
    end = "" if iteration < total else "\n"
    if spent_eps_delta:
        spent_eps = spent_eps_delta.spent_eps
        spent_delta = spent_eps_delta.spent_delta
        print("\r{}/{} - ".format(iteration, total) + metrics + " - spent eps: " +
               f"{spent_eps:.4f}" + " - spent delta: " + f"{spent_delta:.8f}"
               " - time spent: " + f"{time_taken}" "\n", end=end)
    else:
        print("\r{}/{} - ".format(iteration, total) + metrics + " - spent eps: " +
              " - time spent: " + f"{time_taken}" "\n", end=end)
