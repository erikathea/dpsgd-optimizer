import numpy as np
import tensorflow as tf
import collections
import time
import matplotlib.pyplot as plt
from common import load_mnist, shuffle_split_data, make_dense_model, random_batch, print_status_bar
from dpsgd.accountant import *
from dpsgd.sanitizer import *
from pathlib import Path

ROOT_DIR = Path(__name__).absolute().parent
MODELS_DIR = Path.joinpath(ROOT_DIR, 'log')
RESULTS_DIR = Path.joinpath(ROOT_DIR, 'log')

EpsDelta = collections.namedtuple("EpsDelta", ["spent_eps", "spent_delta"])
BATCH_SIZE = 32
LEARNING_RATE = 0.01
L2NORM_BOUND = 4.0
SIGMA = 4.0
DATASET = 'mnist'
MODEL_TYPE = 'dense'
USE_PRIVACY = True
PLOT_RESULTS = True
N_EPOCHS = 10

def main():
    X_train, y_train, X_test, y_test = load_mnist()
    num_classes = 10
    image_size = 28
    n_channels = 1
                    
    # Create train/valid set
    X_train, y_train, X_valid, y_valid = shuffle_split_data(X_train, y_train)

    model = make_dense_model((image_size, image_size, n_channels),
                                  image_size*image_size, num_classes)
  
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.optimizers.SGD(LEARNING_RATE)

    # Set constants for this loop
    eps = 1.0
    delta = 1e-7
    max_eps = 64.0
    max_delta = 1e-3
    target_eps = [64.0]
    target_delta = [1e-5] #unused
    
    # Create accountant, sanitizer and metrics
    accountant = AmortizedAccountant(len(X_train))
    sanitizer = AmortizedGaussianSanitizer(accountant, [L2NORM_BOUND / BATCH_SIZE, True])
    
    # Setup metrics
    train_mean_loss = tf.keras.metrics.Mean()
    valid_mean_loss = tf.keras.metrics.Mean()
    train_acc_scores, valid_acc_scores = list(), list()
    train_loss_scores, valid_loss_scores = list(), list()
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    valid_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    train_metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    valid_metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    test_metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    
    # Run training loop
    start_time = time.time()
    spent_eps_delta = EpsDelta(0, 0)
    should_terminate = False
    n_steps = len(X_train) // BATCH_SIZE
    for epoch in range(1, N_EPOCHS + 1):
        if should_terminate:
            spent_eps = spent_eps_delta.spent_eps
            spent_delta = spent_eps_delta.spent_delta
            print(f"Used privacy budget for {spent_eps:.4f}" +
                   f" eps, {spent_delta:.8f} delta. Stopping ...")
            break
        print(f"Epoch {epoch}/{N_EPOCHS}")
        for step in range(1, n_steps + 1):
            X_batch, y_batch = random_batch(X_train, y_train)
            with tf.GradientTape() as tape:
                y_pred = model(X_batch, training=True)
                main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
                loss = tf.add_n([main_loss] + model.losses)
                train_acc_metric.update_state(y_batch, y_pred)
            gradients = tape.gradient(loss, model.trainable_variables)
            if USE_PRIVACY:
                sanitized_grads = []
                eps_delta = EpsDelta(eps, delta)
                for px_grad in gradients:
                    sanitized_grad = sanitizer.sanitize(px_grad, eps_delta, SIGMA)
                    sanitized_grads.append(sanitized_grad)
                spent_eps_delta = accountant.get_privacy_spent(target_eps=target_eps)[0]
                optimizer.apply_gradients(zip(sanitized_grads, model.trainable_variables))
                if (spent_eps_delta.spent_eps > max_eps or spent_eps_delta.spent_delta > max_delta):
                    should_terminate = True
            else:
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_mean_loss(loss)
            for metric in train_metrics:
                metric(y_batch, y_pred)
            if step % 200 == 0:
                time_taken = time.time() - start_time
                for metric in valid_metrics:
                    X_batch, y_batch = random_batch(X_valid, y_valid)
                    y_pred = model(X_batch, training=False)
                    main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
                    loss = tf.add_n([main_loss] + model.losses)
                    valid_mean_loss(loss)
                    valid_acc_metric.update_state(y_batch, y_pred)
                    metric(y_batch, y_pred)
                if USE_PRIVACY:
                    print_status_bar(step * BATCH_SIZE, len(y_train), train_mean_loss, time_taken,
                                     train_metrics + valid_metrics, spent_eps_delta,) 
                else:
                    print_status_bar(step * BATCH_SIZE, len(y_train), train_mean_loss, time_taken,
                                     train_metrics + valid_metrics)
            if should_terminate:
                break
            
        # Update training scores
        train_acc = train_acc_metric.result()
        train_loss = train_mean_loss.result()
        train_acc_scores.append(train_acc)
        train_loss_scores.append(train_loss)
        train_acc_metric.reset_state()
        train_mean_loss.reset_state()
        
        # Update validation scores
        valid_acc = valid_acc_metric.result()
        valid_loss = valid_mean_loss.result()
        valid_acc_scores.append(valid_acc)
        valid_loss_scores.append(valid_loss)
        valid_acc_metric.reset_state()
        valid_mean_loss.reset_state()
        
        if epoch % 10 == 0:
            for metric in test_metrics:
                y_pred = model(X_test, training=False)
                metric(y_test, y_pred)
            metrics = " - ".join(["{}: {:.4f}".format(m.name, m.result())
                                  for m in test_metrics or []])
            print(f"====== Epoch {10} test accuracy: {metrics}")
    
    # Evaluate model
    for metric in test_metrics:
        y_pred = model(X_test, training=False)
        metric(y_test, y_pred)
    metrics = " - ".join(["{}: {:.4f}".format(m.name, m.result())
                          for m in test_metrics or []])
    print(f"Training completed, test metrics: {metrics}")
    
    # Save model
    version = "DPSGD" if USE_PRIVACY else "SGD"
    model.save(MODELS_DIR/f"{version}-{N_EPOCHS}-{MODEL_TYPE}-{DATASET}.keras")
    
    # Make plots
    if PLOT_RESULTS:
        epochs_range = range(1, N_EPOCHS+1)
        plt.figure(figsize=(8,6))
        plt.plot(epochs_range, train_loss_scores, color='blue', label='Training loss')
        plt.plot(epochs_range, valid_loss_scores, color='red', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(RESULTS_DIR/f"{version}-Loss-{N_EPOCHS}-{MODEL_TYPE}-{DATASET}.png")
        plt.close()
         
        plt.figure(figsize=(8,6))
        plt.plot(epochs_range, train_acc_scores, color='blue', label='Training accuracy')
        plt.plot(epochs_range, valid_acc_scores, color='red', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(RESULTS_DIR/f"{version}-Accuracy-{N_EPOCHS}-{MODEL_TYPE}-{DATASET}.png")
        plt.close()

if __name__ == "__main__":
    main()
