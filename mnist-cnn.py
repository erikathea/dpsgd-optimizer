from __future__ import division

import abc
import collections
import math
import matplotlib.pyplot as plt
import numpy
from pathlib import Path
import sys
import time

import tensorflow as tf



ROOT_DIR = Path(__name__).absolute().parent
MODELS_DIR = Path.joinpath(ROOT_DIR, 'log')
RESULTS_DIR = Path.joinpath(ROOT_DIR, 'log')


ClipOption = collections.namedtuple("ClipOption", ["l2norm_bound", "clip"])

EpsDelta = collections.namedtuple("EpsDelta", ["spent_eps", "spent_delta"])

BATCH_SIZE = 32
LEARNING_RATE = 0.01
L2NORM_BOUND = 4.0
SIGMA = 4.0
DATASET = 'mnist'
MODEL_TYPE = 'cnn'
USE_PRIVACY = True
PLOT_RESULTS = True
N_EPOCHS = 20

def mnist_dp_cnn_model():
    dpmodel = tf.keras.models.Sequential()
    dpmodel.add(tf.keras.layers.BatchNormalization())
    dpmodel.add(tf.keras.layers.Conv2D(32, kernel_size = 3, activation='relu'))
    dpmodel.add(tf.keras.layers.BatchNormalization())
    dpmodel.add(tf.keras.layers.Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
    dpmodel.add(tf.keras.layers.BatchNormalization())
    dpmodel.add(tf.keras.layers.Dropout(0.4))
    dpmodel.add(tf.keras.layers.Conv2D(64, kernel_size = 3, activation='relu'))
    dpmodel.add(tf.keras.layers.BatchNormalization())
    dpmodel.add(tf.keras.layers.Conv2D(64, kernel_size = 3, activation='relu'))
    dpmodel.add(tf.keras.layers.BatchNormalization())
    dpmodel.add(tf.keras.layers.Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
    dpmodel.add(tf.keras.layers.BatchNormalization())
    dpmodel.add(tf.keras.layers.Dropout(0.4))
    dpmodel.add(tf.keras.layers.Conv2D(128, kernel_size = 4, activation='relu'))
    dpmodel.add(tf.keras.layers.BatchNormalization())
    dpmodel.add(tf.keras.layers.Flatten())
    dpmodel.add(tf.keras.layers.Dropout(0.4))
    dpmodel.add(tf.keras.layers.Dense(10, activation='softmax'))
    return dpmodel

def shuffle_split_data(X, y):
    arr_rand = numpy.random.rand(X.shape[0])
    split = arr_rand < numpy.percentile(arr_rand, 70)
    X_train = X[split]
    y_train = y[split]
    X_test =  X[~split]
    y_test = y[~split]
    return X_train, y_train, X_test, y_test

def random_batch(X, y, batch_size=64):
    idx = numpy.random.randint(len(X), size=batch_size)
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

def add_gaussian_noise(t, sigma, name=None):
  noisy_t = t + tf.random.normal(tf.shape(t), stddev=sigma)
  return noisy_t

class AmortizedGaussianSanitizer(object):
    def __init__(self, accountant, default_option):
        self._accountant = accountant
        self._default_option = default_option
        self._options = {}
        self.epsilons = []
        self.deltas = []

    def set_option(self, tensor_name, option):
        self._options[tensor_name] = option

    def sanitize(self, x, eps_delta, sigma=None, option=ClipOption(None, None), tensor_name=None, num_examples=None, add_noise=True):
        eps, delta = eps_delta
        if sigma is None:
            with tf.control_dependencies([tf.Assert(tf.greater(eps, 0), ["eps needs to be greater than 0"]),
                                          tf.Assert(tf.greater(delta, 0), ["delta needs to be greater than 0"])]):
                sigma = tf.sqrt(2.0 * tf.math.log(1.25 / delta)) / eps

        l2norm_bound, clip = option
        if l2norm_bound is None:
            l2norm_bound, clip = self._default_option
            if ((tensor_name is not None) and (tensor_name in self._options)):
                l2norm_bound, clip = self._options[tensor_name]
        if clip:
            x = tf.clip_by_norm(x, clip_norm=l2norm_bound)

        if add_noise:
            if num_examples is None:
                num_examples = tf.slice(tf.shape(x), [0], [1])
            privacy_accum_op = self._accountant.accumulate_privacy_spending(eps_delta, sigma, num_examples)
            with tf.control_dependencies([privacy_accum_op]):
                saned_x = add_gaussian_noise(x, sigma * l2norm_bound)
            self.epsilons.append(eps)  # Store epsilon value
            self.deltas.append(delta)  # Store delta value
        else:
            saned_x = tf.reduce_sum(x, 0)
        return saned_x



class AmortizedAccountant(object):

  def __init__(self, total_examples):
    assert total_examples > 0
    self._total_examples = total_examples
    self._eps_squared_sum = tf.Variable(tf.zeros([1]), trainable=False, name="eps_squared_sum")
    self._delta_sum = tf.Variable(tf.zeros([1]), trainable=False,name="delta_sum")

  def accumulate_privacy_spending(self, eps_delta, unused_sigma,num_examples):
    eps, delta = eps_delta
    with tf.control_dependencies(
        [tf.Assert(tf.greater(delta, 0),
                   ["delta needs to be greater than 0"])]):
      amortize_ratio = (tf.cast(num_examples, tf.float32) * 1.0 /
                        self._total_examples)
      amortize_eps = tf.reshape(tf.math.log(1.0 + amortize_ratio * (tf.exp(eps) - 1.0)), [1])
      amortize_delta = tf.reshape(amortize_ratio * delta, [1])
      return tf.group(*[tf.compat.v1.assign_add(self._eps_squared_sum,
                                      tf.square(amortize_eps)),
                        tf.compat.v1.assign_add(self._delta_sum, amortize_delta)])

  def get_privacy_spent(self, target_eps=None):
    unused_target_eps = target_eps
    eps_squared_sum, delta_sum = ([self._eps_squared_sum, self._delta_sum])
    return [EpsDelta(math.sqrt(eps_squared_sum), float(delta_sum))]

class MomentsAccountant(object):

  __metaclass__ = abc.ABCMeta

  def __init__(self, total_examples, moment_orders=32):
    assert total_examples > 0
    self._total_examples = total_examples
    self._moment_orders = (moment_orders
                           if isinstance(moment_orders, (list, tuple))
                           else range(1, moment_orders + 1))
    self._max_moment_order = max(self._moment_orders)
    assert self._max_moment_order < 100, "The moment order is too large."
    self._log_moments = [tf.Variable(numpy.float64(0.0),
                                     trainable=False,
                                     name=("log_moments-%d" % moment_order))
                         for moment_order in self._moment_orders]

  @abc.abstractmethod
  def _compute_log_moment(self, sigma, q, moment_order):
    pass

  def accumulate_privacy_spending(self, unused_eps_delta,
                                  sigma, num_examples):
    q = tf.cast(num_examples, tf.float64) * 1.0 / self._total_examples

    moments_accum_ops = []
    for i in range(len(self._log_moments)):
      moment = self._compute_log_moment(sigma, q, self._moment_orders[i])
      moments_accum_ops.append(tf.compat.v1.assign_add(self._log_moments[i], moment))
    return tf.group(*moments_accum_ops)

  def _compute_delta(self, log_moments, eps):
    min_delta = 1.0
    for moment_order, log_moment in log_moments:
      if math.isinf(log_moment) or math.isnan(log_moment):
        sys.stderr.write("The %d-th order is inf or Nan\\n" % moment_order)
        continue
      if log_moment < moment_order * eps:
        min_delta = min(min_delta,
                        math.exp(log_moment - moment_order * eps))
    return min_delta

  def _compute_eps(self, log_moments, delta):
    min_eps = float("inf")
    for moment_order, log_moment in log_moments:
      if math.isinf(log_moment) or math.isnan(log_moment):
        sys.stderr.write("The %d-th order is inf or Nan\\n" % moment_order)
        continue
      min_eps = min(min_eps, (log_moment - math.log(delta)) / moment_order)
    return min_eps

  def get_privacy_spent(self, target_eps=None, target_deltas=None):
    assert (target_eps is None) ^ (target_deltas is None)
    eps_deltas = []
    log_moments = self._log_moments
    log_moments_with_order = zip(self._moment_orders, log_moments)
    if target_eps is not None:
      for eps in target_eps:
        eps_deltas.append(
            EpsDelta(eps, self._compute_delta(log_moments_with_order, eps)))
    else:
      assert target_deltas
      for delta in target_deltas:
        eps_deltas.append(
            EpsDelta(self._compute_eps(log_moments_with_order, delta), delta))
    return eps_deltas

class GaussianMomentsAccountant(MomentsAccountant):

  def __init__(self, total_examples, moment_orders=32):
    super(self.__class__, self).__init__(total_examples, moment_orders)
    self._binomial_table = GenerateBinomialTable(self._max_moment_order)

  def _differential_moments(self, sigma, s, t):
    assert t <= self._max_moment_order, ("The order of %d is out "
                                         "of the upper bound %d."
                                         % (t, self._max_moment_order))
    binomial = tf.slice(self._binomial_table, [0, 0],
                        [t + 1, t + 1])
    ii, jj = numpy.mgrid[:t+1,:t+1]
    signs = 1.0 - 2 * ((ii - jj) % 2)
    exponents = tf.constant([j * (j + 1.0 - 2.0 * s) / (2.0 * sigma * sigma)
                             for j in range(t + 1)], dtype=tf.float64)
    x = tf.multiply(binomial, signs)
    y = tf.multiply(x, tf.exp(exponents))
    z = tf.reduce_sum(y, 1)
    return z

  def _compute_log_moment(self, sigma, q, moment_order):
    assert moment_order <= self._max_moment_order, ("The order of %d is out "
                                                    "of the upper bound %d."
                                                    % (moment_order,
                                                       self._max_moment_order))
    binomial_table = tf.slice(self._binomial_table, [moment_order, 0],
                              [1, moment_order + 1])
    qs = tf.exp(tf.constant([i * 1.0 for i in range(moment_order + 1)],
                            dtype=tf.float64) * tf.cast(
                                tf.math.log(q), dtype=tf.float64))
    moments0 = self._differential_moments(sigma, 0.0, moment_order)
    term0 = tf.reduce_sum(binomial_table * qs * moments0)
    moments1 = self._differential_moments(sigma, 1.0, moment_order)
    term1 = tf.reduce_sum(binomial_table * qs * moments1)
    return tf.squeeze(tf.math.log(tf.cast(q * term0 + (1.0 - q) * term1,
                                     tf.float64)))

def GenerateBinomialTable(m):
  table = numpy.zeros((m + 1, m + 1), dtype=numpy.float64)
  for i in range(m + 1):
    table[i, 0] = 1
  for i in range(1, m + 1):
    for j in range(1, m + 1):
      v = table[i - 1, j] + table[i - 1, j -1]
      assert not math.isnan(v) and not math.isinf(v)
      table[i, j] = v
  return tf.convert_to_tensor(table)


def main():
    
    num_classes = 10
    image_size = 28
    n_channels = 1

    # X = image; Y = label
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    X_train = numpy.array(X_train, dtype=numpy.float32) / 255
    X_test = numpy.array(X_test, dtype=numpy.float32) / 255

    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    y_train = numpy.array(y_train, dtype=numpy.int32)
    y_test = numpy.array(y_test, dtype=numpy.int32)

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    # Create train/valid set
    X_train, y_train, X_valid, y_valid = shuffle_split_data(X_train, y_train)

    model = mnist_dp_cnn_model()
  
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE)
    # loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.optimizers.SGD(LEARNING_RATE)

    # Set constants for this loop
    eps = 1.0
    delta = 1e-7
    max_eps = 64.0
    max_delta = 1e-3
    target_eps = [64.0]
    target_delta = [1e-5] #unused
    # Setup metrics
    train_mean_loss = tf.keras.metrics.Mean()
    valid_mean_loss = tf.keras.metrics.Mean()
    train_acc_scores, valid_acc_scores = list(), list()
    train_loss_scores, valid_loss_scores = list(), list()
    train_acc_metric = tf.keras.metrics.CategoricalCrossentropy()
    valid_acc_metric = tf.keras.metrics.CategoricalCrossentropy()
    train_metrics = [tf.keras.metrics.CategoricalCrossentropy()]
    valid_metrics = [tf.keras.metrics.CategoricalCrossentropy()]
    test_metrics = [tf.keras.metrics.CategoricalCrossentropy()]
    
    # Create accountant, sanitizer and metrics
    accountant = AmortizedAccountant(len(X_train))
    sanitizer = AmortizedGaussianSanitizer(accountant, [L2NORM_BOUND / BATCH_SIZE, True])

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

            sanitized_grads = []
            eps_delta = EpsDelta(eps, delta)
            for px_grad in gradients:
                sanitized_grad = sanitizer.sanitize(px_grad, eps_delta, SIGMA)
                sanitized_grads.append(sanitized_grad)
            spent_eps_delta = accountant.get_privacy_spent(target_eps=target_eps)[0]
            optimizer.apply_gradients(zip(sanitized_grads, model.trainable_variables))
            if (spent_eps_delta.spent_eps > max_eps or spent_eps_delta.spent_delta > max_delta):
                should_terminate = True

            train_mean_loss(loss)
            for metric in train_metrics:
                metric(y_batch, y_pred)
            if step % 100 == 0:
                time_taken = time.time() - start_time
                for metric in valid_metrics:
                    X_batch, y_batch = random_batch(X_valid, y_valid)
                    y_pred = model(X_batch, training=False)
                    main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
                    loss = tf.add_n([main_loss] + model.losses)
                    valid_mean_loss(loss)
                    valid_acc_metric.update_state(y_batch, y_pred)
                    metric(y_batch, y_pred)

                print_status_bar(step * BATCH_SIZE, len(y_train), train_mean_loss, time_taken,
                                     train_metrics + valid_metrics, spent_eps_delta,) 

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
