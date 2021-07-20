#%%
import tensorflow as tf
import numpy as np

#%%
class MaxAbsError(tf.keras.metrics.Metric):

  def __init__(self, name='maxae', **kwargs):
    super(MaxAbsError, self).__init__(name=name, **kwargs)
    self.maxae = self.add_weight(name='maxae', initializer='zeros')
    self.count = self.add_weight(name='count', initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    abs_diff = tf.math.abs(y_true - y_pred)
    max_abs_diff = tf.math.reduce_max(abs_diff, axis = -2)
    self.count.assign_add(tf.math.reduce_prod(tf.cast(max_abs_diff.shape, tf.float32)))
    self.maxae.assign_add(tf.reduce_sum(max_abs_diff))

  def result(self):
    return tf.math.divide_no_nan(self.maxae, self.count)
# %%
