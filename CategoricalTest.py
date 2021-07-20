#%%
import time
import numpy as np
import tensorflow as tf
import sys

from datetime import datetime
from itertools import product

import random

from tensorflow.python.keras import backend as K
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.keras.layers import Activation

from tensorboard.plugins.hparams import api as hp

import TFLearn_modules.additional_metrics as mt

import TFLearn_modules.plotting_callbacks as pc

import TFLearn_modules.MACS_dataset_gen as mcs

from TFLearn_modules.bam_dataset_generator import _jagged_slicer



#%% common use:
class LogLearningRate(tf.keras.callbacks.Callback):
    def __init__(self,            
                 logdir = None):
        super(LogLearningRate, self).__init__()
        self.writer = tf.summary.create_file_writer(logdir)
    
    def on_epoch_end(self, epoch, logs = None):
        with self.writer.as_default():
            tf.summary.scalar("learning rate",
                K.get_value(self.model.optimizer.lr), epoch)

class Write_model_weigths(tf.keras.callbacks.Callback):
    def __init__(self,            
                 log_dir = None):
        super(Write_model_weigths, self).__init__()
        self.dir = log_dir
        self.written_start = False

    def on_train_batch_begin(self, batch, logs = None):
        if not self.written_start and batch > 0:
            self.written_start = True
            np.save(self.dir + "/starting_weights_conv.npy", 
                self.model.get_weights()[0])
            np.save(self.dir + "/starting_weights_final_dense.npy", 
                self.model.get_layer(index = 4).get_weights()[0])

    def on_train_end(self, logs = None):
        np.save(self.dir + "/final_weights_conv.npy", 
            self.model.get_weights()[0])
        np.save(self.dir + "/final_weights_final_conv.npy", 
            self.model.get_layer(index = 4).get_weights()[0])

# callback for expanding of pool layer
class UpdateConvAtPlateu(tf.keras.callbacks.Callback):
    def __init__(self, logdir, patience=0, update_layer=0, epsilon = 0.01):
        super(UpdateConvAtPlateu, self).__init__()
        self.patience = patience
        self.epsilon = epsilon
        self.update_layer = update_layer
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None
        self.writer = tf.summary.create_file_writer(logdir)

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")

        current_padding = self.model.get_layer(
            index = self.update_layer).mask_padding

        with self.writer.as_default():
            tf.summary.scalar("conv layer masking", 
            data = current_padding, 
            step = epoch)

        if np.less(current, self.best*(1-self.epsilon)):
            self.wait = 0
        else:
            self.wait += 1

        if np.less(current, self.best):
            self.best = current

        if self.wait >= self.patience and current_padding > 0:
            if current_padding > 0:
                self.model.get_layer(
                    index = self.update_layer).update_conv_mask()
            self.wait = 0

# callback for expanding of pool layer
class UpdateConvSchedule(tf.keras.callbacks.Callback):
    def __init__(self, logdir, interval=10, update_layer=0):
        super(UpdateConvSchedule, self).__init__()
        self.interval = interval
        self.update_layer = update_layer
        self.writer = tf.summary.create_file_writer(logdir)

    def on_epoch_end(self, epoch, logs=None):
        current_padding = self.model.get_layer(
            index = self.update_layer).mask_padding

        with self.writer.as_default():
            tf.summary.scalar("conv layer masking", 
            data = current_padding, 
            step = epoch)

        if (epoch+1) % self.interval == 0 and current_padding > 0:
            self.model.get_layer(
                index = self.update_layer).update_conv_mask()


# maxpool layer for strands
class DnaFwdRvPool(tf.keras.layers.Layer):
    def __init__(self):
        super(DnaFwdRvPool, self).__init__()

        self.pool = tf.keras.layers.MaxPool2D(pool_size = (1,2), strides = (1,1))

    def call(self, x):
        # max pool fwd and rev activation
        # in shape: (batch_size, FASTA_LEN, 2, filters)
        # out shape: (batch_size, FASTA_LEN, filters)
        x = self.pool(x)
        return tf.squeeze(x, axis=-2)


#%% one hot expanding convolutional layer
class DnaMotiveExpandingConvolution_OH(tf.keras.layers.Conv1D):
    def __init__(self, filters, kernel_size, reg, act, 
                 ki = 'glorot_uniform', init_mask = 0.5, **kwargs):
        super(DnaMotiveExpandingConvolution_OH, self).__init__(
            filters, kernel_size, padding = "same", activation=act, 
            kernel_regularizer = reg, kernel_initializer= ki, **kwargs)
        
        self.mask_padding = int(kernel_size//(2/init_mask))

    def build(self, input_shape):
        super().build(input_shape)

        self.mask = self.add_weight(
            shape=self.kernel.shape,
            initializer="ones",
            dtype = tf.float32,
            trainable=False)

        self.set_conv_mask()

    def _call_ops(self, inputs, training):
        if training is None:
            training = K.learning_phase()

        if training:
            outputs = self._convolution_op(inputs, 
                self.kernel * self.mask)
        else:
            outputs = self._convolution_op(inputs, self.kernel)

        outputs = nn.bias_add(
            outputs, self.bias, data_format=self._tf_data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def call(self, x, training=None):
        # calculate convolution along rev_comp strand
        # in shape: (batch_size, FASTA_LEN, 4)
        # out shape: (batch_size, FASTA_LEN, filters)
        rev_x = tf.reverse(x, axis = [-2,-1])
        rev_x = self._call_ops(rev_x, training)
        rev_x = tf.reverse(rev_x, axis = [-2])

        # calculate convolution along fwd strand
        # in shape: (batch_size, FASTA_LEN, 4)
        # out shape: (batch_size, FASTA_LEN, filters)
        x = self._call_ops(x, training)

        # stack fwd and rv convolutions
        # in shape: 2x(batch_size, FASTA_LEN, filters)
        # out shape: (batch_size, FASTA_LEN ,2, filters)
        return tf.stack([x, rev_x], axis = -2)

    def update_conv_mask(self):
        if self.mask_padding > 0:
            self.mask_padding = max(self.mask_padding - 2, 0)
            self.set_conv_mask() 

    def set_conv_mask(self):
        m = tf.concat(
            [tf.zeros((self.mask_padding,4,1)), 
                tf.ones((self.kernel.shape[0] - 2*self.mask_padding,4,1)), 
                tf.zeros((self.mask_padding,4,1))], axis = 0)
        self.set_weights([ *self.get_weights()[:-1], m])

#%% binding model, one-hot, expanding with no intermediate convolution
class TFbindingModel_MACS_cat(tf.keras.Model):
    def __init__(self, 
                    number_conv_filter, 
                    conv_kernel_size, 
                    bam_len, 
                    fasta_len,
                    number_of_dense_layers, 
                    dropout_rate, 
                    conv_reg = [None, None, None],
                    act = ['relu', 'relu', None],
                    conv_init = 'glorot_uniform',
                    dense_init = 'glorot_uniform'):
        super(TFbindingModel_MACS_cat, self).__init__()

        self.DNAconv = DnaMotiveExpandingConvolution_OH(number_conv_filter, 
            conv_kernel_size, conv_reg[0], act[0], conv_init)
        self.strand_pool = DnaFwdRvPool()
        self.max_pool_steps = tf.keras.layers.MaxPool1D(pool_size = fasta_len//bam_len)
        self.dense = [tf.keras.layers.Dense(bam_len, activation = act[1], 
                                            kernel_regularizer = conv_reg[1],
                                            kernel_initializer=dense_init) 
                        for i in range(number_of_dense_layers)]
        self.dropout = [tf.keras.layers.Dropout(dropout_rate)
                        for i in range(number_of_dense_layers)]
        self.flatten = tf.keras.layers.Flatten()
        self.dense_final = tf.keras.layers.Dense(1,
                                        kernel_initializer=dense_init)

    def call(self, inputs):
        # (batch_size, FASTA_LEN, 4)
        x = self.DNAconv(inputs)
        x = self.strand_pool(x)
        # (batch_size, FASTA_LEN, filters)
        x = self.max_pool_steps(x)
        # (batch_size, BAM_LEN, filters)
        x = tf.transpose(x, perm = [0,2,1])
        # (batch_size, filters, BAM_LEN)
        for i in range(len(self.dense)):
            x = self.dense[i](x)
            # (batch_size, filters, BAM_LEN)
            x = self.dropout[i](x)
            # (batch_size, filters, BAM_LEN)
        x = self.flatten(x)
        # (batch_size, BAM_LEN, filters)
        return self.dense_final(x)

#%% binding model, one-hot, expanding with no intermediate convolution
class TFbindingModel_MACS_cat2(tf.keras.Model):
    def __init__(self, 
                    number_conv_filter, 
                    conv_kernel_size, 
                    bam_len, 
                    fasta_len,
                    number_of_dense_layers, 
                    dropout_rate, 
                    conv_reg = [None, None, None],
                    act = ['relu', 'relu', None],
                    conv_init = 'glorot_uniform',
                    dense_init = 'glorot_uniform'):
        super(TFbindingModel_MACS_cat2, self).__init__()

        self.DNAconv = DnaMotiveExpandingConvolution_OH(number_conv_filter, 
            conv_kernel_size, conv_reg[0], 'sigmoid', conv_init)
        self.strand_pool = DnaFwdRvPool()
        self.max_pool_steps = tf.keras.layers.MaxPool1D(pool_size = fasta_len)

    def call(self, inputs):
        # (batch_size, FASTA_LEN, 4)
        x = self.DNAconv(inputs)
        x = self.strand_pool(x)
        return self.max_pool_steps(x)
        
#%% binding model, one-hot, expanding with no intermediate convolution
class TFbindingModel_MACS_cat3(tf.keras.Model):
    def __init__(self, 
                    number_conv_filter, 
                    conv_kernel_size, 
                    bam_len, 
                    fasta_len,
                    number_of_dense_layers, 
                    dropout_rate, 
                    conv_reg = [None, None, None],
                    act = ['relu', 'relu', None],
                    conv_init = 'glorot_uniform',
                    dense_init = 'glorot_uniform'):
        super(TFbindingModel_MACS_cat3, self).__init__()

        self.DNAconv = DnaMotiveExpandingConvolution_OH(number_conv_filter, 
            conv_kernel_size, conv_reg[0], act[0], conv_init)
        self.strand_pool = DnaFwdRvPool()
        self.conv = tf.keras.layers.Conv1D(1, conv_kernel_size, 
            padding = "same", activation=act[1], 
            kernel_regularizer = conv_reg[1], kernel_initializer=dense_init)
        self.max_pool_steps = tf.keras.layers.MaxPool1D(pool_size = fasta_len//bam_len)
        self.flatten = tf.keras.layers.Flatten()
        self.dense_final = tf.keras.layers.Dense(1,
                                        kernel_initializer=dense_init)

    def call(self, inputs):
        # (batch_size, FASTA_LEN, 4)
        x = self.DNAconv(inputs)
        x = self.strand_pool(x)
        # (batch_size, FASTA_LEN, filters)
        x = self.conv(x)
        x = self.max_pool_steps(x)
        # (batch_size, BAM_LEN, filters)
        x = tf.transpose(x, perm = [0,2,1])
        x = self.flatten(x)
        # (batch_size, BAM_LEN, filters)
        return self.dense_final(x)

#%%
script_start = time.time()

# %%
BATCH_SIZE = 256
# size of steps (in bp) over which to calculate the bam signal
SIGNAL_STEP_SIZE = 20

# length of intervals to yield
# ideally divisible by 2, 10 and SIGNAL_STEP_SIZE
#INTERVAL_LENGTH = 600

CHROMOSOME_SELECTION = ["chr" + str(i) for i in range(1,6)]
CHROMOSOME_SIZES = "/Users/streeck/Desktop/EncodeTFTest2020_02-18/hg19.chrom.sizes.tsv"
FASTA_OH = np.load("/Users/streeck/Desktop/EncodeTFTest2020_02-18/hg19_one_hot_soft_masked_chr_1_5_1step.npy")

#%% Datasets
MACS_file_CTCF = "/Users/streeck/Desktop/EncodeTFTest2020_02-18/Hyperparam_bams/MACS/CTCF_peaks.narrowPeak"

#%% hyperparameters

activations = [Activation(lambda x: tf.keras.activations.relu(x, alpha=0.01)), 
                       Activation(lambda x: tf.keras.activations.relu(x, alpha=0.01)), None]

regularizers = [None, None, None]

loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

learning_rate = 0.01

conv_kernel_size = 29

dropout = 0

number_dense_layers = 0

filters = 1



#%% setup
logdir = "./logs_{}/".format(datetime.now().strftime("%Y_%m_%d-%H%M%S"))

session_num = 0

interval_length = 500

each = 20

STEPS_PER_EPOCH = 100

EPOCHS = 50


# test some combinations of peaks to take
interval_length = 500
cut_top = 100
n_regions = 10000



#%% run

dataset = mcs.MACS_cat_dataset(FASTA_OH,
            MACS_file_CTCF,
            CHROMOSOME_SELECTION,
            CHROMOSOME_SIZES,
            interval_length,
            background_frac = 0.5,
            cut_top = cut_top, 
            n_regions = n_regions)



for i in range(each):               

    # make the dataset
    if (time.time() - script_start)/(60*60) > 24:
        print("took too long, aborted")
        break

    train_data = tf.data.Dataset.from_generator(
        lambda: dataset.make_generator(n_batches = 30000),
        output_signature=(
            tf.TensorSpec(shape=(64, interval_length, 4), dtype=tf.float64),
            tf.TensorSpec(shape=(64), dtype=tf.float64)))

    train_data = train_data.prefetch(tf.data.AUTOTUNE)

    ## 2
    train_data = tf.data.Dataset.from_generator(
        lambda: dataset.make_generator(n_batches = 30000),
        output_signature=(
            tf.TensorSpec(shape=(64, interval_length, 4), dtype=tf.float64),
            tf.TensorSpec(shape=(64), dtype=tf.float64)))

    train_data = train_data.prefetch(tf.data.AUTOTUNE)

    # no pool oh expanding
    session_num += 1

    hparams = {
        "session": session_num,
        "model": "max pool"}


    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=logdir + "session_{}/".format(session_num),
            profile_batch = '980, 1000'),
        hp.KerasCallback(
            logdir + "session_{}/".format(session_num), 
            hparams),
        pc.PlotBindingCallback(train_data,
            logdir + "session_{}/".format(session_num), 
            plotting_freq = 50),
        UpdateConvSchedule(
            logdir + "session_{}/".format(session_num)),
        LogLearningRate(
            logdir + "session_{}/".format(session_num)),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', 
            factor=0.3, patience=15, min_lr=1e-9)]

    model = TFbindingModel_MACS_cat2(
        filters, 
        conv_kernel_size, 
        interval_length//SIGNAL_STEP_SIZE, 
        interval_length,
        number_dense_layers, 
        dropout, 
        conv_reg = regularizers,
        act = activations,
        conv_init = 'he_uniform',
        dense_init = 'he_uniform')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss= tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics = ['binary_accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.AUC(name='prc', curve='PR')]
    )

    print('--- Starting trial %d ---' % session_num)                    
    print(hparams)
    paragraph_start = time.time()

    model.fit(train_data, steps_per_epoch= STEPS_PER_EPOCH, epochs=EPOCHS, 
        callbacks=callbacks)

    print("time taken: %.2f minutes" % ((time.time() - paragraph_start)/(60)))

 
# %%
