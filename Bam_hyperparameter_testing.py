#%%
import time
import numpy as np
import tensorflow as tf
import sys

from datetime import datetime

from tensorflow.python.keras import backend as K
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.keras.layers import Activation

from tensorboard.plugins.hparams import api as hp

import matplotlib.pyplot as plt

from TFLearn_modules.tfbinding_generator import _log_norm
from TFLearn_modules.data_checking import _matrix_forom_figure
import TFLearn_modules.data_checking as dc
import TFLearn_modules.additional_metrics as mt

import TFLearn_modules.plotting_callbacks as pc
import TFLearn_modules.tfbinding_generator as tfb

import random



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
            np.save(self.dir + "/starting_weights_dense.npy", 
                self.model.get_layer(index = 3).get_weights()[0])
            np.save(self.dir + "/starting_weights_final_conv.npy", 
                self.model.get_layer(index = 5).get_weights()[0])

    def on_train_end(self, logs = None):
        np.save(self.dir + "/final_weights_conv.npy", 
            self.model.get_weights()[0])
        np.save(self.dir + "/final_weights_dense.npy", 
            self.model.get_layer(index = 3).get_weights()[0])
        np.save(self.dir + "/final_weights_final_conv.npy", 
            self.model.get_layer(index = 5).get_weights()[0])

def PlotModelLayerOuts(data, model):
    n_layers = len(model.layers)
    model_layers = [model.get_layer(index = i) for i in range(n_layers)]
    model_layers.insert(5, lambda x: tf.transpose(x, perm = [0,2,1]))
    model_layers.insert(3, lambda x: tf.transpose(x, perm = [0,2,1]))
    n_layers += 2

    x = data

    fig = plt.figure(figsize=(6, 4*(n_layers)), dpi = 100) 

    for i in range(n_layers):
        ax = fig.add_subplot(n_layers, 1, i + 1)

        x = model_layers[i](x)

        ax.hist(tf.reshape(x, (-1)).numpy(), bins = 50)

        ax.text(.5, .9,
            'layer {} outputs'.format(i),
            horizontalalignment='center',
            transform=ax.transAxes)

    fig.canvas.draw()
    return _matrix_forom_figure(fig)
    

class PlotModelLayerOutsCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset, logs, 
                 plotting_freq = 25, sample_size = 4):
        super(PlotModelLayerOutsCallback, self).__init__()

        self.plotting_freq = plotting_freq
        
        x = []
        for i, _ in dataset.take(sample_size):
            x.append(i)
        self.x = np.concatenate(x)
            
        self.writer = tf.summary.create_file_writer(logs)

    def on_epoch_end(self, epoch, logs = None):

        hists =  PlotModelLayerOuts(self.x, self.model)

        with self.writer.as_default():
            tf.summary.image("ModelWeightsByLayer", hists, step=epoch)

        plt.close('all')


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
class TFbindingModel_OH_Exp(tf.keras.Model):
    def __init__(self, 
                    number_conv_filter, 
                    conv_kernel_size,
                    number_of_bams, 
                    bam_len, 
                    fasta_len,
                    number_of_dense_layers, 
                    dropout_rate, 
                    conv_reg = [None, None, None],
                    act = ['relu', 'relu', None],
                    conv_init = 'glorot_uniform',
                    dense_init = 'glorot_uniform'):
        super(TFbindingModel_OH_Exp, self).__init__()

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
        self.conv_final = tf.keras.layers.Conv1D(number_of_bams, 10,  activation=act[2],
                                        padding = "same", kernel_regularizer = conv_reg[2],
                                        kernel_initializer=tf.keras.initializers.constant(value=0.01))

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
        x = tf.transpose(x, perm = [0,2,1])
        # (batch_size, BAM_LEN, filters)
        return self.conv_final(x)
        
#%% new mappers

def generate_y_center(training_data):
    y = np.concatenate([i[1] for i in train_data.take(20)])
    y = np.reshape(y, newshape=(-1,y.shape[-1]))
    y_center = np.reshape(np.mean(y, axis = 0), (1,1,y.shape[-1]))
    return lambda x: x - y_center

def generate_x_center(training_data):
    y = np.concatenate([i[0] for i in train_data.take(20)])
    y_center = np.mean(y)
    return lambda x: x - y_center


#%%
script_start = time.time()

# %%
BATCH_SIZE = 64
# size of steps (in bp) over which to calculate the bam signal
SIGNAL_STEP_SIZE = 20

# length of intervals to yield
# ideally divisible by 2, 10 and SIGNAL_STEP_SIZE
#INTERVAL_LENGTH = 600

CHROMOSOME_SELECTION = ["chr" + str(i) for i in range(1,6)]
CHROMOSOME_SIZES = "/home/streeck/Desktop/2021_03_19_test/hg19.chrom.sizes.tsv"
FASTA_OH = np.load("/home/streeck/Desktop/2021_03_19_test/hg19_one_hot_soft_masked_chr_1_5.npy")


#%% CTCF
CTCF_CHIP_FILES = ["/home/streeck/Desktop/2021_03_19_test/CTCF_ChIP_1_1.bg", 
    "/home/streeck/Desktop/2021_03_19_test/CTCF_ChIP_1_2.bg", 
    "/home/streeck/Desktop/2021_03_19_test/CTCF_ChIP_2_1.bg", 
    "/home/streeck/Desktop/2021_03_19_test/CTCF_ChIP_2_2.bg"]
CTCF_INPUT_FILES = ["/home/streeck/Desktop/2021_03_19_test/CTCF_INPUT_1_1.bg",
    "/home/streeck/Desktop/2021_03_19_test/CTCF_INPUT_2_1.bg"]

#%% SPI1
SPI1_CHIP_FILES = ["/home/streeck/Desktop/2021_03_19_test/SPI1_ChIP_1_1.bg", 
    "/home/streeck/Desktop/2021_03_19_tests/SPI1_ChIP_1_2bam.bg"]
SPI1_INPUT_FILES = ["/home/streeck/Desktop/2021_03_19_test/SPI1_INPUT_1_1.bg"]

#%% CEBPB
CEBPB_CHIP_FILES = ["/home/streeck/Desktop/2021_03_19_test/CEBPB_ChIP_1_1.bg", 
    "/home/streeck/Desktop/2021_03_19_test/CEBPB_ChIP_1_2.bg"]
CEBPB_INPUT_FILES = ["/home/streeck/Desktop/2021_03_19_test/CEBPB_INPUT_1_1.bg"]

#%% true motive from jaspar
jaspar_CTCF = np.array([[87, 167, 281, 56, 8, 744, 40, 107, 851, 5, 333, 54, 12, 56, 104, 372, 82, 117, 402],
[291, 145, 49, 800, 903, 13, 528, 433, 11, 0, 3, 12, 0, 8, 733, 13, 482, 322, 181],
[76, 414, 449, 21, 0, 65, 334, 48, 32, 903, 566, 504, 890, 775, 5, 507, 307, 73, 266],
[459, 187, 134, 36, 2, 91, 11, 324, 18, 3, 9, 341, 8, 71, 67, 17, 37, 396, 59]])
jaspar_CTCF = np.transpose(jaspar_CTCF, (1,0))

jaspar_CTCF = jaspar_CTCF[3:-3,:]

jaspar_SPI1 = np.array([[3946, 4976, 5053, 5044, 4868, 49, 1793, 29, 0, 5105, 5122, 0, 21, 2535],
[555, 26, 2, 0, 4, 246, 4068, 1, 1, 2, 1, 184, 69, 288],
[1142, 377, 8, 0, 15, 5255, 727, 5300, 5330, 0, 1, 5246, 12, 733],
[476, 82, 30, 99, 524, 2, 1, 2, 0, 1, 5, 1, 4962, 1439]])

jaspar_SPI1 = np.transpose(jaspar_SPI1, (1,0))

jaspar_SPI1 = jaspar_SPI1[3:-3,:]

motive_dict = {
    "CTCF": jaspar_CTCF,
    "SPI1": jaspar_SPI1
}


#%% hyperparameters


training_dataset_dict = {
    "CTCF": (CTCF_CHIP_FILES, CTCF_INPUT_FILES)#,
    #"SPI1": (SPI1_CHIP_FILES, SPI1_INPUT_FILES),
    #"CEBPB": (CEBPB_CHIP_FILES, CEBPB_INPUT_FILES)
}

activ = {
    "relu": ['relu', 'relu', None],
    "relu_rect": [Activation(lambda x: tf.keras.activations.relu(x, alpha=0.01)), 'relu', None],
    "relu_rect_both": [Activation(lambda x: tf.keras.activations.relu(x, alpha=0.01)), 
                       Activation(lambda x: tf.keras.activations.relu(x, alpha=0.01)), None],
    "tanh": ['tanh', 'relu', None]
}

initializers_conv = {
    'glorot_uniform': 'glorot_uniform',
    'he_uniform': 'he_uniform',
}

initializers_dense = {
    'glorot_uniform': 'glorot_uniform',
    'he_uniform': 'he_uniform'
}

mappings = {
    "identity": lambda x, y: (x, y),
    "center_x": lambda x, y: (generate_x_center(train_data)(x), y),
    "center_y": lambda x, y: (x, generate_y_center(train_data)(y)),
    "center_both": lambda x, y: (generate_x_center(train_data)(x), 
        generate_y_center(train_data)(y)),
    "log_y": lambda x, y: (x, tf.math.log(y+tf.ones((64, 50,4)))),
}

regularizers = {
    "none": [None, None, None],
    "l2_first": ['l2', None, None],
    "l2_second": [None, 'l2', None],
    "l2_all": ['l2', 'l2', 'l2']
}



# 15 options: [mapping, act, init_conv, init_dense, reg, loss, learning rate]
options = [
           ("log_y", "relu_rect_both", 'he_uniform', 'he_uniform', "none", 'mean_squared_error', 0.001),
           ("center_x", "relu_rect_both", 'he_uniform', 'he_uniform', "none", 'mean_squared_error', 0.001),
           ("center_y", "relu_rect_both", 'he_uniform', 'he_uniform', "none", 'mean_squared_error', 0.001),
           ("center_both", "relu_rect_both", 'he_uniform', 'he_uniform', "none", 'mean_squared_error', 0.001),
           ("identity", "relu_rect_both", 'he_uniform', 'he_uniform', "l2_first", 'mean_squared_error', 0.001),
           ("identity", "relu_rect_both", 'he_uniform', 'he_uniform', "l2_second", 'mean_squared_error', 0.001),
           ("identity", "relu_rect_both", 'he_uniform', 'he_uniform', "l2_all", 'mean_squared_error', 0.001),
           ("identity", "relu_rect_both", 'he_uniform', 'he_uniform', "none", "poisson", 0.001),
           ("identity", "relu_rect_both", 'glorot_uniform', 'he_uniform', "none", 'mean_squared_error', 0.001),
           ("identity", "relu_rect_both", 'he_uniform', 'glorot_uniform', "none", 'mean_squared_error', 0.001),
           ("identity", "tanh", 'he_uniform', 'he_uniform', "none", 'mean_squared_error', 0.001),
           ("identity", "relu_rect", 'he_uniform', 'he_uniform', "none", 'mean_squared_error', 0.001),
           ("identity", "relu", 'he_uniform', 'he_uniform', "none", 'mean_squared_error', 0.001),
           ("identity", "relu_rect_both", 'he_uniform', 'he_uniform', "none", 'mean_squared_error', 0.001),
           ("identity", "relu_rect_both", 'he_uniform', 'he_uniform', "none", 'mean_squared_error', 0.01)]




# %%

#%% constant hyperparameters
learning_rate_scedules = {
    "ReduceOnPlateau": [tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.3, patience=15, min_lr=1e-9)],
    "None": []}

lrs = "ReduceOnPlateau"
reg = [None, None, None]
interval_length = 1000
loss = 'mean_squared_error'
learning_rate = 0.001
conv_kernel_size = 29
dropout = 0
number_dense_layers = 1
filters = 1



#%% setup
logdir = "./logs_{}/".format(datetime.now().strftime("%Y_%m_%d-%H%M%S"))
session_num = 0

each = 10

STEPS_PER_EPOCH = 100

EPOCHS = 100


dataset_key = "CTCF"


print("working on dataset {}".format(dataset_key))
paragraph_start = time.time()

# read in ChIP bedgraph files
bam_files, input_files = training_dataset_dict[dataset_key]
bam_signal = tfb.get_signal_from_bg(bam_files, 
    CHROMOSOME_SELECTION, SIGNAL_STEP_SIZE)

# read in input bedgraph files
input_signal = tfb.get_signal_from_bg(input_files, 
    CHROMOSOME_SELECTION, SIGNAL_STEP_SIZE)

# match size of input and ChIP
input_signal = np.repeat(input_signal, 2, axis = -1)

masks = (tfb.generate_border_mask(CHROMOSOME_SIZES, CHROMOSOME_SELECTION, 
    SIGNAL_STEP_SIZE, interval_length), 
tfb.generate_bias_mask(bam_signal, 
    input_signal, 20000, interval_length//SIGNAL_STEP_SIZE), 
tfb.generate_background_mask(input_signal, interval_length),
tfb.generate_N_mask(FASTA_OH, SIGNAL_STEP_SIZE))

mask = tfb.combine_masks(*masks, 0.1)

for mapping, act, kic, kid, reg, loss, learning_rate in options:
    for i in range(each):  
        

        # make the dataset
        file_count = bam_signal.shape[1]
        if (time.time() - script_start)/(60*60) > 70:
            print("took too long, aborted")
            break

        train_data = tf.data.Dataset.from_generator(
            lambda: tfb.batch_slice_generator_performance(FASTA_OH, bam_signal, mask, interval_length,
                                            SIGNAL_STEP_SIZE, BATCH_SIZE, 30000),
            output_signature=(
                tf.TensorSpec(shape=(64, interval_length, 4), dtype=tf.float64),
                tf.TensorSpec(shape=(64, interval_length//SIGNAL_STEP_SIZE, file_count), dtype=tf.float64)))

        if mapping == "center_x":
            c_x = tf.math.reduce_mean(tf.stack([i[0] for i in train_data.take(20)]))
            ds_map = lambda x, y: (x-c_x, y)

        if mapping == "center_y":
            c_y = tf.math.reduce_mean(tf.stack([i[1] for i in train_data.take(20)]))
            ds_map = lambda x, y: (x, y-c_y)

        if mapping == "center_both":
            c_x = tf.math.reduce_mean(tf.stack([i[0] for i in train_data.take(20)]))
            c_y = tf.math.reduce_mean(tf.stack([i[1] for i in train_data.take(20)]))
            ds_map = lambda x, y: (x-c_x, y-c_y)
        
        if mapping == "identity":
            ds_map = lambda x, y: (x, y)
            
            
        if mapping == "log_y":
            ds_map = lambda x, y: (x, tf.math.log(y+1))
      
        train_data = train_data.map(ds_map).prefetch(tf.data.AUTOTUNE)
        
        # no pool oh expanding
        session_num += 1

        hparams = {
            "session": session_num,
            "mapping": mapping,
            "activation":act,
            "kernel init conv":kic,
            "kernel init denes":kid,
            "regul": reg,
            "loss": loss,
            "init_lr": learning_rate}

  
        callbacks = [
            tf.keras.callbacks.TensorBoard(
                log_dir=logdir + "session_{}/".format(session_num),
                profile_batch = '980, 1000'),
            hp.KerasCallback(
                logdir + "session_{}/".format(session_num), 
                hparams),
            pc.PlotBindingCallback(train_data,
                logdir + "session_{}/".format(session_num),
                motive_dict[dataset_key], plotting_freq = 50),
            pc.PlotExampleCallback(train_data, 
                logdir + "session_{}/".format(session_num), 
                plotting_freq = 50),
            UpdateConvSchedule(
                logdir + "session_{}/".format(session_num)),
            LogLearningRate(
                logdir + "session_{}/".format(session_num)),
            PlotModelLayerOutsCallback(train_data, 
                logdir + "session_{}/".format(session_num)),
            Write_model_weigths(
                logdir + "session_{}/".format(session_num))]

        callbacks += learning_rate_scedules[lrs]
        
        model = TFbindingModel_OH_Exp(
            filters, 
            conv_kernel_size,
            bam_signal.shape[1], 
            interval_length//SIGNAL_STEP_SIZE, 
            interval_length,
            number_dense_layers, 
            dropout, 
            conv_reg = regularizers[reg],
            act = activ[act],
            conv_init = initializers_conv[kic],
            dense_init = initializers_dense[kid])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss= loss,
            metrics = ['mse', mt.MaxAbsError()]
        )

        print('--- Starting trial %d' % session_num)                    
        print(hparams)
        paragraph_start = time.time()

        model.fit(train_data, steps_per_epoch= STEPS_PER_EPOCH, epochs=EPOCHS, 
            callbacks=callbacks)

        time_writer = tf.summary.create_file_writer(logdir + "session_{}/".format(session_num))

        with time_writer.as_default():
            tf.summary.scalar("minutes run time", data = ((time.time() - paragraph_start)/(60)), step = 199)
        
        print("time taken: %.2f minutes" % ((time.time() - paragraph_start)/(60)))

       

# %%
