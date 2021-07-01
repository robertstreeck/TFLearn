#%%

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import matplotlib; matplotlib.use('agg')
from .tfbinding_generator import _generate_example_for_plot
from scipy.special import gammaln
from .data_checking import _combine_image_matrix
from .data_checking import _matrix_forom_figure

#%%

def _plot_multiregion_callback(y_true, y_pred):
    regions = y_true.shape[0]
    bams = y_true.shape[2]

    fig = plt.figure(figsize=(6*bams, 3*regions), dpi = 100)

    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)

    x = np.arange(y_true.shape[1])

    for region in range(regions):
        for bam in range(bams):
            ax = fig.add_subplot(y_true.shape[0], y_true.shape[2], 
                region*y_true.shape[2] + bam + 1)

            yt = y_true[region,:,bam]
            yp = y_pred[region,:,bam]

            ax.fill_between(x, yt, yp, where=yt >= yp, facecolor='#a1d99b30')
            ax.fill_between(x, yt, yp, where=yt <= yp, facecolor='#fc927230')

            ax.plot(yt, label='true', color = "#377eb8")

            ax.plot(yp, label='pred', color = "#984ea3")

            ax.legend(loc = 'upper left')

            ax.text(.5, .9,
                'Region {}, Bam {}'.format(region+1, bam+1),
                horizontalalignment='center',
                transform=ax.transAxes)

    return _matrix_forom_figure(fig)

class PlotExampleCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset, logs, number_examples = 5, plotting_freq = 1):
        super(PlotExampleCallback, self).__init__()

        self.plotting_freq = plotting_freq

        x, y = next(iter(dataset.take(1)))

        x = x.numpy()
        y = y.numpy()
        y_argsort = np.argsort(np.max(np.sum(y, axis = 2), axis = 1))

        self.x_test = x[y_argsort[-number_examples:],...]
        self.y_true = y[y_argsort[-number_examples:],...]

        self.writer = tf.summary.create_file_writer(logs)

    def on_epoch_end(self, epoch, logs = None):

        if (epoch+1)%self.plotting_freq != 0:
            return

        y_pred = self.model(self.x_test, training = False)

        example_tracks = _plot_multiregion_callback(self.y_true, y_pred)

        with self.writer.as_default():
            tf.summary.image("Example Track", example_tracks, step=epoch)

        plt.close('all')

#%%
def _interpolate_colors(x, rgb_max):
    rgb_min = np.ones(shape = (1,1,3))
    np.reshape(rgb_max, (1,1,3))
    x = x[...,np.newaxis]
    rgb_step = rgb_max - rgb_min
    return rgb_min + x*rgb_step

def _base_heatmap(x):
    
    # order A, C, G, T

    BASE_COLS = np.array(
    [[16, 150, 72],
    [37, 92, 153],
    [247, 179, 43],
    [214, 40, 57]])/255    

    fig = plt.figure(figsize=(10, 4*x.shape[0]), dpi = 100)    

    hm = x - np.min(x, axis = -1)[...,np.newaxis]
    hm = hm / np.sum(hm, axis = -1)[...,np.newaxis]

    hm = np.stack([_interpolate_colors(hm[...,i], BASE_COLS[i,]) for i in range(4)], axis = -2)

    for f in range(x.shape[0]):
        ax = fig.add_subplot(x.shape[0], 1, f + 1)
        hm_slice = np.transpose(hm[f,...], (1,0,2))
        _ = ax.imshow(hm_slice, aspect = "auto")
        ax.set_yticks(np.arange(4))
        ax.set_yticklabels(["A", "C", "G", "T"])
        lab = np.round(np.transpose(x[f,...], (1,0)), 2)
        for i in range(lab.shape[0]):
            for j in range(lab.shape[1]):
                text = ax.text(j, i, lab[i, j],
                            ha="center", va="center", color="black")

    fig.tight_layout()
    fig.canvas.draw()
    return _matrix_forom_figure(fig)

#%%
def _rel_freq(x):
    x = x - np.min(x, axis = 1)[:,np.newaxis]
    return x/x.sum(axis = 0)

def _shannon_ent(x):
    x = _rel_freq(x)
    h = x * np.ma.log2(x).filled(0)
    return -h.sum(axis = 0)

def _information_cont(x, use_correction = False):
    if use_correction:
        en = (3/(2*len(x)*np.log(2)))
    else:
        en = 0
    return 2 - _shannon_ent(x) - en

def _plot_seq_mpl(motives):
    fig = plt.figure(figsize=(10, 4*motives.shape[0]), dpi = 100) 

    for f in range(motives.shape[0]):
        y = np.transpose(motives[f,...])
        m = _rel_freq(y)*_information_cont(y)
        x = np.arange(y.shape[1])

        ax = fig.add_subplot(motives.shape[0], 1, f + 1)

        a = ax.bar(x, m[0,:], color = "#109648")
        c = ax.bar(x, m[1,:], bottom = m[0,:], color = "#255c99")
        g = ax.bar(x, m[2,:], bottom = m[0,:] + m[1,:], color = "#f7b32b")
        t = ax.bar(x, m[3,:], bottom = m[0,:] + m[1,:] + m[2,:], color = "#d62839")

        ax.set_xlabel('position')
        ax.set_ylim(0, 2.1)
        ax.legend((a,c,g,t), ("A", "C", "G", "T"))

    fig.canvas.draw()
    return _matrix_forom_figure(fig)

# %%
def _draw_matrix_fig(mat):
    mat = np.squeeze(mat, axis = 0)
    fig, ax = plt.subplots(figsize = tuple(i//300 for i in mat.shape[0:2]), dpi = 300)
    ax.set_axis_off()
    ax.imshow(mat)

def _make_propper(a):
    # makes a numpy array compatible with the model
    a = tf.convert_to_tensor(a[np.newaxis,...])
    return tf.cast(a, dtype=tf.float64)

def _extract_filter_motive(filter_active_argmax, filter_strand, region, motive_len_h):
    filter_strand = filter_strand[filter_active_argmax,...]
    filter_strand = np.argmax(filter_strand, axis = -1)

    out = []
    for j, i in enumerate(filter_active_argmax):
        if filter_strand[j] == 0:
            out_temp = region[i-motive_len_h:i+motive_len_h]
        else:
            out_temp = np.flip(region[1+i-motive_len_h:i+motive_len_h+1])
        out.append(out_temp)

    out = np.stack(out)
    out = (out - np.min(out))//np.max(out)
    out = out//np.max(out)
    return np.sum(out, axis = 0)


def _extract_top_motives(conv_layer, pool_layer, region, n_top = 50):
    motive_len_h = conv_layer.get_weights()[0].shape[0]//2

    # extract from the model the convolutional activation
    strand = conv_layer(_make_propper(region))
    # max pool fwd and rv activation
    active = pool_layer(strand)

    # squeeze batch size
    active = np.squeeze(active.numpy(), axis = 0)
    strand = np.squeeze(strand.numpy(), axis = 0)

    # mask ends
    active[:motive_len_h,:] = -np.inf
    active[-motive_len_h:,:] = -np.inf

    # get the position of the 50 windows containing the max activation
    out = []
    for f in range(active.shape[-1]):
        filter_active_argmax = active[:,f].argsort()[-n_top:]
        out.append(
            _extract_filter_motive(
                filter_active_argmax,
                strand[...,f],
                region, motive_len_h
                )
            )

    return np.stack(out)

def _pad_region(region, pad_size):
    padded_region = np.concatenate([
        np.zeros(shape = (pad_size, 4)),
        region,
        np.zeros(shape = (pad_size, 4))],
        axis = 0
    )
    return padded_region

def _top_hits_maxl(results, true_filter):
    true_filter = true_filter+1
    true_filter = true_filter/np.sum(true_filter, axis = 1)[:,np.newaxis]
    sample_count = np.sum(results)//results.shape[0]
    padded_results = np.concatenate([results,
        np.ones(shape = (true_filter.shape[0]-1,4)) * sample_count/4],
        axis = 0)
    poly_fw = np.stack([
        np.convolve(padded_results[:,i], 
            np.log(true_filter[:,i]), mode = "valid")
        for i in range(4)], axis = -1)
    convoluted_ll_fw = (np.sum(poly_fw, axis = 1) 
        )
    padded_results = np.flip(padded_results)
    poly_rv = np.stack([
        np.convolve(padded_results[:,i], 
            np.log(true_filter[:,i]), mode = "valid")
        for i in range(4)], axis = -1)
    convoluted_ll_rv = (np.sum(poly_rv, axis = 1) 
        )
    return (np.max(convoluted_ll_fw), np.max(convoluted_ll_rv))

def _orient_pwm(pwm):
    pwm_sum = np.sum(pwm, axis = 0)
    if pwm_sum[0] + pwm_sum[1] > np.sum(pwm_sum)/2:
        return np.flip(pwm)
    return pwm

# %%
class PlotBindingCallback(tf.keras.callbacks.Callback):
    def __init__(self, train_data, logs, true_pwm = None,
                 sample_size = 128, conv_layer = 0, pool_layer = 1,
                 n_top = 500, plotting_freq = 1):
        super(PlotBindingCallback, self).__init__()

        scanning_region = np.concatenate([i[0] for i in train_data.take(sample_size)])
        self.scanning_region = np.reshape(scanning_region, (-1,4)).astype(np.int32)

        self.conv_layer = conv_layer

        self.pool_layer = pool_layer

        self.n_top = n_top

        self.writer = tf.summary.create_file_writer(logs)

        self.plotting_freq = plotting_freq

        self.pwm = true_pwm

    def on_epoch_end(self, epoch, logs = None):

        if (epoch+1)%self.plotting_freq != 0:
            return

        conv_layer = self.model.get_layer(index = self.conv_layer)
        pool_layer = self.model.get_layer(index = self.pool_layer)

        pad_length = conv_layer.get_weights()[0].shape[0]

        top_motives = _extract_top_motives(conv_layer, pool_layer, 
            _pad_region(self.scanning_region, pad_length), 
            self.n_top)

        ll = []
        if self.pwm is not None:
            for i in range(top_motives.shape[0]):
                ll.append(_top_hits_maxl(top_motives[i,...], self.pwm))
                if ll[i][1] > ll[i][0]:
                    top_motives[i,...] = np.flip(top_motives[i,...])
        
        if self.pwm is None:
            for i in range(top_motives.shape[0]):
                top_motives[i,...] = _orient_pwm(top_motives[i,...])

        with self.writer.as_default():            
            im = _combine_image_matrix([[_base_heatmap(top_motives),
                _plot_seq_mpl(top_motives)]]).astype(np.uint8)
            tf.summary.image("Sequence Motive Summary", im, step=epoch)
        
        plt.close('all')

