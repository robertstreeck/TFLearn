#%%

import numpy as np
from scipy import stats
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt

from .tfbinding_generator import _jagged_slicer, _log_norm

import tensorflow as tf


#%%
def _matrix_forom_figure(fig):
    '''
    takes fig object and returns a np array 
    for tensorboard plotting, with shape
    (1, shape_y*dpi, shape_x*dpi, 3)
    '''
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape((1,) + fig.canvas.get_width_height()[::-1] + (3,))
    return data
#%%
'''
take:

a bam array of ChIP read counts of shape
(number_of_intervals, bam_files)

optional:
a matching input array of shape
(number_of_intervals, 1 OR bam_files)

optional:
a one hot fasta of shape
((number_of_intervals * step_size, 4)

optional
a matrix of the true pattern

'''
def select_top_centers(log_signal, top_bins, bin_width):
    sum_signal = np.sum(log_signal, axis = 1)[bin_width:-bin_width]
    args_sorted = np.flip(sum_signal.argsort())
    '''
    too slow on large arrays
    mask = np.ones(args_sorted.shape, dtype = np.bool)
    i = 0
    while i < top_bins:
        args_sorted[i] = args_sorted[mask][i]
        mask[(args_sorted > args_sorted[i] - half_width) & (args_sorted < args_sorted[i] + half_width)] = False
        mask[i] = True
        i += 1
    '''
    return args_sorted[0:top_bins] + bin_width

def stacked_quantile_sums(signal, center, width, spacing, groups):
    half_width = width//2
    stacked_signals = _jagged_slicer(signal, center-half_width, width)

    if spacing == "log":
        up = np.power(10, np.linspace(0, np.log10(center.shape[0]), groups)).astype(int)
        low = np.append([0], up[0:groups-1])
    else:
        low = np.arange(groups) * center.shape[0]//groups
        up = low + center.shape[0]//groups

    # dim: (groups, width, number_of_signal_tracks)
    return np.stack([np.mean(stacked_signals[low[i]:up[i]], axis=0) for i in range(groups)])

def numpy_maxpool_1d(x, pool_size):
    return np.max(np.stack(np.split(x, pool_size)), axis = -1)

def convolve_activation(fasta, center, convolution, width, step_size, spacing, groups):
    '''
    this still does not really work, propably because finding the
    correct convolutional matrix is difficult
    '''
    fasta_width = width * step_size
    half_width = fasta_width//2
    convolve_fasta = _jagged_slicer(fasta, (center*step_size)-half_width, fasta_width)

    for i in range(4):
        convolve_fasta[...,i] = np.apply_along_axis(
            lambda x: np.convolve(x, convolution[:,i], mode = "same"),
            axis = 1, arr=convolve_fasta[...,i])
    
    convolve_fasta = np.sum(convolve_fasta, axis = -1)

    convolve_fasta = np.apply_along_axis(
            lambda x: numpy_maxpool_1d(x, width),
            axis = 1, arr=convolve_fasta)

    if spacing == "log":
        up = np.power(10, np.linspace(0, np.log10(center.shape[0]), groups)).astype(int)
        low = np.append([0], up[0:groups-1])
    else:
        low = np.arange(groups) * center.shape[0]//groups
        up = low + center.shape[0]//groups

    # dim: (groups, width, number_of_signal_tracks)
    return np.stack([np.mean(convolve_fasta[low[i]:up[i]], axis=0) for i in range(groups)])


def plot_tracks_by_quantile(chip_bam, 
                 input_bam = 'None', 
                 fasta_one_hot = 'None',
                 convolution_matrix = 'None',
                 top_bins = 500,
                 spacing = "log",
                 groups = 5,
                 bin_width = 1000,
                 plot_signal = "log",
                 select = "log_dif"):
    # sanitize        

    if not np.isscalar(input_bam):        
        chip_norm = _log_norm(chip_bam)
        input_norm = _log_norm(input_bam)
        log_signal = chip_norm - input_norm
    else:
        chip_norm = _log_norm(chip_bam)
        log_signal = chip_norm

    if select == "log_dif":
        top_centers = select_top_centers(log_signal, top_bins, bin_width)
    else:
        top_centers = select_top_centers(chip_norm, top_bins, bin_width)

    fig = plt.figure(figsize=(32, 16), dpi = 300)

    x = np.arange(bin_width)

    if plot_signal == "log":
        chip_plot = stacked_quantile_sums(
            chip_norm, top_centers, bin_width, spacing, groups)
        if not np.isscalar(input_bam): 
            input_plot = stacked_quantile_sums(
                input_norm, top_centers, bin_width, spacing, groups)
    elif plot_signal == "raw":
        chip_plot = stacked_quantile_sums(
            chip_bam, top_centers, bin_width, spacing, groups)
        if not np.isscalar(input_bam): 
            input_plot = stacked_quantile_sums(
                input_bam, top_centers, bin_width, spacing, groups)
    else:
        chip_plot = stacked_quantile_sums(
            log_signal, top_centers, bin_width, spacing, groups)
    
    for region in range(groups):
        for bam in range(chip_plot.shape[2]):
            ax = fig.add_subplot(groups, chip_plot.shape[2], 
                region*chip_plot.shape[2] + bam + 1)

            yt = chip_plot[region,:,bam]

            if plot_signal in ["raw", "log"] and not np.isscalar(input_bam):
                yp = input_plot[region,:,min(bam, input_plot.shape[2]-1)]

                ax.fill_between(x, yt, yp, where=yt >= yp, facecolor='#a1d99b30')
                ax.fill_between(x, yt, yp, where=yt <= yp, facecolor='#fc927230')

                ax.plot(yp, label='input', color = "#984ea3")

            ax.plot(yt, label='ChIP', color = "#377eb8")

            if convolution_matrix != 'None' and fasta_one_hot != 'None':
                ax2 = ax.twinx()
                convolved_fasta = convolve_activation(fasta_one_hot, 
                    top_centers, convolution_matrix, bin_width, 
                    fasta_one_hot.shape[0]//log_signal.shape[0], spacing, groups)
                ax2.plot(convolved_fasta[region,:], label='input', color = "#a6761d")

            ax.legend(loc = 'upper left')

            ax.text(.5, .9,
                'Quantile {}, Bam {}'.format(region+1, bam+1),
                horizontalalignment='center',
                transform=ax.transAxes)

    fig.tight_layout()
    return fig

#%%  
def log_count_histogram(chip_bam, input_bam = 'None', 
                        mask = 'None', bins = 50, 
                        log = True, reduce = 10000):
    
    if not np.isscalar(input_bam):
        total_plots = chip_bam.shape[1] + input_bam.shape[1]
    else:
        total_plots = chip_bam.shape[1]

    fig = plt.figure(figsize=(6,4*total_plots), dpi = 100)

    if reduce:
        choose = np.random.choice(np.arange(chip_bam.shape[0]), reduce, replace=False)
        chip_bam = chip_bam[choose,:]
        if not np.isscalar(input_bam):
            input_bam = input_bam[choose,:]
        if not np.isscalar(mask):
            mask = mask[choose]

    if log:
        chip_bam = _log_norm(chip_bam)
        if not np.isscalar(input_bam):
            input_bam = _log_norm(input_bam)

    for i in range(chip_bam.shape[1]):
        ax = fig.add_subplot(total_plots, 1, i+1)
        if not np.isscalar(mask):
            hist_list = [
                np.squeeze(chip_bam[mask.astype("bool"),i]),
                np.squeeze(chip_bam[np.logical_not(mask.astype("bool")),i])
            ]
            ax.hist(hist_list, bins=bins, stacked=True, 
                label = ["not masked", "masked"])
        else:
            ax.hist(chip_bam[:,i], bins=bins)
        ax.text(.8, .9,
                "ChIP {}".format(i),
                horizontalalignment='center',
                transform=ax.transAxes)
    
    if not np.isscalar(input_bam):
            for i in range(input_bam.shape[1]):
                ax = fig.add_subplot(total_plots, 1, chip_bam.shape[1] + i+1)
                if not np.isscalar(mask):
                    hist_list = [
                        np.squeeze(input_bam[mask.astype("bool"),i]),
                        np.squeeze(input_bam[np.logical_not(mask.astype("bool")),i])
                    ]
                    ax.hist(hist_list, bins=bins, stacked=True, 
                        label = ["not masked", "masked"])
                else:
                    ax.hist(input_bam[:,i], bins=bins)
                ax.text(.8, .9,
                        "Input {}".format(i),
                        horizontalalignment='center',
                        transform=ax.transAxes)
    
    fig.tight_layout()
    return fig

 

# %%
def count_fingerprint(chip_bam, input_bam = 'None', reduce = 10000):

    if reduce:
        choose = np.random.choice(np.arange(chip_bam.shape[0]), reduce, replace=False)
        chip_bam = chip_bam[choose,:]
        if not np.isscalar(input_bam):
            input_bam = input_bam[choose,:]

    chip_bam = np.cumsum(np.sort(chip_bam, axis = 0), axis = 0)
    chip_bam = chip_bam/chip_bam[-1,:]
    if not np.isscalar(input_bam):
        input_bam = np.cumsum(np.sort(input_bam, axis = 0), axis = 0)
        input_bam = input_bam/input_bam[-1,:]

    fig, ax = plt.subplots(figsize=(5, 5), dpi = 100)

    ax.plot(np.linspace(0,1,reduce), label = "diagonal", linestyle = '--')

    for i in range(chip_bam.shape[1]):
        ax.plot(chip_bam[:,i], label = "ChIP {}".format(i))

    if not np.isscalar(input_bam):
        for i in range(input_bam.shape[1]):
            ax.plot(input_bam[:,i], label = "Input {}".format(i))

    ax.legend(loc = 'upper left')
    ax.set_title("Fingerprint of ChIP and Input signals")

    fig.tight_layout()
    return fig 


# %%
def pie_chart_mask(mask):
    zero_bins = np.sum(mask == 0)
    background_prop = np.sum(mask == stats.mode(mask[mask > 0])[0][0])
    sizes = [100*zero_bins/mask.shape[0], 
        100*background_prop/mask.shape[0],
        100*(mask.shape[0] - zero_bins - background_prop)/mask.shape[0]]
    labels = ["zero", "background", "signal"]
    fig, ax = plt.subplots(figsize=(5, 5), dpi = 100)
    ax.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.set_title("Fractions of bins by masking state")
    return fig
    
# %%
def bar_chart_mask(mask_tuple, full_mask):
    border_mask, bias_mask, background_mask, N_mask = mask_tuple
    n_bin = bias_mask.shape[0]
    bg_prop = 0.1 / n_bin
    bg_mask = bg_prop* np.ones(shape = n_bin)
    bm = bias_mask * 0.9
    bm = bm + bg_mask

    values = [np.sum(bm != 0)]
    values += [np.sum(bm*border_mask != 0)]
    values += [np.sum(bm*background_mask*border_mask != 0)]
    values += [np.sum(bm*background_mask*border_mask*N_mask != 0)]
    values += [np.sum(full_mask != 0)]

    #values = np.log10(values)

    names = ["bias", "* border", "* background", "* N mask", "= full mask"]
    fig, ax = plt.subplots(figsize=(5, 5), dpi = 100)
    ax.bar(names, values)
    ax.plot(names, values, color = "orange")
    ax.set_xlabel('Applied Masks')
    ax.set_ylabel('Number of unmasked regions')
    ax.set_title('Cummulative count of bin with non-zero mask')
    return fig

# %%
def prop_histogram(mask, bins = 50, log = True):
    non_zero = mask[mask > stats.mode(mask[mask > 0])[0][0]]
    if log:
        non_zero = np.log10(non_zero)
    fig, ax = plt.subplots(figsize=(5, 5), dpi = 100)
    ax.hist(non_zero, bins = bins)
    ax.set_xlabel('Draw propability')
    ax.set_ylabel('Number of regions with signal')
    ax.set_title('Propability of draws for non-zero, non-background region')
    return fig

#%%
def plot_mask_quantiles(signal, mask, 
                        signal_interval_length,
                        groupsize = 1000, 
                        groups = 3):
    mask_argsort = np.flip(np.argsort(mask)[-groupsize*groups:])
    plot_array = _jagged_slicer(signal, 
        mask_argsort, 
        signal_interval_length)

    means = np.stack([
        np.mean(plot_array[i*groupsize:(i+1)*groupsize], axis = 0)
        for i in range(groups)])

    sds = np.stack([
        np.std(plot_array[i*groupsize:(i+1)*groupsize], axis = 0)
        for i in range(groups)])

    fig = plt.figure(figsize=(6, 3*signal.shape[-1]), dpi = 100)

    x = np.arange(signal_interval_length)

    linecols = plt.cm.Dark2(np.arange(groups))

    for bam in range(signal.shape[-1]):
        ax = fig.add_subplot(signal.shape[-1], 1, bam + 1)

        for group in range(groups):
            ax.fill_between(x, 
                means[group,:,bam] - sds[group,:,bam],
                means[group,:,bam] + sds[group,:,bam],
                facecolor = linecols[group,:], alpha = .2)

        for group in range(groups):
            ax.plot(means[group,:,bam],
                color=linecols[group,:],
                label = "Top {}".format((group+1)*groupsize))

            ax.legend(loc = 'upper left',
                fontsize="x-small")

            ax.text(.5, .9,
                'ChIP replicate {}'.format(bam+1),
                horizontalalignment='center',
                transform=ax.transAxes)
    return fig

#%%
def plot_dataset_sample(dataset, n_batches = 20):
    samples = np.concatenate(
        [i[1] for i in dataset.take(n_batches)], axis = 0)
    sample_mean = np.mean(samples, axis = 0)
    sample_sd = np.std(samples, axis = 0)

    x = np.arange(sample_mean.shape[0])

    fig = plt.figure(figsize=(6, 3*samples.shape[-1]), dpi = 100)
    for bam in range(samples.shape[-1]):
        ax = fig.add_subplot(samples.shape[-1], 1, bam + 1)

        ax.fill_between(x, 
            sample_mean[:,bam] - sample_sd[:,bam],
            sample_mean[:,bam] + sample_sd[:,bam],
            facecolor = "grey", alpha = .2)

        ax.plot(sample_mean[:,bam], color="red",
            label = "mean")

        group_min = samples[np.argmin(np.sum(samples[...,bam], axis = 0)),:,bam]
        ax.plot(group_min, color="blue")

        group_max = samples[np.argmax(np.sum(samples[...,bam], axis = 0)),:,bam]
        ax.plot(group_max, color="green")

        ax.legend(loc = 'upper left',
            fontsize="x-small")

        ax.text(.5, .9,
            'ChIP replicate {}'.format(bam+1),
            horizontalalignment='center',
            transform=ax.transAxes)

    fig.canvas.draw()
    
    return fig

#%%
def _combine_image_matrix(image_matrix_list):
    '''
    image_matrix_list is a nested list of arrays to be plotted
    '''
    width, height = 0,0
    for row in image_matrix_list:
        height += max(p.shape[1] for p in row)
        width = max(width, sum(p.shape[2] for p in row))

    combined_image = np.ones(shape = (1, height ,width,3))*255
    line_pos = 0
    for row in image_matrix_list:
        col_pos = 0
        max_line_hight = 0
        for plot in row:
            h, w = plot.shape[1:3]
            combined_image[:,line_pos:line_pos+h, col_pos:col_pos+w,:] = plot
            col_pos += w
            max_line_hight = max(max_line_hight, h)
        line_pos += max_line_hight
    
    return combined_image.astype(np.int)


# %%
def mask_analytics_for_tensorboard(mask, mask_tuple, chip,
                                 signal_interval_length,
                                 target_dir):
    plots = [[_matrix_forom_figure(pie_chart_mask(mask)),
        _matrix_forom_figure(bar_chart_mask(mask_tuple, mask))],
        [_matrix_forom_figure(prop_histogram(mask)),
        _matrix_forom_figure(plot_mask_quantiles(
                chip, mask, signal_interval_length))]]

    writer = tf.summary.create_file_writer(target_dir)

    with writer.as_default():
        tf.summary.image("Mask Analytics ", 
                _combine_image_matrix(plots).astype(np.uint8), step=0)

    plt.close('all')

def raw_data_analytics_tb(chip, input, target_dir, mask = "None"):
    plots = [[_matrix_forom_figure(count_fingerprint(chip, input)),
        _matrix_forom_figure(log_count_histogram(chip, input, mask))],
        [ _matrix_forom_figure(plot_tracks_by_quantile(chip, input))]]

    writer = tf.summary.create_file_writer(target_dir)

    with writer.as_default():
        tf.summary.image("Mask Analytics", 
                _combine_image_matrix(plots).astype(np.uint8), step=0)

    plt.close('all')

def dataset_analytics_tb(dataset, target_dir):

    writer = tf.summary.create_file_writer(target_dir)

    with writer.as_default():
        tf.summary.image("Random Sample from dataset", 
            _matrix_forom_figure(plot_dataset_sample(dataset)), step=0)

    plt.close('all')

# %%

def write_plots_for_tensorboard(chip, input, mask, mask_tuple, 
                                signal_interval_length, target_dir,
                                dataset):

    writer = tf.summary.create_file_writer(target_dir)

    with writer.as_default():
        tf.summary.image("Random Sample from dataset", 
            _matrix_forom_figure(plot_dataset_sample(dataset)), step=0)
        tf.summary.image("Top Regions from Mask", 
            _matrix_forom_figure(plot_mask_quantiles(
                chip, mask, signal_interval_length)), step=0)
        tf.summary.image("Quantile Plot", 
            _matrix_forom_figure(plot_tracks_by_quantile(
                chip, input)), step=0)
        tf.summary.image("Count Histogram", 
            _matrix_forom_figure(log_count_histogram(
                chip, input, mask)), step=0)
        tf.summary.image("Fingerprint", 
            _matrix_forom_figure(count_fingerprint(
                chip, input)), step=0)
        tf.summary.image("Masking Pie Chart", 
            _matrix_forom_figure(pie_chart_mask(mask)), step=0)
        tf.summary.image("Masking Cummulative Groups", 
            _matrix_forom_figure(bar_chart_mask(mask_tuple, mask)), step=0)
        tf.summary.image("Propability Histogram", 
            _matrix_forom_figure(prop_histogram(mask)), step=0)

    plt.close('all')
    
    
    
    
    
    