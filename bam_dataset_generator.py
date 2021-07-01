#%%
import pandas as pd
import numpy as np

# for bias weight calculation
from scipy import stats
from scipy.signal import gaussian

from .MACS_Import import _import_MACS_narrowpeak, _MACS_positive_regions

'''
Generator based data set for learning:
General idea:

Read bedgraph signal files:
One large signal track is generated of dim 
(sum_of_chromosomes/SIGNAL_STEP_SIZE, number_of_bams)

Read preprocessed array of one hot encoded fasta
dim: (sum_of_chromosomes, 4)

Generate weigths for randomly choosing intervals:
1. mask the regions exluded for analysis:
    - borders of the stiched together chromosomes
    - Regions with Ns
    - Regions with high input signal
2. choose a base propability of every bin to be selected
3. calculate the bias for high signal bins as follows:
    - for every track:
        - log2 ration ChIP/Input (with pseudo counts)
        - calculate z-score
        - select the x highest bins/steps
        - set all others to 0
        - convolute signal and shift by INTERVAL_LENGTH/2
        - normalize to sum = 1
    - sum bam tracks
=> combine mask (including bias ratio)

In Generator:
Use weights to choose bin b
yield 
(
    fasta[b*SIGNAL_STEP_SIZE:b*SIGNAL_STEP_SIZE+INTERVAL_LENGTH,:]
    bam[b:b+INTERVAL_LENGTH/SIGNAL_STEP_SIZE,:]

)

'''

#%% reading in bed graph files

'''
bed graph files are required
They can be generated using deeptools
e.g.
bamCoverage 
    -b path/to/bam 
    -o path/to/output 
    -of bedgraph 
    -bs StepSize 
    -p n_cores 
    -e --centerReads
'''

def _bg_worker(bg_path, selection, signal_step_size):
    '''
    worker to read in bed graph file
    on the way it:
        - filters chromosomes to selection
        - fixes the spacing to get evenly spaced signal tracks
            along intervals of SIGNAL_STEP_SIZE
    '''
    bg = pd.read_csv(bg_path, sep = "\t", header = None, 
                            names = ["chr", "start", "end", "int"])
    bg = bg[bg.chr.isin(selection)]
    return np.repeat(bg.int, 1+(bg.end-bg.start-1)//signal_step_size)

def get_signal_from_bg(bg_list, selection, signal_step_size):
    '''
    calls _bg_worker for each file in bg_list
    then returns array of shape
    (number of genomic bins, number of bg files)
    '''
    signals = [_bg_worker(bg_path, selection, signal_step_size) 
                for bg_path in bg_list]
    return np.stack(signals, axis = 1)


#%% Generate selection bias weigths

def generate_border_mask(chr_sizes, selection, 
                        signal_step_size, interval_length):
    '''
    creates a mask to exclude chromosome borders from being selected

    chr_sizes: path for file containing chromosome sizes
    selection: list of chromosomes to include 

    returns mask of dim:
    (sum_of_chromosome_length/SIGNAL_STEP_SIZE)
    Is 0 where the chromosome end is less then 
    INTERVAL_LENGTH/SIGNAL_STEP_SIZE away
    else 1
    '''
    chr_sizes = pd.read_csv(
        chr_sizes, sep = "\t", header = None, 
        index_col = 0, names = ["length"])
    mask = np.array([])
    for chr in selection:
        up = (int(chr_sizes.loc[chr])-interval_length)//signal_step_size
        if int(chr_sizes.loc[chr])%signal_step_size == 0:
            zero_fill = interval_length//signal_step_size 
        else:
            zero_fill = interval_length//signal_step_size + 1
        mask = np.append(mask, [1]*up)
        mask = np.append(mask, [0]*zero_fill)
    return mask

def _log_norm(array):
    '''
    Returns the array with normalized log transformed read data
    '''
    array = array + 1
    array_sum = np.sum(array, axis=0, keepdims = True)
    array = 1e6*array/array_sum
    return np.log2(array)

def _row_top_mask(array, n):
    '''
    use like this:
    np.apply_along_axis(_row_top_mask, axis = 0, array, n)

    generates a rowwise mask:
    is 0 if that bins signal is less then the n-th value of that row
    else 1
    '''
    cutoff = np.sort(array)[-n]
    mask = np.ones(array.shape)
    mask[array < cutoff] = 0
    return mask

def _row_shifted_gauss_filter(array, signal_interval_length, 
                                width_scaling = 1, sigma_scaling= 10):
    '''
    takes a signal row from an array
    applies a gaussian blur to that row with sigma = INTERVAL_LENGTH//10
    then shifts everything be INTERVAL_LENGTH/2
    '''
    width = signal_interval_length * width_scaling
    sigma = signal_interval_length//sigma_scaling
    gaussian_fil = gaussian(width, sigma)
    conv = np.convolve(array, gaussian_fil, mode="same")
    return np.concatenate((conv[signal_interval_length//2:], 
                           np.zeros(signal_interval_length//2)))

def _MACS_positive_mask(MACS_df, interval_length, 
                        chr_sizes, selection,
                        signal_step_size):

    ## adjust the coordinates according to chrom length
    ### read in files and select chromosomes
    chr_sizes_df = pd.read_csv(
        chr_sizes, sep = "\t", header = None, 
        index_col = 0, names = ["length"])
    chr_sizes_df = chr_sizes_df.loc[chr_sizes_df.index.isin(selection)]

    # extend chromosomes to match signal step size
    chr_sizes_df.length += ((signal_step_size - chr_sizes_df.length)%signal_step_size) 
    chr_sizes_df["cum_length"] = [0] + list(chr_sizes_df.length.cumsum())[:-1]

    for chrom in chr_sizes_df.index:
        if chrom in list(MACS_df.chr):
            MACS_df.loc[MACS_df.chr == chrom, "start"] += (
                chr_sizes_df.loc[chrom, "cum_length"]
            )
            MACS_df.loc[MACS_df.chr == chrom, "end"] += (
                chr_sizes_df.loc[chrom, "cum_length"]
            )

    ## empty mask array
    positive_mask = np.zeros((sum(chr_sizes_df.length)))

    for row in MACS_df.itertuples():
        positive_mask[row.start:row.end] = 1

    return np.max(
            np.reshape(positive_mask, 
                (positive_mask.shape[0]//signal_step_size, signal_step_size)), 
            axis = 1)

def generate_bias_mask(bam_signal, input_signal, ntop, 
              signal_interval_length, MACS_regions_array = None,
              width_scaling = 1, sigma_scaling= 10):
    '''
    uses all functions above to calculate the     
    bias weights for high signal bins

    - for every track:
        - log2 ration ChIP/Input (with pseudo counts)
        - calculate z-score
        - select the x highest bins/steps
        - set all others to 0
        - convolute signal and shift by signal_interval_length/2
        - normalize to sum = 1
    - sum bam tracks

    returns mask of dim:
    (sum_of_chromosome_length/SIGNAL_STEP_SIZE)
    '''

    #calculate log2 ration ChIP/Input (with pseudo counts)
    if input_signal.ndim == 1:
        input_signal = input_signal[:,np.newaxis]
        
    bam_norm = _log_norm(bam_signal)
    input_norm = _log_norm(input_signal)
    log_ratio_signal = bam_norm - input_norm

    # z-score the log2 ratio rows   
    log_ratio_signal = log_ratio_signal/(np.std(log_ratio_signal, axis = 0)[np.newaxis,:])
    log_ratio_signal += -np.min(log_ratio_signal, axis = 0)[np.newaxis,:]

    if MACS_regions_array is not None:
        top_mask = MACS_regions_array[:,np.newaxis]
    
    if MACS_regions_array is None:
        # create mask of bins with signal >= that of the ntop-th bin
        top_mask = np.apply_along_axis(_row_top_mask, 0, 
                                    log_ratio_signal, ntop)

    # apply mask to z-scored normalized signal,
    # then gauss-blur and shift
    filtered_signal = np.apply_along_axis(
        _row_shifted_gauss_filter, 0, log_ratio_signal*top_mask,
        signal_interval_length= signal_interval_length, 
        width_scaling = width_scaling, 
        sigma_scaling= sigma_scaling)  

    # normalize rows to 1
    filtered_sum = np.sum(filtered_signal, axis=0)[np.newaxis,:]
    filtered_signal = filtered_signal/filtered_sum
    filtered_signal = np.sum(filtered_signal, axis=1)

    # return the sum of signal tracks normalized to 1
    return filtered_signal/np.sum(filtered_signal)

#%% filter regions with high background

def _filter_row_wise(row, sigma_factor):
    '''
    in row find elements with more reads than
    sigma * 95percentile
    '''
    row_mean = np.mean(row)
    row_sd = np.std(row)

    max_cutoff = sigma_factor*np.quantile(row, 0.95)

    return row > max_cutoff

def generate_background_mask(input_signal, interval_length, sigma_factor = 3):
    '''
    mask all regions (by conv) where any of the input bams have signal
    larger than sigma * 95percentile
    '''
    if input_signal.ndim == 1:
        input_signal = input_signal[:,np.newaxis]

    mask = np.apply_along_axis(
        lambda x: _filter_row_wise(x, sigma_factor), 
        axis = 0, arr = input_signal)

    mask = np.convolve(np.sum(mask, axis = 1), 
        np.ones(shape = (interval_length//2)), 
        mode="same").astype(bool)
    return np.ones(shape = mask.shape) - mask

#%%
def generate_N_mask(fasta, step_size):
    '''
    mask all regions where the starting interval contains an N
    '''
    n_mask = np.sum(fasta, axis=1) != 0
    return np.min(np.reshape(
        n_mask, (fasta.shape[0]//step_size, step_size)), axis = 1)
   

#%%

def combine_masks(border_mask, bias_mask, background_mask, N_mask, bg_frac):
    '''
    combines the masks for the final weigth of drawing
    border mask: output of created_border_mask
    bias_mask: output of bias_prop
    bg_frac: the fraction of intervals that should be samples uniformly

    returns the finsal weights for interval drawing
    shape: (sum_of_chromosome_length/SIGNAL_STEP_SIZE)
    '''
    n_bin = bias_mask.shape[0]
    bg_prop = bg_frac / n_bin
    bg_mask = bg_prop* np.ones(shape = n_bin)
    bias_mask *= 1 - bg_frac
    mask = (bias_mask + bg_mask)*border_mask*background_mask * N_mask
    return mask/np.sum(mask)


#%% generator

'''
This version requires one-hot encoded fasta in memory
(generated by get_one_hot_fasta function)
=> larger memory footprint (32x), but faster (3x)
'''

def _jagged_slicer(array, choice, len):
    return np.stack([array[i:i+len,:] for i in choice])

def batch_slice_generator_performance(fasta, bam, weights, 
                          interval_length, signal_step_size, 
                          bs = 64, n_batches = 10000):
    along = np.arange(0, weights.shape[0], 1)    
    while True:
        sb = np.random.choice(along, size = (n_batches, bs), p=weights)
        sf = sb * signal_step_size
        for i in range(n_batches):
            yield (_jagged_slicer(fasta, sf[i,:], interval_length), 
                _jagged_slicer(bam, sb[i,:], interval_length//signal_step_size))

#%% dataset_class

class bam_signal_dataset(object):
    '''
    This class combines all the above functions to make it easy to
    produce a generator for learning
    it takes:
    fasta: a one hot encoded fasta
    chip_bam_files: a list of chip bedgraph files
    input_bam_files: a list of input bedgraph files
    chr_selection: a list of chromosome names
    chr_sizes_file: a file that contains chromosome sizes
    fasta_interval_length: the length of the target interval in bp
    signal_step_size: the step size of the bedgraph file
    narrowpeaks: optional, a MACS2 narrowpeaks file to use 
        for region selection
    top_positiv_regions: number of top regions to use as positive regions,
        overwritten by the narrowpeaks option,
    background_frac: fraction of samples that will be derived from background

    after init, generators for use with tf can be 
        made using the make_dataset method 
    '''

    def __init__(self,
                 fasta,
                 chip_bam_files,
                 input_bam_files,
                 chr_selection,
                 chr_sizes_file,
                 fasta_interval_length,
                 signal_step_size,
                 narrowpeaks = None,
                 top_positiv_regions = 20000,
                 background_frac = 0.3):

        self.fasta = fasta

        self.bam_signal = get_signal_from_bg(chip_bam_files, 
            chr_selection, signal_step_size)  

        self.signal_step_size = signal_step_size

        self.interval_length = fasta_interval_length  

        input_signal = get_signal_from_bg(input_bam_files, 
            chr_selection, signal_step_size)

        if narrowpeaks is not None:  
            MACS_df = _import_MACS_narrowpeak(narrowpeaks, chr_selection)
            narrowpeaks = _MACS_positive_mask(MACS_df, 
                fasta_interval_length, chr_sizes_file, chr_selection,
                signal_step_size) 

        masks = (
            generate_border_mask(chr_sizes_file, chr_selection, 
                signal_step_size, fasta_interval_length), 
            generate_bias_mask(self.bam_signal, 
                np.sum(input_signal, axis = 1), top_positiv_regions, 
                fasta_interval_length//signal_step_size, narrowpeaks),
            generate_background_mask(input_signal, fasta_interval_length),
            generate_N_mask(fasta, signal_step_size)
        )

        print(
            "fraction of regions masked because of Ns: %.2f \n" % (
            (masks[3].shape[0] - np.sum(masks[3]))/masks[3].shape[0]
            )
        )

        print(
            "fraction of regions masked because of chromosome borders: %.2f \n" % (
            (masks[0].shape[0] - np.sum(masks[0]))/masks[0].shape[0]
            )
        )

        print(
            "fraction of regions masked because of high input signal: %.2f \n" % (
            (masks[2].shape[0] - np.sum(masks[2]))/masks[2].shape[0]
            )
        )

        biased_regions = (masks[1] > 0) * masks[0] * masks[2] * masks[3]

        print(
            "fraction of regions selected as signal: %.2f \n" % (
            np.sum(biased_regions)/masks[1].shape[0]
            )
        )


        bg_regions = (np.ones(masks[0].shape) * 
            (masks[1] == 0) * masks[0] * masks[2] * masks[3]
        )

        print(
            "fraction of regions selected as background: %.2f \n" % (
            np.sum(bg_regions)/masks[1].shape[0]
            )
        )

        self.mask = combine_masks(*masks, background_frac)
    
    def make_generator(self,
                       batch_size = 64,
                       n_batches = 10000):
        return batch_slice_generator_performance(
            self.fasta, self.bam_signal, self.mask,
            self.interval_length, self.signal_step_size,
            batch_size, n_batches
        )
# %%
