#%%
import pandas as pd
import numpy as np
import tensorflow as tf

# for bam reading
#import deeptools.countReadsPerBin as crpb
from multiprocessing import Pool
from itertools import repeat
import os

# for fasta one hot encoding
from sklearn.preprocessing import OneHotEncoder

# for bias weight calculation
from scipy import stats
from scipy.signal import gaussian

# to exclude high bad regions

'''
Generator based data set for learning:
General idea:

Read bams:
One large signal track is generated of dim 
(sum_of_chromosomes/SIGNAL_STEP_SIZE, number_of_bams)

Generate weigths for randomly choosing intervals:
1. mask the borders of the stiched together chromosomes
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

Read Fasta into one_hot
dim: (sum_of_chromosomes, 4)

In Generator:
Use weights to choose bin b
yield 
(
    fasta[b*SIGNAL_STEP_SIZE:b*SIGNAL_STEP_SIZE+INTERVAL_LENGTH,:]
    bam[b:b+INTERVAL_LENGTH/SIGNAL_STEP_SIZE,:]

)

'''

#%% Some parameters

# size of steps (in bp) over which to calculate the bam signal
#SIGNAL_STEP_SIZE = 20

# length of intervals to yield
# ideally divisible by 2, 10 and SIGNAL_STEP_SIZE
#INTERVAL_LENGTH = 5000

#%% for test
#CHROMOSOME_SELECTION = ["2L", "2R", "3R"]
#CHROMOSOME_SIZES = "/Users/streeck/Genomes/DmelBDGP6.91/chrNameLength.txt"
#FASTA_FILE = "/Users/streeck/Genomes/DmelBDGP6.91/Drosophila_melanogaster.BDGP6.dna.toplevel.fa"

#%% bam reading function

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

def generate_bias_mask(bam_signal, input_signal, ntop, 
              signal_interval_length, width_scaling = 1, 
              sigma_scaling= 10):
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
    log_ratio_signal = stats.zscore(log_ratio_signal, axis = 0)

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

#%%

def _filter_row_wise(row, sigma_factor):
    row_mean = np.mean(row)
    row_sd = np.std(row)

    max_cutoff = sigma_factor*np.quantile(row, 0.95)

    return row > max_cutoff

def generate_background_mask(input_signal, interval_length, sigma_factor = 3):

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

#%% Fasta file reading

def _ReadFasta(FastaFile):
    '''
    reads fasta file
    returns dictionary of style:
    {"chromosome name": "ATGC..."}
    '''
    FastaSeqs = {}
    with open(FastaFile, "r") as f:
        for line in f:
            if line.startswith(">"):
                try:
                    FastaSeqs[seqid] = seq
                except:
                    pass
                seq = ""
                seqid = str.split(line)[0][1:]
            else:
                seq = seq + line[:-1]
    return FastaSeqs

def get_one_hot_fasta(FastaFile, selection, signal_step_size):
    '''
    Takes a fasta file path and a selection of chromosomes
    returns an one-hot encoded numpy array of the sequence
    with 0 padding at chromosome borders to be compatible with 
    bam step size
    '''

    # read fasta
    fasta = _ReadFasta(FastaFile)

    # set up sklearn one hot encoder
    OneHotFasteEncoder = OneHotEncoder(
        categories = [["A", "C", "G", "T"]], 
        handle_unknown="ignore", sparse = False)
    OneHotFasteEncoder.fit(np.reshape(np.array(list("ACGT")), (-1,1)))

    # empty array for concatenating
    one_hot = np.empty((0,4)).astype(np.int8)

    # for selected chromosomes
    for chr in selection:
        # generate one hot encoded arrays
        fasta_read = np.reshape(np.array(list(fasta[chr])), (-1,1))
        fasta_read = np.array(OneHotFasteEncoder.transform(fasta_read)).astype(np.int8)

        # they need to be padded to be compatible with the bams
        pad = signal_step_size - fasta_read.shape[0]%signal_step_size
        pad = np.zeros((pad, 4)).astype(np.int8)

        # concatenate
        one_hot = np.concatenate([one_hot, fasta_read, pad]) 

    return one_hot


def _string_to_sklearn(string):
    return np.reshape(np.array(list(string)), (-1,1))

def _one_hotEncoder(string):
    OneHotFasteEncoder = OneHotEncoder(
        categories = [["A", "C", "G", "T"]], 
        handle_unknown="ignore", sparse = False)
    OneHotFasteEncoder.fit(_string_to_sklearn("ACGT"))

    oh = OneHotFasteEncoder.transform(_string_to_sklearn(string))

    return np.array(oh).astype(np.int8)    

def get_one_hot_fasta_reduced(FastaFile, selection, signal_step_size):
    '''
    Takes a fasta file path and a selection of chromosomes
    returns an one-hot encoded numpy array of the sequence
    with 0 padding at chromosome borders to be compatible with 
    bam step size
    '''
    # empty array for concatenating
    append = False
    fasta = ""
    with open(FastaFile, "r") as f:
        for line in f:
            if line.startswith(">"):
                if str.split(line)[0][1:] in selection:
                    append = True
                    fasta += "N" * ((signal_step_size - len(fasta))%signal_step_size)
                else:
                    append = False
            elif append:
                fasta = fasta + line[:-1].upper()

    fasta += "N" * ((signal_step_size - len(fasta))%signal_step_size)           
    return _one_hotEncoder(fasta)

def get_one_hot_parallel(FastaFile, selection, 
                            signal_step_size, chromosome_sizes, 
                            n_cores = 1):
    chromosome_sizes = pd.read_csv(
        chromosome_sizes, sep = "\t", header = None, 
        index_col = 0, names = ["length"])

    n_chunks = n_cores

    split = np.array_split(selection, n_chunks)

    while np.any([np.sum(chromosome_sizes.loc[chunk])>2e9 for chunk in split]):
        if n_chunks > 100:
            break
        n_chunks = (1 + n_chunks//n_cores)*n_cores
        split = np.array_split(selection, n_chunks)

    worker = lambda x: get_one_hot_fasta_reduced(FastaFile, x, 
        signal_step_size)
    
    pool = Pool(int(n_cores))
    stack = pool.starmap(get_one_hot_fasta_reduced, 
        zip(repeat(FastaFile),
            split, repeat(signal_step_size)))
    pool.close()
    pool.join()
    return np.concatenate(stack)


def get_string_fasta(FastaFile, selection, signal_step_size):
    '''
    Takes a fasta file path and a selection of chromosomes
    returns an fasta string of the sequence
    with "A" padding at chromosome borders to be 
    compatible with bam step size
    '''

    # read fasta
    fasta = _ReadFasta(FastaFile)

    fasta_string = ""

    # for selected chromosomes
    for chr in selection:

        # they need to be padded to be compatible with the bams
        pad = signal_step_size - len(fasta[chr])%signal_step_size
        pad = "A"*pad

        # join
        fasta_string += fasta[chr] + pad

    return fasta_string


#%% generator

'''
This version requires one-hot encoded fasta in memory
(generated by get_one_hot_fasta function)
=> larger memory footprint (32x), but faster (3x)
'''

def _jagged_slicer(array, choice, len):
    return np.stack([array[i:i+len,:] for i in choice])

def batch_slice_generator(fasta, bam, weights, 
                          interval_length, signal_step_size, 
                          bs = 64):
    along = np.arange(0, weights.shape[0], 1)
    while True:
        sb = np.random.choice(along, size = bs, p=weights)
        sf = sb * signal_step_size
        yield (_jagged_slicer(fasta, sf, interval_length), 
                _jagged_slicer(bam, sb, interval_length//signal_step_size))

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

'''
def _jagged_slicer(array, choice, len):
    slicer = np.linspace(choice, 
        choice + len, num = len).astype(np.int)
    return np.stack([array[i:i+len,:] for i in choice])

def batch_slice_generator_performance_v2(fasta, bam, weights, 
                          interval_length, signal_step_size, 
                          bs = 64, n_batches = 10000):
    along = np.arange(0, weights.shape[0], 1)    
    while True:
        sb = np.random.choice(along, size = (n_batches, bs), p=weights)
        sf = np.linspace(sb*signal_step_size, 
            sb*signal_step_size + interval_length,
            num = interval_length, endpoint = False)
        sb = np.linspace(sb, 
            sb + interval_length//signal_step_size,
            num = interval_length//signal_step_size, endpoint = False) 
        for i in range(n_batches):
            yield (_jagged_slicer(fasta, sf[i,:], interval_length), 
                _jagged_slicer(bam, sb[i,:], interval_length//signal_step_size))
'''
#%%
'''
This version generates one-hot encoding from fasta string
(generated by get_string_fasta function)
an then maps during yielding
=> smaller memory (32x smaller), but slower (3x)
'''

OneHotFasteEncoder = OneHotEncoder(
    categories = [["A", "C", "G", "T"]], 
    handle_unknown="ignore", sparse = False)
OneHotFasteEncoder.fit(np.reshape(np.array(list("ACGT")), (-1,1)))

def _hot_encoder_destacker(s,c,l):
    '''
    to prevent function nesting (because of sklear input/output limitations)
    takes string s and slices from c to c+l
    then generates one-hot encoding for np output
    '''
    f = s[c:c+l]
    f = list(f)
    f = np.array(f)
    f = f[:,np.newaxis]
    f = OneHotFasteEncoder.transform(f)
    return np.array(f)

def _jagged_string_slicer(string, choice, len):
    return np.stack([_hot_encoder_destacker(string, i, len) for i in choice])

def string_batch_slice_generator(fasta_string, bam, weights, 
                                 interval_length, signal_step_size, 
                                 bs = 64):
    along = np.arange(0, weights.shape[0], 1)
    while True:
        sb = np.random.choice(along, size = bs, p=weights)
        sf = sb * signal_step_size 
        yield (_jagged_string_slicer(fasta_string, sf, interval_length), 
                _jagged_slicer(bam, sb, interval_length//signal_step_size))

def _generate_example_for_plot(fasta, signal, 
                               mask, interval_length, 
                               number_examples):
    '''
    generates examples to use for plotting
    '''

    slice_length = mask.shape[0]//number_examples
    signal_step_size = fasta.shape[0]//signal.shape[0]

    fasta_examples = np.zeros(
        shape = (number_examples, interval_length, 4))
    
    signal_examples = np.zeros(
        shape = (number_examples, 
            interval_length//signal_step_size,  
            signal.shape[-1]))

    for i in range(number_examples):

        signal_lower = (np.argmax(mask[i*slice_length:(i+1)*slice_length])
            + (i * slice_length))
        signal_upper = signal_lower + interval_length//signal_step_size
        signal_examples[i,...] = signal[signal_lower:signal_upper,:]

        fasta_lower = signal_lower*signal_step_size
        fasta_upper = fasta_lower + interval_length
        fasta_examples[i,...] = fasta[fasta_lower:fasta_upper,:]

    return fasta_examples, signal_examples

        
