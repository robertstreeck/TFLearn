#%%
import pandas as pd
import numpy as np

from .MACS_Import import make_MACS_dataset

from .bam_dataset_generator import generate_border_mask, generate_N_mask, _jagged_slicer


#%% cat batch generator

def cat_batch_slice_generator(fasta, y_cat, weights, 
                          interval_length, bs = 64, n_batches = 10000):
    along = np.arange(0, weights.shape[0], 1)    
    while True:
        sf = np.random.choice(along, size = (n_batches, bs), p=weights)
        for i in range(n_batches):
            yield (_jagged_slicer(fasta, sf[i,:], interval_length), 
                y_cat[sf[i,:]])


#%% dataset_class

class MACS_cat_dataset(object):
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
                 narrowpeaks,
                 chr_selection,
                 chr_sizes_file,
                 fasta_interval_length,
                 background_frac = 0.3,
                 cut_top = 100,
                 n_regions = 10000):

        self.fasta = fasta

        self.interval_length = fasta_interval_length

        self.y_cat, mask = make_MACS_dataset(narrowpeaks, 
                      fasta_interval_length, 
                      chr_sizes_file, 
                      chr_selection,
                      rel_freq = background_frac,
                      cut_top = cut_top, 
                      n_regions = n_regions)

        masks = (
            generate_border_mask(chr_sizes_file, chr_selection, 
                1, fasta_interval_length),
            generate_N_mask(fasta, 1)
        )

        self.mask = mask * masks[0] * masks[1]

        self.mask = self.mask/np.sum(self.mask)

        print(
            "fraction of regions masked because of Ns: %.2f \n" % (
            (masks[1].shape[0] - np.sum(masks[1]))/masks[1].shape[0]
            )
        )

        print(
            "fraction of regions masked because of chromosome borders: %.2f \n" % (
            (masks[0].shape[0] - np.sum(masks[0]))/masks[0].shape[0]
            )
        )

        print(
            "fraction of regions considered as peaks: %.2f \n" % (
            np.sum(masks[0] * masks[1] * self.y_cat)/self.y_cat.shape[0]
            )
        )
    
    def make_generator(self,
                       batch_size = 64,
                       n_batches = 10000):
        return cat_batch_slice_generator(self.fasta, self.y_cat, 
                        self.mask, self.interval_length, bs = batch_size, 
                        n_batches = n_batches)
