#%%
import pandas as pd
import numpy as np


# %%
def _import_MACS_narrowpeak(filepath, selection):

    narrow_peak_column_format = ["chr", "start", "end", "name", "score", "strand", "fc", "logp", "logq", "summit"]

    df = pd.read_csv(filepath, sep = "\t", names = narrow_peak_column_format)

    return df.loc[df.chr.isin(selection)]

def _MACS_positive_regions(MACS_df, 
                           interval_length, 
                           chr_sizes, 
                           selection,
                           cut_top = 100,
                           n_regions = 10000):

    # trim MACS_df according to params
    MACS_df = MACS_df.sort_values("logp", ascending=False)
    MACS_df = MACS_df[cut_top:(cut_top+n_regions)]

    # calculate correct start and end positions such that if:
    ## the length of the MACS interval is shorter than the target length
    ## the interval is extended to always include the full MACS interval,
    ## else it is made to always be fully contained in the MACS interval
    MACS_df.end -= interval_length

    short = MACS_df.end - MACS_df.start < 0
    (MACS_df.loc[short, "start"], MACS_df.loc[short, "end"]) = (
        MACS_df.loc[short, "end"], MACS_df.loc[short, "start"]
    )

    ## fix some problems that might arise
    MACS_df.loc[MACS_df.start < 0, "start"] = 0
    MACS_df.loc[MACS_df.end - MACS_df.start == 0, "end"] += 1

    ## adjust the coordinates according to chrom length
    chr_sizes_df = pd.read_csv(
        chr_sizes, sep = "\t", header = None, 
        index_col = 0, names = ["length"])
    chr_sizes_df = chr_sizes_df.loc[chr_sizes_df.index.isin(selection)]
    chr_sizes_df["cum_length"] = [0] + list(chr_sizes_df.length.cumsum())[:-1]

    for chrom in chr_sizes_df.index:
        if chrom in list(MACS_df.chr):
            MACS_df.loc[MACS_df.chr == chrom, "start"] += (
                chr_sizes_df.loc[chrom, "cum_length"]
            )
            MACS_df.loc[MACS_df.chr == chrom, "end"] += (
                chr_sizes_df.loc[chrom, "cum_length"]
            )

    ## calculate relative freq to use for even sampeling
    MACS_df["prop_dens"] = 1/(MACS_df.end - MACS_df.start)

    ## empty mask array
    positive_mask = np.zeros((sum(chr_sizes_df.length)))

    for row in MACS_df.itertuples():
        positive_mask[row.start:row.end] += row.prop_dens

    return positive_mask

def _MACS_negative_regions(MACS_df, interval_length, chr_sizes, selection):

    # look for all regions that start less then 
    # interval length before a MACS call
    MACS_df.start -= interval_length

    ## fix some problems that might arise
    MACS_df.loc[MACS_df.start < 0, "start"] = 0

    ## adjust the coordinates according to chrom length
    chr_sizes_df = pd.read_csv(
        chr_sizes, sep = "\t", header = None, 
        index_col = 0, names = ["length"])
    chr_sizes_df = chr_sizes_df.loc[chr_sizes_df.index.isin(selection)]
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
    negative_mask = np.ones((sum(chr_sizes_df.length)))

    for row in MACS_df.itertuples():
        negative_mask[row.start:row.end] = 0

    return negative_mask

def make_MACS_dataset(MACS_filepath, 
                      interval_length, 
                      chr_sizes, 
                      chr_selection,
                      rel_freq = 0.3,
                      cut_top = 100,
                      n_regions = 10000):

    MACS_df = _import_MACS_narrowpeak(MACS_filepath, chr_selection)

    positive_mask = _MACS_positive_regions(MACS_df, interval_length, 
        chr_sizes, chr_selection, cut_top = cut_top, n_regions = n_regions)

    negative_mask = _MACS_negative_regions(MACS_df, interval_length, 
        chr_sizes, chr_selection)

    y_true = positive_mask > 0

    prop_mask = (
        positive_mask * (1-rel_freq) / np.sum(positive_mask) +
        negative_mask * rel_freq / np.sum(negative_mask)
    )

    return y_true, prop_mask

# %%
