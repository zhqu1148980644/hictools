"""
Used for command line.
"""
import io
import os
import sys
import click
import cooler
import ray
from api import ChromMatrix
from peaks import get_chunk_slices, hiccups, fetch_regions
import logging
import numpy as np
from collections import OrderedDict
from functools import partial

logger = logging.getLogger()
console_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(console_handler)


@click.group()
def cli():
    pass


@cli.command()
@click.argument('mcool', type=str, nargs=1)
@click.option('--nproc', '-n', default=25, type=int, nargs=1,
              help="Number of cores for calculation")
def extract_features(mcool, nproc):
    """Extract all features from a .mcool file.

    These files will be created in the current folder:
            compartment_100k.bed
            peaks_10k.bed
            tad_score_10k.bed
            tad_10k.bed

    Caution: .mcool file must contain /resultions/10000 and /resolutions/100000 group.
    """

    def expected_fetcher(key, slices, expected_dict):
        if isinstance(expected_dict[key], ray._raylet.ObjectID):
            expected_dict[key] = ray.get(expected_dict[key])
        return expected_dict[key][slices]

    def observed_fetcher(key, slices, cool, start_dict):
        row_st, row_ed = slices[0].start + start_dict[key], slices[0].stop + start_dict[key]
        col_st, col_ed = slices[1].start + start_dict[key], slices[1].stop + start_dict[key]
        return cool.matrix()[slice(row_st, row_ed), slice(col_st, col_ed)]

    def factors_fetcher(key, slices, factor_dict):
        return factor_dict[key][slices[0]], factor_dict[key][slices[1]]

    def chunks_gen(chromsizes, band_width, height, ov_length):
        for chrom, size in chromsizes.items():
            for slices in get_chunk_slices(size, band_width, height, ov_length):
                yield chrom, slices

#    sys.stderr = io.StringIO()
    ray.init(num_cpus=nproc)
    # calculate expected and compartments in 100kb resolution.
    logger.info("Start reading cool in 100000 resolution.")
    try:
        co = cooler.Cooler(mcool + "::/resolutions/100000")
    except Exception as e:
        raise RuntimeError("Pass: group /resolutions/100000 does not exist in {}".format(mcool))

    chrom_dict = OrderedDict()
    for chrom in co.chromnames:
        chrom_dict[chrom] = ChromMatrix.remote(co, chrom)

    compartment_dict = OrderedDict()
    for key, chrom in chrom_dict.items():
        compartment_dict[key] = chrom.compartments.remote()
    compartments = [ray.get(compartment_dict[key]) for key in chrom_dict.keys()]

    decay_dict = OrderedDict()
    for key, chrom in chrom_dict.items():
        decay_dict[key] = chrom.decay.remote()
    decays = [ray.get(decay_dict[key]) for key in decay_dict.keys()]

    compartments = np.concatenate(compartments)
    decays = np.concatenate(decays)

    bins = co.bins()[['chrom', 'start', 'end']][:].copy()
    bins['expected'] = decays
    bins['eigvals'] = compartments

    bins.to_csv('compartment_100k.bed', sep='\t', header=True, index=False, na_rep="nan")
    logger.info("Creating compartment_100k.bed done.")

    # calculate peaks and tads in 10kb resolution.
    # tad_score_10k.bed
    logger.info("Start reading cool in 10000 resolution.")
    try:
        co = cooler.Cooler(mcool + "::/resolutions/10000")
    except Exception as e:
        raise RuntimeError("Pass: group /resolutions/10000 does not exist in {}".format(mcool))

    chrom_dict = OrderedDict()
    for chrom in co.chromnames:
        chrom_dict[chrom] = ChromMatrix.remote(co, chrom)

    di_score_dict = OrderedDict()
    for key, chrom in chrom_dict.items():
        di_score_dict[key] = chrom.di_score.remote()
    di_scores = np.concatenate([ray.get(di_score_dict[key]) for key in chrom_dict.keys()])

    insu_score_dict = OrderedDict()
    for key, chrom in chrom_dict.items():
        insu_score_dict[key] = chrom.insu_score.remote()
    insu_scores = np.concatenate([ray.get(insu_score_dict[key]) for key in chrom_dict.keys()])

    bins = co.bins()[['chrom', 'start', 'end']][:].copy()
    bins['insu_score'] = insu_scores
    bins['di_score'] = di_scores
    bins.to_csv('tad_score_10k.bed', sep='\t', header=True, index=False, na_rep="nan")
    logger.info("Creating tad_score_10k.bed done.")
    # peaks_10k.bed
    expected_dict = OrderedDict()
    start_dict = OrderedDict()
    factor_dict = OrderedDict()
    chrom_size_dict = OrderedDict()
    for key, chrom in chrom_dict.items():
        expected_dict[key] = chrom.expected.remote()
        chrom_bin = co.bins().fetch(key)
        start_dict[key] = chrom_bin.index.min()
        factor_dict[key] = np.array(chrom_bin['weight'])
        chrom_size_dict[key] = factor_dict[key].size

    expected_fetcher = partial(expected_fetcher,
                               expected_dict=expected_dict)
    observed_fetcher = partial(observed_fetcher,
                               cool=co,
                               start_dict=start_dict)
    factors_fetcher = partial(factors_fetcher,
                              factor_dict=factor_dict)

    chunks = chunks_gen(chrom_size_dict, 500, 500, 10)
    kernels = fetch_regions(2, 5, kernel=True)

    peaks = hiccups(expected_fetcher=expected_fetcher,
                    observed_fetcher=observed_fetcher,
                    factors_fetcher=factors_fetcher,
                    chunks=chunks,
                    kernels=kernels,
                    outer_radius=5,
                    max_dis=5000000,
                    resolution=10000,
                    bin_index=False)
    peaks.to_csv('peaks_10k.bed', sep='\t', header=True, index=False, na_rep="nan")
    logger.info("Creating peaks_10k.bed done.")

    ray.shutdown()


if __name__ == "__main__":
    print('asdasd')
    cli()
