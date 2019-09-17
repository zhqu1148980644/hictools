"""Command line tools."""

import sys
from collections import OrderedDict
from functools import partial
import logging

import click
import cooler
import numpy as np

from .api import ChromMatrix as _ChromMatrix
from .compartment import corr_sorter, plain_sorter
from .peaks import (
    hiccups,
    fetch_kernels,
    expected_fetcher,
    observed_fetcher,
    factors_fetcher,
    chunks_gen
)
from .utils.utils import (
    CPU_CORE,
    RayWrap,
    get_logger
)
from .io import records2bigwigs
from . import config

click.option = partial(click.option, show_default=True)
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


def fetch_chrom_dict(cool):
    ray = RayWrap()
    ChromMatrix = ray.remote(_ChromMatrix)
    log = get_logger()

    co = cooler.Cooler(cool)
    records = co.bins()[['chrom', 'start', 'end']][:].copy()

    chrom_dict = OrderedDict()
    for chrom in co.chromnames:
        weight = np.array(co.bins().fetch(chrom)['weight'])
        badbin_ratio = np.isnan(weight) / weight.size
        if badbin_ratio > 0.5:
            log.warning(f"Skipped chromosome: {chrom} due to high percentage of bad regions.")
            continue
        chrom_dict[chrom] = ChromMatrix.remote(co, chrom)

    return co, records, chrom_dict


@click.group(context_settings=CONTEXT_SETTINGS)
@click.option("--log-file", help="The log file, default output to stderr.")
@click.option("--debug", is_flag=True,
              help="Open debug mode, disable ray.")
def cli(log_file, debug):
    log = logging.getLogger()  # root logger
    if log_file:
        handler = logging.FileHandler(log_file)
    else:
        handler = logging.StreamHandler(sys.stderr)

    fomatter = logging.Formatter(
        fmt=config.LOGGING_FMT,
        datefmt=config.LOGGING_DATE_FMT
    )
    handler.setFormatter(fomatter)
    log.addHandler(handler)

    if debug:
        config.DEBUG = True
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)


# ----------------------------------------------peaks--------------------------------------------

@cli.group()
def peaks():
    """Call peaks(loops) by using hiccups or cloops methods."""
    pass


@peaks.command()
@click.argument(
    'cool', type=str, nargs=1
)
@click.argument(
    "output", type=click.File('w'), default=sys.stdout
)
@click.option(
    "--max-dis", type=int, nargs=1, default=5000000,
    help='Max distance of loops. Due to the natural property of loops(distance less than 8M) '
         'and computation bound of hiccups algorithm, hiccups algorithm are applied only in '
         ' a band region over the main diagonal to speed up the whole process.'
)
@click.option(
    "--inner-radius", "-p", type=int, default=2,
    help='Radius of innner square in hiccups.'
         'Pixels in this region will not be counted in the calcualtion of background.'
         'According to original paper, p set to 1 at 25kb resolution; 2 at 10kb; 4 at 5kb.'
)
@click.option(
    "--outer-radius", "-w", type=int, default=5,
    help='Radius of outer square in hiccps.'
         'Only pixels in this region will be counted in the calculation of background.'
         'According to original paper, w set to 3 at 25kb resolution; 5 at 10kb; 7 at 5kb.'
)
@click.option(
    "--chunk-size", '-h', type=int, default=500,
    help='Height of each chunk(submatrix).'
)
@click.option(
    "--fdrs", nargs=4, default=(0.1, 0.1, 0.1, 0.1),
    type=click.Tuple([float, float, float, float]),
    help='Tuple of fdrs to control the false discovery rate for each background.'
)
@click.option(
    "--sigs", nargs=4, default=(0.1, 0.1, 0.1, 0.1),
    type=click.Tuple([float, float, float, float]),
    help='Tuple of padjs thresholds for each background.'
)
@click.option(
    "--fold_changes", nargs=4, default=(1.5, 1.5, 1.5, 1.5),
    type=click.Tuple([float, float, float, float]),
    help='Fold change threshold for each region. '
         'Valid peak\'s fold changes should pass all four fold-change-thresholds.'
)
@click.option(
    "--ignore-single-gap", nargs=1, type=bool, default=True,
    help='If ignore small gaps when filtering peaks close to gap regions.'
)
@click.option(
    "--bin-index", nargs=1, type=bool, default=False,
    help='Return actual genomic positions of peaks if set to False.'
)
@click.option(
    '--nproc', '-n', type=int, nargs=1, default=25,
    help='Number of cores for calculation'
)
def hiccups(cool, output,
            max_dis, inner_radius, outer_radius,
            chunk_size, fdrs, sigs, fold_changes,
            ignore_single_gap, bin_index, nproc):
    """Call peaks by using hiccups method."""
    ray = RayWrap(num_cpus=nproc)
    co, _, chrom_dict = fetch_chrom_dict(cool)

    num_diags = max_dis // co.binsize
    expected_dict = OrderedDict()
    start_dict = OrderedDict()
    factor_dict = OrderedDict()
    chrom_size_dict = OrderedDict()
    for key, chrom in chrom_dict.items():
        expected_dict[key] = chrom.expected.remote(ndiags=num_diags + 20)
        chrom_bin = co.bins().fetch(key)
        start_dict[key] = chrom_bin.index.min()
        factor_dict[key] = np.array(chrom_bin['weight'])
        chrom_size_dict[key] = factor_dict[key].size

    expected = partial(expected_fetcher, expected_dict=expected_dict)
    observed = partial(observed_fetcher, cool=co, start_dict=start_dict)
    factors = partial(factors_fetcher, factor_dict=factor_dict)
    chunks = chunks_gen(
        chromsizes=chrom_size_dict,
        band_width=num_diags,
        height=chunk_size,
        ov_length=2 * outer_radius)
    kernels = fetch_kernels(inner_radius, outer_radius)

    peaks_df = hiccups(
        expected_fetcher=expected,
        observed_fetcher=observed,
        factors_fetcher=factors,
        chunks=chunks,
        kernels=kernels,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        num_cpus=nproc,
        max_dis=max_dis,
        resolution=co.binsize,
        fdrs=fdrs,
        sigs=sigs,
        fold_changes=fold_changes,
        ignore_single_gap=ignore_single_gap,
        bin_index=bin_index
    )

    peaks_df.to_csv(output, sep='\t', header=True, index=False, na_rep="nan")


@peaks.command()
@click.argument(
    "file", nargs=1, type=click.File('r')
)
def cloops(file):
    click.echo("Not implemented yet.")
    pass


@peaks.command()
@click.argument(
    "file", nargs=1, type=click.File('r')
)
def peaks2d(file):
    click.echo("Not implemented yet.")
    pass


# ----------------------------------------------tads----------------------------------------
@cli.group()
def tads():
    """Tools for topological associated domain analysis."""
    pass


@tads.command()
@click.argument(
    'cool', type=str, nargs=1
)
@click.argument(
    'output', type=click.File('w'), default=sys.stdout
)
@click.option(
    '--balance', type=bool, nargs=1, default=True,
    help='Use ICE normalized contact matrix.'
)
@click.option(
    '--window-size', type=int, nargs=1, default=20,
    help='Length of upstream array and downstream array.'
)
@click.option(
    "--ignore-diags", type=int, nargs=1, default=3
)
@click.option(
    '--nproc', '-n', type=int, nargs=1, default=5,
    help='Number of cores for calculation'
         'This step is not only time consuming but also memory-intensive.'
)
def di_score(cool, output, balance, window_size, ignore_diags, nproc):
    ray = RayWrap(num_cpus=nproc)
    _, records, chrom_dict = fetch_chrom_dict(cool)

    standard_di_dict = OrderedDict()
    adap_di_dict = OrderedDict()
    for key, chrom in chrom_dict.items():
        standard_di_dict[key] = chrom.di_score.remote(
            balance=balance,
            window_size=window_size,
            ignore_diags=ignore_diags,
            method='standard'
        )
        adap_di_dict[key] = chrom.di_score.remote(
            balance=balance,
            window_size=window_size,
            ignore_diags=ignore_diags,
            method='adaptive'
        )
    standard_di = np.concatenate(
        [ray.get(standard_di_dict[key]) for key in chrom_dict.keys()]
    )
    adp_di = np.concatenate(
        [ray.get(adap_di_dict[key]) for key in chrom_dict.keys()]
    )

    records['standard_di'] = standard_di
    records['adptive_di'] = adp_di
    records.to_csv(output, sep='\t', header=True, index=False, na_rep="nan")


@tads.command()
@click.argument(
    'cool', type=str, nargs=1
)
@click.argument(
    'output', type=click.File('w'), default=sys.stdout
)
@click.option(
    '--balance', type=bool, nargs=1, default=True,
    help='Use ICE normalized contact matrix.'
)
@click.option(
    '--window-size', type=int, nargs=1, default=20,
    help='Length of upstream array and downstream array.'
)
@click.option(
    '--normalize', type=bool, nargs=1, default=True,
    help='If normalize the insulation score with log2 ratio of insu_score and mean insu_score.'
)
@click.option(
    "--ignore-diags", type=int, nargs=1, default=3
)
@click.option(
    '--nproc', '-n', type=int, nargs=1, default=5,
    help='Number of cores for calculation'
         'This step is not only time consuming but also memory-intensive.'
)
def insu_score(cool, output, balance, window_size, normalize, ignore_diags, nproc):
    ray = RayWrap(num_cpus=nproc)
    _, records, chrom_dict = fetch_chrom_dict(cool)

    insu_score_dict = OrderedDict()
    for key, chrom in chrom_dict.items():
        insu_score_dict[key] = chrom.insu_score.remote(
            balance=balance,
            window_size=window_size,
            ignore_diags=ignore_diags,
            normalize=normalize
        )
    insu_scores = np.concatenate(
        [ray.get(insu_score_dict[key]) for key in chrom_dict.keys()]
    )

    records['insu_score'] = insu_scores
    records.to_csv(output, sep='\t', header=True, index=False, na_rep="nan")


@tads.command()
@click.argument(
    'cool', type=str, nargs=1
)
@click.argument(
    'output', type=click.File('w'), default=sys.stdout
)
@click.option(
    '--balance', type=bool, nargs=1, default=True,
    help='Use ICE normalized contact matrix.'
)
@click.option(
    '--nproc', '-n', type=int, nargs=1, default=5,
    help='Number of cores for calculation'
         'This step is not only time consuming but also memory-intensive.'
)
def di_hmm(cool, output, balance, nproc):
    click.echo("Not implemented yet.")


# ----------------------------------------------expected-----------------------------------------
@cli.command()
@click.argument(
    'cool', type=str, nargs=1
)
@click.argument(
    'output', type=click.File('w'), default=sys.stdout
)
@click.option(
    '--balance', type=bool, nargs=1, default=True,
    help='Use ICE normalized contact matrix.'
)
@click.option(
    '--nproc', '-n', type=int, nargs=1, default=CPU_CORE - 4,
    help='Number of cores for calculation'
)
def expected(cool, output, balance, nproc):
    """Compute expected values from a .cool file."""
    ray = RayWrap(num_cpus=nproc)
    co, records, chrom_dict = fetch_chrom_dict(cool)

    decay_dict = OrderedDict()
    for key, chrom in chrom_dict.items():
        decay_dict[key] = chrom.decay.remote(balance=balance)
    decays = np.concatenate(
        [ray.get(decay_dict[key]) for key in decay_dict.keys()]
    )

    records['expected'] = decays
    records.to_csv(output, sep='\t', header=True, index=False, na_rep="nan")


# ----------------------------------------compartments-------------------------------------
@cli.group()
def compartments():
    """Tools for topological associated domain analysis."""
    pass


@compartments.command()
@click.argument(
    'cool', type=str, nargs=1
)
@click.argument(
    'output', type=click.File('w'), default=sys.stdout
)
@click.option(
    '--balance', type=bool, nargs=1, default=True,
    help='Use ICE normalized contact matrix.'
)
@click.option(
    '--method', type=click.Choice(['pca', 'eigen']), nargs=1, default='pca',
    help='Choose method for detecting A/B compartment. Currently support \'eigen\' and \'pca\'.'
)
@click.option(
    '--numvecs', type=int, nargs=1, default=3,
    help='Choose number of eigen vectors to check.'
)
@click.option(
    "--ignore-diags", type=int, nargs=1, default=3,
    help='Choose number of diagonals to ignore.'
)
@click.option(
    '--sort', type=bool, nargs=1, default=True,
    help='Sort compartments based on corr matrix.'
)
@click.option(
    '--out-fmt', type=click.Choice(['tab', 'bigwig']), default='tab',
    help="Output format, if specify 'bigwig', package 'pyBigWig' is needed."
)
@click.option(
    '--nproc', '-n', type=int, nargs=1, default=25,
    help='Number of cores for calculation'
)
def decomposition(cool, output, balance, method,
                  numvecs, ignore_diags, sort, out_fmt, nproc):
    """Compute A/B compartment from a .cool file based on decomposition of intra-interaction matrix."""
    log = get_logger()
    log.info("Call compartments")
    log.debug(locals())

    ray = RayWrap(num_cpus=nproc)
    co, records, chrom_dict = fetch_chrom_dict(cool)

    compartment_dict = OrderedDict()
    for key, chrom in chrom_dict.items():
        compartment_dict[key] = chrom.compartments.decomposition.remote(
            method=method,
            balance=balance,
            numvecs=numvecs,
            sort_fn=corr_sorter if sort else plain_sorter,
            full=True,
            ignore_diags=ignore_diags
        )
        log.debug(compartment_dict[key])
    coms = np.hstack(
        [ray.get(compartment_dict[key]) for key in chrom_dict.keys()]
    )

    if len(coms.shape) == 2:
        for i in range(coms.shape[0]):
            records['com_{}'.format(i)] = coms[i]
    else:
        records['com_0'] = coms

    if out_fmt == 'tab':
        records.to_csv(output, sep='\t', header=True, index=False, na_rep="nan")
    elif out_fmt == 'bigwig':
        records2bigwigs(records, output.name)
    else:
        raise IOError("Only support tab or bigwig output format.")


if __name__ == "__main__":
    cli()
