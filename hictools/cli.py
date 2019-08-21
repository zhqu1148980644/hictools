"""
Command line tools.
"""

import sys
from collections import OrderedDict
from functools import partial

import click
import cooler
import numpy as np

from .api import ChromMatrix as _ChromMatrix
from .compartment import corr_sorter, plain_sorter
from .peaks import hiccups, fetch_regions, expected_fetcher, observed_fetcher, factors_fetcher, chunks_gen
from .utils import CPU_CORE, RayWrap

DEBUG = False

# logger = logging.getLogger()
# console_handler = logging.StreamHandler(sys.stdout)
# logger.addHandler(console_handler)

click.option = partial(click.option, show_default=True)
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


def fetch_chrom_dict(cool):
    ray = RayWrap()
    ChromMatrix = ray.remote(_ChromMatrix)

    co = cooler.Cooler(cool)

    records = co.bins()[['chrom', 'start', 'end']][:].copy()
    chrom_dict = OrderedDict()
    for chrom in co.chromnames:
        chrom_dict[chrom] = ChromMatrix.remote(co, chrom)

    if not DEBUG:
        sys.stderr = open('/tmp/hictools.log', 'a')
    return co, records, chrom_dict


@click.group()
@click.option("--debug", is_flag=True,
    help="Open debug mode, disable ray.")
def cli(debug):
    if debug:
        global DEBUG
        DEBUG = True


@cli.group()
def peaks():
    """Call peaks(loops) by using hiccups or cloops methods."""
    pass


@peaks.command(context_settings=CONTEXT_SETTINGS)
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
    "--fdrs", nargs=4, default=(0.01, 0.01, 0.01, 0.01),
    type=click.Tuple([float, float, float, float]),
    help='Tuple of fdrs to control the false discovery rate for each background.'
)
@click.option(
    "--sigs", nargs=4, default=(0.01, 0.01, 0.01, 0.01),
    type=click.Tuple([float, float, float, float]),
    help='Tuple of padjs thresholds for each background.'
)
@click.option(
    "--single-fcs", nargs=4, default=(1.75, 1.5, 1.5, 1.75),
    type=click.Tuple([float, float, float, float]),
    help='Padjs threshold for each region. '
         'Valid peak\'s padjs should pass all four fold-hange thresholds.'
)
@click.option(
    "--double-fcs", nargs=4, default=(2.5, 0., 0., 2.5),
    type=click.Tuple([float, float, float, float]),
    help='Padjs threshold for each region. '
         'Valid peak\'s padjs should pass either one of four fold-change thresholds.'
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
def call_by_hiccups(cool, output,
                    max_dis, inner_radius, outer_radius, chunk_size,
                    fdrs, sigs, single_fcs, double_fcs,
                    ignore_single_gap, bin_index,
                    nproc):
    """Call peaks by using hiccups method."""
    ray = RayWrap(num_cpus=nproc)
    co, _, chrom_dict = fetch_chrom_dict(cool)

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

    expected = partial(expected_fetcher, expected_dict=expected_dict)
    observed = partial(observed_fetcher, cool=co, start_dict=start_dict)
    factors = partial(factors_fetcher, factor_dict=factor_dict)
    chunks = chunks_gen(
        chromsizes=chrom_size_dict,
        band_width=max_dis // co.binsize,
        height=chunk_size,
        ov_length=2 * outer_radius)
    kernels = fetch_regions(inner_radius, outer_radius, kernel=True)

    peaks_df = hiccups(
        expected_fetcher=expected,
        observed_fetcher=observed,
        factors_fetcher=factors,
        chunks=chunks,
        kernels=kernels,
        num_cpus=nproc,
        max_dis=max_dis,
        resolution=co.binsize,
        fdrs=fdrs,
        sigs=sigs,
        single_fcs=single_fcs,
        double_fcs=double_fcs,
        ignore_single_gap=ignore_single_gap,
        bin_index=bin_index
    )

    peaks_df.to_csv(output, sep='\t', header=True, index=False, na_rep="nan")


@peaks.command(context_settings=CONTEXT_SETTINGS)
@click.argument(
    "file", nargs=1, type=click.File('r')
)
def call_by_cloops(file):
    click.echo("Not implemented yet.")
    pass


@cli.group()
def tad():
    """Tools for topological associated domain analysis."""
    pass


@tad.command(context_settings=CONTEXT_SETTINGS)
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
    standard_di = np.concatenate([ray.get(standard_di_dict[key])
                                  for key in chrom_dict.keys()])
    adp_di = np.concatenate([ray.get(adap_di_dict[key])
                             for key in chrom_dict.keys()])

    records['standard_di'] = standard_di
    records['adptive_di'] = adp_di
    records.to_csv(output, sep='\t', header=True, index=False, na_rep="nan")


@tad.command(context_settings=CONTEXT_SETTINGS)
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
    insu_scores = np.concatenate([ray.get(insu_score_dict[key])
                                  for key in chrom_dict.keys()])

    records['insu_score'] = insu_scores
    records.to_csv(output, sep='\t', header=True, index=False, na_rep="nan")


@tad.command(context_settings=CONTEXT_SETTINGS)
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
def call_tad(cool, output, balance, nproc):
    click.echo("Not implemented yet.")


@cli.command(context_settings=CONTEXT_SETTINGS)
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
    decays = np.concatenate([ray.get(decay_dict[key])
                             for key in decay_dict.keys()])

    records['expected'] = decays
    records.to_csv(output, sep='\t', header=True, index=False, na_rep="nan")


@cli.command(context_settings=CONTEXT_SETTINGS)
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
    '--nproc', '-n', type=int, nargs=1, default=25,
    help='Number of cores for calculation'
)
def compartment(cool, output, balance, method, numvecs, ignore_diags, sort, nproc):
    """Compute A/B compartment from a .cool file."""
    ray = RayWrap(num_cpus=nproc)
    co, records, chrom_dict = fetch_chrom_dict(cool)

    compartment_dict = OrderedDict()
    for key, chrom in chrom_dict.items():
        compartment_dict[key] = chrom.compartments.remote(
            method=method,
            balance=balance,
            vec_range=numvecs,
            com_range=numvecs,
            sort_fn=corr_sorter if sort else plain_sorter,
            full=True,
            ignore_diags=ignore_diags
        )
        print(compartment_dict[key])
    compartments = np.hstack([ray.get(compartment_dict[key])
                              for key in chrom_dict.keys()])

    if len(compartments.shape) == 2:
        for i in range(compartments.shape[0]):
            records['com_{}'.format(i)] = compartments[i]
    else:
        records['com_0'] = compartments

    records.to_csv(output, sep='\t', header=True, index=False, na_rep="nan")


if __name__ == "__main__":
    cli()
