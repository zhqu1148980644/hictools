from subprocess import check_call
from os.path import join
import fire

DOWNLOAD_DIR = "./download"
RESOLUTIONS = [5000, 10000, 25000, 40000, 100000, 1000000]
COOL_URL = "ftp://cooler.csail.mit.edu/coolers/hg19/Rao2014-GM12878-MboI-allreps-filtered.1kb.cool"
MCOOL = "./test.mcool"
RESULT_DIR = "./results"
LOG_FILE = "./benchmark.log"
TEST_RESOLUTIONS = [40000, 10000, 5000]
assert set(TEST_RESOLUTIONS) in set(RESOLUTIONS)


import sys
import logging
log = logging.getLogger("benchmark")
LOGGING_FMT = "%(asctime)s: %(message)s"
LOGGING_DATE_FMT = "%m/%d/%y %H:%M:%S"
fomatter = logging.Formatter(fmt=LOGGING_FMT, datefmt=LOGGING_DATE_FMT)
hd_err = logging.StreamHandler(sys.stderr)
hd_err.setFormatter(fomatter)
log.addHandler(hd_err)
hd_f = logging.FileHandler(LOG_FILE)
hd_f.setFormatter(fomatter)
log.addHandler(hd_f)
log.setLevel(logging.DEBUG)



def timethis(func=None, proc_name=None):
    '''
    Decorator that reports the execution time.
    '''
    from functools import wraps, partial
    from datetime import datetime
    if func is None:
        return partial(timethis, proc_name=proc_name)
    proc_name = proc_name or func.__name__

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = datetime.now()
        log.info(f"[{proc_name}] START")
        result = func(*args, **kwargs)
        end = datetime.now()
        log.info(f"[{proc_name}] END")
        log.info( f"[{proc_name}] RUN_TIME: {end - start}")
        return result
    return wrapper


def mk_dir(dir_):
    from pathlib import Path
    outdir = Path(dir_)
    if not outdir.exists():
        outdir.mkdir()
    return outdir


@timethis
def prepare_cool(url=COOL_URL):
    log.info(f"download cool file from {url}")
    down_dir = mk_dir(DOWNLOAD_DIR)
    from os.path import split
    cool_file = split(url)[-1]
    import wget
    cool_path = str(down_dir/cool_file)
    wget.download(url, cool_path)

    log.info("Zoomify cool")
    from cooler.api import Cooler
    c = Cooler(cool_path)
    resos = [str(r) for r in RESOLUTIONS if r >= c.binsize]
    check_call(["cooler", "zoomify", "--balance", "-p", "30", "-r", ",".join(resos), cool_path])
    import os
    target = MCOOL
    if os.path.exists(target):
        os.unlink(target)
    import re
    mcool_path = re.sub(".cool$", ".mcool", cool_path)
    os.symlink(mcool_path, target)


def get_cool_uri(mcool, reso):
    return mcool + f"::/resolutions/{reso}"


@timethis
def call_peaks(reso, output, p=5, w=7, num_cpus=10):
    uri = get_cool_uri(MCOOL, reso)
    cmd = ['hictools', 'peaks', 'call-by-hiccups']
    cmd += ['-p', str(p), '-w', str(w), '-n', str(num_cpus)]
    cmd += [uri, output]
    check_call(cmd)


def test_call_peaks(num_cpus=20):
    mk_dir(RESULT_DIR)
    resos = TEST_RESOLUTIONS
    for reso, (p, w) in zip(resos, [(1, 3), (2, 5), (4, 7)]):
        log.info(f"Resolution: {reso}\tNumber of CPUs: {num_cpus}")
        output = join(RESULT_DIR, f"peaks_{reso}.txt")
        call_peaks(reso, output, p, w, num_cpus)


@timethis
def call_compartment(reso, output, method="pca", num_cpus=10):
    uri = get_cool_uri(MCOOL, reso)
    cmd = ['hictools', 'compartment']
    cmd += ['--method', method, '-n', str(num_cpus)]
    cmd += [uri, output]
    check_call(cmd)


def test_call_compartment(num_cpus=20):
    mk_dir(RESULT_DIR)



fire.Fire()
