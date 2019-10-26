import os
import random
import subprocess
import sys

import cooler
import h5py
import pytest
import wget

sys.path.insert(0, '../')

COOL_URL = "ftp://cooler.csail.mit.edu/coolers/hg19/Rao2014-K562-MboI-allreps-filtered.10kb.cool"
COOL = 'data/' + COOL_URL[COOL_URL.rfind('/') + 1:]
MCOOL = COOL.replace('.cool', '.mcool')
SUB_MCOOL = 'data/' + "test.mcool"


def reso_uri(reso=10000):
    return f"::resolutions/{reso}"


@pytest.fixture(scope='module')
def load_pyx():
    import os
    import subprocess
    import numpy as np
    back_up = os.getcwd()
    os.chdir('../')
    np_inc = np.get_include()
    env = os.environ.copy()
    env.update({ "C_INCLUDE_PATH": np_inc, "CPLUS_INCLUDE_PATH": np_inc })
    p = subprocess.Popen("cythonize -q -f -b -i **/*.pyx",
            shell=True, env=env, stdout=-1, stderr=-1)
    p.wait()
    os.chdir(back_up)


@pytest.fixture(scope='module')
def get_cool():
    from hictools.io import extract_cool
    def resolution(reso):
        return f'{os.path.abspath(SUB_MCOOL)}{reso_uri(reso)}'

    if (not os.path.exists(COOL)
            and not os.path.exists(SUB_MCOOL)
            and not os.path.exists(MCOOL)):
        wget.download(COOL_URL, COOL)

    if not os.path.exists(SUB_MCOOL):
        if not os.path.exists(MCOOL):
            try:
                subprocess.check_call(
                    f"cooler zoomify --balance -p 30 -r 10000,100000 {COOL}",
                    shell=True,
                    executable='/bin/bash'
                )
            except Exception as e:
                os.remove(MCOOL)
                raise e
        try:
            sub_mcool = h5py.File(SUB_MCOOL)
            grp_reso = sub_mcool.create_group('resolutions')
            grp_reso.create_group('10000')
            grp_reso.create_group('100000')
            sub_mcool.close()

            chroms = ('chr18', 'chr19', 'chr21')
            # Create 10k and 100k resolution cool file.
            extract_cool(
                cool=MCOOL + reso_uri(),
                sub_cool=SUB_MCOOL + reso_uri(),
                chroms=chroms
            )
            extract_cool(
                cool=MCOOL + reso_uri(),
                sub_cool=SUB_MCOOL + reso_uri(),
                chroms=chroms
            )
        except Exception as e:
            os.remove(SUB_MCOOL)
            raise e

    return resolution


@pytest.fixture(scope='module')
def get_chrom(get_cool, load_pyx):
    from hictools.api import ChromMatrix

    def resolution(reso):
        return ChromMatrix(cool_dict[reso], random.choice(chroms))

    cool_path = get_cool(100000)
    co = cooler.Cooler(cool_path)
    # chroms = []
    # for chrom in co.chromnames:
    #     weight = np.array(co.bins().fetch('weight'))
    #     bad_ratio = np.isnan(weight) / weight.size
    #     if bad_ratio < 0.5:
    #         chroms.append(chrom)
    chroms = tuple(co.info['Included chroms'])

    cool_dict = {
        10000: cooler.Cooler(get_cool(10000)),
        100000: cooler.Cooler(get_cool(100000))
    }

    return resolution
