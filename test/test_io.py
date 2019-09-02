import os
import sys

sys.path.append('../')
from hictools.io import extract_cool

COOL = '/store/qzhong/test/Rao2014-K562-MboI-allreps-filtered.10kb.cool'


def _test_extract_cool(tmp_path):
    if os.path.isfile(COOL):
        sub_cool = tmp_path / 'test.cool'
        extract_cool(
            cool=COOL,
            sub_cool=sub_cool.absolute().as_posix(),
            chroms=('chr21', 'chr20')
        )
        assert sub_cool.exists()


def test_test(get_cool):
    print(get_cool(10000))
