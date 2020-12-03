import os
import sys

sys.path.insert(0, '../')

COOL = '/store/qzhong/test/Rao2014-K562-MboI-allreps-filtered.10kb.cool'


def test_extract_cool(tmp_path):
    from hictools.utils.io import extract_cool
    if os.path.isfile(COOL):
        sub_cool = tmp_path / 'test.cool'
        extract_cool(
            cool_path=COOL,
            sub_cool_path=sub_cool.absolute().as_posix(),
            chroms=('chr21', 'chr20')
        )
        assert sub_cool.exists()


def test_test(get_cool):
    print(get_cool(10000))
