"""Tools for io handling."""

import os
from collections import OrderedDict, namedtuple
from typing import Iterable, Tuple

import h5py
import numpy as np
import pandas as pd

from .utils import get_logger


def records2bigwigs(df: pd.DataFrame, prefix: str):
    """ Dump dataframe to bigwig files

    :param df: records dataframe, contain fields: chrom, start, end.
    :param prefix: prefix of output bigwig files.
    """
    import pyBigWig
    required_fields = ['chrom', 'start', 'end']
    assert all((f in df) for f in required_fields), \
        f"records dataframe need fields: {', '.join(required_fields)}"

    val_cols = tuple(str(col)
                     for col in df.columns
                     if col not in required_fields)
    bigwigs = {}
    for col in val_cols:
        path_ = prefix + '.' + col + '.bw'
        if os.path.exists(path_):
            os.remove(path_)
        bigwigs[col] = pyBigWig.open(path_, 'w')

    chroms = df['chrom'].drop_duplicates()
    chroms2maxend = {chrom: df[df['chrom'] == chrom]['end'].max() for chrom in chroms}
    headers = list(chroms2maxend.items())
    for bw in bigwigs.values():
        bw.addHeader(headers)
    for col in val_cols:
        df_ = df[~df[col].isna()]
        bigwigs[col].addEntries(
            chroms=list(df_['chrom']),
            starts=list(df_['start']),
            ends=list(df_['end']),
            values=list(df_[col])
        )

    for bw in bigwigs.values():
        bw.close()


def fetch_coolinfo(cool: h5py.File, chroms: Iterable) -> Tuple[list, dict]:
    bin1_offset = cool['indexes']['bin1_offset']
    chrom_offset = cool['indexes']['chrom_offset']
    chrom_names = cool['chroms']['name']
    chrom_lengths = cool['chroms']['length']

    sub_chroms = list(chroms)
    chrom_info = OrderedDict()
    Info = namedtuple('Info', 'bin_st bin_ed pixel_offset length index')
    for i, (chrom, length) in enumerate(zip(chrom_names, chrom_lengths)):
        chrom = chrom.decode()
        if chrom not in sub_chroms:
            continue
        sub_chroms.remove(chrom)
        bin_st, bin_ed = chrom_offset[i], chrom_offset[i + 1]
        pixel_offset = bin1_offset[bin_st: bin_ed + 1]
        chrom_info[chrom] = Info(bin_st, bin_ed, pixel_offset, length, i)

    return sub_chroms, chrom_info


def extract_cool(cool_path: str, sub_cool_path: str, chroms: Iterable, intra_only: bool = True):
    """Extract a certain subset data(chromosomes) from a given .cool file.
    For extracting sub regions: see: https://github.com/Nanguage/CoolClip
    """

    def open_mcool(filename, mode='r') -> h5py.File:
        if '::' not in filename:
            return h5py.File(filename, mode)

        filename, grp_name = filename.split('::')
        return h5py.File(filename, mode)[grp_name]

    log = get_logger('Extract sub_cool.')
    log.info(f'Extract sub data from {cool_path}.')
    cool = open_mcool(cool_path, mode='r')

    sub_chroms, chrom_info = fetch_coolinfo(cool, chroms)
    if sub_chroms:
        log.warning(f"Chromosomes: {sub_chroms} not find in {cool.name}.")

    if os.path.isfile(sub_cool_path):
        sub_cool = open_mcool(sub_cool_path, mode='a')
        included_chroms = sub_cool.attrs.get('Included chroms', np.empty(0))
        if np.all(included_chroms == list(chrom_info.keys())):
            sub_cool.file.close()
            return
        else:
            sub_cool.clear()
    else:
        sub_cool = open_mcool(sub_cool_path, mode='a')

    # copy groups.
    cool.copy('chroms', sub_cool)
    cool.copy('bins', sub_cool)
    cool.copy('indexes', sub_cool)
    for k, v in cool.attrs.items():
        sub_cool.attrs[k] = v
    sub_cool.attrs['Included chroms'] = list(chrom_info.keys())

    # create pixels group.
    bin2_id = cool['pixels']['bin2_id']
    count = cool['pixels']['count']
    old_indptr = cool['indexes']['bin1_offset']
    new_indptr = np.zeros_like(old_indptr)
    new_bin1_id, new_bin2_id, new_count = [], [], []
    for chrom, info in chrom_info.items():
        if intra_only:
            offset = info.pixel_offset
            offset_st, offset_ed = offset[0], offset[-1]
            all_bin2 = bin2_id[offset_st: offset_ed]
            all_count = count[offset_st: offset_ed]
            dtype = all_bin2.dtype
            _st, _ed = np.int64(info.bin_st), np.int64(info.bin_ed - 1)
            for row_id, st, ed in zip(
                    range(info.bin_st, info.bin_ed),
                    offset[:-1] - offset_st,
                    offset[1:] - offset_st
            ):
                bin2 = all_bin2[st: ed]
                bin2_st = np.searchsorted(bin2, _st, 'left')
                bin2_ed = np.searchsorted(bin2, _ed, 'right')
                sub_bin2 = bin2[bin2_st: bin2_ed]
                new_bin2_id.append(sub_bin2)
                new_bin1_id.append(np.full(sub_bin2.size, row_id, dtype=dtype))
                new_count.append(all_count[st + bin2_st: st + bin2_ed])
                new_indptr[row_id + 1] = bin2_ed - bin2_st
                # mask = (bin2 >= info.bin_st) & (bin2 < info.bin_ed)
                # sub_bin2 = bin2[mask]
                # size = sub_bin2.size
                # new_bin2_id.append(sub_bin2)
                # new_bin1_id.append(np.full(size, row_id, dtype=dtype))
                # new_count.append(all_count[st: ed][mask])
                # new_indptr[row_id + 1] = size
        else:
            raise NotImplementedError('Not implemented. Only support intra-only extraction.')

    new_datasets = (
        np.concatenate(new_bin1_id),
        np.concatenate(new_bin2_id),
        np.concatenate(new_count)
    )

    pixels_grp = sub_cool.create_group('pixels')
    for name, dataset in zip(('bin1_id', 'bin2_id', 'count'), new_datasets):
        pixels_grp.create_dataset(
            name=name,
            data=dataset,
            maxshape=dataset.shape,
            compression='gzip',
            compression_opts=6,
            chunks=cool[f'pixels/{name}'].chunks
        )

    # rewrite indexes.bin1_offset dataset.
    del sub_cool['indexes']['bin1_offset']
    sub_cool.create_dataset(name='indexes/bin1_offset', data=np.cumsum(new_indptr))
    sub_cool.attrs['nnz'] = sub_cool['pixels']['bin1_id'].size
    sub_cool.file.close()
    cool.file.close()

    return True
