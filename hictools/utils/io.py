"""Tools for io handling."""

import os
import shutil
import subprocess
from collections import OrderedDict, namedtuple
from typing import Iterable, Tuple

import h5py
import numpy as np
import pandas as pd

from .utils import get_logger


class auto_open(object):
    # TODO(zhongquan789@gmail.com)  1.Add log system. 2.Add exception handling. 3.Add stderr handling.
    """
    Wrapper for built-in function open.
    Additional support for automatically handling bam-sam and gzip-text file transversion.
    """

    def __init__(self, file: str,
                 mode: str = 'r',
                 nproc: int = 4,
                 command: str = None,
                 convert: bool = True):
        """

        :param file: str.  File name.
        :param mode: str. FILE mode. support r/w/rb/wb/. default: 'r'
        :param nproc: int. Numbers of process used for file format transversion. default: 1
        :param command: str. User defined command to replace default command. default: None
        :param convert: bool. If the automatically file format transversion is activated. default: True
        """

        self._file = file
        self._nproc = nproc
        self._pipe = None
        self._stream = None
        self._convert = convert
        self.mode = mode
        self.command = command

        if mode not in ('r', 'w', 'rb', 'wb'):
            raise ValueError('Only support r w rb wb mode.')

        self._create_stream()

    @staticmethod
    def _popen(command: str, file: str, mode: str) -> subprocess.Popen:
        """

        :param command: str. Command used as the paramether args in subprocess.Popen.
        :param file: str. File name used as the parameter file in built-in function open.
        :param mode: str. Mode used as the parameter mode in built-in function open.
        :return:
        """

        text = len(mode) != 2 or mode[1] != 'b'
        with open(file, mode) as f:
            if mode[0] == 'w':
                stdin, stdout = subprocess.PIPE, f
            elif mode[0] == 'r':
                stdin, stdout = f, subprocess.PIPE
            else:
                raise ValueError('mode only support write and read')
            return subprocess.Popen(
                command,
                stdin=stdin,
                stdout=stdout,
                shell=True,
                bufsize=-1,
                universal_newlines=text
            )

    @classmethod
    def _handle_bam(cls, file: str, mode: str, nproc: int) -> subprocess.Popen:
        """

        :param file: str. Bam/Sam file name.
        :param mode: str. Mode used as the parameter mode in built-in function open.
        :param nproc: int. Numbers of process used in samtools.
        :return:
        """

        if shutil.which('samtools') is None:
            raise ValueError('samtools not exist in PATH.')

        if mode[0] == 'w':
            command = "samtools view -bS -@ {} -".format(nproc)
        else:
            command = "samtools view -h -@ {}".format(nproc)

        return cls._popen(command, file, mode)

    @classmethod
    def _handle_gzip(cls, file: str, mode: str, nproc: int) -> subprocess.Popen:
        """

        :param file: str. gz-end file or text file which will be converted to text and gz file respectively.
        :param mode: str. Mode used as the parameter mode in built-in function open.
        :param nproc: int. Numbers pf process used in pbgzip.
        :return:
        """
        if shutil.which('pbgzip') is None:
            raise ValueError('pbgzip not found')

        if mode[0] == 'w':
            command = "bgzip -c -@ {}".format(nproc)
        else:
            command = "bgzip -dc -@ {}".format(nproc)

        return cls._popen(command, file, mode)

    def _create_stream(self):
        """Dispatch file handler to create certain stream object according to their file name. e.g. .bam .gz
        """
        if self.command is not None:
            self._pipe = self._popen(self.command, self._file, self.mode)

        elif self._convert and self._file.endswith("bam"):
            self._pipe = self._handle_bam(self._file, self.mode, self._nproc)

        elif self._convert and self._file.endswith('gz'):
            self._pipe = self._handle_gzip(self._file, self.mode, self._nproc)

        else:
            self._stream = open(self._file, self.mode)
            return self._stream

        self._stream = self._pipe.stdin if self.mode[0] == 'w' else self._pipe.stdout

    def __enter__(self):
        """Emulating context_manager-like behavior.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Make some cleaning stuff.
        """
        if self._pipe is not None:
            self._pipe.communicate()
        else:
            self._stream.close()

    def __getattr__(self, attr):
        """Interface for inside stream object.
        """
        return getattr(self._stream, attr)

    def __iter__(self):
        """Interface for inside stream object.
        """
        return self._stream

    def __dir__(self):
        return list(set(list(self.__dict__.keys()) + dir(self._stream)))

    def __repr__(self):
        return repr(self._stream)


def stream_to_file(filename: str, stream):
    """Read from stream and write into file named filename line by line.

    :param filename: str. File name of the disired output file.
    :param stream:
    :return:
    """
    with auto_open(filename, 'w') as f:
        for line in stream:
            f.write(line)


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
