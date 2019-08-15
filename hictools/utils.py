"""
Utils for other modules.
"""
import functools
import multiprocessing
import shutil
import subprocess
import warnings
from typing import Union

import numpy as np
from scipy import sparse

CPU_CORE = multiprocessing.cpu_count()


class LazyProperty(object):
    """Lazy property for caching computed properties"""

    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner):
        if instance is None:
            return self
        else:
            value = self.func(instance)
            setattr(instance, self.func.__name__, value)
            return value


def lazy_method(func):
    """Lazy method for caching results of time-consuming methods"""

    @functools.wraps(func)
    def lazy(self, *args, **kwargs):
        key = "_lazy_{}_{}_{}".format(func.__name__, args, kwargs)
        if hasattr(self, key):
            return getattr(self, key)
        else:
            value = func(self, *args, **kwargs)
            setattr(self, key, value)
            return value

    return lazy


def suppress_warning(func=None, warning_msg=RuntimeWarning):
    """Ignore the given type of warning omitted from the function. The default warning is RuntimeWarning."""
    if func is None:
        return functools.partial(suppress_warning, warning_msg=warning_msg)

    @functools.wraps(func)
    def inner(*args, **kwargs):
        with warnings.catch_warnings():
            if warning_msg is None:
                warnings.simplefilter('ignore')
            else:
                warnings.simplefilter('ignore', warning_msg)
            results = func(*args, **kwargs)
        return results

    return inner


def mask_array(mask, *args) -> np.ndarray:
    """Mask all ndarray in args with a given Boolean array.

    :param mask: np.ndarray. Boolean array where desired values are marked with True.
    :param args: tuple. tuple of np.ndarray. Masking will be applied to each ndarray.
    :return: np.ndarray. A generator yield a masked ndarray each time.
    """
    for mat in args:
        if isinstance(mat, (tuple, list)):
            yield tuple(mask_array(mask, *mat))
        else:
            if len(mat.shape) == 1:
                yield mat[mask]
            else:
                yield mat[:, mask]


def index_array(index, *args):
    """Index all ndarray in args with a given Integer array. Be cautious of the order of each value in indexed ndarray.

    :param index: np.ndarray. Integer array with indexs of desired values'.
    :param args: tuple. tuple of np.ndarray. Indexing will be applied to each ndarray.
    :return: np.ndarray. A generator yield indexed ndarray each time.
    """
    yield from mask_array(index, *args)


def check_slice(s: Union[slice, int]):
    if not isinstance(s, slice):
        return slice(0, s, 1)
    else:
        return s


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

        text = not (True if (len(mode) == 2 and mode[1] == 'b') else False)
        with open(file, mode) as file:
            if mode[0] == 'w':
                stdin, stdout = subprocess.PIPE, file
            elif mode[0] == 'r':
                stdin, stdout = file, subprocess.PIPE
            else:
                raise ValueError('mode only support write and read')
            pipe = subprocess.Popen(
                command,
                stdin=stdin,
                stdout=stdout,
                shell=True,
                bufsize=-1,
                universal_newlines=text
            )

            return pipe

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

        if self.mode[0] == 'w':
            self._stream = self._pipe.stdin
        else:
            self._stream = self._pipe.stdout

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


def remove_small_gap(gap_mask: np.ndarray, gap_size: int = 1) -> np.ndarray:
    """Remove gaps with length shorter than the specified length threshold in a Boolean array.

    :param gap_mask: np.ndarray. Boolen array(mask) in which gap region are marked with True.
    :param gap_size: int. Gap length threshold to define a gap as small gap.
    :return: np.ndarray. New mask with small gaps are removed. ie: True to False.
    """
    # TODO(zhongquan789@126.com) support for gap_size
    gap_indexs = np.where(gap_mask)[0]
    single_gap = []
    for i in range(1, len(gap_indexs) - 1):
        adjacent_f = gap_indexs[i] == (gap_indexs[i - 1] + 1)
        adjacent_b = gap_indexs[i] == (gap_indexs[i + 1] - 1)
        if not (adjacent_f or adjacent_b):
            single_gap.append(gap_indexs[i])
        if gap_indexs[0] != (gap_indexs[1] - 1):
            single_gap.append(gap_indexs[0])
        if gap_indexs[-1] != (gap_indexs[-2] + 1):
            single_gap.append(gap_indexs[-1])
    gap_mask = np.full_like(gap_mask, False)
    gap_mask[list(set(gap_indexs) - set(single_gap))] = True

    return gap_mask


@suppress_warning
def is_symmetric(mat: Union[np.ndarray, sparse.spmatrix],
                 rtol: float = 1e-05,
                 atol: float = 1e-08) -> bool:
    """Check if the input matrix is symmetric.

    :param mat: np.ndarray/scipy.sparse.spmatrix.
    :param rtol: float. The relative tolerance parameter. see np.allclose.
    :param atol: float. The absolute tolerance parameter. see np.allclose
    :return: bool. True if the input matrix is symmetric.
    """
    if isinstance(mat, np.ndarray):
        data, data_t = mat, mat.T
        return np.allclose(data, data_t, rtol=rtol, atol=atol, equal_nan=True)
    elif sparse.isspmatrix(mat):
        mat = mat.copy()
        mat.data[np.isnan(mat.data)] = 0
        return (np.abs(mat - mat.T) > rtol).nnz == 0
    else:
        raise ValueError('Only support for np.ndarray and scipy.sparse_matrix')


def fill_diag(mat: np.ndarray,
              offset: int = 1,
              fill_value: float = 1.0,
              copy: bool = False) -> np.ndarray:
    """
    Fill specified value in a given diagonal of a 2d ndarray.
    Reference: https://stackoverflow.com/questions/9958577/changing-the-values-of-the-diagonal-of-a-matrix-in-numpy
    :param mat: np.ndarray.
    :param offset: int. The diagonal's index. 0 means the main diagonal.
    :param fill_value: float. Value to fill the diagonal.
    :param copy: bool. Set True to fill value in the copy of input matrix.
    :return: np.ndarray. Matrix with the 'offset' diagonal filled with 'fill_value'.
    """

    if copy:
        mat = mat.copy()
    length = mat.shape[1]
    st = max(offset, -length * offset)
    ed = max(0, length - offset)
    mat.ravel()[st: ed: length + 1] = fill_value

    return mat


if __name__ == "__main__":
    pass
