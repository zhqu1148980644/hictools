"""
Utils for other modules.
"""
import logging
import functools
import itertools
import multiprocessing
import shutil
import subprocess
import warnings
from collections import UserDict
from functools import partial, wraps
from typing import Union, Iterable

import numpy as np
from scipy import sparse
import pandas as pd

CPU_CORE = multiprocessing.cpu_count()


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


def records2bigwigs(df: pd.DataFrame, prefix: str):
    """ Dump dataframe to bigwig files
    :param df: records dataframe, contain fields: chrom, start, end.
    :param prefix: prefix of output bigwig files.
    """
    import pyBigWig
    required_fields = ['chrom', 'start', 'end']
    assert all([(f in df) for f in required_fields]), \
        f"records dataframe need fields: {', '.join(required_fields)}"

    val_cols = [c for c in list(df.columns) if c not in required_fields]
    bigwigs = {}
    for col in val_cols:
        import os
        from os.path import exists
        path_ = prefix + '.' + col + '.bw'
        if exists(path_):
            os.remove(path_)
        bigwigs[col] = pyBigWig.open(path_, 'w')

    chroms = df['chrom'].drop_duplicates()
    chroms2maxend = {c: df[df['chrom'] == c]['end'].max() for c in chroms}
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
              offset: int = 0,
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
    ed = max(0, length - offset) * length
    mat.ravel()[st: ed: length + 1] = fill_value

    return mat


def fill_diags(mat: np.ndarray,
               ignore_diags: Union[int, Iterable] = 1,
               fill_values: Union[float, Iterable] = 1.,
               copy: bool = False) -> np.ndarray:
    if isinstance(ignore_diags, int):
        ignore_diags = range(-ignore_diags + 1, ignore_diags)
    if isinstance(fill_values, float):
        fill_values = itertools.repeat(fill_values)
    if copy:
        mat = mat.copy()

    for diag_index, fill_value in zip(ignore_diags, fill_values):
        fill_diag(mat, diag_index, fill_value)

    return mat


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


# Dprecated. use functools.lru_cache
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


class NodupsDict(UserDict):
    def __setitem__(self, key, value):
        if key in self:
            raise RuntimeError(f"Can't register method with the same name: '{key}' multiple times.")
        super().__setitem__(key, value)


class multi_methods(object):
    """Dispatch multi methods through attributes fetching"""

    def __new__(cls, func=None, **kwargs):
        if func is None:
            return partial(cls, **kwargs)
        else:
            return super().__new__(cls)

    def __init__(self, func=None, **kwargs):
        self._func = func
        self._global_config = kwargs
        self._methods = NodupsDict()

    def __get__(self, instance, owner):
        @wraps(self._func)
        def sub_method(self, *args, **kwargs):
            if args or kwargs:
                for _name in fn_names:
                    sub_method.__dict__[_name] = partial(
                        sub_method.__dict__[_name],
                        *args,
                        **kwargs
                    )
            return sub_method

        if '__doc__' not in self.__dict__:
            self.__dict__['__doc__'] = self._doc

        fn_names = self._methods.keys()
        bound_method = type(self.register)
        for name in fn_names:
            method = self._methods[name]
            if instance is not None:
                sub_method.__dict__[name] = bound_method(method, instance)
            else:
                sub_method.__dict__[name] = method

        sub_method.__dict__['__doc__'] = self._doc
        if instance is not None:
            return bound_method(sub_method, instance)
        else:
            return sub_method

    def __set__(self):
        raise PermissionError("Not allowed.")

    @LazyProperty
    def _doc(self):
        num = len(self._methods)
        methods = '\n' + '\n'.join(f"{name}:\n{fn.__doc__}"
                                   for name, fn in self._methods.items())
        doc_template = self._func.__doc__

        return doc_template.format(num=num, methods=methods)

    def __call__(self, func=None, **kwargs):
        if func is None:
            return partial(self, **kwargs)
        else:
            return self.register(func, **kwargs)

    def register(self, func=None, **kwargs):
        if func is None:
            return partial(self.register, **kwargs)
        else:

            func.__name__ = func.__name__.strip('_')
            self._methods[func.__name__] = func


class RayWrap(object):
    """
    An wrap of ray, for redirect ray log and debug easily.
    If not enable_ray, the code will execute serially.
    """

    import ray

    _cache = {}  # store mapping from task id to result obj

    def __init__(self, *args,
                 enable_ray: bool = None,
                 log_file: str = "./ray.log",
                 **kwargs):
        if enable_ray is None:
            from . import config
            debug = config.DEBUG
            self.enable_ray = (not debug)
        else:
            self.enable_ray = enable_ray
        self.log_file = log_file
        if self.enable_ray:
            from contextlib import redirect_stderr
            if not self.ray.is_initialized():
                with open(log_file, 'a') as f:
                    with redirect_stderr(f):
                        self.ray.init(*args, **kwargs)

    def remote(self, obj):
        if self.enable_ray:
            return self.ray.remote(obj)
        else:
            import inspect
            if inspect.isclass(obj):
                return self._mimic_actor(obj)
            elif inspect.isfunction(obj):
                return self._mimic_func(obj)
            else:
                raise TypeError("Only support remote fcuntion or class(Actor)")

    def _mimic_actor(self, cls):
        """mimic Actor's behavior"""
        import inspect
        log = get_logger()

        class _Actor(cls):
            def __init__(obj, *args, **kwargs):
                super().__init__(*args, **kwargs)

                def make_remote(name, mthd):
                    def remote_(*args, **kwargs):  # mimic actor.func.remote()
                        log.debug(f"Remote method '{cls.__name__}.{name}' is called.")
                        id_ = f"{cls.__name__}[{id(obj)}].{name}_{args}_{kwargs}"
                        res = mthd(*args, **kwargs)
                        self._cache[id_] = res
                        return id_

                    return remote_

                for name, attr in inspect.getmembers(obj):
                    if not inspect.ismethod(attr) or name.startswith('__'):
                        continue
                    mthd = attr
                    mic_mthd = mimic_method(mthd)
                    mic_mthd.remote = make_remote(name, mthd)
                    setattr(obj, name, mic_mthd)

            @classmethod
            def remote(cls_, *args, **kwargs):
                obj = cls_(*args, **kwargs)
                obj.ray = self
                return obj

        return _Actor

    def _mimic_func(self, obj):
        """ mimic remote function """

        def wrapper(*args, **kwargs):
            print(f"Remote function '{obj.__name__}' is called.")
            id_ = f"{obj.__name__}_{args}_{kwargs}"
            res = obj(*args, **kwargs)
            self._cache[id_] = res
            return id_

        return wrapper

    def get(self, id_):
        if self.enable_ray:
            return self.ray.get(id_)
        else:
            return self._cache[id_]


class mimic_method(object):
    """ Used for mimic class's method.
    Use this substitude original bound method,
    you can specify attributes to it, like:
    >>> a = A()
    >>> a.mth.b = 1  # this is not allowed!(raise AttributeError)
    >>> a.mth = mimic_method(a.mth)
    >>> a.mth.b = 1  # this is allowed
    """

    def __init__(self, mth):
        self.mth = mth

    def __call__(self, *args, **kwargs):
        return self.mth(*args, **kwargs)


def get_logger(name: str = None) -> logging.Logger:
    """
    :param name: the name of the Logger object, if not set will
    set a default name according to it's caller.
    """
    from inspect import currentframe, getframeinfo, ismethod

    def get_caller():
        """
        Get caller function of the `get_logger`.
        reference: https://stackoverflow.com/a/4493322/8500469
        """
        cal_f = currentframe().f_back.f_back
        func_name = getframeinfo(cal_f)[2]
        outer_f = cal_f.f_back
        func = outer_f.f_locals.get(
            func_name,
            outer_f.f_globals.get(func_name))
        if func is None:  # call from click command
            func = cal_f.f_globals.get(func_name)
        if (func is None) and ('self' in outer_f.f_locals):  # call from method
            try:
                func = getattr(outer_f.f_locals.get('self'), func_name)
            except AttributeError:
                pass
        return func

    if name is None:  # set a default name to logger
        caller = get_caller()
        assert caller is not None, "Caller not Found."
        import click
        if isinstance(caller, click.core.Command):  # click command
            name = 'CLI.' + caller.name
        else:  # function & method
            name = caller.__module__ + '.'
            if '__wrapped__' in caller.__dict__:
                cname = caller.__wrapped__.__qualname__
            else:
                cname = caller.__qualname__
            name += cname

    log = logging.getLogger(name)

    return log


if __name__ == "__main__":
    pass
