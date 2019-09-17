"""Utils for other modules."""
import inspect
import logging
import functools
import itertools
import multiprocessing
import warnings
import numbers
from collections import UserDict
from functools import partial, wraps
from typing import Union, Iterable, Callable
from contextlib import redirect_stderr

import numpy as np
from scipy import sparse

from ..config import *

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
    """Fill specified value in a given diagonal of a 2d ndarray.\n
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
    if isinstance(fill_values, numbers.Number):
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

        fn_names = self._methods.keys()
        bound_method = type(self.register)
        for name in fn_names:
            method = self._methods[name]
            if instance is not None:
                sub_method.__dict__[name] = bound_method(method, instance)
            else:
                sub_method.__dict__[name] = method

        self.__doc__ = self._doc
        sub_method.__doc__ = self._doc
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
        doc_template = self.__dict__.get('__doc__', None)
        if doc_template is None:
            doc_template = "{methods}"

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
    """An wrap of ray, for redirect ray log and debug easily.
    If not enable_ray, the code will execute serially.
    """

    import ray

    _cache = {}  # store mapping from task id to result obj

    def __init__(self,
                 *args,
                 enable_ray: bool = None,
                 log_file: str = "./ray.log",
                 **kwargs):
        if enable_ray is None:
            enable_ray = not config.DEBUG
        self.enable_ray = enable_ray
        self.log_file = log_file
        if enable_ray:
            if not self.ray.is_initialized():
                with open(log_file, 'a') as f:
                    with redirect_stderr(f):
                        self.ray.init(*args, **kwargs)

    def remote(self, obj):
        if self.enable_ray:
            return self.ray.remote(obj)
        else:
            if inspect.isclass(obj):
                return self._mimic_actor(obj)
            elif inspect.isfunction(obj):
                return self._mimic_func(obj)
            else:
                raise TypeError("Only support remote fcuntion or class(Actor)")

    def _mimic_actor(self, cls):
        """mimic Actor's behavior"""

        class _Actor(cls):
            def __init__(obj, *args, **kwargs):
                super().__init__(*args, **kwargs)
                obj.ray = self

            @classmethod
            def remote(cls_, *args, **kwargs):
                obj = cls_(*args, **kwargs)
                return obj

        for name, attr in inspect.getmembers(_Actor):
            if not inspect.isfunction(attr) or name.startswith('__'):
                continue
            setattr(_Actor, name, MethodWithRemote(attr))

        return _Actor

    def _mimic_func(self, obj: Callable):
        """ mimic remote function """
        log = get_logger()

        def _remote(*args, **kwargs):
            log.debug(f"Remote function '{obj.__qualname__}' is called.")
            id_ = f"{obj.__qualname__}_{args}_{kwargs}"
            self._cache[id_] = obj(*args, **kwargs)
            return id_

        @wraps(obj)
        def wrapper(*args, **kwargs):
            return obj(*args, **kwargs)

        wrapper.remote = _remote

        return wrapper

    def get(self, id_):
        if self.enable_ray:
            return self.ray.get(id_)
        else:
            return self._cache[id_]


class MethodWithRemote(object):
    """For mimic ray Actor's method.remote API."""

    def __init__(self, mth):
        self.mth = mth

    def __get__(self, instance, owner):
        self.instance = instance
        self.owner = owner
        if instance is None:
            return self.mth
        else:
            return self

    def remote(self, *args, **kwargs):  # mimic actor.func.remote()
        log = get_logger("Mimic remote")
        log.debug(f"Remote method '{self.mth.__qualname__}' is called.")
        id_ = f"{self.owner.__name__}[{id(self.instance)}]" + \
              f".{self.mth.__name__}_{args}_{kwargs}"
        res = self.mth(self.instance, *args, **kwargs)
        self.instance.ray._cache[id_] = res
        return id_

    def __call__(self, *args, **kwargs):
        msg = "Actor methods cannot be called directly." + \
              f"Instead of running 'object.{self.mth.__name__}()', " + \
              f"try 'object.{self.mth.__name__}.remote()'."
        raise Exception(msg)


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
