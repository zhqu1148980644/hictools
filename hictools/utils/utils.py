"""Utils for other modules."""
import re
import functools
import inspect
import logging
import multiprocessing
import warnings
from collections import UserDict
from contextlib import redirect_stderr
from functools import partial, wraps
import typing as T

import numpy as np

from .. import config

CPU_CORE = multiprocessing.cpu_count()


# TODO fix
# def infer_mat(mat,
#               mask: np.ndarray = None,
#               mask_ratio: float = 0.2,
#               check_symmetric: bool = False,
#               copy: bool = False) -> tuple:
#     """Maintain non-zero contacts outside bad regions in a triangular sparse matrix.\n
#     When calculating decay, always keep contacts outside bad regions to non-nan, and keep contacts within bad regions to nan.\n
#     This step could take considerable time as dense matrix enable the fast computaion of decay whereas sparse matrix
#     can reduce space occupancy and speed up the calculation of OE matrix.\n
#
#     :param mat: np.ndarray/scipy.sparse.sparse_matrix.
#     :param mask: np.ndarray.
#     :param mask_ratio: float.
#     :param span_fn: callable.
#     :param check_symmetric: bool.
#     :param copy: bool.
#     :return: tuple(scipy.sparse.coo_matrix, np.ndarray, np.ndarray).
#     """
#
#     def find_mask(nan_mat: np.ndarray):
#         last = None
#         last_row = -1
#         while 1:
#             row = np.random.randint(mat.shape[0])
#             if row != last_row and not np.alltrue(nan_mat[row]):
#                 if last is None:
#                     last = nan_mat[row]
#                 elif np.all(last == nan_mat[row]):
#                     return ~last
#                 else:
#                     return None
#             last_row = row
#
#     def mask_by_ratio(mat: np.ndarray) -> np.ndarray:
#         col_mean = np.nanmean(mat, axis=0)
#         return col_mean > (np.mean(col_mean) * mask_ratio)
#
#     if check_symmetric and not is_symmetric(mat):
#         raise ValueError('Matrix is not symmetric.')
#
#     if copy:
#         mat = mat.copy()
#
#     if not isinstance(mat, np.ndarray) and not isinstance(mat, sparse.coo_matrix):
#         mat = mat.tocoo(copy=False)
#
#     if mask is None:
#         if not isinstance(mat, np.ndarray):
#             mat_cache = mat
#             mat = mat.toarray()
#
#         nan_mat = np.isnan(mat)
#         contain_nan = nan_mat.any()
#         if contain_nan:
#             mask = find_mask(nan_mat)
#             if mask is None:
#                 mask = mask_by_ratio(mat)
#         else:
#             mask = mask_by_ratio(mat)
#         nan_mask = ~(mask[:, np.newaxis] * mask[np.newaxis, :])
#         if contain_nan and nan_mat[~nan_mask].any():
#             mat[nan_mat] = 0
#         mat[nan_mask] = np.nan
#         decay = np.array(tuple(get_decay(mat)))
#
#         if not isinstance(mat, np.ndarray):
#             mat = sparse.triu(mat_cache)
#             mat.eliminate_zeros()
#             mat.data[np.isnan(nan_mask[mat.nonzero()])] = 0
#             mat.data[np.isnan(mat.data)] = 0
#             mat.eliminate_zeros()
#         else:
#             mat[nan_mask] = 0
#             mat = sparse.triu(mat, format='coo'), mask, decay
#
#     else:
#         if not isinstance(mat, np.ndarray):
#             nan_mask = ~(mask[:, np.newaxis] * mask[np.newaxis, :])
#             mat.data[np.isnan(mat.data)] = 0
#
#             dense_mat = mat.toarray()
#             dense_mat[nan_mask] = np.nan
#             decay = np.array(tuple(get_decay(dense_mat)))
#
#             mat = sparse.triu(mat)
#             mat.eliminate_zeros()
#             mat.data[np.isnan(nan_mask[mat.nonzero()])] = 0
#             mat.eliminate_zeros()
#         else:
#             nan_mat = np.isnan(mat)
#             contain_nan = nan_mat.any()
#             nan_mask = ~(mask[:, np.newaxis] * mask[np.newaxis, :])
#             if contain_nan & nan_mat[~nan_mask].any():
#                 mat[nan_mat] = 0
#             mat[nan_mask] = np.nan
#             decay = np.array(tuple(get_decay(mat)))
#             mat[nan_mask] = 0
#             mat = sparse.triu(mat, format='coo')
#
#     return mat, mask, decay


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
            raise RuntimeError(
                f"Can't register method with the same name: '{key}' multiple times.")
        super().__setitem__(key, value)


class multi_methods(object):
    # TODO Add support for descriptor.G
    # How to compitable with ray?
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
        def sub_method(obj, *args, **kwargs):
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

    def remote(self, obj, **kwargs):
        """Entry point."""
        if self.enable_ray:
            return self.ray.remote(**kwargs)(obj)
        else:
            if inspect.isclass(obj):
                return self._mimic_actor(obj)
            elif inspect.isfunction(obj):
                return self._mimic_func(obj)
            else:
                raise TypeError("Only support remote fcuntion or class(Actor)")

    def _mimic_actor(self, cls):
        """Mimic Actor's behavior."""

        class _Actor(cls):
            def __init__(obj, *args, **kwargs):
                super().__init__(*args, **kwargs)
                obj.ray = self

            @classmethod
            def remote(cls_, *args, **kwargs):
                obj = cls_(*args, **kwargs)
                return obj

        for name, attr in inspect.getmembers(_Actor):
            if not inspect.isfunction(attr) or name.startswith('_'):
                continue
            setattr(_Actor, name, MethodWithRemote(attr))

        return _Actor

    def _mimic_func(self, obj: T.Callable):
        """Mimic remote function."""
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

    def remote(self, *args, **kwargs):
        log = get_logger("Mimic remote")
        log.debug(f"Remote method '{self.mth.__qualname__}' is called.")
        id_ = f"{self.owner.__name__}[{id(self.instance)}]" + \
              f".{self.mth.__name__}_{args}_{kwargs}"
        res = self.mth(self.instance, *args, **kwargs)
        self.instance.ray._cache[id_] = res
        return id_

    def __call__(self, *args, **kwargs):
        msg = "Actor methods cannot be called directly. " + \
              f"Instead of running 'object.{self.mth.__name__}()', " + \
              f"try 'object.{self.mth.__name__}.remote()'."
        raise Exception(msg)


def get_logger(name: str = None) -> logging.Logger:
    """Get a logging.Logger object.
    :param name: the name of the Logger object, if not set will
    set a default name according to it's caller.
    """
    from inspect import currentframe, getframeinfo
    from os.path import basename, splitext

    # set a default name to logger
    if name is None:
        call_frame = currentframe().f_back
        file_ = splitext(basename(call_frame.f_code.co_filename))[0]
        name_ = call_frame.f_code.co_name
        name  = f"{file_}:{name_}" if name != '<module>' else file_

    log = logging.getLogger(name)

    return log


import click


def parse_docstring(doc:str) -> T.Iterable[T.Tuple[str, str, str, str]]:
    """Parsing sphinx style doctring.

    sphinx docstring format example:

        :param steps: int. Number of steps in each batch.
         ----- -----  ---  ------------------------------
         kind  name   tp   desc

    """
    kind = None
    for line in doc.split("\n"):
        line = line.strip()
        #import ipdb; ipdb.set_trace()
        m = re.match(":(.+) (.*?): (.+)\. (.*)$", line) or \
            re.match(":(.+) ?(.*?): (.+)\.(.*)$",  line)
        if m:
            if kind is not None:
                yield (kind, name, tp, desc)
            kind, name, tp, desc = m.groups()
        else:
            if kind == 'param':
                desc += " " + line
    yield (kind, name, tp, desc)


def paste_doc(source:T.Union[str, T.Callable]):
    """Copy docstring or Click command help,
    avoiding document one thing many times.

    For example, copy `api_func1`'s doc to command `cli_func1`:

        @copy_doc(api_func1)
        @click.command()
        @click.option("--arg1")
        def cli_func1(arg1):
            ...

    """
    if isinstance(source, str):
        doc = source
    else:
        doc = source.__doc__
    params = filter(lambda i: i[0] == 'param', parse_docstring(doc))
    params = {name: desc for (kind, name, tp, desc) in params}

    process_opt = lambda opt: opt.lstrip('-').replace('-', '_')
    def in_params(arg):
        for opt in arg.opts:
            opt = process_opt(opt)
            if opt in params:
                return params[opt]
        return None

    def decorate(target:T.Union[T.Callable, click.Command]):
        if isinstance(target, click.Command):
            # copy doc string to argument's help
            for arg in target.params:
                if not isinstance(arg, click.Option): continue
                desc = in_params(arg)
                if desc and (not arg.help):
                    arg.help = desc
            return target
        else:  # copy docstring to another function
            return NotImplemented
    return decorate


if __name__ == "__main__":
    pass
