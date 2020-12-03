"""Utils for other modules."""
import re
import inspect
import functools
import logging
import multiprocessing
import warnings
from typing import Callable, Tuple, Union, Iterable, Mapping, Any
import click

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


def parse_docstring(doc: str) -> Iterable[Tuple[str, str, str, str]]:
    """Parsing sphinx style doctring.

    sphinx docstring format example:

        :param steps: int. Number of steps in each batch.
         ----- -----  ---  ------------------------------
         kind  name   tp   desc

    """
    kind = name = tp = desc = ""
    for line in doc.split("\n"):
        line = line.strip()
        # import ipdb; ipdb.set_trace()
        m = re.match(r":(.+) (.*?): (.+)\. (.*)$", line) or \
            re.match(r":(.+) ?(.*?): (.+)\.(.*)$", line)
        if m:
            if kind is not None:
                yield (kind, name, tp, desc)
            kind, name, tp, desc = m.groups()
        else:
            if kind == 'param':
                desc += " " + line
    yield (kind, name, tp, desc)


def paste_doc(source: Union[str, Callable]):
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

    def process_opt(opt):
        return opt.lstrip('-').replace('-', '_')

    def in_params(arg):
        for opt in arg.opts:
            opt = process_opt(opt)
            if opt in params:
                return params[opt]
        return None

    def decorate(target: Union[Callable, click.Command]):
        if isinstance(target, click.Command):
            # copy doc string to argument's help
            for arg in target.params:
                if not isinstance(arg, click.Option):
                    continue
                desc = in_params(arg)
                if desc and (not arg.help):
                    arg.help = desc
            return target
        else:  # copy docstring to another function
            return NotImplemented

    return decorate


def get_logger(name: str = None) -> logging.Logger:
    """Get a logging.Logger object.
    :param name: the name of the Logger object, if not set will
    set a default name according to it's caller.
    """
    from inspect import currentframe
    from os.path import basename, splitext

    # set a default name to logger
    if name is None:
        call_frame = currentframe().f_back
        file_ = splitext(basename(call_frame.f_code.co_filename))[0]
        name_ = call_frame.f_code.co_name
        name = f"{file_}:{name_}" if name != '<module>' else file_

    log = logging.getLogger(name)

    return log


class MethodWrapper(object):

    def __init__(self,
                 input_handler: Callable[[Callable, Tuple, dict, dict], Tuple[Tuple, dict]] = None,
                 output_handler: Callable[[Callable, Tuple, dict, dict, Any], Any] = None):
        self.input_handler = input_handler
        self.output_handler = output_handler

    def __call__(self, attr, attr_obj):
        if callable(attr_obj):
            fn = attr_obj
            origin_params = inspect.signature(fn).parameters

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                # handle input args
                kwargs, extra_kwargs = self._extract_extra_kargs(
                    origin_params, kwargs)
                if self.input_handler:
                    args, kwargs = self.input_handler(
                        fn, args, kwargs, extra_kwargs)
                    kwargs, extra_kwargs = self._extract_extra_kargs(
                        origin_params, kwargs)
                # call original function
                result = fn(*args, **kwargs)
                # handle outputs of original function
                if self.output_handler:
                    result = self.output_handler(
                        fn, args, kwargs, extra_kwargs, result)

                return result

            return wrapper
        else:
            if self.output_handler is not None:
                attr_obj = self.output_handler(None, attr_obj)
            return attr_obj

    @staticmethod
    def _extract_extra_kargs(params: Mapping[str, inspect.Parameter], kwargs: dict):
        extra_kwargs, remain_kwargs = {}, {}
        for k, v in kwargs.items():
            if k in params:
                remain_kwargs[k] = v
            else:
                extra_kwargs[k] = v
        return remain_kwargs, extra_kwargs

    @classmethod
    def wrap_attr(cls, wrap_cls, predicate: Callable[[str, object], bool], wrapper: 'MethodWrapper'):
        import inspect
        for attr, _ in inspect.getmembers(wrap_cls):
            # raw object without obj.xx
            attr_obj = inspect.getattr_static(wrap_cls, attr)
            if callable(attr_obj) and predicate(attr, attr_obj):
                setattr(wrap_cls, attr, wrapper(attr, attr_obj))


if __name__ == "__main__":
    pass
