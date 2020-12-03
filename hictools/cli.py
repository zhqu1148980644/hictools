"""Command line tools."""
import sys
import logging
from functools import partial

import click

from . import config
from .hgserver.cli import hgserver

click.option = partial(click.option, show_default=True)
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
@click.option("--log-file", help="The log file, default output to stderr.")
@click.option("--debug", is_flag=True,
              help="Open debug mode, disable ray.")
def cli(log_file, debug):
    log = logging.getLogger()  # root logger
    if log_file:
        handler = logging.FileHandler(log_file)
    else:
        handler = logging.StreamHandler(sys.stderr)

    fomatter = logging.Formatter(
        fmt=config.LOGGING_FMT,
        datefmt=config.LOGGING_DATE_FMT
    )
    handler.setFormatter(fomatter)
    log.addHandler(handler)

    if debug:
        config.DEBUG = True
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)


cli.add_command(hgserver)

if __name__ == "__main__":
    cli()
