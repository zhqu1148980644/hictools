import os
import socket
import shutil
import asyncio
import contextlib
import subprocess
from pathlib import Path
from functools import partial
from concurrent.futures import ProcessPoolExecutor

import click
import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse

from .server import Server
from .store import default_monitor

click.option = partial(click.option, show_default=True)
logging_level = "error"


def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('1.1.1.1', 1))
        ip = s.getsockname()[0]
    except:
        ip = "127.0.0.1"
    finally:
        s.close()

    return ip


IP = get_ip()


def echo(text, fg="white", bg="black"):
    return click.echo(click.style(text, fg, bg))


@click.group()
@click.option('--log_level', default="error", type=click.Choice(
    ['critial', 'error', 'warning', 'info', 'debug'], case_sensitive=False))
def hgserver(log_level):
    """View results with higlass.

    Steps:\n
    >> hictools hgserver serve --port 7777 --paths ./\n
    >> hictools hgserver view --api_port 7777 --port 8888
    """
    global logging_level
    logging_level = log_level


@hgserver.command()
@click.option('--api_port', type=int, default=0, help="Apiserver port opened by 'hictools hgserver serve'")
@click.option('--host', type=str, default="0.0.0.0")
@click.option('--port', type=int, default=8888)
def view(api_port, host, port):
    """Start higlass web app.

    Example: hictools hgserver view --api_port 7777 --port 8888
    """
    # Could use
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get('/', response_class=HTMLResponse)
    def index():
        html = open(f'{Path(__file__).parent.resolve()}/ui/index.html').read()
        api = f"'http://{IP}:{api_port}/api/v1'" if api_port else "''"
        html = html.replace('{}', api)
        return html

    echo(f"Go visit http://{IP}:{port} in browser.", "green")
    uvicorn.run(app, host=host, port=port, log_level=logging_level)


@hgserver.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True
))
@click.option('--store_uri', type=str, default="sqlite:////tmp/test.db", help='Database URI.')
@click.option('--host', type=str, default="0.0.0.0")
@click.option('--port', type=int, default=7777)
@click.option('--paths', type=click.Path(exists=True, readable=True), multiple=True,
              help="Path to monitor file changes.")
@click.pass_context
def serve(ctx, store_uri, host, port, paths):
    uvicorn.main.parse_args(ctx, args=['TMP'] + ctx.args)
    kwargs = ctx.params.copy()
    for key in ('store_uri', 'paths', 'app', 'no_access_log'):
        del kwargs[key]
    kwargs.update({
        'host': host,
        'port': port,
        'log_level': logging_level
    })
    if paths:
        echo('Monitering folders:', "green")
        for path in paths:
            echo(f"\t{path}", "blue")

    if kwargs['uds'] is None:
        echo(f"Openning api server: http://{IP}:{port}/api/v1", 'green')
    else:
        echo(f"Openning api server: {kwargs['uds']}", 'green')
    echo(f"Tilesets Database: {store_uri}", "green")

    try:
        loop = asyncio.get_event_loop()
        if paths:
            executor = ProcessPoolExecutor(1)
            loop.run_in_executor(executor, partial(
                run_monitor, store_uri, paths))
        server = Server()
        server.run(store_uri, **kwargs)
    except Exception as e:
        echo(str(e), "red")
        echo("\nStoping services .......", "yellow")
        for task in asyncio.Task.all_tasks():
            task.cancel()


def run_monitor(store_uri, paths):
    from .store import default_monitor as monitor
    for path in paths:
        monitor(path)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(monitor.run(store_uri))


@hgserver.command()
@click.argument('paths', type=click.Path(exists=True, readable=True), nargs=-1)
@click.option('--store_uri', type=str, default="sqlite:////tmp/test.db", help='Database URI.')
def monitor(paths, store_uri):
    """
    Monitor folders to automatically supply tilesets.
    """
    echo('Monitering folders:', "green")
    for path in paths:
        default_monitor(path)
        echo(f"\t{path}", "blue")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(default_monitor.run(store_uri=store_uri))


if __name__ == "__main__":
    hgserver()
