"""
Description:
class: TileSet     Determin which function needs to be called when fetching tiles, tileset_infos.
class: TileSetDB   Used by both store and server. Handling the communication with database.
TileSetMonitor:  Monitor file changes in watcher file. Auto conversion. Auto registration.
tilesets_store:  only contain info of each tileset
apiserver:
    As fetching data may be computing expensive, multiple apiserver can be deployed to facilitate data rendering.
    get:
        Fetch tileset info from tilesets_store
        Remove tileset if datafile doesn't exists
        Call decent fucntions to get tiles, tileset_infos
    post: Store new tileset in database.

            server:                   store:
            apiserver1
nginx <-->  apiserver2    <------>    (tilesets_store  <------->  TileSetMonitor)
  |         apiserver3
  |         ...
  |
client: browser
"""
import socket
import asyncio
import contextlib
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


def echo(text, fg="white", bg="black"):
    return click.echo(click.style(text, fg, bg))


def run_server(host, port, store_uri):
    server = Server()
    server.run(host, port, store_uri, log_level=logging_level)


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


@click.group()
@click.option('--log_level', default="error", type=click.Choice(
    ['critial', 'error', 'warning', 'info', 'debug'], case_sensitive=False))
def hgserver(log_level):
    global logging_level
    logging_level = log_level


@hgserver.command()
@click.option('--host', '-h', type=str, default="0.0.0.0")
@click.option('--port', '-p', type=int, nargs=1, default=6666,
              help="Port to serve higlass web app")
def view(host, port):
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
        return open(f'{Path(__file__).parent.resolve()}/ui/index.html').read()

    click.launch(f"http://localhost:{port}/")
    uvicorn.run(app, host=host, port=port, log_level=logging_level)


@hgserver.command()
@click.argument('paths', type=click.Path(exists=True, readable=True), nargs=-1)
@click.option('--store_uri', type=str, default="sqlite:///test.db",
              help='Database URI. Example: sqlite:///path_to_hold_my_database/tilesets.db')
@click.option('--host', '-h', type=str, default="0.0.0.0")
@click.option('--ports', '-p', type=int, multiple=True, default=[5555])
def serve(paths, store_uri, host, ports):
    loop = asyncio.get_event_loop()
    for path in paths:
        default_monitor(path)
    monitor = asyncio.ensure_future(
        default_monitor.run(store_uri=store_uri),
        loop=loop
    )
    executor = ProcessPoolExecutor(len(ports))
    server_tasks = []
    for port in ports:
        server_tasks.append(loop.run_in_executor(
            executor,
            partial(run_server, host, port, store_uri))
        )

    # show message
    ip = get_ip()
    echo('Opening api servers:', "green")
    for port in ports:
        echo(f"\thttp://{ip}:{port}/api/v1", "blue")
    echo('Monitering folders::', "green")
    for path in paths:
        echo(f"\t{path}", "blue")

    try:
        loop.run_forever()
    except:
        echo("\nStoping services .......", "yellow")
        for task in asyncio.Task.all_tasks():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                loop.run_until_complete(task)

    echo(f"Stop serving apiserves: {ports}", "yellow")
    echo(f"Stop monitoring folders: {paths}", "yellow")


@hgserver.group()
def control():
    pass


@control.command()
@click.option('--store_uri', type=str, default="sqlite:///test.db", help='Database URI.')
@click.option('--host', '-h', type=str, default="0.0.0.0")
@click.option('--port', '-p', type=int, default=5555)
def start_apiserver(host, port, store_uri):
    server = Server()
    server.run(host, port, store_uri)


@control.command()
@click.argument('paths', type=click.Path(exists=True, readable=True), nargs=-1)
@click.option('--store_uri', type=str, default="sqlite:///test.db", help='Database URI.')
def start_monitor(paths, store_uri):
    for path in paths:
        default_monitor(path)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(default_monitor.run(store_uri=store_uri))


if __name__ == "__main__":
    hgserver()
