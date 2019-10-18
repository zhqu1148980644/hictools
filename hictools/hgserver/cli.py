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

import asyncio
from functools import partial
from concurrent.futures import ProcessPoolExecutor

import click

from .server import Server
from .store import default_monitor

click.option = partial(click.option, show_default=True)


@click.group()
def hgserver():
    pass


@hgserver.command()
def view():
    pass


@hgserver.command()
@click.argument('paths', type=click.Path(exists=True, readable=True), nargs=-1)
@click.option('--store_uri', type=str, nargs=1, default="sqlite:///test.db", help='Database URI.')
@click.option('--host', '-h', type=str, nargs=1, default="0.0.0.0")
@click.option('--ports', '-p', type=int, multiple=True, default=[5555])
def start_server(paths, store_uri, host, ports):
    server = Server()
    loop = asyncio.get_event_loop()
    executor = ProcessPoolExecutor(len(ports))
    server_tasks = []
    for port in ports:
        server_tasks.append(loop.run_in_executor(
            executor,
            partial(server.run, host, port, store_uri))
        )
    for path in paths:
        default_monitor(path)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(default_monitor.run(store_uri=store_uri))


@hgserver.command()
@click.option('--store_uri', type=str, nargs=1, default="sqlite:///test.db", help='Database URI.')
@click.option('--host', '-h', type=str, nargs=1, default="0.0.0.0")
@click.option('--port', '-p', type=int, nargs=1, default=5555)
def start_apiserver(host, port, store_uri):
    server = Server()
    server.run(host, port, store_uri)


@hgserver.command()
@click.argument('paths', type=click.Path(exists=True, readable=True), nargs=-1)
@click.option('--store_uri', type=str, default="sqlite:///test.db", help='Database URI.')
def start_monitor(paths, store_uri):
    for path in paths:
        default_monitor(path)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(default_monitor.run(store_uri=store_uri))


@hgserver.command()
def list_monitors():
    pass


@hgserver.command()
def list_apiservers():
    pass


if __name__ == "__main__":
    hgserver()
