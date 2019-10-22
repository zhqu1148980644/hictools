"""
worker_processes auto;
pid var/run/nginx.pid;
events {
    worker_connections 4096;
}

http {
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    gzip on;
    gzip_disable "msie6";
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_buffers 16 8k;
    gzip_http_version 1.1;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;
    log_format log '$remote_addr [$time_local] "$request" $status'
                           '"$upstream_addr" $upstream_response_time $upstream_http_etag';

    upstream api_server {
        {server}
    }

    server {
        listen  *:{port};
        charset utf8;
        server_name www.hgserver.com;
        access_log /tmp/hgserver_access.log log;
        error_log /tmp/hgserver_error.log;

        location / {
            proxy_pass http://api_server;
        }
    }
}
"""
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

from ..cli import cli
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


def run_server(store_uri, **kwargs):
    server = Server()
    server.run(store_uri, log_level=logging_level, **kwargs)


def addr_to_kwargs(addr, default="0.0.0.0"):
    # Too naive. ipv6, path with : ....
    if ':' not in addr:
        try:
            port = int(addr)
            addr = f"{default}:{addr}"
        except:
            if not Path(addr).parent.exists():
                raise ValueError(f'Path: {Path(addr).parent} not exists')
            addr = f"unix:{Path(addr).resolve()}"
    host, port = addr.split(':', 1)
    if host == "unix":
        return addr, {'uds': port}
    else:
        return addr, {'host': host, 'port': int(port)}


def control_nginx(default_config, write_to='/tmp/tmp_hgserver_nginx.conf'):
    # Maybe we should use some pypi package tp handle nginx config file.
    def start_nginx(port, socket_urls):
        if not shutil.which('nginx'):
            raise RuntimeError(
                'No nginx detected. Please install nginx with "conda install nginx".')

        stop_nginx()
        for i in range(len(socket_urls)):
            socket_urls[i] = socket_urls[i].replace('0.0.0.0', 'localhost')
        servers = "\n".join(f"server {url};" for url in socket_urls) + '\n'
        config = default_config.replace('{server}', servers)
        config = config.replace('{port}', str(port))
        with open(write_to, 'w') as f:
            f.write(config)
        subprocess.check_call(['nginx', '-c', write_to])

    def stop_nginx():
        try:
            subprocess.call(['nginx', '-c', write_to, '-s',
                             'stop'], stderr=subprocess.DEVNULL)
            os.remove(write_to)
        except:
            pass

    return start_nginx, stop_nginx


@cli.group()
@click.option('--log_level', default="error", type=click.Choice(
    ['critial', 'error', 'warning', 'info', 'debug'], case_sensitive=False))
def hgserver(log_level):
    """View results with higlass.
    Steps:\n
    1. serve
    2. view
    """
    global logging_level
    logging_level = log_level


@hgserver.command()
@click.argument('addr', type=str, default="0.0.0.0:6666", nargs=1)
def view(addr):
    """Start higlass web app.

    Example: view 6666
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
        return open(f'{Path(__file__).parent.resolve()}/ui/index.html').read()

    addr, kwargs = addr_to_kwargs(addr)
    if kwargs.get('uds') is not None:
        raise ValueError("Only support for TCP.")
    # click.launch(f"http://localhost:{kwargs['port']}/")
    echo(f"Go visit http://{IP}:{kwargs['port']} in browser.", "green")
    uvicorn.run(app, log_level=logging_level, **kwargs)


@hgserver.command()
@click.argument('paths', type=click.Path(exists=True, readable=True), nargs=-1)
@click.option('--port', type=int, default=4321,
              help="Api sever port served by nginx.")
@click.option('--store_uri', type=str, default="sqlite:////tmp/test.db",
              help='Database URI. Example: sqlite:///path_to_hold_my_database/tilesets.db')
@click.option('--num_worker', '-n', type=int, default=0,
              help="Number of randomly opened backend api workers.")
@click.option('--addr', '-a', type=str, default=["0.0.0.0:5555"], multiple=True,
              help="Api server backend address. Eaxmple: 0.0.0.0:5555  unix:/tmp/apiserver.sock")
def serve(paths, port, store_uri, addr, num_worker):
    """Start api server served by nginx with multiple backend api workers.

    Example:  hgserver ./ --addr 0.0.0.0:5555 --addr 5556 --num_worker 2.

    This command will open 4 workers with two you explicitly specified and two randomly
    opened.
    """
    # pre check
    tmp_addrs = list(addr)
    for i in range(num_worker):
        tmp_addrs.append(f'unix:/tmp/hgserver_api_{i}.sock')
    addrs = dict(addr_to_kwargs(addr) for addr in tmp_addrs)
    paths = list(paths)
    if not paths:
        paths = [os.getcwd()]
    for i in range(len(paths)):
        paths[i] = Path(paths[i]).resolve()

    start_nginx, stop_nginx = control_nginx(__doc__)
    try:
        start_nginx(port, list(addrs.keys()))
    except RuntimeError as e:
        print(r)
        return

    # Dsipactch web workers and monitor
    loop = asyncio.get_event_loop()
    for path in paths:
        default_monitor(path)
    monitor = asyncio.ensure_future(
        default_monitor.run(store_uri=store_uri),
        loop=loop
    )
    executor = ProcessPoolExecutor(len(addrs))
    server_tasks = []
    for addr, kwargs in addrs.items():
        server_tasks.append(loop.run_in_executor(
            executor,
            partial(run_server, store_uri, **kwargs))
        )

    # show message
    echo('Monitering folders:', "green")
    for path in paths:
        echo(f"\t{path}", "blue")
    echo(f"Openning api server: http://{IP}:{port}/api/v1", 'green')
    echo('Opening sockets:', "green")
    for addr in addrs:
        echo(f"\t{addr}", "blue")
    echo(f"Database: {store_uri}", "green")

    try:
        loop.run_forever()
    except:
        echo("\nStoping services .......", "yellow")
        for task in asyncio.Task.all_tasks():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                loop.run_until_complete(task)
    finally:
        stop_nginx()


@hgserver.group()
def control():
    pass


@control.command()
@click.option('--store_uri', type=str, default="sqlite:////tmp/test.db", help='Database URI.')
@click.option('--addr', '-a', type=str, default="0.0.0.0:5555", nargs=1,
              help="Eaxmple: 0.0.0.0:5555  unix:/tmp/apiserver.sock")
def start_apiserver(store_uri, addr):
    server = Server()
    addr, kwargs = addr_to_kwargs(addr)
    server.run(store_uri, **kwargs)


@control.command()
@click.argument('paths', type=click.Path(exists=True, readable=True), nargs=-1)
@click.option('--store_uri', type=str, default="sqlite:////tmp/test.db", help='Database URI.')
def start_monitor(paths, store_uri):
    for path in paths:
        default_monitor(path)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(default_monitor.run(store_uri=store_uri))


if __name__ == "__main__":
    hgserver()
