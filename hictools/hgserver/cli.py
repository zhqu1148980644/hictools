import os
import socket
import uvloop
from typing import Union
from pathlib import Path
from functools import partial
import concurrent
from concurrent.futures import ProcessPoolExecutor

import click

uvloop.install()

click.option = partial(click.option, show_default=True)
logging_level = "error"



def echo(text, fg="white", bg="black"):
    return click.echo(click.style(text, fg, bg))


def get_open_port():
    # reference: https://stackoverflow.com/a/2838309/10336496
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port


def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('1.1.1.1', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()

    return ip


def fetch_valid_uri(store_uri: str) -> Union[str, None]:
    from sqlalchemy.engine.url import make_url
    uri = None
    try:
        uri = str(make_url(store_uri))
    except Exception:
        pass
    try:
        realpath = str(Path(store_uri).expanduser().resolve())
        if Path(realpath).parent.exists():
            uri = str(make_url("sqlite:///" + realpath))
    except Exception:
        pass

    return uri


IP = get_ip()


@click.group()
@click.option('--log_level', default="error", type=click.Choice(
    ['critial', 'error', 'warning', 'info', 'debug'], case_sensitive=False))
def hgserver(log_level):
    """View results with higlass.

    Steps:\n
    >> hictools hgserver serve --port 7777 --paths ./\n
    >> hictools hgserver view --api_port 7777
    """
    global logging_level
    logging_level = log_level


@hgserver.command()
@click.option('--api_port', type=int, default=0, help="Apiserver port opened by 'hictools hgserver serve'")
@click.option('--host', type=str, default="0.0.0.0")
@click.option('--port', type=int, default=0, help="'--port 0' represents automatically select for an opening port.")
def view(api_port, host, port):
    """Start higlass web app.
    Make sure to run 'hictools hgserver serve' subcommand in order to fetch an API_PORT before this step.

    Example: hictools hgserver view --api_port API_PORT
    """
    import uvicorn
    from fastapi import FastAPI
    from starlette.middleware.cors import CORSMiddleware
    from starlette.responses import HTMLResponse

    # Could use
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if port == 0:
        port = get_open_port()

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
@click.option('--store_uri',
              type=str,
              default="~/.hictools_hgserver.db",
              help='Database URI.')
@click.option('--host',
              type=str,
              default="0.0.0.0")
@click.option('--port',
              type=int,
              default=0,
              help="'--port 0' represents automatically select for an opening port.")
@click.option('--paths',
              type=click.Path(exists=True, readable=True),
              multiple=True,
              help="Path to monitor file changes.")
@click.option('--workers',
              type=click.IntRange(min=1),
              default=10,
              help="Number of process workers for supplying tilesets.")
@click.pass_context
def serve(ctx, store_uri, host, port, paths, workers):
    """Monitor folders and serve an api server used for higlass.\n
        Files added into the monitoring folders would be automatically registered in the api server.
    """
    import uvicorn
    from .server import Server

    # get sqlalchemy engine uri
    store_uri = fetch_valid_uri(store_uri)
    if store_uri is None:
        echo(f"Invalid database uri {store_uri}", "red")
        return

    # get opening port
    if port == 0:
        port = get_open_port()

    uvicorn.main.parse_args(ctx, args=['TMP'] + ctx.args)
    kwargs = ctx.params.copy()
    kwargs.update({
        'store_uri': store_uri,
        'host': host,
        'port': port,
        'log_level': logging_level,
    })
    # uvicorn only support for workers and reload for file app

    # get valid monitoring paths
    if not paths:
        paths = [os.getcwd()]
    paths = [p for p in paths if Path(p).is_dir() and Path(p).exists()]

    futures = []
    try:
        with ProcessPoolExecutor(workers + 1) as executor:
            # run monitor in process
            fut = executor.submit(partial(run_monitor, store_uri, paths))
            futures.append(fut)

            # serving
            for _ in range(1, workers + 1):
                fut = executor.submit(partial(run_server, kwargs))
                futures.append(fut)

            # echo
            echo('Monitering folders:', "green")
            for path in paths:
                echo(f"\t{Path(path).resolve()}", "blue")

            if kwargs['uds'] is None:
                echo(f"Openning api server: http://{IP}:{port}/api/v1", 'green')
            else:
                echo(f"Openning api server: {kwargs['uds']}", 'green')
            echo(f"Tilesets Database: {store_uri}", "green")
            if kwargs['uds'] is None:
                echo(f"Run 'hictools hgserver view --api_port {port} \
                       to visualize in your web browser.", "green")
            # wait to be done
            concurrent.futures.wait(futures)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)
    finally:
        echo("\nStoping services .......", "yellow")
        for fut in futures:
            fut.cancel()



def run_server(kwargs):
    from .server import Server

    server = Server()
    server.run(**kwargs)


def run_monitor(store_uri, paths):
    import asyncio
    from .store import default_monitor as monitor

    for path in paths:
        monitor(path)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(monitor.run(store_uri))


if __name__ == "__main__":
    hgserver()
