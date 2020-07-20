import re
import inspect
import asyncio
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import partial
from hashlib import blake2b
from pathlib import Path

import clodius.tiles.bam as bam_tiles
import clodius.tiles.cooler as cooler_tiles
import sqlalchemy
from databases import Database
from sqlalchemy import Column, String, JSON, select, MetaData
from watchgod import awatch
from watchgod.watcher import Change, DefaultWatcher

# May caues error when ThreadPoolExecutor are used.


class FileMonitor(object):
    EventType = Change
    added = Change.added
    modified = Change.modified
    deleted = Change.deleted

    class Event(object):
        __slots__ = ('type', 'path')

        def __init__(self, type, path):
            self.type = type
            self.path = path

    def __init__(self, root=None, handlers={},
                 watcher_cls=DefaultWatcher,
                 executor=ProcessPoolExecutor()):
        self.root = [] if root is None else [root]
        self.handlers = OrderedDict(handlers)
        self.executor = executor
        self.queue = asyncio.Queue()
        self.watcher_cls = watcher_cls
        self._listeners = []

    async def run(self):
        await self.init()
        self._wrap_callbacks()
        self.loop = asyncio.get_running_loop()
        for root in self.root:
            self._listeners.append(asyncio.ensure_future(
                self.listen_events(root), loop=self.loop)
            )
        await self.handle_events()

    async def init(self):
        pass

    async def listen_events(self, root):
        watcher = awatch(root, watcher_cls=self.watcher_cls)
        init_files = self.watcher_cls(root).files.keys()
        for file in init_files:
            self.queue.put_nowait(
                self.Event(self.added, Path(file).resolve())
            )
        async for events in watcher:
            for event_type, path in events:
                self.queue.put_nowait(
                    self.Event(event_type, Path(path).resolve())
                )

    async def handle_events(self):
        while True:
            event = await self.queue.get()
            if not await self.check_event(event):
                continue
            for pattern, handler in self.handlers.items():
                if pattern.fullmatch(str(event.path)):
                    await self.execute(handler, event)

    async def check_event(self, event):
        return True

    async def execute(self, handler, event):
        if inspect.iscoroutinefunction(handler):
            res = await handler(self, event)
            await handler.callbacks(self, event, res)
        else:
            fut = self.loop.run_in_executor(
                self.executor,
                partial(handler, event)
            )
            fut.add_done_callback(partial(handler.callbacks, self, event))

    def __call__(self, pattern=".*", handler=None):
        def add_callback(callback):
            handler.callbacks.append(callback)

        # Add watching root path.
        if Path(pattern).is_dir() and Path(pattern).exists():
            self.root.append(Path(pattern).resolve())
            return
        # Add handlers. check callbacks, paramter validation
        if handler is None:
            return partial(self, pattern)

        handler.callbacks = []
        handler.done = add_callback
        self.handlers[re.compile(pattern)] = handler

        return handler

    def _wrap_callbacks(self):
        async def call(callbacks, watcher, event, res):
            for callback in callbacks:
                if inspect.iscoroutinefunction(callback):
                    res = await callback(watcher, event, res)
                else:
                    res = callback(watcher, event, res)
            return res

        def gen_coro(callbacks):
            async def coro(watcher, event, res):
                return await call(
                    callbacks, watcher, event, res
                )
            return coro

        def gen_noncoro(callbacks):
            def noncoro(watcher, event, fut):
                return asyncio.ensure_future(
                    call(callbacks, watcher, event, fut.result()),
                    loop=self.loop
                )
            return noncoro

        for handler in self.handlers.values():
            if inspect.iscoroutinefunction(handler):
                handler.callbacks = gen_coro(handler.callbacks)
            else:
                handler.callbacks = gen_noncoro(handler.callbacks)


class TileSetDB(object):
    # TODO Too much code, need to abstract.
    meta = MetaData()
    table = sqlalchemy.Table(
        'tileset', meta,
        Column("uuid", String(100), primary_key=True),
        Column("name", String(100)),
        Column("datafile", String(100)),
        Column('datatype', String(100)),
        Column('filetype', String(100)),
        Column('tileset', JSON)
    )
    uuid_tileset_query = select([table.c.uuid, table.c.tileset])
    columns = [column
               for column in table.columns.keys()
               if column != 'tileset']

    def __init__(self, store_uri="sqlite:////tmp/test.db"):
        self.store_uri = store_uri
        self.db = None

    async def connect(self, store_uri=None):
        if store_uri is not None:
            self.store_uri = store_uri
        try:
            engine = None
            engine = sqlalchemy.create_engine(self.store_uri)
            if not self.table.exists(engine):
                self.table.metadata.create_all(engine)
            if not self.table.exists(engine):
                raise RuntimeError("Table error.")
        except Exception as e:
            raise e
        finally:
            if engine is not None:
                engine.dispose()

        if self.db is None or not self.db.is_connected:
            self.db = Database(self.store_uri)
            await self.db.connect()

    async def disconnect(self):
        if self.db is not None and self.db.is_connected:
            await self.db.disconnect()

    async def query(self, query=None, **kwargs):
        if query is None:
            query = select([self.table.c.uuid, self.table.c.tileset])
        for key, value in kwargs.items():
            if key not in self.columns:
                raise KeyError(key)
            if not value:
                continue
            if isinstance(value, list):
                query = getattr(query, 'where')(
                    getattr(self.table.c, key).in_(value))
            else:
                query = getattr(query, 'where')(
                    getattr(self.table.c, key) == value)
        return {uuid: tileset
                async for uuid, tileset in self.db.iterate(query)}

    # TODO handling ordered by

    async def items(self, query=None, **kwargs):
        return (await self.query(query, **kwargs)).items()

    async def update(self, tilesets):
        wrapped_tilesets = []
        for uuid, tileset in tilesets.items():
            wrapped_tileset = {
                column: tileset[column] for column in self.columns
            }
            wrapped_tileset['tileset'] = tileset
            wrapped_tilesets.append(wrapped_tileset)

        return await self.db.execute_many(
            query=self.table.insert(),
            values=wrapped_tilesets
        )

    async def remove(self, uuids):
        if not isinstance(uuids, list):
            uuids = [uuids]
        return await self.db.execute(
            self.table
            .delete()
            .where(self.table.c.uuid.in_(uuids))
        )


class TileSet(object):
    # should be coupled with TileSetDB.?
    # Overhead ?
    _kwargs = {
        'uuid': None,
        'name': None, 'datafile': None, 'datatype': None, 'filetype': None,
        'created': None, 'description': "", 'private': False,
        'project': "", 'project_name': "", 'indexfile': "",
        'corrdSystem': "", 'coordSystem2': "", "_hash": None,
    }
    info_colums = ('name', 'datatype', 'coordSystem', 'coordSystem2')

    def __init__(self, **kwargs):
        tmp_kwargs = self._kwargs.copy()
        tmp_kwargs.update(kwargs)
        for key, value in tmp_kwargs.items():
            setattr(self, key, value)

        if self.created is None:
            self.created = str(datetime.utcnow())
        if self._hash is None:
            self._hash = self.hash(self.datafile)
        if self.name is None:
            self.name = Path(self.datafile).stem
        unfilled_keys = [key
                         for key in self._kwargs.keys()
                         if self.__dict__[key] is None]
        if unfilled_keys:
            raise ValueError(f"These fileds must be filled: {unfilled_keys}")

    @staticmethod
    def hash(path):
        h = blake2b(digest_size=16)
        try:
            path = Path(path)
            hash_string = f"{path.resolve()}{path.stat().st_size}{path.stat().st_mtime}"
            h.update(bytes(hash_string, encoding="utf-8"))
        except:
            return ""
        return h.hexdigest()

    def todict(self):
        return self.__dict__

    def update(self, kwargs):
        self.__dict__.update(kwargs)

    @classmethod
    def meta(cls, tileset_dict):
        return tileset_dict

    # Use deco way?
    @classmethod
    def tileset_info(cls, tileset_dict):
        filetype = tileset_dict.get("filetype", "")
        if filetype == "bam":
            info = bam_tiles.tileset_info(tileset_dict['datafile'])
            info['max_tile_width'] = int(1e5)
        elif filetype == "cooler":
            info = cooler_tiles.tileset_info(tileset_dict['datafile'])
        elif filetype == "bigwig":
            import clodius.tiles.bigwig as bigwig_tiles
            info = bigwig_tiles.tileset_info(tileset_dict['datafile'])
        else:
            info = {'error': f"Unknown tileset filetype: {filetype}"}

        for column in cls.info_colums:
            info[column] = tileset_dict.get(column)

        return info

    @classmethod
    def tiles(cls, tileset_dict, tids):
        filetype = tileset_dict.get('filetype', "")
        if filetype == "bam":
            data = bam_tiles.tiles(
                tileset_dict.datafile, tids,
                index_filename=tileset_dict.index_filename
            )
        elif filetype == "cooler":
            data = cooler_tiles.tiles(tileset_dict['datafile'], tids)
        elif filetype == "bigwig":
            import clodius.tiles.bigwig as bigwig_tiles
            data = bigwig_tiles.tiles(tileset_dict['datafile'], tids)
        else:
            data = {'error': f"Unknown tileset filetype: {filetype}"}

        return data
