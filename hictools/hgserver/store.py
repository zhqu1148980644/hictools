import os
from datetime import datetime
import subprocess
from pathlib import Path

import slugid
import clodius.tiles.bam as bam_tiles
import clodius.tiles.cooler as cooler_tiles
import sqlalchemy
from databases import Database
from sqlalchemy import Column, String, JSON, select, MetaData
from .monitor import FileMonitor, FileWatcher


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

    def __init__(self, store_uri: str = None):
        self.store_uri = store_uri
        self.db = None

    async def connect(self, store_uri=None):
        if store_uri is not None:
            self.store_uri = store_uri
        engine = None
        try:
            engine = sqlalchemy.create_engine(self.store_uri)
            if not self.table.exists(engine):
                self.table.metadata.create_all(engine)
            if not self.table.exists(engine):
                raise RuntimeError("Sqlalchemy engine creation error.")
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
        from hashlib import blake2b
        h = blake2b(digest_size=16)
        try:
            path = Path(path)
            hash_string = f"{path.resolve()}{path.stat().st_size}{path.stat().st_mtime}"
            h.update(bytes(hash_string, encoding="utf-8"))
        except Exception as e:
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
            return bam_tiles.tiles(
                tileset_dict.datafile, tids,
                index_filename=tileset_dict.index_filename
            )
        elif filetype == "cooler":
            return cooler_tiles.tiles(tileset_dict['datafile'], tids)
        elif filetype == "bigwig":
            import clodius.tiles.bigwig as bigwig_tiles
            return bigwig_tiles.tiles(tileset_dict['datafile'], tids)
        else:
            return {'error': f"Unknown tileset filetype: {filetype}"}


class TilesetsMonitor(FileMonitor):
    def __init__(self, tilsets_db, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tilesets = tilsets_db

    async def run(self, store_uri=None):
        await self.tilesets.connect(store_uri=store_uri)
        await super().run()

    async def check_event(self, event):
        print(event.type, event.path)
        path = str(event.path.resolve())

        # Do nothing when the same file added(or restarted the program)
        if event.type == self.added:
            for uuid, tileset in await self.tilesets.items(datafile=path):
                path_equal = tileset['datafile'] == path
                if path_equal and TileSet.hash(path) == tileset['_hash']:
                    return False

        # delete all recorded file with the same datafile prefix(except this modified file)
        # delete all recorded tilesets with the same datafile prefix
        elif event.type == self.modified:
            uuids = []
            for uuid, tileset in await self.tilesets.items(name=event.path.stem):
                if tileset['datafile'] != str(event.path):
                    os.remove(tileset.datafile)
                uuids.append(uuid)
            await self.tilesets.remove(uuids)

        # delete all recorded tilesets with the same datafile path.
        elif event.type == self.deleted:
            uuids = []
            for uuid, tileset in await self.tilesets.items(datafile=path):
                uuids.append(uuid)
            await self.tilesets.remove(uuids)
            return False

        return True


watch = default_monitor = TilesetsMonitor(
    tilsets_db=TileSetDB(),
    watcher_cls=FileWatcher
)


@watch(r'.*\.mcool$')
async def cooler(watcher, event):
    uuid = slugid.nice()
    await watcher.tilesets.update({
        uuid: TileSet(
            uuid=uuid,
            datafile=str(event.path),
            datatype="matrix",
            filetype="cooler",
        ).todict()
    })


@watch(r".*\.(bigwig|bw|bigWig|BigWig)$")
async def bigwig(watcher, event):
    uuid = slugid.nice()
    await watcher.tilesets.update({
        slugid.nice(): TileSet(
            uuid=uuid,
            datafile=str(event.path),
            datatype="vector",
            filetype="bigwig"
        ).todict()
    })


@watch(r".*\.(sam|bam)$")
def bam(event):
    bam_index_path = Path(str(event.path) + ".bai")
    if not bam_index_path.exists():
        subprocess.call(["samtools", "index", str(event.path)])
    return bam_index_path


@bam.done
async def bam_register(watcher, event, bam_index_path):
    if bam_index_path.exists():
        uuid = slugid.nice()
        await watcher.tilesets.update({
            uuid: TileSet(
                uuid=uuid,
                datafile=str(event.path),
                datatype="reads",
                filetype="bam",
                indexfile=str(bam_index_path)
            ).todict()
        })
