import os
import subprocess
from pathlib import Path

import slugid
from watchgod.watcher import DefaultWatcher

from .utils import TileSet, TileSetDB, FileMonitor


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


class FileWatcher(DefaultWatcher):
    def should_watch_dir(self, entry):
        return False


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


@watch(r".*\.(bigwig|bw)$")
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


@watch(r'.*\.bed$')
def bed():
    pass


@watch(r".*\.bedpe$")
def bedpe():
    pass


@watch(r".*\.(chrom\.sizes|chromsizes|chromsize)$")
async def chromsizes(watcher, event):
    pass
