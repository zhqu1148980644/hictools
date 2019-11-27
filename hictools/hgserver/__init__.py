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
        Call decent functions to get tiles, tileset_infos
    post: Store new tileset in database.

            server:                   store:
            apiserver2    <------>    (tilesets_store  <------->  TileSetMonitor)
client: browser
"""
