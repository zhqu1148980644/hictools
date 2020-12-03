import re
import inspect
import asyncio
from typing import Union, Sequence, Mapping
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

from watchgod import awatch
from watchgod.watcher import Change, DefaultWatcher


class FileWatcher(DefaultWatcher):
    def should_watch_dir(self, entry):
        return False


# May caues error when ThreadPoolExecutor are used.
class FileMonitor(object):
    EventType = Change
    added = Change.added
    modified = Change.modified
    deleted = Change.deleted

    class Event(object):
        __slots__ = ('type', 'path')

        def __init__(self, event_type: Change, path: Union[str, Path]):
            if isinstance(path, str):
                path = Path(path)
            self.type = event_type
            self.path = path

    def __init__(self, root: Union[Sequence, None, str] = None, handlers: Union[Mapping, None] = None,
                 watcher_cls=DefaultWatcher,
                 executor=ProcessPoolExecutor()):
        if root is None:
            root = []
        else:
            if isinstance(root, str):
                root = [root]
            root = [r for r in root if isinstance(r, str) and Path(r).exists()]
        self.root = root
        self.handlers = OrderedDict(handlers) if handlers else {}
        self.executor = executor
        self.queue: asyncio.Queue = asyncio.Queue()
        self.watcher_cls = watcher_cls
        self._listeners: list = []

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
