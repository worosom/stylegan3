import logging
import time
import asyncio
import fractions
import threading
from typing import Optional, Set

from av import VideoFrame

from aiortc.mediastreams import MediaStreamTrack
from aiortc.contrib.media import PlayerStreamTrack
import numpy as np

from .renderer import AsyncRenderer

logger = logging.getLogger(__name__)


class GeneratorTrack(AsyncRenderer):
    kind = 'video'

    def __init__(self):
        super().__init__()
        self.noise_mode = 'const'
        self._queue = asyncio.Queue()
        self._start = None
        self.id = 0
        self.pts = 0
        self.start_time = None
        self.time_base = fractions.Fraction(1, 30)
        self.placeholder = VideoFrame.from_ndarray(np.zeros((1024, 1024, 3), dtype=np.uint8), format='rgb24')
        self.placeholder.pts = 0
        self.placeholder.time_base = self.time_base

    async def recv(self):
        gen_kwargs = await self._queue.get()
        if not self._queue.full():
            self.set_args(**gen_kwargs)
            if self.start_time is None:
                self.start_time = round(time.time() * 1000)
            result = self.get_result()
            if 'error' in result:
                print(result.error)
                exit()
            img = result.image
            self.frame = VideoFrame.from_ndarray(img, format='rgb24')
        self.frame.pts = round(time.time() * 1000) - self.start_time
        self.frame.time_base = self.time_base
        return self.frame
        

class Generator:
    def __init__(self):
        self.__thread: Optional[threading.Thread] = None
        self.__thread_quit: Optional[threading.Event] = None

        # examine streams
        self.__streams = []
        self.__video = GeneratorTrack()
        self.__streams.append(self.__video)

        self._throttle_playback = False

    def data(self, data):
        asyncio.run_coroutine_threadsafe(self.__video._queue.put(data), asyncio.get_event_loop())

    @property
    def video(self) -> MediaStreamTrack:
        """
        A :class:`aiortc.MediaStreamTrack` instance if the file contains video.
        """
        return self.__video

    def __log_debug(self, msg: str, *args) -> None:
        logger.debug(f"Generator(%s) {msg}", self.__container.name, *args)
