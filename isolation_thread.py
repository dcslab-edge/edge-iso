
import logging
import time

from threading import Thread
from libs.isolation.policies import IsolationPolicy
from libs.isolation.isolators import Isolator, CacheIsolator, SchedIsolator, CPUFreqThrottleIsolator
from typing import Type
from heracles_func import State


class IsolationThread(Thread):

    def __init__(self, isolation_type: Type[Isolator]) -> None:
        super().__init__(daemon=True)
        self._isolator_type: Type[Isolator] = isolation_type
        #self._alloc_target_wl = None
        #self._dealloc_target_wl = None
        self._group = None
        self._heracles = None

    @property
    def group(self):
        return self._group

    @group.setter
    def group(self, new_group):
        self._group = new_group

    @property
    def heracles(self):
        return self._heracles

    @heracles.setter
    def heracles(self, new_heracles):
        self._heracles = new_heracles

    def run(self) -> None:
        logger = logging.getLogger(__name__)
        while True:
            state = self.heracles.state
            #self._group: IsolationPolicy = self.heracles.group
            logger.critical(f'[isolation_thread:run] state: {state}')
            logger.critical(f'[isolation_thread:run] self._group: {self._group}')
            logger.critical(f'[isolation_thread:run] self._isolator_type: {self._isolator_type}')
            if isinstance(self._isolator_type, SchedIsolator):
                # Core allocation
                cur_isolator = self._isolator_type
                for be_wl in self._group.best_effort_workloads:
                    if state is State.GROW_CORES:
                        self._group.alloc_target_wl = be_wl
                        cur_isolator.alloc_target_wl = be_wl
                        self._group.dealloc_target_wl = None
                        cur_isolator.dealloc_target_wl = None
                        cur_isolator.weaken()
                    elif state is State.STOP_GROWTH:
                        self._group.alloc_target_wl = None
                        cur_isolator.alloc_target_wl = None
                        self._group.dealloc_target_wl = be_wl
                        cur_isolator.dealloc_target_wl = be_wl
                        cur_isolator.strengthen()
                    cur_isolator.enforce()

            elif isinstance(self._isolator_type, CacheIsolator):
                # LLC allocation
                cur_isolator = self._isolator_type
                for be_wl in self._group.best_effort_workloads:
                    if state is State.GROW_LLC:
                        self._group.alloc_target_wl = be_wl
                        cur_isolator.alloc_target_wl = be_wl
                        self._group.dealloc_target_wl = None
                        cur_isolator.dealloc_target_wl = None
                        cur_isolator.weaken()
                    elif state is State.STOP_GROWTH:
                        self._group.alloc_target_wl = None
                        cur_isolator.alloc_target_wl = None
                        self._group.dealloc_target_wl = be_wl
                        cur_isolator.dealloc_target_wl = be_wl
                        cur_isolator.strengthen()
                    cur_isolator.enforce()

            elif isinstance(self._isolator_type, CPUFreqThrottleIsolator):
                # per-core DVFS
                cur_isolator = self._isolator_type
                for be_wl in self._group.best_effort_workloads:
                    if state is not State.STOP_GROWTH:
                        self._group.alloc_target_wl = be_wl
                        cur_isolator.alloc_target_wl = be_wl
                        self._group.dealloc_target_wl = None
                        cur_isolator.dealloc_target_wl = None
                        cur_isolator.weaken()
                    elif state is State.STOP_GROWTH:
                        self._group.alloc_target_wl = None
                        cur_isolator.alloc_target_wl = None
                        self._group.dealloc_target_wl = be_wl
                        cur_isolator.dealloc_target_wl = be_wl
                        cur_isolator.strengthen()
                    cur_isolator.enforce()

            else:
                # unknown isolator
                print(f'isolator is not chosen. isolator:{self._isolator_type}')
            time.sleep(2)

