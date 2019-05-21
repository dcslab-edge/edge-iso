# coding: UTF-8

import logging

from typing import Set

from .base import IsolationPolicy
from .. import ResourceType
from ..isolators import IdleIsolator, CycleLimitIsolator, CPUFreqThrottleIsolator, \
    GPUFreqThrottleIsolator, SchedIsolator
from ...workload import Workload
from ...utils.machine_type import NodeType


class EdgePolicy(IsolationPolicy):
    def __init__(self, fg_wl: Workload, bg_wls: Set[Workload]) -> None:
        super().__init__(fg_wl, bg_wls)
        self._is_mem_isolated = False
        self._is_freq_throttled_max = False

    @property
    def new_isolator_needed(self) -> bool:
        return isinstance(self._cur_isolator, IdleIsolator)

    def choose_next_isolator(self) -> bool:
        logger = logging.getLogger(__name__)
        logger.debug('looking for new isolation...')

        """
        *  contentious_resource() returns the most contentious resources of the foreground 
        *  It returns either ResourceType.MEMORY or ResourceType.Cache
        *  JetsonTX2 : ResourceType.Cache -> CycleLimitIsolator, 
        *              ResourceType.Memory -> GPUFreqThrottleIsolator, SchedIsolator
        *
        *  Desktop   : ResourceType.Cache -> CycleLimitIsolator, ResourceType.Memory -> SchedIsolator
        * 
        """

        """
        * When choosing isolator, policy considers whether the foreground task uses GPU. 
        * If yes, it will use CPUFreqThrottle Isolator
        * Otherwise, it will use GPUFreqThrottle Isolator
        """

        for resource, diff_value in self.contentious_resources():
            if resource is ResourceType.CACHE:
                    isolator = self._isolator_map[CycleLimitIsolator]
            elif resource is ResourceType.MEMORY:
                #self.foreground_workload.check_gpu_task()
                if self._node_type == NodeType.IntegratedGPU:
                    isolator = self._isolator_map[SchedIsolator]
                    """
                    if self.foreground_workload.is_gpu_task == 1:   # if fg is gpu task, ...
                    # FIXME: choosing an isolator by considering whether the FG task using GPU or not.
                        isolator = self._isolator_map[CPUFreqThrottleIsolator]
                        if isolator.is_max_level is True:
                            isolator = self._isolator_map[SchedIsolator]
                    elif self.foreground_workload.is_gpu_task == 0: # if fg is cpu task, ...
                        isolator = self._isolator_map[GPUFreqThrottleIsolator]
                        if isolator.is_max_level is True:
                            isolator = self._isolator_map[SchedIsolator]
                    """
                elif self._node_type == NodeType.CPU:
                    isolator = self._isolator_map[SchedIsolator]
            else:
                raise NotImplementedError(f'Unknown ResourceType: {resource}')

            if diff_value < 0 and not isolator.is_max_level or \
                    diff_value > 0 and not isolator.is_min_level:
                self._cur_isolator = isolator
                logger.info(f'Starting {self._cur_isolator.__class__.__name__}...')
                return True

        logger.debug('A new Isolator has not been selected')
        return False
