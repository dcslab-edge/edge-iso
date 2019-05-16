# coding: UTF-8

import logging
from typing import ClassVar, Set

from .edge import EdgePolicy
from .. import ResourceType
from ..isolators import IdleIsolator, CycleLimitIsolator, CPUFreqThrottleIsolator, \
    GPUFreqThrottleIsolator, SchedIsolator, AffinityIsolator
from ...workload import Workload
from ...utils.machine_type import NodeType


class EdgeWViolationPolicy(EdgePolicy):
    VIOLATION_THRESHOLD: ClassVar[int] = 3

    def __init__(self, fg_wl: Workload, bg_wls: Set[Workload]) -> None:
        super().__init__(fg_wl, bg_wls)

        self._violation_count: int = 0

    def _check_violation(self) -> bool:
        if isinstance(self._cur_isolator, AffinityIsolator):
            return False

        resource: ResourceType = self.contentious_resource()
        if self._node_type == NodeType.IntegratedGPU and self.foreground_workload.is_gpu_task == 0:
            return \
                resource is ResourceType.CACHE and not isinstance(self._cur_isolator, CycleLimitIsolator) \
                or resource is ResourceType.MEMORY and not (isinstance(self._cur_isolator, GPUFreqThrottleIsolator)
                                                            or isinstance(self._cur_isolator, SchedIsolator))
        elif self._node_type == NodeType.IntegratedGPU and self.foreground_workload.is_gpu_task == 1:
            return \
                resource is ResourceType.CACHE and not isinstance(self._cur_isolator, CycleLimitIsolator) \
                or resource is ResourceType.MEMORY and not (isinstance(self._cur_isolator, CPUFreqThrottleIsolator)
                                                            or isinstance(self._cur_isolator, SchedIsolator))
        
    @property
    def new_isolator_needed(self) -> bool:
        if isinstance(self._cur_isolator, IdleIsolator):
            return True

        if self._check_violation():
            logger = logging.getLogger(__name__)
            logger.info(f'violation is occurred. current isolator type : {self._cur_isolator.__class__.__name__}')

            self._violation_count += 1

            if self._violation_count >= EdgeWViolationPolicy.VIOLATION_THRESHOLD:
                logger.info('new isolator is required due to violation')
                self.set_idle_isolator()
                self._violation_count = 0
                return True

        return False

    def choose_next_isolator(self) -> bool:
        if super().choose_next_isolator():
            self._violation_count = 0
            return True

        return False
