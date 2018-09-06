# coding: UTF-8

import logging

from .base_policy import ResourceType
from .greedy_diff_policy import GreedyDiffPolicy
from ..isolators import CacheIsolator, IdleIsolator, MemoryIsolator, SchedIsolator
from ...workload import Workload


class GreedyDiffWViolationPolicy(GreedyDiffPolicy):
    VIOLATION_THRESHOLD = 3

    def __init__(self, fg_wl: Workload, bg_wl: Workload) -> None:
        super().__init__(fg_wl, bg_wl)

        self._violation_count: int = 0

    def _check_violation(self) -> bool:
        resource: ResourceType = self.contentious_resource()

        return \
            resource is ResourceType.CACHE and not isinstance(self._cur_isolator, CacheIsolator) \
            or resource is ResourceType.MEMORY and (not isinstance(self._cur_isolator, MemoryIsolator)
                                                    and not isinstance(self._cur_isolator, SchedIsolator))

    @property
    def new_isolator_needed(self) -> bool:
        if isinstance(self._cur_isolator, IdleIsolator):
            return True

        if self._check_violation():
            logger = logging.getLogger(__name__)
            logger.info(f'violation is occurred. current isolator type : {self._cur_isolator.__class__.__name__}')

            self._violation_count += 1

            if self._violation_count >= GreedyDiffWViolationPolicy.VIOLATION_THRESHOLD:
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