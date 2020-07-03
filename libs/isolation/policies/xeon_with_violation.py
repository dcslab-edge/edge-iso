# coding: UTF-8

import logging
from typing import ClassVar, Set

from .xeon import XeonPolicy
from .. import ResourceType
from ..isolators import IdleIsolator, CycleLimitIsolator, CPUFreqThrottleIsolator, CacheIsolator,\
    GPUFreqThrottleIsolator, SchedIsolator, AffinityIsolator
from ...workload import Workload
from ...utils.machine_type import NodeType


class XeonWViolationPolicy(XeonPolicy):
    VIOLATION_THRESHOLD: ClassVar[int] = 1

    def __init__(self, lc_wls: Set[Workload], be_wls: Set[Workload]) -> None:
        super().__init__(lc_wls, be_wls)
        # TODO: Get alloc_target_wl
        # self._perf_target_wl = self.cur_isolator.perf_target_wl

        self._violation_count: int = 0

    def _check_violation(self) -> bool:
        logger = logging.getLogger(__name__)
        if isinstance(self._cur_isolator, AffinityIsolator):
            logger.info(f'[_check_violation] ret: False')
            return False

        dom_res_cont = self._dom_res_cont
        logger.info(f'[_check_violation] dominant contentious resource: {dom_res_cont}, '
                    f'cur_isolator: {self._cur_isolator}')
        """
        ret = dom_res_cont is ResourceType.CPU and not isinstance(self._cur_isolator, AffinityIsolator) \
              or dom_res_cont is ResourceType.CACHE and not isinstance(self._cur_isolator, CycleLimitIsolator) \
              or dom_res_cont is ResourceType.MEMORY and not (isinstance(self._cur_isolator, SchedIsolator)
                                                              or isinstance(self._cur_isolator, CycleLimitIsolator))
        logger.info(f'[_check_violation] ret: {ret}')
        """
        ret = dom_res_cont is ResourceType.CPU and not isinstance(self._cur_isolator, AffinityIsolator) \
              or dom_res_cont is ResourceType.CACHE and not isinstance(self._cur_isolator, CacheIsolator) \
              or dom_res_cont is ResourceType.MEMORY and not (isinstance(self._cur_isolator, CPUFreqThrottleIsolator)
                                                              or isinstance(self._cur_isolator, SchedIsolator))
        logger.info(f'[_check_violation] ret: {ret}')
        return ret

        # resources = self.contentious_resources(self._perf_target_wl)
        # for resource, diff_value in resources:
        #     logger.info(f'[_check_violation] contentious resource: {resource}, cur_isolator: {self._cur_isolator}')
        #     # FIXME: last condition statement can be misleading
        #     ret = \
        #         resource is ResourceType.CPU and not isinstance(self._cur_isolator, AffinityIsolator) \
        #         or resource is ResourceType.CACHE and not isinstance(self._cur_isolator, CycleLimitIsolator) \
        #         or resource is ResourceType.MEMORY and not (isinstance(self._cur_isolator, SchedIsolator)
        #                                                     or isinstance(self._cur_isolator, CycleLimitIsolator))
        #     logger.info(f'[_check_violation] ret: {ret}')
        #     return ret
        #logger.info(f'[_check_violation] ret: {ret}')
        #return ret

        # return \
        #         resource is ResourceType.CACHE and not isinstance(self._cur_isolator, CycleLimitIsolator) \
        #         or resource is ResourceType.MEMORY and not (isinstance(self._cur_isolator, SchedIsolator)
        #                                                     or isinstance(self._cur_isolator, CycleLimitIsolator))

        """
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
        """

    @property
    def new_isolator_needed(self) -> bool:
        logger = logging.getLogger(__name__)
        cur_isolator = self.cur_isolator
        if isinstance(self._cur_isolator, IdleIsolator):
            return True

        if self._dom_res_cont is ResourceType.CPU:
            logger.info('new isolator is required due to violation')
            self.set_idle_isolator()
            self._violation_count = 0
            cur_isolator.violated_counts = 0
            return True

        if self._check_violation():
            logger.info(f'violation occurs. current isolator type : {self._cur_isolator.__class__.__name__}')

            self._violation_count += 1
            cur_isolator.violated_counts += 1

            if self._violation_count >= XeonWViolationPolicy.VIOLATION_THRESHOLD:
                logger.info('new isolator is required due to violation')
                self.set_idle_isolator()
                self._violation_count = 0
                cur_isolator.violated_counts = 0
                return True

        return False

    def choose_next_isolator(self) -> bool:
        if super().choose_next_isolator():
            self._violation_count = 0
            self.cur_isolator.violated_counts = 0
            return True

        return False