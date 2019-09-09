# coding: UTF-8

import logging
from typing import Optional, Set, Dict

from .. import ResourceType
from .base import Isolator
from ...metric_container.basic_metric import MetricDiff
from ...utils.cgroup import Cpu
from ...workload import Workload


class CycleLimitIsolator(Isolator):

    def __init__(self, latency_critical_wls: Set[Workload], best_effort_wls: Set[Workload]) -> None:
        super().__init__(latency_critical_wls, best_effort_wls)

        # FIXME: working for multiple workloads
        self._cur_steps: Dict[Workload, int] = dict()
        for wl in latency_critical_wls:
            self._cur_steps[wl] = Cpu.MAX_PERCENT
        for wl in best_effort_wls:
            self._cur_steps[wl] = Cpu.MAX_PERCENT

        self._stored_config: Optional[Dict[Workload, int]] = None

    def _get_metric_type_from(self, metric_diff: MetricDiff) -> float:
        logger = logging.getLogger(__name__)
        res_cont_type = super().cur_dominant_resource_cont
        logger.info(f'[_get_metric_type_from] res_cont_type: {res_cont_type}')
        if res_cont_type is ResourceType.CACHE:
            return metric_diff.llc_hit_ratio - metric_diff.diff_slack
        elif res_cont_type is ResourceType.MEMORY:
            return metric_diff.local_mem_util_ps - metric_diff.diff_slack

    def strengthen(self) -> 'CycleLimitIsolator':
        if self.dealloc_target_wl is not None:
            self._cur_steps[self.dealloc_target_wl] -= Cpu.STEP
        return self

    def weaken(self) -> 'CycleLimitIsolator':
        if self.alloc_target_wl is not None:
            self._cur_steps[self.alloc_target_wl] += Cpu.STEP
        return self

    @property
    def is_max_level(self) -> bool:
        # FIXME: hard coded
        logger = logging.getLogger(__name__)
        logger.info(f'[is_max_level] self.dealloc_target_wl: {self.dealloc_target_wl}')
        if self.dealloc_target_wl is None:
            return False
        else:
            return self._cur_steps[self.dealloc_target_wl] - Cpu.STEP < Cpu.MIN_PERCENT

    @property
    def is_min_level(self) -> bool:
        # FIXME: hard coded
        logger = logging.getLogger(__name__)
        logger.info(f'[is_min_level] self.alloc_target_wl: {self.alloc_target_wl}')
        if self.alloc_target_wl is None:
            return False
        else:
            return Cpu.MAX_PERCENT < self._cur_steps[self.alloc_target_wl] + Cpu.STEP

    def enforce(self) -> None:
        logger = logging.getLogger(__name__)
        wls = [self.alloc_target_wl, self.dealloc_target_wl]
        #logger.info(f'{self.alloc_target_wl.name}-{self.alloc_target_wl.pid}')
        #logger.info(f'{self.dealloc_target_wl.name}-{self.dealloc_target_wl.pid}')
        for wl in wls:
            if wl is not None:
                logger.info(f'limit_percentages of bound_cores of {wl.name}\'s {wl.bound_cores} is '
                            f'{self._cur_steps[wl]}%')
        for wl in wls:
            if wl is not None:
                Cpu.limit_cycle_percentage(wl.group_name, self._cur_steps[wl])
        self.alloc_target_wl = None
        self.dealloc_target_wl = None

    def reset(self) -> None:
        for wl in self._all_wls:
            if wl.is_running:
                Cpu.limit_cycle_percentage(wl.group_name, Cpu.MAX_PERCENT)


    def store_cur_config(self) -> None:
        self._stored_config = self._cur_steps

    def load_cur_config(self) -> None:
        super().load_cur_config()
        self._cur_steps = self._stored_config
        self._stored_config = None
