# coding: UTF-8

import logging
from typing import Optional, Set, Dict

from .. import ResourceType
from .base import Isolator
from ...metric_container.basic_metric import MetricDiff
from ...utils import CPUDVFS
from ...workload import Workload


class CPUFreqThrottleIsolator(Isolator):
    def __init__(self, latency_critical_wls: Set[Workload], best_effort_wls: Set[Workload]) -> None:
        super().__init__(latency_critical_wls, best_effort_wls)

        # FIXME: hard coded
        # Assumption: FG is latency-sensitive process (GPU) and BG is compute-intensive process (CPU)
        #self._cur_step: int = CPUDVFS.MAX_IDX
        #self._stored_config: Optional[int] = None
        #self._cpufreq_range = CPUDVFS.get_freq_range()
        self._cur_steps: Dict[Workload, int] = dict()
        for wl in latency_critical_wls:
            self._cur_steps[wl] = CPUDVFS.MAX_IDX
        for wl in best_effort_wls:
            self._cur_steps[wl] = CPUDVFS.MIN_IDX       # FIXME: Heracles's be workloads' freq starts with the lowest freq.
        self._stored_config: Optional[Dict[Workload, int]] = None
        self._cpufreq_range = CPUDVFS.get_freq_range()

    def _get_metric_type_from(self, metric_diff: MetricDiff) -> float:
        #return metric_diff.local_mem_util_ps
        return metric_diff.local_mem_util_ps - metric_diff.diff_slack

    def _get_res_type_from(self) -> ResourceType:
        return ResourceType.MEMORY

    def strengthen(self) -> 'CPUFreqThrottleIsolator':
        if self.dealloc_target_wl is not None:
            self._cur_steps[self.dealloc_target_wl] -= CPUDVFS.STEP_IDX
        return self

    def weaken(self) -> 'CPUFreqThrottleIsolator':
        if self.alloc_target_wl is not None:
            self._cur_steps[self.alloc_target_wl] += CPUDVFS.STEP_IDX
        return self

    @property
    def is_max_level(self) -> bool:
        # FIXME: hard coded
        logger = logging.getLogger(__name__)
        logger.critical(f'[is_max_level] self.dealloc_target_wl: {self.dealloc_target_wl}')
        if self.dealloc_target_wl is None:
            return False
        else:
            return self._cur_steps[self.dealloc_target_wl] - CPUDVFS.STEP_IDX < CPUDVFS.MIN_IDX

    @property
    def is_min_level(self) -> bool:
        # FIXME: hard coded
        logger = logging.getLogger(__name__)
        logger.critical(f'[is_min_level] self.alloc_target_wl: {self.alloc_target_wl}')
        if self.alloc_target_wl is None:
            return False
        else:
            cur_dvfs_idx = self._cur_steps[self.alloc_target_wl]
            logger.critical(f'[is_min_level] self._cur_steps[self.alloc_target_wl]: {cur_dvfs_idx}')
            return CPUDVFS.MAX_IDX < cur_dvfs_idx + CPUDVFS.STEP_IDX

    def enforce(self) -> None:
        # logger = logging.getLogger(__name__)
        # freq = self._cpufreq_range[self._cur_step]
        # # FIXME: It assumes all bgs are running on a single CPU socket, so we throttle freq.s for the one bg
        # for bg_wl in self._best_effort_wls:
        #     logger.info(f'frequency of CPU cores of {bg_wl.name}\'s {bg_wl.bound_cores} is {freq / 1_000_000}GHz')
        #     bg_wl.cpu_dvfs.set_freq_cgroup(freq)
        logger = logging.getLogger(__name__)
        wls = [self.alloc_target_wl, self.dealloc_target_wl]
        #logger.info(f'len: {len(self._gpufreq_range)}')
        for wl in wls:
            if wl is not None:
                #logger.info(f'[enforce] self._cur_steps[wl]: {self._cur_steps[wl]}')
                freq = self._cpufreq_range[self._cur_steps[wl]]
                logger.critical(f'[enforce:DVFS][HW] CPU core frequencies of {wl.name}\'s is {freq/1_000_000}GHz')

        for wl in wls:
            if wl is not None:
                freq = self._cpufreq_range[self._cur_steps[wl]]
                CPUDVFS.set_freq(freq, wl.bound_cores)

    def reset(self) -> None:
        max_freq = self._cpufreq_range[CPUDVFS.MAX_IDX]
        for wl in self._all_wls:
            if wl.is_running:
                CPUDVFS.set_freq(max_freq, wl.bound_cores)
        # FIXME: It assumes all bgs are running on a single CPU socket, so we throttle freq.s for the one bg
        #for bg_wl in self._best_effort_wls:
        #    bg_wl.cpu_dvfs.set_freq_cgroup(max_freq)

    def store_cur_config(self) -> None:
        self._stored_config = self._cur_steps

    def load_cur_config(self) -> None:
        super().load_cur_config()
        self._cur_steps = self._stored_config
        self._stored_config = None
