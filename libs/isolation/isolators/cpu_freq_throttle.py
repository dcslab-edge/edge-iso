# coding: UTF-8

import logging
from typing import Optional, Set

from .base import Isolator
from ...metric_container.basic_metric import MetricDiff
from ...utils import CPUDVFS
from ...workload import Workload


class CPUFreqThrottleIsolator(Isolator):
    def __init__(self, foreground_wl: Workload, background_wls: Set[Workload]) -> None:
        super().__init__(foreground_wl, background_wls)

        # FIXME: hard coded
        # Assumption: FG is latency-sensitive process (GPU) and BG is compute-intensive process (CPU)
        self._cur_step: int = CPUDVFS.MAX_IDX
        self._stored_config: Optional[int] = None
        self._cpufreq_range = CPUDVFS.get_freq_range()

    def _get_metric_type_from(self, metric_diff: MetricDiff) -> float:
        return metric_diff.local_mem_util_ps

    def strengthen(self) -> 'CPUFreqThrottleIsolator':
        self._cur_step -= CPUDVFS.STEP_IDX
        return self

    def weaken(self) -> 'CPUFreqThrottleIsolator':
        self._cur_step += CPUDVFS.STEP_IDX
        return self

    @property
    def is_max_level(self) -> bool:
        # FIXME: hard coded
        return self._cur_step - CPUDVFS.STEP_IDX < CPUDVFS.MIN_IDX

    @property
    def is_min_level(self) -> bool:
        # FIXME: hard coded
        return CPUDVFS.MAX_IDX < self._cur_step + CPUDVFS.STEP_IDX

    def enforce(self) -> None:
        logger = logging.getLogger(__name__)
        freq = self._cpufreq_range[self._cur_step]
        # FIXME: It assumes all bgs are running on a single CPU socket, so we throttle freq.s for the one bg
        for bg_wl in self._background_wls:
            logger.info(f'frequency of CPU cores of {bg_wl.name}\'s {bg_wl.bound_cores} is {freq / 1_000_000}GHz')
            bg_wl.cpu_dvfs.set_freq_cgroup(freq)

    def reset(self) -> None:
        max_freq = self._cpufreq_range[CPUDVFS.MAX_IDX]
        # FIXME: It assumes all bgs are running on a single CPU socket, so we throttle freq.s for the one bg
        for bg_wl in self._background_wls:
            bg_wl.cpu_dvfs.set_freq_cgroup(max_freq)

    def store_cur_config(self) -> None:
        self._stored_config = self._cur_step

    def load_cur_config(self) -> None:
        super().load_cur_config()

        self._cur_step = self._stored_config
        self._stored_config = None
