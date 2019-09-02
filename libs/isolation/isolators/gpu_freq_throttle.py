# coding: UTF-8

import logging
from typing import Optional, Set, Dict

from .base import Isolator
from ...metric_container.basic_metric import MetricDiff
from ...utils import CPUDVFS, GPUDVFS
from ...workload import Workload


class GPUFreqThrottleIsolator(Isolator):
    def __init__(self, latency_critical_wls: Set[Workload], best_effort_wls: Set[Workload]) -> None:
        super().__init__(latency_critical_wls, best_effort_wls)

        # Note, Assumption: FG is latency-sensitive process (CPU) and BG is compute-intensive process (GPU)
        # FIXME: GPUDVFS influences all GPU cores
        self._cur_steps: Dict[Workload, int] = dict()
        for wl in latency_critical_wls:
            self._cur_steps[wl] = GPUDVFS.MAX_IDX
        for wl in best_effort_wls:
            self._cur_steps[wl] = GPUDVFS.MAX_IDX
        self._stored_config: Optional[Dict[Workload, int]] = None
        self._gpufreq_range = GPUDVFS.get_freq_range()

    def _get_metric_type_from(self, metric_diff: MetricDiff) -> float:
        return metric_diff.local_mem_util_ps - metric_diff.diff_slack

    def strengthen(self) -> 'GPUFreqThrottleIsolator':
        self._cur_steps[self.dealloc_target_wl] -= GPUDVFS.STEP_IDX
        return self

    def weaken(self) -> 'GPUFreqThrottleIsolator':
        self._cur_steps[self.alloc_target_wl] += GPUDVFS.STEP_IDX
        return self

    @property
    def is_max_level(self) -> bool:
        # FIXME: hard coded
        return self._cur_steps[self.dealloc_target_wl] - GPUDVFS.STEP_IDX < GPUDVFS.MIN_IDX

    @property
    def is_min_level(self) -> bool:
        # FIXME: hard coded
        return CPUDVFS.MAX_IDX < self._cur_steps[self.alloc_target_wl] + GPUDVFS.STEP_IDX

    def enforce(self) -> None:
        logger = logging.getLogger(__name__)
        wls = [self.alloc_target_wl, self.dealloc_target_wl]
        for wl in wls:
            freq = self._gpufreq_range[self._cur_steps[wl]]
            logger.info(f'GPU core frequencies of {wl.name}\'s is '
                        f'{self._gpufreq_range[freq]/1_000_000_000}GHz')

        for wl in wls:
            freq = self._gpufreq_range[self._cur_steps[wl]]
            GPUDVFS.set_freq(freq)
        self.alloc_target_wl = None
        self.dealloc_target_wl = None

    def reset(self) -> None:
        max_freq = self._gpufreq_range[GPUDVFS.MAX_IDX]
        GPUDVFS.set_freq(max_freq)

    def store_cur_config(self) -> None:
        self._stored_config = self._cur_steps

    def load_cur_config(self) -> None:
        super().load_cur_config()

        self._cur_steps = self._stored_config
        self._stored_config = None
