# coding: UTF-8

import logging
from typing import Optional, Set

from .base import Isolator
from ...metric_container.basic_metric import MetricDiff
from ...utils import CPUDVFS, GPUDVFS
from ...workload import Workload


class GPUFreqThrottleIsolator(Isolator):
    def __init__(self, foreground_wl: Workload, background_wls: Set[Workload]) -> None:
        super().__init__(foreground_wl, background_wls)

        # FIXME: hard coded
        # Assumption: FG is latency-sensitive process (CPU) and BG is compute-intensive process (GPU)
        self._cur_step: int = GPUDVFS.MAX_IDX
        self._stored_config: Optional[int] = None
        self._gpufreq_range = GPUDVFS.get_freq_range()

    @classmethod
    def _get_metric_type_from(cls, metric_diff: MetricDiff) -> float:
        return metric_diff.local_mem_util_ps

    def strengthen(self) -> 'GPUFreqThrottleIsolator':
        self._cur_step -= GPUDVFS.STEP_IDX
        return self

    def weaken(self) -> 'GPUFreqThrottleIsolator':
        self._cur_step += GPUDVFS.STEP_IDX
        return self

    @property
    def is_max_level(self) -> bool:
        # FIXME: hard coded
        return self._cur_step - GPUDVFS.STEP_IDX < GPUDVFS.MIN_IDX

    @property
    def is_min_level(self) -> bool:
        # FIXME: hard coded
        return CPUDVFS.MAX_IDX < self._cur_step + GPUDVFS.STEP_IDX

    def enforce(self) -> None:
        logger = logging.getLogger(__name__)
        freq = self._gpufreq_range[self._cur_step]
        for bg_wl in self._background_wls:
            logger.info(f'frequency of GPU cores of {bg_wl.name}\'s {bg_wl.bound_cores} is {freq / 1_000_000_000}GHz')
        GPUDVFS.set_freq(freq)

    def reset(self) -> None:
        max_freq = self._gpufreq_range[GPUDVFS.MAX_IDX]
        GPUDVFS.set_freq(max_freq)

    def store_cur_config(self) -> None:
        self._stored_config = self._cur_step

    def load_cur_config(self) -> None:
        super().load_cur_config()

        self._cur_step = self._stored_config
        self._stored_config = None
