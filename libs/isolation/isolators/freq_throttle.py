# coding: UTF-8

import logging
from typing import Optional

from .base import Isolator
from ...metric_container.basic_metric import MetricDiff
from ...utils import DVFS, GPUDVFS
from ...workload import Workload


class FreqThrottleIsolator(Isolator):
    def __init__(self, foreground_wl: Workload, background_wl: Workload) -> None:
        super().__init__(foreground_wl, background_wl)

        # FIXME: hard coded
        # Assumption: FG is latency-sensitive process (CPU) and BG is compute-intensive process (GPU)
        self._cur_step: int = DVFS.MAX_IDX
        self._stored_config: Optional[int] = None
        self._gpufreq_range = GPUDVFS.get_freq_range()

    @classmethod
    def _get_metric_type_from(cls, metric_diff: MetricDiff) -> float:
        return metric_diff.local_mem_util_ps

    def strengthen(self) -> 'FreqThrottleIsolator':
        self._cur_step -= GPUDVFS.STEP_IDX
        return self

    def weaken(self) -> 'FreqThrottleIsolator':
        self._cur_step += GPUDVFS.STEP_IDX
        return self

    @property
    def is_max_level(self) -> bool:
        # FIXME: hard coded
        return self._cur_step - GPUDVFS.STEP_IDX < GPUDVFS.MIN_IDX

    @property
    def is_min_level(self) -> bool:
        # FIXME: hard coded
        return DVFS.MAX_IDX < self._cur_step + GPUDVFS.STEP_IDX

    def enforce(self) -> None:
        logger = logging.getLogger(__name__)
        logger.info(f'frequency of bound_cores {self._background_wl.bound_cores} is {self._cur_step / 1_000_000}GHz')
        freq = self._gpufreq_range[self._cur_step]
        GPUDVFS.set_freq(freq, self._background_wl.bound_cores)

    def reset(self) -> None:
        max_freq = self._gpufreq_range[GPUDVFS.MAX_IDX]
        GPUDVFS.set_freq(max_freq, self._background_wl.orig_bound_cores)

    def store_cur_config(self) -> None:
        self._stored_config = self._cur_step

    def load_cur_config(self) -> None:
        super().load_cur_config()

        self._cur_step = self._stored_config
        self._stored_config = None
