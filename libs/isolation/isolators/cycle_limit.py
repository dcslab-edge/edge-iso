# coding: UTF-8

import logging
from typing import Optional, Set

from .base import Isolator
from ...metric_container.basic_metric import MetricDiff
from ...utils.cgroup import Cpu
from ...workload import Workload


class CycleLimitIsolator(Isolator):
    def __init__(self, foreground_wl: Workload, background_wls: Set[Workload]) -> None:
        super().__init__(foreground_wl, background_wls)

        # FIXME: hard coded
        self._cur_step: int = Cpu.MAX_PERCENT
        self._stored_config: Optional[int] = None

    @classmethod
    def _get_metric_type_from(cls, metric_diff: MetricDiff) -> float:
        return metric_diff.local_mem_util_ps

    def strengthen(self) -> 'CycleLimitIsolator':
        self._cur_step -= Cpu.STEP
        return self

    def weaken(self) -> 'CycleLimitIsolator':
        self._cur_step += Cpu.STEP
        return self

    @property
    def is_max_level(self) -> bool:
        # FIXME: hard coded
        return self._cur_step - Cpu.STEP < Cpu.MIN_PERCENT

    @property
    def is_min_level(self) -> bool:
        # FIXME: hard coded
        return Cpu.MAX_PERCENT < self._cur_step + Cpu.STEP

    def enforce(self) -> None:
        logger = logging.getLogger(__name__)
        for bg_wl in self._background_wls:
            logger.info(f'frequency of bound_cores of {bg_wl.name}\'s {bg_wl.bound_cores} is '
                        f'{self._cur_step / 1_000_000}GHz')

        for bg_wl in self._background_wls:
            Cpu.limit_cycle_percentage(bg_wl.group_name, self._cur_step)

    def reset(self) -> None:
        for bg_wl in self._background_wls:
            Cpu.limit_cycle_percentage(bg_wl.group_name, Cpu.MAX_PERCENT)

    def store_cur_config(self) -> None:
        self._stored_config = self._cur_step

    def load_cur_config(self) -> None:
        super().load_cur_config()

        self._cur_step = self._stored_config
        self._stored_config = None
