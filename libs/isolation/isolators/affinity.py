# coding: UTF-8

import logging
from typing import Optional, Set

from .base import Isolator
from ...metric_container.basic_metric import MetricDiff
from ...workload import Workload


class AffinityIsolator(Isolator):
    def __init__(self, latency_critical_wls: Workload, best_effort_wls: Set[Workload]) -> None:
        super().__init__(latency_critical_wls, best_effort_wls)

        self._cur_step: int = self._latency_critical_wls.orig_bound_cores[-1]

        self._stored_config: Optional[int] = None

    def _get_metric_type_from(self, metric_diff: MetricDiff) -> float:
        return metric_diff.instruction_ps

    def strengthen(self) -> 'AffinityIsolator':
        self._cur_step += 1
        return self

    @property
    def is_max_level(self) -> bool:
        # FIXME: hard coded
        for bg_wl in self._best_effort_wls:
            if self._cur_step + 1 == bg_wl.bound_cores[0]:
                return True
        return False

    @property
    def is_min_level(self) -> bool:
        return self._latency_critical_wls.orig_bound_cores == self._latency_critical_wls.bound_cores

    def weaken(self) -> 'AffinityIsolator':
        self._cur_step -= 1
        return self

    def enforce(self) -> None:
        logger = logging.getLogger(__name__)
        logger.info(f'affinity of foreground is {self._latency_critical_wls.orig_bound_cores[0]}-{self._cur_step}')

        self._latency_critical_wls.bound_cores = range(self._latency_critical_wls.orig_bound_cores[0], self._cur_step + 1)

    def reset(self) -> None:
        if self._latency_critical_wls.is_running:
            self._latency_critical_wls.bound_cores = self._latency_critical_wls.orig_bound_cores

    def store_cur_config(self) -> None:
        self._stored_config = self._cur_step

    def load_cur_config(self) -> None:
        super().load_cur_config()

        self._cur_step = self._stored_config
        self._stored_config = None
