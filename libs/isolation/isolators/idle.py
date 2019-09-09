# coding: UTF-8

from .base import Isolator
from .. import NextStep
from ...metric_container.basic_metric import MetricDiff
from ...workload import Workload
from typing import Set, Dict


class IdleIsolator(Isolator):
    def __init__(self, latency_critical_wls: Set[Workload], best_effort_wls: Set[Workload]) -> None:
        super().__init__(latency_critical_wls, best_effort_wls)

    def _get_metric_type_from(self, metric_diff: MetricDiff) -> float:
        pass

    def strengthen(self) -> 'Isolator':
        pass

    @property
    def is_max_level(self) -> bool:
        return True

    @property
    def is_min_level(self) -> bool:
        return False

    def weaken(self) -> 'Isolator':
        pass

    def enforce(self) -> None:
        pass

    def _first_decision(self, _) -> NextStep:
        self._fg_next_step = NextStep.IDLE
        self._bg_next_step = NextStep.IDLE
        return NextStep.IDLE

    def decide_next_step(self) -> NextStep:
        return self._monitoring_result()

    def _monitoring_result(self, **kwargs) -> NextStep:
        self._fg_next_step = NextStep.IDLE
        self._bg_next_step = NextStep.IDLE
        return NextStep.IDLE

    def reset(self) -> None:
        pass

    def store_cur_config(self) -> None:
        pass

    def load_cur_config(self) -> None:
        pass
