# coding: UTF-8

import logging
from abc import ABCMeta, abstractmethod
from typing import Any, ClassVar, Optional, Set, Dict

from .. import NextStep
from .. import ResourceType
from ...metric_container.basic_metric import MetricDiff
from ...workload import Workload


class Isolator(metaclass=ABCMeta):
    _DOD_THRESHOLD: ClassVar[float] = 0.005
    _FORCE_THRESHOLD: ClassVar[float] = 0.05

    def __init__(self, latency_critical_wls: Set[Workload], best_effort_wls: Set[Workload]) -> None:
        self._prev_metric_diff: MetricDiff = None

        self._latency_critical_wls = latency_critical_wls
        self._best_effort_wls = best_effort_wls
        self._target_wl = None

        # FIXME: All FGs, BGs can have different NextStep (Currently, All WLs are homogeneous)
        self._lc_wl_next_steps: Dict[Workload, NextStep] = dict()
        self._be_wl_next_steps: Dict[Workload, NextStep] = dict()

        for k in self._lc_wl_next_steps.keys():
            self._lc_wl_next_steps[k] = NextStep.IDLE

        for k in self._be_wl_next_steps.keys():
            self._be_wl_next_steps[k] = NextStep.IDLE

        self._is_first_decision: bool = True
        self._cur_dominant_resource_cont: ResourceType = None

        self._stored_config: Optional[Any] = None

    def __del__(self):
        self.reset()

    @property
    def target_wl(self) -> None:
        return self._target_wl

    @target_wl.setter
    def target_wl(self, new_target_wl: Workload) -> None:
        self._target_wl = new_target_wl

    @abstractmethod
    def strengthen(self) -> 'Isolator':
        """
        Adjust the isolation parameter to allocate more resources to latency-critical workloads.
        (Does not actually isolate)

        :return: current isolator object for method chaining
        :rtype: Isolator
        """
        pass

    @property
    @abstractmethod
    def is_max_level(self) -> bool:
        pass

    @property
    @abstractmethod
    def is_min_level(self) -> bool:
        pass

    @abstractmethod
    def weaken(self) -> 'Isolator':
        """
        Adjust the isolation parameter to allocate less resources to latency-critical workloads.
        (Does not actually isolate)

        :return: current isolator object for method chaining
        :rtype: Isolator
        """
        pass

    @abstractmethod
    def enforce(self) -> None:
        """Actually applies the isolation parameter that set on the current object"""
        pass

    @property
    def cur_dominant_resource_cont(self) -> ResourceType:
        return self._cur_dominant_resource_cont

    @cur_dominant_resource_cont.setter
    def cur_dominant_resource_cont(self, resource: ResourceType) -> None:
        self._cur_dominant_resource_cont = resource

    def yield_isolation(self) -> None:
        """
        Declare to stop the configuration search for the current isolator.
        Must be called when the current isolator yields the initiative.
        """
        self._is_first_decision = True

    def _first_decision(self, cur_metric_diff: MetricDiff) -> NextStep:
        # FIXME: first decision for latency-critical workloads

        curr_diff = self._get_metric_type_from(cur_metric_diff)

        logger = logging.getLogger(__name__)
        logger.debug(f'current diff: {curr_diff:>7.4f}')

        if curr_diff < 0:
            if self.is_max_level:
                return NextStep.STOP
            else:
                return NextStep.STRENGTHEN
        elif curr_diff <= self._FORCE_THRESHOLD:
            return NextStep.STOP
        else:
            if self.is_min_level:
                return NextStep.STOP
            else:
                return NextStep.WEAKEN

    def _monitoring_result(self, prev_metric_diff: MetricDiff, cur_metric_diff: MetricDiff) -> NextStep:
        # FIXME: monitoring_result for latency-critical workloads

        curr_diff = self._get_metric_type_from(cur_metric_diff)
        prev_diff = self._get_metric_type_from(prev_metric_diff)
        diff_of_diff = curr_diff - prev_diff

        logger = logging.getLogger(__name__)
        logger.debug(f'diff of diff is {diff_of_diff:>7.4f}')
        logger.debug(f'current diff: {curr_diff:>7.4f}, previous diff: {prev_diff:>7.4f}')

        if abs(diff_of_diff) <= self._DOD_THRESHOLD \
                or abs(curr_diff) <= self._DOD_THRESHOLD:
            return NextStep.STOP

        elif curr_diff > 0:
            if self.is_min_level:
                return NextStep.STOP
            else:
                return NextStep.WEAKEN

        else:
            if self.is_max_level:
                return NextStep.STOP
            else:
                return NextStep.STRENGTHEN

    @abstractmethod
    def _get_metric_type_from(self, metric_diff: MetricDiff) -> float:
        pass
    """
    def choose_workload_for_isolation(self) -> Workload:
        logger = logging.getLogger(__name__)

        target_lc_wl = None
        min_inst_diff = 0
        for lc_wl in self._latency_critical_wls:
            curr_metric_diff = lc_wl.calc_metric_diff()
            curr_inst_diff = curr_metric_diff.instruction_ps
            if curr_inst_diff < min_inst_diff:
                min_inst_diff = curr_inst_diff
                target_lc_wl = lc_wl

        if target_lc_wl is None:
            logger.info(f'target_lc_wl is None!')
        if target_lc_wl is not None:
            logger.info(f'lowest_instruction_diff target_lc_wl: {target_lc_wl.name}-{target_lc_wl.pid}, '
                        f'inst_diff: {min_inst_diff}')

        return target_lc_wl
    """

    def decide_next_step(self) -> NextStep:
        # FIXME: Fix code to work for `multiple` latency-critical workloads
        """
        Deciding the next step for current isolator
        isolation is performed at a time
        :return:
        """

        #target_lc_wl = self.choose_workload_for_isolation()

        if self._target_wl is None:
            return NextStep.IDLE

        curr_metric_diff = self._target_wl.calc_metric_diff()

        if self._is_first_decision:
            self._is_first_decision = False
            next_step = self._first_decision(curr_metric_diff)

        else:
            next_step = self._monitoring_result(self._prev_metric_diff, curr_metric_diff)

        self._prev_metric_diff = curr_metric_diff

        return next_step

    @abstractmethod
    def reset(self) -> None:
        """Restore to initial configuration"""
        pass

    def change_lc_wls(self, new_workloads: Set[Workload]) -> None:
        #self._foreground_wl = new_workload
        self._latency_critical_wls = new_workloads
        #self._prev_metric_diff = new_workload.calc_metric_diff()

    def change_be_wls(self, new_workloads: Set[Workload]) -> None:
        self._best_effort_wls = new_workloads

    @abstractmethod
    def store_cur_config(self) -> None:
        """Store the current configuration"""
        pass

    def load_cur_config(self) -> None:
        """Load the current configuration"""
        if self._stored_config is None:
            raise ValueError('Store configuration first!')
