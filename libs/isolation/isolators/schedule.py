# coding: UTF-8

import logging
import random
from typing import Optional, Set, Dict, Tuple, Iterable, ClassVar, Any

#from ..policies.base import IsolationPolicy

from .base import Isolator
from ...metric_container.basic_metric import MetricDiff
from ...workload import Workload
from ...utils.machine_type import MachineChecker, NodeType
from ...metric_container.basic_metric import BasicMetric
from .. import NextStep, ResourceType


class SchedIsolator(Isolator):
    def __init__(self, latency_critical_wls: Set[Workload], best_effort_wls: Set[Workload]) -> None:
        super().__init__(latency_critical_wls, best_effort_wls)

        _FORCE_THRESHOLD: ClassVar[float] = 0.05

        # FIXME: hard coded (self._MAX_CORES is fixed depending on node type)
        self._node_type = MachineChecker.get_node_type()
        if self._node_type == NodeType.IntegratedGPU:
            self._all_cores = tuple([0, 3, 4, 5])
            self._MIN_CORES = 1
        elif self._node_type == NodeType.CPU: # BC5 case
            self._all_cores = tuple(range(0, 8, 1))
            self._MIN_CORES = 1

        self._cur_steps: Dict[Workload, Tuple[int]] = dict()
        # self._cur_memory_bw: Dict[Workload, float] = dict()
        # self._cur_memory_bw_diff: Dict[Workload, float] = dict()
        # self._cur_instr_diff: Dict[Workload, float] = dict()
        # for lc_wl in self._latency_critical_wls:
        #     self._cur_steps[lc_wl] = lc_wl.orig_bound_cores
        #     self._cur_memory_bw[lc_wl] = BasicMetric.calc_avg(lc_wl, 30).llc_miss_ps    # LLC misses
        #     self._cur_memory_bw_diff[lc_wl] = lc_wl.calc_metric_diff().local_mem_util_ps
        #     self._cur_instr_diff[lc_wl] = lc_wl.calc_metric_diff().instruction_ps
        # for be_wl in self._best_effort_wls:
        #     self._cur_steps[be_wl] = be_wl.orig_bound_cores
        #     self._cur_memory_bw[be_wl] = BasicMetric.calc_avg(be_wl, 30).llc_miss_ps    # LLC misses
        #     self._cur_memory_bw_diff[be_wl] = be_wl.calc_metric_diff().local_mem_util_ps
        #     self._cur_instr_diff[be_wl] = be_wl.calc_metric_diff().instruction_ps

        self._chosen_alloc: Optional[int] = None
        self._chosen_dealloc: Optional[int] = None
        self._cur_alloc: Optional[Tuple[int]] = None
        self._cur_dealloc: Optional[Tuple[int]] = None

        self._available_cores: Optional[Tuple[int]] = Isolator.available_cores
        self._allocated_cores: Optional[Tuple[int]] = None
        for step in self._cur_steps.values():
            self._allocated_cores += step

        #self._available_cores = set(self._all_cores) - set(self._allocated_cores)

        self._stored_config: Optional[Dict[Workload, Tuple[int]]] = None

    def _get_metric_type_from(self, metric_diff: MetricDiff) -> float:
        return metric_diff.local_mem_util_ps - metric_diff.diff_slack

    def strengthen(self) -> 'SchedIsolator':
        """
        This function deallocates cores from the most memory-intensive workload
        :return:
        """
        logger = logging.getLogger(__name__)

        # re-assign dealloc_target_wl based on "mem_bw"
        # It selects the workload which is able to deallocable
        #self.choosing_wl_for(strengthen=True, sort_criteria="mem_bw", lowest=True)
        wl = self.dealloc_target_wl
        if wl is not None:
            logger.info(f"It can deallocate {wl.name}-{wl.pid}")
            logger.info(f"self.dealloc_target_wl : {wl}")
            self._chosen_dealloc = random.choice(wl.bound_cores)
            logger.info(f"self._chosen_dealloc : {wl}")
            self._cur_dealloc = tuple(filter(lambda x: x is not self._chosen_dealloc, wl.bound_cores))
        elif wl is None:
            logger.info(f"There is no dealloc_target_wl. (No workload)")
            logger.info(f"self.dealloc_target_wl : {wl}")
            self._chosen_dealloc = None
            self._cur_dealloc = None
        return self

    def weaken(self) -> 'SchedIsolator':
        """
        This function allocates cores to the memory-intensive workload
        (Prioritizing memory-intensive workloads,
        because there is no contention? or excess resource?
        How can we determine this?)
        :return:
        """
        logger = logging.getLogger(__name__)

        # finding workload for highest memory diff workload
        # TODO: Performance Testing with Simple Scenario
        #self.choosing_wl_for(strengthen=False, sort_criteria="mem_bw_diff", lowest=False)
        wl = self.alloc_target_wl
        if wl is not None:
            logger.info(f"It can allocate {wl.name}-{wl.pid}")
            logger.info(f"self.alloc_target_wl : {wl}")
            self.get_available_cores()
            self._chosen_alloc = random.choice(self._available_cores)
            logger.info(f"self._chosen_dealloc : {wl}")
        elif wl is None:
            logger.info(f"There is no alloc_target_wl. (No workload)")
            logger.info(f"self.alloc_target_wl : {wl}")
            self._chosen_alloc = None
        self._cur_alloc = tuple(self._cur_steps[wl]+tuple(self._chosen_alloc))
        return self

    @property
    def is_max_level(self) -> bool:
        # At least, a process needs one core for its execution - strengthen condition 1 (in `self._choosing_wl_for()`)
        # FIXME: hard-coded CPUSET.STEP (e.g., 1)
        logger = logging.getLogger(__name__)
        logger.info(f'[is_max_level] self.dealloc_target_wl: {self.dealloc_target_wl}')
        if self.dealloc_target_wl is None:
            return False
        else:
            return len(self.dealloc_target_wl.bound_cores) - 1 < self._MIN_CORES

    @property
    def is_min_level(self) -> bool:
        # At least, available core has one core - weaken condition 1 (in `self._choosing_wl_for()`)
        logger = logging.getLogger(__name__)
        logger.info(f'[is_min_level] self.alloc_target_wl: {self.alloc_target_wl}')
        # FIXME: self._available_cores is valid in the following code segment?
        self.get_available_cores()
        return len(self._available_cores) <= 0

    def enforce(self) -> None:
        logger = logging.getLogger(__name__)
        # FIXME: hard coded & Assuming that bg task is placed to CPUs which have higher CPU IDs (e.g., 4,5)
        if self._cur_alloc is not None:
            logger.info(f'affinity of {self.alloc_target_wl.name}-{self.alloc_target_wl.pid} is '
                        f'{self._cur_alloc}')
            self.alloc_target_wl.bound_cores = self._cur_alloc
            self._cur_steps[self.alloc_target_wl] = self._cur_alloc
            self._update_other_values("alloc")

        elif self._cur_dealloc is not None:
            logger.info(f'affinity of {self.dealloc_target_wl.name}-{self.dealloc_target_wl.pid} is '
                        f'{self._cur_dealloc}')
            self.dealloc_target_wl.bound_cores = self._cur_dealloc
            self._cur_steps[self.dealloc_target_wl] = self._cur_dealloc
            self._update_other_values("dealloc")

    def reset(self) -> None:
        """
        reset() is used to set the original configuration for foregrounds (for measuring solo-run data),
        and it is also used to run bg task alone.
        :return:
        """
        for wl in self._all_wls:
            if wl.is_running:
                wl.bound_cores = wl.orig_bound_cores

    def store_cur_config(self) -> None:
        self._stored_config = self._cur_steps

    def load_cur_config(self) -> None:
        """
        This function should restore the configurations of multiple background tasks.

        :return:
        """
        super().load_cur_config()

        self._cur_steps = self._stored_config

        self._stored_config = None

    def get_available_cores(self) -> None:
        self._available_cores = Isolator.available_cores()

    def update_available_cores(self) -> None:
        Isolator.set_available_cores(self._available_cores)

    # def update_workload_res_info(self) -> None:
    #     for lc_wl in self._latency_critical_wls:
    #         self._cur_memory_bw[lc_wl] = BasicMetric.calc_avg(lc_wl, 30).llc_miss_ps
    #         self._cur_memory_bw_diff[lc_wl] = lc_wl.calc_metric_diff().local_mem_util_ps
    #         self._cur_instr_diff[lc_wl] = lc_wl.calc_metric_diff().instruction_ps
    #     for be_wl in self._best_effort_wls:
    #         self._cur_memory_bw[be_wl] = BasicMetric.calc_avg(be_wl, 30).llc_miss_ps
    #         self._cur_memory_bw_diff[be_wl] = be_wl.calc_metric_diff().local_mem_util_ps
    #         self._cur_instr_diff[be_wl] = be_wl.calc_metric_diff().instruction_ps
    #
    # #def update_memory_bandwidth(self) -> None:
    # #    for lc_wl in self._latency_critical_wls:
    # #        self._cur_memory_bw[lc_wl] = BasicMetric.calc_avg(lc_wl, 30).llc_miss_ps
    # #    for be_wl in self._best_effort_wls:
    # #        self._cur_memory_bw[be_wl] = BasicMetric.calc_avg(be_wl, 30).llc_miss_ps
    #
    # def sorting_workloads_by(self, target: Dict[Workload, Any], lowest: bool) -> Iterable:
    #     """
    #     This function
    #     :param target: This parameter is a target to be sorted
    #     :param criteria: This parameter determines the key for sorting
    #     :param lowest: This parameter decides the order of sorting
    #     (e.g., if true, then sorting in ascending order)
    #     :return:
    #     """
    #     logger = logging.getLogger(__name__)
    #
    #     # Sorting workloads in descending order
    #     sorted_wls = tuple(sorted(target.items(), key=lambda x: x[1], reverse=lowest))
    #     #if criteria is "mem_bw":
    #     #    sorted_wls = tuple(sorted(self._cur_memory_bw.items(), key=lambda x: x[1], reverse=lowest))
    #     #elif criteria is "mem_bw_diff":
    #     #    sorted_wls = tuple(sorted(self._cur_memory_bw_diff.items(), key=lambda x: x[1], reverse=lowest))
    #
    #     if sorted_wls is None:
    #         logger.info(f"sorted_wls :{sorted_wls}")
    #
    #     return sorted_wls
    #
    # def choosing_wl_for(self, strengthen: bool, sort_criteria: str, lowest: bool) -> None:
    #     """
    #     This function is originally for picking a workload to strengthen/weaken
    #     :param strengthen: It indicates the intention for choosing workload (weaken or strengthen)
    #     :param sort_criteria: It indicates the sorting criteria (i.e., mem_bw for strengthen, mem_bw_diff for weaken)
    #     :param lowest: It indicates the sort direction (e.g., ascending or descending)
    #     :return:
    #     """
    #     # TODO: This function only choose a workload of the highest memory diff
    #     # FIXME: This function should override the `self._alloc_target_wl` & `self._dealloc_target_wl`
    #     logger = logging.getLogger(__name__)
    #     excluded: Iterable = ()
    #     if strengthen is True:
    #         # Choosing allocable wl, so we exclude `dealloc_target_wl`
    #         excluded = (self.perf_target_wl, )
    #     elif strengthen is False:
    #         # Choosing deallocable wl, so we exclude `alloc_target_wl`
    #         excluded = (self.perf_target_wl, )
    #
    #     #self.update_memory_bandwidth()  # Update Resource Usage of All Workloads
    #     self.update_workload_res_info()  # Update Resource Usage of All Workloads
    #     sort_target = None
    #     if sort_criteria is "mem_bw":
    #         sort_target = self._cur_memory_bw
    #     elif sort_criteria is "mem_bw_diff":
    #         sort_target = self._cur_memory_bw_diff
    #     elif sort_criteria is "instruction":
    #         sort_target = self._cur_instr_diff
    #     wls_info = tuple(sorted(sort_target.items(), key=lambda x: x[1], reverse=lowest))
    #     #wls_info = self.sorting_workloads_by(sort_target, sort_criteria, lowest)
    #     chosen = False
    #     idx = 0
    #
    #     if wls_info is None:
    #         logger.info(f"There is no any workloads to sort!")
    #         self.dealloc_target_wl = None
    #         return
    #
    #     # Choosing the highest (or the lowest) memory bandwidth / memory bandwidth diff
    #     candidate = tuple(filter(lambda x: x[0] not in excluded, wls_info))
    #     num_candidates = len(candidate)
    #     while not chosen and idx < num_candidates-1:
    #         candidate = tuple(filter(lambda x: x[0] not in excluded, wls_info))
    #         cur_target_wl: Workload = candidate[idx][0]   # idx is the ordinal position from the very first one
    #         # cur_memory_bw_diff is criteria for choosing a deallocable candidate for weakening
    #         # cur_memory_bw is criteria for choosing an allocable candidate for strengthening
    #
    #         if strengthen is True:
    #             # strengthening condition 1 (at least, number of cores > 1)
    #             # NOTE: Currently `strengthen` needs at least a single core
    #             if len(cur_target_wl.bound_cores) > 1:
    #                 self.dealloc_target_wl = cur_target_wl
    #                 chosen = True
    #             else:
    #                 excluded += cur_target_wl
    #         elif strengthen is False:
    #             # weakening condition 1 (at least, available core > 0)
    #             if len(self._available_cores) > 0:
    #                 self.alloc_target_wl = cur_target_wl
    #                 chosen = True
    #             else:
    #                 excluded += cur_target_wl
    #         idx += 1
    #
    #     if not chosen:
    #         logger.info(f"There is no chosen workload to strengthen or weaken, "
    #                     f"strengthen:{strengthen}, chosen: {chosen}")
    #         self.alloc_target_wl = None
    #         self.dealloc_target_wl = None

    def _update_other_values(self, action: str) -> None:
        if action is "alloc":
            self._available_cores = tuple(filter(lambda x: x is not self._chosen_alloc, self._available_cores))
            self.update_available_cores()
            self._cur_alloc = None
            self._chosen_alloc = None
        elif action is "dealloc":
            self._available_cores = tuple(self._available_cores + self._chosen_dealloc)
            self.update_available_cores()
            self._cur_dealloc = None
            self._chosen_dealloc = None
