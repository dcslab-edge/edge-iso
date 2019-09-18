# coding: UTF-8

import logging

from typing import Set, Tuple, Type

from .base import IsolationPolicy, Isolator
from .. import ResourceType
from ..isolators import IdleIsolator, CycleLimitIsolator, CPUFreqThrottleIsolator, \
    GPUFreqThrottleIsolator, SchedIsolator, AffinityIsolator
from ...workload import Workload
from ...utils.machine_type import NodeType


class EdgePolicy(IsolationPolicy):
    def __init__(self, lc_wls: Set[Workload], be_wls: Set[Workload]) -> None:
        super().__init__(lc_wls, be_wls)
        self._is_mem_isolated = False
        self._is_freq_throttled_max = False
        self._leftover = lc_wls | be_wls

        self._perf_target_wl = None
        self._alloc_target_wl = None
        self._dealloc_target_wl = None

    @property
    def new_isolator_needed(self) -> bool:
        """
        This function choosing "victim" and checks if current isolator is `IdleIsolator`
        :return:
        """
        return isinstance(self._cur_isolator, IdleIsolator)

    def choose_next_isolator(self) -> bool:
        logger = logging.getLogger(__name__)
        logger.debug('[choose_next_isolator] looking for new isolation...')

        if self._dom_res_cont is ResourceType.CPU:
            isolator = self._isolator_map[AffinityIsolator]
        elif self._dom_res_cont is ResourceType.CACHE:
            isolator = self._isolator_map[CycleLimitIsolator]
        elif self._dom_res_cont is ResourceType.MEMORY:
            isolator = self._isolator_map[SchedIsolator]
            if isolator.is_max_level is True:
                isolator = self._isolator_map[CycleLimitIsolator]
        else:
            raise NotImplementedError(f'Unknown ResourceType: {self._dom_res_cont}')

        logger.info(f'[choose_next_isolator] Trying to choose {isolator.__class__.__name__}...')
        logger.info(f'[choose_next_isolator] self._dom_res_diff: {self._dom_res_diff}')
        logger.info(f'[choose_next_isolator] isolator.is_max_level: {isolator.is_max_level}')
        logger.info(f'[choose_next_isolator] isolator.is_min_level: {isolator.is_min_level}')
        # FIXME: condition for updating isolator to self._cur_isolator
        #if not (isolator.alloc_target_wl is None and isolator.dealloc_target_wl is None):
        if self._dom_res_diff < 0 and not isolator.is_max_level or \
                self._dom_res_diff > 0 and not isolator.is_min_level:
            self._cur_isolator = isolator
            # FIXME: Here is only place to update cur_dominant_resource_cont
            # There is a mismatch between `cur_dominant_resource_cont` and `self._dom_res_cont`
            self._cur_isolator.cur_dominant_resource_cont = self._dom_res_cont
            self._cur_isolator.perf_target_wl = self._perf_target_wl

            logger.info(f'===================================================')
            logger.info(f'Chosen isolator: {self._cur_isolator.__class__.__name__}...')
            logger.info(f'Dominant Resource Contention (isolator): {self._cur_isolator.cur_dominant_resource_cont}')
            logger.info(f'Dominant Resource Contention (policy): {self._dom_res_cont}')
            logger.info(f'Diff (policy): {self._dom_res_diff}')
            logger.info(f'Perf workload (isolator): [{self._cur_isolator.perf_target_wl}]')
            logger.info(f'Perf workload (policy): [{self._perf_target_wl}]')
            logger.info(f'===================================================')
            return True

        logger.info(f'===================================================')
        logger.info(f'However, controller fails to choose isolator!')
        logger.info(f'===================================================')
        return False

    # def choose_next_isolator(self) -> bool:
    #     logger = logging.getLogger(__name__)
    #     logger.debug('looking for new isolation...')
    #
    #     """
    #     *  contentious_resource() returns the most contentious resources of the foreground
    #     *  It returns either ResourceType.MEMORY or ResourceType.Cache
    #     *  JetsonTX2 : ResourceType.Cache -> CycleLimitIsolator,
    #     *              ResourceType.Memory -> GPUFreqThrottleIsolator, SchedIsolator
    #     *
    #     *  Desktop   : ResourceType.Cache -> CycleLimitIsolator, ResourceType.Memory -> SchedIsolator
    #     *
    #     """
    #
    #     """
    #     * When choosing isolator, policy considers whether the foreground task uses GPU.
    #     * If yes, it will use CPUFreqThrottle Isolator - We don't use it, since it's not per-core dvfs
    #     * Otherwise, it will use GPUFreqThrottle Isolator
    #     """
    #
    #     # if there are available free cores in the node, ...
    #     free_cores_set = self.update_allocated_cores()
    #     logger.info(f'[choose_next_isolator] free_cores_set: {free_cores_set}')
    #
    #     # TODO: Affinity Isolator should work properly when WEAKEN and STRENGTHEN.
    #     # WEAKEN: When free CPUs exist
    #     # STRENGTHEN: When workloads having excess CPUs exist
    #     # `check_excess_cpus` returns the number of workloads which have fewer threads than cores
    #     if len(free_cores_set) > 0 or self.check_excess_cpus_wls():
    #         if AffinityIsolator in self._isolator_map:
    #             if not (self._isolator_map[AffinityIsolator].is_max_level
    #                     or self._isolator_map[AffinityIsolator].is_min_level):
    #
    #                 # Select an isolator
    #                 self._cur_isolator = self._isolator_map[AffinityIsolator]
    #                 self._cur_isolator.cur_dominant_resource_cont = ResourceType.CPU
    #
    #                 # Update perf_target_wl
    #                 self._cur_isolator.perf_target_wl = self._perf_target_wl
    #
    #                 for res, diff in self.contentious_resources(self._perf_target_wl):
    #                     if res is ResourceType.CPU:
    #                         self._dom_res_cont = ResourceType.CPU
    #                         self._dom_res_diff = diff
    #                 logger.info(f'===================================================')
    #                 logger.info(f'Choosing {self._cur_isolator.__class__.__name__}...')
    #                 logger.info(f'Dominant Resource Contention (isolator): '
    #                             f'{self._cur_isolator.cur_dominant_resource_cont}')
    #                 logger.info(f'Dominant Resource Contention (policy): {self._dom_res_cont}')
    #                 logger.info(f'Diff (policy): {self._dom_res_diff}')
    #                 logger.info(f'Perf workload (isolator): [{self._cur_isolator.perf_target_wl}]')
    #                 logger.info(f'Perf workload (policy): [{self._perf_target_wl}]')
    #                 logger.info(f'===================================================')
    #                 return True
    #
    #     # TODO: Resource Fungibility(?), diff_slack for argument of contentious_resources
    #     # FIXME: Currently, diff_slack is FIXED to 0.2 for testing!
    #     # TODO: self.contentious_resources is not workload's function!
    #     # lc_wl = self.most_contentious_workload()
    #
    #     # Choose Isolator (Based on resource type)
    #     for resource, diff_value in self.contentious_resources(self._perf_target_wl):
    #         if resource is ResourceType.CACHE:
    #             isolator = self._isolator_map[CycleLimitIsolator]
    #         elif resource is ResourceType.MEMORY:
    #             # self.foreground_workload.check_gpu_task()
    #             if self._node_type == NodeType.IntegratedGPU:
    #                 isolator = self._isolator_map[SchedIsolator]
    #                 if isolator.is_max_level is True:
    #                     isolator = self._isolator_map[CycleLimitIsolator]
    #                     """
    #                     if self.foreground_workload.is_gpu_task == 1:   # if fg is gpu task, ...
    #                     # FIXME: choosing an isolator by considering whether the FG task using GPU or not.
    #                         isolator = self._isolator_map[CPUFreqThrottleIsolator]
    #                         if isolator.is_max_level is True:
    #                             isolator = self._isolator_map[SchedIsolator]
    #                     elif self.foreground_workload.is_gpu_task == 0: # if fg is cpu task, ...
    #                         isolator = self._isolator_map[GPUFreqThrottleIsolator]
    #                         if isolator.is_max_level is True:
    #                             isolator = self._isolator_map[SchedIsolator]
    #                     """
    #             elif self._node_type == NodeType.CPU:
    #                 isolator = self._isolator_map[SchedIsolator]
    #         elif resource is ResourceType.CPU:
    #             logger.info(f'[contentious_resources_loop] resource: {resource}, diff_value: {diff_value}')
    #             continue
    #         else:
    #             raise NotImplementedError(f'Unknown ResourceType: {resource}')
    #
    #         logger.info(f'[contentious_resources_loop] resource: {resource}, diff_value: {diff_value}, isolator: {isolator}')
    #
    #         # Update cur isolator and its dominant resource cotention info. (res_type, diff)
    #         #self._cur_isolator = isolator   # Update Current Isolator
    #         #self._cur_isolator.cur_dominant_resource_cont = resource
    #         #self._cur_isolator.perf_target_wl = self._perf_target_wl
    #
    #         self._dom_res_cont = resource
    #         self._dom_res_diff = diff_value
    #
    #         logger.info(f'[contentious_resources_loop] diff_value: {diff_value}')
    #         logger.info(f'[contentious_resources_loop] isolator.is_max_level: {isolator.is_max_level}')
    #         logger.info(f'[contentious_resources_loop] isolator.is_min_level: {isolator.is_min_level}')
    #
    #         if not (self._cur_isolator.alloc_target_wl is None and
    #                 self._cur_isolator.dealloc_target_wl is None):
    #             #self._cur_isolator.cur_dominant_resource_cont = resource
    #             #self._cur_isolator.perf_target_wl = self._perf_target_wl
    #
    #             logger.info(f'===================================================')
    #             logger.info(f'Choosing {self._cur_isolator.__class__.__name__}...')
    #             logger.info(f'Dominant Resource Contention (isolator): {self._cur_isolator.cur_dominant_resource_cont}')
    #             logger.info(f'Dominant Resource Contention (policy): {self._dom_res_cont}')
    #             logger.info(f'Diff (policy): {self._dom_res_diff}')
    #             logger.info(f'Perf workload (isolator): [{self._cur_isolator.perf_target_wl}]')
    #             logger.info(f'Perf workload (policy): [{self._perf_target_wl}]')
    #             logger.info(f'===================================================')
    #             return True
    #
    #     logger.debug('A new Isolator has not been selected')
    #     return False