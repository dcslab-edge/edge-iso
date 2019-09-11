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

    @property
    def new_isolator_needed(self) -> bool:
        return isinstance(self._cur_isolator, IdleIsolator)

    def choose_next_isolator(self) -> bool:
        logger = logging.getLogger(__name__)
        logger.debug('looking for new isolation...')

        """
        *  contentious_resource() returns the most contentious resources of the foreground 
        *  It returns either ResourceType.MEMORY or ResourceType.Cache
        *  JetsonTX2 : ResourceType.Cache -> CycleLimitIsolator, 
        *              ResourceType.Memory -> GPUFreqThrottleIsolator, SchedIsolator
        *
        *  Desktop   : ResourceType.Cache -> CycleLimitIsolator, ResourceType.Memory -> SchedIsolator
        * 
        """

        """
        * When choosing isolator, policy considers whether the foreground task uses GPU. 
        * If yes, it will use CPUFreqThrottle Isolator - We don't use it, since it's not per-core dvfs 
        * Otherwise, it will use GPUFreqThrottle Isolator
        """

        # if there are available free cores in the node, ...
        # FIXME: hard coded `max_cpu_cores`
        #cpu_core_set: Set[int] = {0, 3, 4, 5}
        free_cores_set = self.update_allocated_cores()
        #all_allocated_cores: Set[int] = self.all_lc_cores | self.all_be_cores

        #free_cores_set = cpu_core_set - all_allocated_cores
        logger.info(f'[choose_next_isolator] free_cores_set: {free_cores_set}')
        #logger.debug(f'[choose_next_isolator:before choosing_wl_for] self._leftover: {self._leftover}')
        self.choosing_wl_for(objective="victim", sort_criteria="instr_diff", highest=False) # Pick a workload of low IPS
        logger.info(f'[choose_next_isolator:after choosing_wl_for] self._cur_isolator: {self.cur_isolator}')
        logger.debug(f'[choose_next_isolator:after choosing_wl_for] self._cur_isolator.perf_target_wl: '
                    f'{self.cur_isolator.perf_target_wl.name}-{self.cur_isolator.perf_target_wl.pid}')
        logger.info(f'[choose_next_isolator:after choosing_wl_for] self._perf_target_wl: '
                    f'{self._perf_target_wl.name}-{self._perf_target_wl.pid}')

        self._perf_target_wl = self._cur_isolator.perf_target_wl
        self._perf_target_wl.diff_slack = 0.2

        #logger.debug(f'[choose_next_isolator:after choosing_wl_for] self._leftover: {self._leftover}')
        #self._leftover.remove(self._perf_target_wl)

        # TODO: Affinity Isolator should work properly when WEAKEN and STRENGTHEN.
        # WEAKEN: When free CPUs exist
        # STRENGTHEN: When workloads having excess CPUs exist
        # `check_excess_cpus` returns the number of workloads which have fewer threads than cores
        if len(free_cores_set) > 0 or self.check_excess_cpus_wls():
            if AffinityIsolator in self._isolator_map:
                if not self._isolator_map[AffinityIsolator].is_max_level and not self._isolator_map[AffinityIsolator].is_min_level:
                    self._cur_isolator = self._isolator_map[AffinityIsolator]
                    self._cur_isolator.cur_dominant_resource_cont = ResourceType.CPU

                    # Moving perf_target_wl
                    self._cur_isolator.perf_target_wl = self._perf_target_wl

                    for res, diff in self.contentious_resources(self._perf_target_wl):
                        if res is ResourceType.CPU:
                            self.choose_isolation_target(self._cur_isolator.cur_dominant_resource_cont, diff)
                    logger.info(f'===================================================')
                    logger.info(f'Starting {self._cur_isolator.__class__.__name__}...')
                    logger.info(f'Dominant Resource Contention: {self._cur_isolator.cur_dominant_resource_cont}')
                    logger.info(f'Dealloc workload: [{self._cur_isolator.dealloc_target_wl}]')
                    logger.info(f'Alloc workload: [{self._cur_isolator.alloc_target_wl}]')
                    logger.info(f'Perf workload: [{self._cur_isolator.perf_target_wl}]')
                    logger.info(f'===================================================')
                    #self._leftover = self._lc_wls | self._be_wls
                    return True

        # TODO: Resource Fungibility(?), diff_slack for argument of contentious_resources
        # FIXME: Currently, diff_slack is FIXED to 0.2 for testing!
        # TODO: self.contentious_resources is not workload's function!
        # lc_wl = self.most_contentious_workload()

        # Choose Isolator (Based on resource type)
        for resource, diff_value in self.contentious_resources(self._perf_target_wl):
            if resource is ResourceType.CACHE:
                isolator = self._isolator_map[CycleLimitIsolator]
            elif resource is ResourceType.MEMORY:
                # self.foreground_workload.check_gpu_task()
                if self._node_type == NodeType.IntegratedGPU:
                    isolator = self._isolator_map[SchedIsolator]
                    if isolator.is_max_level is True:
                        isolator = self._isolator_map[CycleLimitIsolator]
                    """
                    if self.foreground_workload.is_gpu_task == 1:   # if fg is gpu task, ...
                    # FIXME: choosing an isolator by considering whether the FG task using GPU or not.
                        isolator = self._isolator_map[CPUFreqThrottleIsolator]
                        if isolator.is_max_level is True:
                            isolator = self._isolator_map[SchedIsolator]
                    elif self.foreground_workload.is_gpu_task == 0: # if fg is cpu task, ...
                        isolator = self._isolator_map[GPUFreqThrottleIsolator]
                        if isolator.is_max_level is True:
                            isolator = self._isolator_map[SchedIsolator]
                    """
                elif self._node_type == NodeType.CPU:
                    isolator = self._isolator_map[SchedIsolator]
            elif resource is ResourceType.CPU:
                logger.info(f'[contentious_resources_loop] resource: {resource}, diff_value: {diff_value}')
                continue
            else:
                raise NotImplementedError(f'Unknown ResourceType: {resource}')

            logger.info(f'[contentious_resources_loop] resource: {resource}, diff_value: {diff_value}, isolator: {isolator}')

            # Choose Workload Target (based on resource type & isolator)
            self._cur_isolator = isolator   # Update Current Isolator
            self.choose_isolation_target(resource, diff_value)
            #self.choose_isolation_target(resource)

            logger.info(f'[contentious_resources_loop] diff_value: {diff_value}')
            logger.info(f'[contentious_resources_loop] isolator.is_max_level: {isolator.is_max_level}')
            logger.info(f'[contentious_resources_loop] isolator.is_min_level: {isolator.is_min_level}')

            #if diff_value < 0 and not isolator.is_max_level or diff_value > 0 and not isolator.is_min_level:
            #self._cur_isolator = isolator
            if not (self._cur_isolator.alloc_target_wl is None and
                    self._cur_isolator.dealloc_target_wl is None):
                self._cur_isolator.cur_dominant_resource_cont = resource
                self._cur_isolator.perf_target_wl = self._perf_target_wl

                # TODO: Testing which dealloc function works properly
                #self._cur_isolator.dealloc_target_wl = self.choose_workload_for_dealloc_v1()
                #dealloc_target_wl = self.choose_workload_for_dealloc_v2(resource)
                #self._cur_isolator.dealloc_target_wl = dealloc_target_wl
                logger.info(f'===================================================')
                logger.info(f'Starting {self._cur_isolator.__class__.__name__}...')
                logger.info(f'Dominant Resource Contention: {self._cur_isolator.cur_dominant_resource_cont}')
                logger.info(f'Dealloc workload: [{isolator.dealloc_target_wl}]')
                logger.info(f'Alloc workload: [{isolator.alloc_target_wl}]')
                logger.info(f'Perf workload: [{isolator.perf_target_wl}]')
                logger.info(f'===================================================')
                #self._leftover = self._lc_wls | self._be_wls
                return True

        logger.debug('A new Isolator has not been selected')
        #self._leftover = self._lc_wls | self._be_wls
        return False

    # Matching res_type and sort_criteria
    # FIXME: The below function may be relocated to policy/base.py to be provided for making other policy
    def choose_isolation_target(self, res_type: ResourceType, diff_value: float) -> None:
        logger = logging.getLogger(__name__)

        # check `diff_value` of perf_target_wl
        if diff_value < 0:
            action = 'strengthen'
        else:   # diff_value >= 0 case
            action = 'weaken'

        # sort_metric[0] : strengthen criteria, sort_metric[1] : weaken criteria
        sort_metric = ["instr_diff"] * 2    # default
        pick_order = [True, False]          # default for pick_order (for
        if res_type is ResourceType.CPU:
            sort_metric = ["instr_diff"] * 2
        elif res_type is ResourceType.CACHE:
            sort_metric = ["llc_hr_diff"] * 2
        elif res_type is ResourceType.MEMORY:
            sort_metric = ["mem_bw", "mem_bw_diff"] # FIXME: NEED TESTING
        else:
            logger.info(f"[choose_isolation_target] Unknown resource type: {res_type}, "
                        f"sort_criteria is determined to instr_diff")

        if self._cur_isolator is not AffinityIsolator:
            pick_order = [True, False]
        elif self._cur_isolator is AffinityIsolator:
            pick_order = [False, True]

        logger.debug(f'[choose_isolation_target] res_type: {res_type}')
        logger.debug(f'[choose_isolation_target] pick_order: {pick_order}')
        logger.debug(f'[choose_isolation_target] sort_metric: {sort_metric}')
        # FIXME: In this code, there is an assumption that two workloads are always selected.
        # FIXME: How about two workloads? -> deciding the direction of `strengthen` and `weaken`
        #logger.debug(f'[choose_isolation_target] self._leftover: {self._leftover}, action: {action}')
        logger.debug(f'[choose_isolation_target] action: {action}')
        if not isinstance(self.cur_isolator, IdleIsolator):
            if action == 'strengthen':
                self.choosing_wl_for(objective="strengthen",
                                     sort_criteria=sort_metric[0],
                                     highest=pick_order[0])
                #self._update_leftover()
            elif action == 'weaken':
                self.choosing_wl_for(objective="weaken",
                                     sort_criteria=sort_metric[1],
                                     highest=pick_order[1])
                #self._update_leftover()

        logger.debug(f"[choose_isolation_target] self._cur_isolator: {self._cur_isolator}")
        logger.debug(f"[choose_isolation_target] alloc_target_wl: {self._cur_isolator.alloc_target_wl}")
        logger.debug(f"[choose_isolation_target] dealloc_target_wl: {self._cur_isolator.dealloc_target_wl}")
        logger.debug(f"[choose_isolation_target] perf_target_wl: {self._cur_isolator.perf_target_wl}")

    def _update_leftover(self) -> None:
        logger = logging.getLogger(__name__)
        logger.debug(f'[update_leftover] self._leftover: {self._leftover}')
        logger.debug(f'[update_leftover] self.cur_isolator.alloc_target_wl: {self.cur_isolator.alloc_target_wl}')
        logger.debug(f'[update_leftover] self.cur_isolator.dealloc_target_wl: {self.cur_isolator.dealloc_target_wl}')
        logger.debug(f'[update_leftover] self.cur_isolator.perf_target_wl: {self.cur_isolator.perf_target_wl}')
        if self.cur_isolator.alloc_target_wl is not None:
            self._leftover.remove(self.cur_isolator.alloc_target_wl)
        elif self.cur_isolator.dealloc_target_wl is not None:
            self._leftover.remove(self.cur_isolator.dealloc_target_wl)
