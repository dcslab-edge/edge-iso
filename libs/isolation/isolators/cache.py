# coding: UTF-8

import logging
from typing import Optional, Tuple, Set, Dict, List

from .. import ResourceType
from .base import Isolator
from ...metric_container.basic_metric import MetricDiff
from ...utils import ResCtrl, numa_topology
from ...workload import Workload


class CacheIsolator(Isolator):
    def __init__(self, latency_critical_wls: Set[Workload], best_effort_wls: Set[Workload]) -> None:
        super().__init__(latency_critical_wls, best_effort_wls)

        #self._prev_step: Optional[int] = None
        #self._cur_step: Optional[int] = None
        #self._stored_config: Optional[Tuple[int, int]] = None
        # initialize cur_steps[wl]
        # strengthen:
        # weaken:
        self._cur_steps: Dict[Workload, List[List[int]]] = dict()
        for wl in latency_critical_wls:
            llc_masks = wl.resctrl.get_llc_mask()
            self._cur_steps[wl] = ResCtrl.get_llc_bit_ranges_from_mask(llc_masks)
        for wl in best_effort_wls:
            llc_masks = wl.resctrl.get_llc_mask()
            self._cur_steps[wl] = ResCtrl.get_llc_bit_ranges_from_mask(llc_masks)

        self._all_wls = latency_critical_wls | best_effort_wls

        self._chosen_alloc: Optional[int] = None
        self._chosen_dealloc: Optional[int] = None
        self._cur_alloc: Optional[List[str]] = None
        self._cur_dealloc: Optional[List[str]] = None
        self._free_llc_ranges: List[List[int]] = [[0, 19], [0, 19]]
        # init `self._free_llc_ranges`
        for wl in latency_critical_wls:
            self._free_llc_ranges = ResCtrl.update_llc_ranges(self._free_llc_ranges,
                                                              self._cur_steps[wl],
                                                              op='-')
        for wl in best_effort_wls:
            self._free_llc_ranges = ResCtrl.update_llc_ranges(self._free_llc_ranges,
                                                              self._cur_steps[wl],
                                                              op='-')

        self._stored_config: Optional[Dict[Workload, List[str]]] = None
        #self._cpufreq_range = CPUDVFS.get_freq_range()

    def _get_metric_type_from(self, metric_diff: MetricDiff) -> float:
        #return metric_diff.llc_hit_ratio
        return metric_diff.llc_hit_ratio - metric_diff.diff_slack

    def _get_res_type_from(self) -> ResourceType:
        return ResourceType.CACHE

    def strengthen(self) -> 'CacheIsolator':
        """
        It deallocates LLC bits of `dealloc_target_wl` and add them to free llc bits
        :return:
        """
        logger = logging.getLogger(__name__)

        wl = self.dealloc_target_wl
        #self.sync_cur_steps()
        if wl is not None:
            logger.info(f"It can deallocate {wl.name}-{wl.pid}")
            logger.info(f"self.dealloc_target_wl : {wl}")

            self._chosen_dealloc = self.release_llc_bit(self._cur_steps[wl])
            logger.info(f"self._chosen_dealloc : {wl}")
            masks = ResCtrl.get_llc_mask_from_ranges(self._cur_steps[wl])
            # FIXME: chosen_dealloc is always in socket 0 (hard-coded)
            self._cur_dealloc = hex(int(masks[0], 16) ^ self._chosen_dealloc)   # Bitwise op: XOR
        elif wl is None:
            logger.info(f"There is no dealloc_target_wl. (No workload)")
            logger.info(f"self.dealloc_target_wl : {wl}")
            self._chosen_dealloc = None
            self._cur_dealloc = None

        return self

    def weaken(self) -> 'CacheIsolator':
        """
        It allocates free llc bits to `alloc_target_wl`
        :return:
        """
        logger = logging.getLogger(__name__)

        wl = self.alloc_target_wl
        #self.sync_cur_steps()
        if wl is not None:
            logger.info(f"It can allocate {wl.name}-{wl.pid}")
            logger.info(f"self.alloc_target_wl : {wl}")

            self._chosen_alloc = self.free_llc_bit(self._cur_steps[wl])
            logger.info(f"self._chosen_alloc : {wl}")
            # self._cur_steps[wl] : [[0, 19],[0, 19]], self._chosen_alloc: 1
            # FIXME: hard-coded for socket 0 (i.e., masks[0])
            masks = ResCtrl.get_llc_mask_from_ranges(self._cur_steps[wl])
            self._cur_alloc = hex(int(masks[0], 16) | self._chosen_alloc)   # Bitwise op: OR
        elif wl is None:
            logger.info(f"There is no alloc_target_wl. (No workload)")
            logger.info(f"self.alloc_target_wl : {wl}")
            self._chosen_alloc = None
            self._cur_alloc = None

        return self

    @property
    def is_max_level(self) -> bool:
        # FIXME: hard coded
        #return self._cur_step is not None and self._cur_step + ResCtrl.STEP >= ResCtrl.MAX_BITS
        logger = logging.getLogger(__name__)
        logger.info(f'[is_max_level] self.dealloc_target_wl: {self.dealloc_target_wl}')
        if self.dealloc_target_wl is None:
            return False
        else:
            ranges = self._cur_steps[self.dealloc_target_wl]
            s = ranges[0][0]
            e = ranges[0][1]
            cur_llc_bits = e - s + 1
            return cur_llc_bits - ResCtrl.STEP < ResCtrl.MIN_BITS

    @property
    def is_min_level(self) -> bool:
        # FIXME: hard coded
        #return self._cur_step is None or self._cur_step - ResCtrl.STEP < ResCtrl.MIN_BITS
        logger = logging.getLogger(__name__)
        logger.info(f'[is_min_level] self.alloc_target_wl: {self.alloc_target_wl}')
        # FIXME: self._available_cores is valid in the following code segment?
        # self.get_available_cores()
        if self.alloc_target_wl is None:
            return False
        else:
            return self.free_llc_bit(self._cur_steps[self.alloc_target_wl]) < 0

    def enforce(self) -> None:
        logger = logging.getLogger(__name__)

        # reset condition? when do we use this code?
        if len(self._cur_steps) is 0:
            logger.info('CAT off')
            self.reset()

        else:
            wls = [self.alloc_target_wl, self.dealloc_target_wl]
            for wl in wls:
                if wl is not None:
                    logger.info(f'{wl.name}\'s LLC range is '
                                f'{self._cur_steps[wl]}')
            for wl in wls:
                if wl is not None:
                    #masks = [ResCtrl.MIN_MASK, ResCtrl.MIN_MASK]
                    llc_masks = wl.resctrl.get_llc_mask()
                    masks = ResCtrl.get_llc_bit_ranges_from_mask(llc_masks)
                    wl.resctrl.assign_llc(*masks)

    def reset(self) -> None:
        masks = [ResCtrl.MIN_MASK] * (max(numa_topology.cur_online_nodes()) + 1)

        for be_wl in self._best_effort_wls:
            if be_wl.is_running:
                be_masks = masks.copy()
                be_masks[be_wl.cur_socket_id()] = ResCtrl.MAX_MASK
                be_wl.resctrl.assign_llc(*be_masks)

        for lc_wl in self._latency_critical_wls:
            if lc_wl.is_running:
                lc_masks = masks.copy()
                lc_masks[lc_wl.cur_socket_id()] = ResCtrl.MAX_MASK
                lc_wl.resctrl.assign_llc(*lc_masks)

    def store_cur_config(self) -> None:
        self._stored_config = self._cur_steps

    def load_cur_config(self) -> None:
        super().load_cur_config()
        self._cur_steps = self._stored_config
        self._stored_config = None

    def free_llc_bit(self, wl_llc_ranges: List[List[int]]) -> int:
        """
        It returns a `free_llc_bit' near to the llc_mask of wl_alloc_target
        :return: free llc it: non-negative integer (0~19), no free llc bit: -1
        """
        free_bit = -1
        # return free `llc_bit` near to `wl_llc_ranges`
        # FIXME: return llc_ranges in `socket 0`
        llc_range = wl_llc_ranges[0]
        s, e = llc_range[0], llc_range[1]
        candidate_bits = [s - ResCtrl.STEP, e + ResCtrl.STEP]   # ResCtrl.STEP == 1

        if candidate_bits[0] > 0 and candidate_bits[0] in self._free_llc_ranges[0]:
            free_bit = candidate_bits[0]
        elif candidate_bits[1] < 19 and candidate_bits[1] in self._free_llc_ranges[0]:
            free_bit = candidate_bits[1]

        return free_bit

    def release_llc_bit(self, wl_llc_ranges: List[List[int]]) -> int:
        release_bit = -1
        # return released `llc_bit` near to `wl_llc_ranges`
        # FIXME: return llc_ranges in `socket 0`
        llc_range = wl_llc_ranges[0]
        s, e = llc_range[0], llc_range[1]
        candidate_bits = [s, e]

        # wl_llc_ranges : dealloc_target_wl
        # alloc_target_wl :
        # FIXME: what if self.alloc_target_wl is None? \
        #  (Assuming that an LC workload can be determined to self.alloc_target_wl
        lc_wls: Set[Workload] = self._latency_critical_wls
        sorted_lc_wls = sorted(lc_wls, key=lambda x: x.calc_metric_diff().llc_hit_ratio, reverse=True)
        wl_alloc = sorted_lc_wls[0]

        alloc_target_llc_range = self._cur_steps[wl_alloc][0]
        a_s = alloc_target_llc_range[0]
        a_e = alloc_target_llc_range[1]
        if candidate_bits[0] > a_e:         # if wl occupies right side of `alloc_target_wl'
            release_bit = candidate_bits[0]
        elif candidate_bits[1] < a_s:       # if wl occupies left side of `alloc_target_wl'
            release_bit = candidate_bits[1]
        return release_bit

