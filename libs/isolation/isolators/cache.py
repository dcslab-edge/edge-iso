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
        logger = logging.getLogger(__name__)
        #self._prev_step: Optional[int] = None
        #self._cur_step: Optional[int] = None
        #self._stored_config: Optional[Tuple[int, int]] = None
        # initialize cur_steps[wl]
        # strengthen:
        # weaken:
        # FIXME: assign first 1/N of llc to latency critical workloads
        self._cur_steps: Dict[Workload, List[List[int]]] = dict()
        all_wls = latency_critical_wls | best_effort_wls
        num_all_wls = len(all_wls)
        num_lc_wls = len(latency_critical_wls)
        num_be_wls = len(best_effort_wls)

        fair_bits = ResCtrl.MAX_BITS // num_all_wls     # f_b == 20 // 2 == 10
        # FIXME: hard-coded for heracles algorithm (LC assignment 18 ways)
        lc_start_bit = ResCtrl.MIN_BITS - 1
        lc_end_bit = 17
        #lc_start_bit = ResCtrl.MIN_BITS - 1
        #lc_end_bit = fair_bits * num_lc_wls - 1         # (20//2) == 10 == 10 -> idx:9

        logger.critical(f'[__init__] num_all_wls: {num_all_wls}, num_lc_wls: {num_lc_wls}, num_be_wls: {num_be_wls}')
        logger.critical(f'[__init__] latency_critical_wls: {latency_critical_wls}')
        for idx, wl in enumerate(latency_critical_wls):
            if idx > 1:
                lc_start_bit = lc_end_bit + 1           # 10
                lc_end_bit = lc_end_bit + fair_bits     # 9+10=19
                logger.critical(f'[__init__] wl: {wl.name}, '
                                f'lc_start_bit: {lc_start_bit}, lc_end_bit: {lc_end_bit}, fair_bits: {fair_bits}')
            llc_masks = wl.resctrl.get_llc_mask()
            #self._cur_steps[wl] = ResCtrl.get_llc_bit_ranges_from_mask(llc_masks)
            # FIXME: hard-coded assignment for single latency-critical workload case
            llc_ranges = ResCtrl.get_llc_bit_ranges_from_mask(llc_masks)
            #logger.critical(f'[__init__] wl: {wl.name}, llc_masks: {llc_masks}, llc_ranges: {llc_ranges}')
            llc_ranges[0][0] = int(lc_start_bit)
            llc_ranges[0][1] = int(lc_end_bit)
            updated_masks = ResCtrl.get_llc_mask_from_ranges(llc_ranges)
            logger.debug(f'[__init__] wl: {wl.name}, llc_masks: {llc_masks}, llc_ranges: {llc_ranges}, updated_masks: {updated_masks}')
            self._cur_steps[wl] = ResCtrl.get_llc_bit_ranges_from_mask(updated_masks)

        # FIXME: hard-coded for heracles algorithm (BE assignment two ways)
        #be_start_bit = lc_end_bit + 1
        #be_end_bit = be_start_bit + fair_bits - 1
        be_start_bit = 18
        be_end_bit = be_start_bit + 1
        logger.debug(f'[__init__] lc_end_bit: {lc_end_bit}, be_start_bit: {be_start_bit}, fair_bits: {fair_bits}')
        for idx, wl in enumerate(best_effort_wls):
            if idx > 1:
                be_start_bit = lc_end_bit + 1
                be_end_bit = be_start_bit + fair_bits
                logger.debug(f'[__init__] wl: {wl.name}, '
                                f'be_start_bit: {be_start_bit}, be_end_bit: {be_end_bit}, fair_bits: {fair_bits}')
            llc_masks = wl.resctrl.get_llc_mask()
            llc_ranges = ResCtrl.get_llc_bit_ranges_from_mask(llc_masks)
            #logger.critical(f'[__init__] wl: {wl.name}, llc_masks: {llc_masks}, llc_ranges: {llc_ranges}')
            llc_ranges[0][0] = int(be_start_bit)
            llc_ranges[0][1] = int(be_end_bit)
            updated_masks = ResCtrl.get_llc_mask_from_ranges(llc_ranges)
            logger.debug(f'[__init__] wl: {wl.name}, llc_masks: {llc_masks}, llc_ranges: {llc_ranges}, updated_masks: {updated_masks}')
            self._cur_steps[wl] = ResCtrl.get_llc_bit_ranges_from_mask(updated_masks)
            #be_start_bit = lc_end_bit + 1
            #be_end_bit = be_start_bit + fair_bits

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
        logger.critical(f"[cache:strengthen] self.dealloc_target_wl : {wl}")
        #self.sync_cur_steps()
        if wl is not None:
            logger.critical(f"[cache:strengthen] It can deallocate {wl.name}-{wl.pid}")
            logger.critical(f"[cache:strengthen] self.dealloc_target_wl : {wl}")
            logger.critical(f"[cache:strengthen] self._cur_steps[wl] : {self._cur_steps[wl]}")

            self._chosen_dealloc = self.release_llc_bit(self._cur_steps[wl])
            #logger.critical(f'self._chosen_dealloc: {self._chosen_dealloc}')
            if self._chosen_dealloc >= 0:
                chosen_dealloc_bin = format(1 << (19 - self._chosen_dealloc), '020b')
            else:
                chosen_dealloc_bin = format(0, '020b')
            logger.info(f"[strengthen:cache] self._chosen_dealloc : {self._chosen_dealloc}, chosen_dealloc_binary: {chosen_dealloc_bin},"
                            f" self.dealloc_target_wl: {wl.name}")
            masks = ResCtrl.get_llc_mask_from_ranges(self._cur_steps[wl])
            # FIXME: chosen_dealloc is always in socket 0 (hard-coded)
            masks[0] = hex(int(masks[0], 16) ^ int(chosen_dealloc_bin, 2))   # Bitwise op: XOR
            self._cur_dealloc = masks

            logger.critical(f"[strengthen:cache] self._cur_dealloc : {self._cur_dealloc}, self.dealloc_target_wl: {wl.name}")
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
        logger.critical(f"[cache:weaken] self.alloc_target_wl : {wl}")
        if wl is not None:
            logger.critical(f"[cache:weaken] It can allocate {wl.name}-{wl.pid}")
            logger.critical(f"[cache:weaken] self.alloc_target_wl : {wl}")
            logger.critical(f"[cache:weaken] self._cur_steps[wl] : {self._cur_steps[wl]}")

            self._chosen_alloc = self.get_free_llc_bit(self._cur_steps[wl])
            logger.critical(f'[cache:weaken] self._chosen_alloc: {self._chosen_alloc}')
            if self._chosen_alloc >= 0:
                chosen_alloc_bin = format(1 << (19 - self._chosen_alloc), '020b')
            else:
                chosen_alloc_bin = format(0, '020b')
            #logger.info(f"self._chosen_alloc : {self._c}")
            logger.info(f"[cache:weaken] self._chosen_alloc : {self._cur_alloc}, chosen_alloc_binary: {chosen_alloc_bin}, "
                            f"self.alloc_target_wl: {wl.name}")
            # self._cur_steps[wl] : [[0, 19],[0, 19]], self._chosen_alloc: 1
            # FIXME: hard-coded for socket 0 (i.e., masks[0])
            masks = ResCtrl.get_llc_mask_from_ranges(self._cur_steps[wl])
            masks[0] = hex(int(masks[0], 16) | int(chosen_alloc_bin, 2))
            self._cur_alloc = masks   # Bitwise op: OR

            logger.critical(f"[cache:weaken] self._cur_alloc : {self._cur_alloc}, self.alloc_target_wl: {wl.name}")
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
            deallocable_ranges = self._cur_steps[self.dealloc_target_wl]
            deallocable_ranges_len = deallocable_ranges[0][1] - deallocable_ranges[0][0]
            return self.get_free_llc_bit(self._cur_steps[self.alloc_target_wl]) < 0 and deallocable_ranges_len == 1
            #return self.get_free_llc_bit(self._cur_steps[self.alloc_target_wl]) < 0     # free bit이 없으면, alloc할 수 없으니 min_level True?

    def enforce(self) -> None:
        logger = logging.getLogger(__name__)

        # reset condition? when do we use this code?
        if len(self._cur_steps) is 0:
            logger.info('CAT off')
            self.reset()

        else:
            wls = [self.alloc_target_wl, self.dealloc_target_wl]
            #for wl in wls:
            #    if wl is not None:
            #        logger.critical(f'[enforce:cache_partition][HW] {wl.name}\'s LLC range is '
            #                    f'{self._cur_steps[wl]}')

            if self._cur_alloc is not None and self.alloc_target_wl is not None:
                logger.critical(f'[enforce:cache_partition][HW] '
                                f'{self.alloc_target_wl.name}-{self.alloc_target_wl.pid}\'s'
                                f' LLC range is {self._cur_alloc}')
                self.alloc_target_wl.resctrl.assign_llc(*self._cur_alloc)
                self._cur_steps[self.alloc_target_wl] = ResCtrl.get_llc_bit_ranges_from_mask(self._cur_alloc)
                self._update_other_values("alloc")

            elif self._cur_dealloc is not None and self.dealloc_target_wl is not None:
                logger.critical(f'[enforce:cache_partition][HW] '
                                f'{self.dealloc_target_wl.name}-{self.dealloc_target_wl.pid}\'s'
                                f' LLC range is {self._cur_dealloc}')
                self.dealloc_target_wl.resctrl.assign_llc(*self._cur_dealloc)
                self._cur_steps[self.dealloc_target_wl] = ResCtrl.get_llc_bit_ranges_from_mask(self._cur_dealloc)
                self._update_other_values("dealloc")

    def reset(self) -> None:
        logger = logging.getLogger(__name__)
        logger.critical(f'[reset] CacheIsolator reset!')
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

    def get_free_llc_bit(self, wl_llc_ranges: List[List[int]]) -> int:
        """
        It returns a `free_llc_bit' near to the llc_mask of wl_alloc_target
        :return: free llc it: non-negative integer (0~19), no free llc bit: -1
        """
        logger = logging.getLogger(__name__)

        free_bit = -1
        # return free `llc_bit` near to `wl_llc_ranges`
        # FIXME: return llc_ranges in `socket 0`
        llc_range = wl_llc_ranges[0]
        s, e = llc_range[0], llc_range[1]
        candidate_bits = [s - ResCtrl.STEP, e + ResCtrl.STEP]   # ResCtrl.STEP == 1
        logger.critical(f'[get_free_llc_bit] self._cur_steps: {self._cur_steps}')
        logger.critical(f'[get_free_llc_bit] self._free_llc_range[0]: {self._free_llc_ranges[0]}, candidate_bits: {candidate_bits}')

        if candidate_bits[0] > 0 and candidate_bits[0] in self._free_llc_ranges[0]:
            free_bit = candidate_bits[0]
        elif candidate_bits[1] < 19 and candidate_bits[1] in self._free_llc_ranges[0]:
            free_bit = candidate_bits[1]

        logger.critical(f'[get_free_llc_bit] free_bit: {free_bit}')
        return free_bit

    def release_llc_bit(self, wl_llc_ranges: List[List[int]]) -> int:
        logger = logging.getLogger(__name__)

        release_bit = -1
        # return released `llc_bit` near to `wl_llc_ranges`
        # FIXME: return llc_ranges in `socket 0`
        llc_range = wl_llc_ranges[0]
        s, e = llc_range[0], llc_range[1]
        candidate_bits = [s, e]                 # dealloc workload's bits

        # wl_llc_ranges : dealloc_target_wl
        # alloc_target_wl :
        # FIXME: what if self.alloc_target_wl is None? \
        #  (Assuming that an LC workload can be determined to self.alloc_target_wl
        lc_wls = set()
        logger.critical(f'[release_llc_bit] self._cur_steps: {self._cur_steps}')
        for wl, _ in self._cur_steps.items():
            if wl.wl_type == "LC":
                lc_wls.add(wl)
        #lc_wls: Set[Workload] = self._latency_critical_wls
        logger.critical(f'[release_llc_bit] lc_wls: {lc_wls}')
        wl_alloc = self.alloc_target_wl
        if wl_alloc is not None:
            # FIXME: hard-coded for two workload scenario (1 LC , 1 BE)
            #sorted_lc_wls = sorted(lc_wls, key=lambda x: x.calc_metric_diff().llc_hit_ratio, reverse=True)
            #sorted_lc_wls = list(lc_wls)
            #logger.critical(f'[release_llc_bit] sorted_lc_wls: {sorted_lc_wls}')
            #wl_alloc = sorted_lc_wls[0]
            #wl_alloc = self.alloc_target_wl

            alloc_target_llc_range = self._cur_steps[wl_alloc][0]
            a_s = alloc_target_llc_range[0]
            a_e = alloc_target_llc_range[1]

            logger.critical(f'[release_llc_bit] self._cur_steps: {self._cur_steps}')
            logger.critical(f'[release_llc_bit] alloc_target_llc_range(a_s, a_e): {alloc_target_llc_range}, wl_alloc: {wl_alloc}')
            logger.critical(f'[release_llc_bit] candidate_bits: {candidate_bits}')

            if candidate_bits[0] >= a_e:         # if wl occupies right side of `alloc_target_wl'
                release_bit = candidate_bits[0]
            elif candidate_bits[1] <= a_s:       # if wl occupies left side of `alloc_target_wl'
                release_bit = candidate_bits[1]

        return release_bit

    def _update_other_values(self, action: str) -> None:
        logger = logging.getLogger(__name__)
        if action is "alloc":
            updated_free_masks = []
            logger.critical(f'[_update_other_values] self._free_llc_ranges: {self._free_llc_ranges}')
            free_masks = ResCtrl.get_llc_mask_from_ranges(self._free_llc_ranges)
            logger.critical(f'[_update_other_values] free_masks: {free_masks}')
            logger.critical(f'[_update_other_values] self._chosen_alloc: {self._chosen_alloc}')
            #logger.critical(f'[_update_other_values] self._cur_alloc: {self._cur_alloc}')
            if self._chosen_alloc >= 0:
                chosen_alloc_bin = format(1 << (19 - self._chosen_alloc), '020b')
            else:
                chosen_alloc_bin = format(0, '020b')
            logger.critical(f'[_update_other_values] self._cur_alloc: {self._cur_alloc}')
            for idx, mask in enumerate(free_masks):     # 111 ^ 101 -> 010
                if idx == 0:
                    updated_mask = hex(int(mask, 16) ^ int(chosen_alloc_bin, 2))
                    updated_free_masks.append(updated_mask)
                elif idx > 0:
                    chosen_alloc_bin = format(0, '020b')
                    updated_mask = hex(int(mask, 16) ^ int(chosen_alloc_bin, 2))
                    updated_free_masks.append(updated_mask)

            logger.critical(f'[_update_other_values] updated_free_masks: {updated_free_masks}')
            self._free_llc_ranges = ResCtrl.get_llc_bit_ranges_from_mask(updated_free_masks)
            self._cur_alloc = None
            self._chosen_alloc = None
        elif action is "dealloc":
            updated_free_masks = []
            logger.critical(f'[_update_other_values] self._free_llc_ranges: {self._free_llc_ranges}')
            free_masks = ResCtrl.get_llc_mask_from_ranges(self._free_llc_ranges)
            logger.critical(f'[_update_other_values] free_masks: {free_masks}')
            logger.critical(f'[_update_other_values] self._chosen_dealloc: {self._chosen_dealloc}')
            if self._chosen_dealloc >= 0:
                chosen_dealloc_bin = format(1 << (19 - self._chosen_dealloc), '020b')
            else:
                chosen_dealloc_bin = format(0, '020b')
            logger.critical(f'[_update_other_values] self._cur_dealloc: {self._cur_dealloc}')
            for idx, mask in enumerate(free_masks):     # 010 | 101 -> 111 , 000 ^ 101 -> 101
                # FIXME: hard-coded for socket 0
                if idx == 0:
                    updated_mask = hex(int(mask, 16) | int(chosen_dealloc_bin, 2))
                    updated_free_masks.append(updated_mask)
                elif idx > 0:
                    chosen_dealloc_bin = format(0, '020b')
                    updated_mask = hex(int(mask, 16) | int(chosen_dealloc_bin, 2))
                    updated_free_masks.append(updated_mask)

            logger.critical(f'[_update_other_values] updated_free_masks: {updated_free_masks}')
            self._free_llc_ranges = ResCtrl.get_llc_bit_ranges_from_mask(updated_free_masks)
            self._cur_dealloc = None
            self._chosen_dealloc = None
