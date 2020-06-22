# coding: UTF-8

from statistics import mean
from typing import Iterable, Tuple, Optional
from itertools import islice

from cpuinfo import cpuinfo
from ..utils.machine_type import MachineChecker, NodeType
from ..isolation import ResourceType

NODE_TYPE = MachineChecker.get_node_type()

# LLC_SIZE = int(cpuinfo.get_cpu_info()['l3_cache_size'].split()[0]) * 1024  # Xeon Server (BC5) LLC (L3 Cache)
if NODE_TYPE == NodeType.IntegratedGPU:
    LLC_SIZE = int(cpuinfo.get_cpu_info()['l2_cache_size'].split()[0]) * 1024   # JETSON TX2 LLC (L2Cache)
elif NODE_TYPE == NodeType.CPU:
    LLC_SIZE = int(cpuinfo.get_cpu_info()['l3_cache_size'].split()[0]) * 1024  # Desktop (SDC) LLC (L3Cache)

class BasicMetric:
    def __init__(self, llc_references, llc_misses, inst, cycles,
                 stall_cycles, intra_coh, inter_coh, wall_cycles, llc_size, local_mem, remote_mem, interval):
                 #gpu_core_util, gpu_core_freq, gpu_emc_util, gpu_emc_freq, interval):
        self._llc_references = llc_references
        self._llc_misses = llc_misses
        self._instructions = inst
        self._cycles = cycles
        self._stall_cycles = stall_cycles
        self._intra_coh = intra_coh
        self._inter_coh = inter_coh
        self._wall_cycles = wall_cycles
        self._llc_size = llc_size
        self._local_mem = local_mem
        self._remote_mem = remote_mem
        # self._gpu_core_util = gpu_core_util
        # self._gpu_core_freq = gpu_core_freq
        # self._gpu_emc_util = gpu_emc_util
        # self._gpu_emc_freq = gpu_emc_freq
        self._interval = interval

    @classmethod
    def calc_avg(cls, metrics: Iterable['BasicMetric'], metric_num: int) -> 'BasicMetric':
            metrics = list(islice(metrics, 0, metric_num))
            return BasicMetric(
                    mean(metric._llc_references for metric in metrics),
                    mean(metric._llc_misses for metric in metrics),
                    mean(metric._instructions for metric in metrics),
                    mean(metric._cycles for metric in metrics),
                    mean(metric._stall_cycles for metric in metrics),
                    mean(metric._intra_coh for metric in metrics),
                    mean(metric._inter_coh for metric in metrics),
                    mean(metric._wall_cycles for metric in metrics),
                    mean(metric._llc_size for metric in metrics),
                    mean(metric._local_mem for metric in metrics),
                    mean(metric._remote_mem for metric in metrics),
                    #mean(metric._gpu_core_util for metric in metrics),
                    #mean(metric._gpu_core_freq for metric in metrics),
                    #mean(metric._gpu_emc_util for metric in metrics),
                    #mean(metric._gpu_emc_freq for metric in metrics),
                    mean(metric._interval for metric in metrics),
            )

    @property
    def llc_references(self):
        return self._llc_references

    @property
    def llc_misses(self):
        return self._llc_misses

    # @property
    # def gpu_core_util(self):
    #     return self._gpu_core_util
    #
    # @property
    # def gpu_core_freq(self):
    #     return self._gpu_core_freq
    #
    # @property
    # def gpu_mem_util(self):
    #     return self._gpu_emc_util
    #
    # @property
    # def gpu_emc_freq(self):
    #     return self._gpu_emc_freq

    @property
    def llc_miss_ps(self) -> float:
        return self._llc_misses * (1000 / self._interval)

    @property
    def instruction(self):
        return self._instructions

    @property
    def instruction_ps(self):
        return self._instructions * (1000 / self._interval)

    @property
    def cycles(self):
        return self._cycles

    @property
    def stall_cycles(self):
        return self._stall_cycles

    @property
    def wall_cycles(self):
        return self._wall_cycles

    @property
    def intra_coh(self):
        return self._intra_coh

    @property
    def inter_coh(self):
        return self._inter_coh

    @property
    def llc_size(self):
        return self._llc_size

    @property
    def local_mem(self):
        return self._local_mem

    @property
    def local_mem_ps(self) -> float:
        return self._local_mem * (1000/ self._interval)

    @property
    def remote_mem(self):
        return self._remote_mem

    @property
    def remote_mem_ps(self) -> float:
        return self._remote_mem * (1000/ self._interval)

    @property
    def ipc(self) -> float:
        return self._instructions / self._cycles

    @property
    def intra_coh_ratio(self) -> float:
        return self._intra_coh / self._llc_references if self._llc_references != 0 else 0

    @property
    def inter_coh_ratio(self) -> float:
        return self._inter_coh / self._llc_references if self._llc_references != 0 else 0

    @property
    def coh_ratio(self) -> float:
        return (self._inter_coh + self._intra_coh) / self._llc_references if self._llc_references != 0 else 0

    @property
    def llc_miss_ratio(self) -> float:
        return self._llc_misses / self._llc_references if self._llc_references != 0 else 0

    @property
    def llc_hit_ratio(self) -> float:
        return 1 - self._llc_misses / self._llc_references if self._llc_references != 0 else 0

    #def __repr__(self) -> str:
    #    return ', '.join(map(str, (
    #        self._llc_references, self._llc_misses, self._instructions, self._cycles,
    #        self._gpu_core_util, self._gpu_core_freq, self._gpu_emc_util, self._gpu_emc_freq,
    #        self._interval)))
    def __repr__(self) -> str:
        return ', '.join(map(str, (
            self._llc_references, self._llc_misses, self ._instructions, self._cycles, self._stall_cycles,
            self._intra_coh, self._inter_coh, self._wall_cycles, self._llc_size, self._local_mem, self._remote_mem,
            self._interval)))


class MetricDiff:
    # FIXME: hard coded (CPU -> SDC Node, In. GPU -> Jetson TX2 Node)
    if NODE_TYPE == NodeType.IntegratedGPU:
        _MAX_MEM_BANDWIDTH_PS = 50 * 1024 * 1024 * 1024     # MemBW specified in Jetson TX2 docs
    if NODE_TYPE == NodeType.CPU:
        #_MAX_MEM_BANDWIDTH_PS = 24 * 1024 * 1024 * 1024     # MemBW measured by Intel VTune
        _MAX_MEM_BANDWIDTH_PS = 68 * 1024 * 1024 * 1024     # MemBW of Xeon Server

    def __init__(self, curr: BasicMetric, prev: BasicMetric, core_norm: float = 1, diff_slack: float = 0.0) -> None:
        self._diff_slack = diff_slack
        self._llc_hit_ratio = curr.llc_hit_ratio - prev.llc_hit_ratio

        if curr.local_mem_ps == 0:
            if prev.local_mem_ps == 0:
                self._local_mem_ps = 0
            else:
                self._local_mem_ps = prev.local_mem_ps / self._MAX_MEM_BANDWIDTH_PS
        elif prev.local_mem_ps == 0:
            # TODO: is it fair?
            self._local_mem_ps = -curr.local_mem_ps / self._MAX_MEM_BANDWIDTH_PS
        else:
            self._local_mem_ps = curr.local_mem_ps / (prev.local_mem_ps * core_norm) - 1

        # NOTE: using `local_mem_ps` instead of `llc_miss_ps`
        # if curr.llc_miss_ps == 0:
        #     if prev.llc_miss_ps == 0:
        #         self._llc_miss_ps = 0
        #     else:
        #         self._llc_miss_ps = prev.llc_miss_ps / self._MAX_MEM_BANDWIDTH_PS
        # elif prev.llc_miss_ps == 0:
        #     # TODO: is it fair?
        #     self._llc_miss_ps = -curr.llc_miss_ps / self._MAX_MEM_BANDWIDTH_PS
        # else:
        #     self._llc_miss_ps = curr.llc_miss_ps / (prev.llc_miss_ps * core_norm) - 1

        self._instruction_ps = curr.instruction_ps / (prev.instruction_ps * core_norm) - 1

    @property
    def llc_hit_ratio(self) -> float:
        return self._llc_hit_ratio

    @property
    def local_mem_util_ps(self) -> float:
        return self._local_mem_ps
        #return self._llc_miss_ps

    @property
    def instruction_ps(self) -> float:
        return self._instruction_ps

    @property
    def diff_slack(self) -> float:
        return self._diff_slack

    def calc_by_diff_slack(self, diff_slack: Optional[float]) -> Tuple[Tuple[ResourceType, float], ...]:
        # NOTE: diff_slack is positive float value
        resource_slacks = ()

        orig_diff = ((ResourceType.CPU, self._instruction_ps),
                     (ResourceType.CACHE, self._llc_hit_ratio),
                     (ResourceType.MEMORY, self._local_mem_ps))     # NOTE: Changed to local_mem_ps

        # Explicitly re-assigning diff_slack
        if diff_slack is not None:
            self._diff_slack = diff_slack

        # Calculating slack from pre-defined diff value
        # `resource_slacks` contains all diffs re-calculated using `diff_slack`
        # diff_slack > 0: making more sensitive to the contention
        # diff_slack < 0: making less sensitive to the contention
        for res, val in orig_diff:
            resource_slacks += ((res, val - self._diff_slack),)

        return resource_slacks

    def verify(self) -> bool:
        return self._local_mem_ps <= 1 and self._instruction_ps <= 1

    def __repr__(self) -> str:
        return f'L3 hit ratio diff: {self._llc_hit_ratio:>6.03f}, ' \
               f'Local Memory access diff: {self._local_mem_ps:>6.03f}, ' \
               f'Instructions per sec. diff: {self._instruction_ps:>6.03f}'
