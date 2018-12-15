# coding: UTF-8

from statistics import mean
from typing import Iterable
from itertools import islice

from cpuinfo import cpuinfo
from ..utils.machine_type import MachineChecker, NodeType

NODE_TYPE = MachineChecker.get_node_type()

# LLC_SIZE = int(cpuinfo.get_cpu_info()['l3_cache_size'].split()[0]) * 1024  # Xeon Server (BC5) LLC (L3 Cache)
if NODE_TYPE == NodeType.IntegratedGPU:
    LLC_SIZE = int(cpuinfo.get_cpu_info()['l2_cache_size'].split()[0]) * 1024   # JETSON TX2 LLC (L2Cache)
elif NODE_TYPE == NodeType.CPU:
    LLC_SIZE = int(cpuinfo.get_cpu_info()['l3_cache_size'].split()[0]) * 1024  # Desktop (SDC) LLC (L3Cache)


class BasicMetric:
    def __init__(self, llc_references, llc_misses, inst, cycles,
                 gpu_core_util, gpu_core_freq, gpu_emc_util, gpu_emc_freq, interval):
        self._llc_references = llc_references
        self._llc_misses = llc_misses
        self._instructions = inst
        self._cycles = cycles
        self._gpu_core_util = gpu_core_util
        self._gpu_core_freq = gpu_core_freq
        self._gpu_emc_util = gpu_emc_util
        self._gpu_emc_freq = gpu_emc_freq
        self._interval = interval

    @classmethod
    def calc_avg(cls, metrics: Iterable['BasicMetric'], metric_num: int) -> 'BasicMetric':
            metrics = list(islice(metrics, 0, metric_num))
            return BasicMetric(
                    mean(metric._llc_references for metric in metrics),
                    mean(metric._llc_misses for metric in metrics),
                    mean(metric._instructions for metric in metrics),
                    mean(metric._cycles for metric in metrics),
                    mean(metric._gpu_core_util for metric in metrics),
                    mean(metric._gpu_core_freq for metric in metrics),
                    mean(metric._gpu_emc_util for metric in metrics),
                    mean(metric._gpu_emc_freq for metric in metrics),
                    mean(metric._interval for metric in metrics),
            )

    @property
    def llc_references(self):
        return self._llc_references

    @property
    def llc_misses(self):
        return self._llc_misses

    @property
    def gpu_core_util(self):
        return self._gpu_core_util

    @property
    def gpu_core_freq(self):
        return self._gpu_core_freq

    @property
    def gpu_mem_util(self):
        return self._gpu_emc_util

    @property
    def gpu_emc_freq(self):
        return self._gpu_emc_freq

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
    def ipc(self) -> float:
        return self._instructions / self._cycles

    @property
    def llc_miss_ratio(self) -> float:
        return self._llc_misses / self._llc_references if self._llc_references != 0 else 0

    @property
    def llc_hit_ratio(self) -> float:
        return 1 - self._llc_misses / self._llc_references if self._llc_references != 0 else 0

    def __repr__(self) -> str:
        return ', '.join(map(str, (
            self._llc_references, self._llc_misses, self._instructions, self._cycles,
            self._gpu_core_util, self._gpu_core_freq, self._gpu_emc_util, self._gpu_emc_freq,
            self._interval)))


class MetricDiff:
    # FIXME: hard coded (CPU -> SDC Node, In. GPU -> Jetson TX2 Node)
    if NODE_TYPE == NodeType.IntegratedGPU:
        _MAX_MEM_BANDWIDTH_PS = 50 * 1024 * 1024 * 1024     # MemBW specified in Jetson TX2 docs
    if NODE_TYPE == NodeType.CPU:
        _MAX_MEM_BANDWIDTH_PS = 24 * 1024 * 1024 * 1024     # MemBW measured by Intel VTune

    def __init__(self, curr: BasicMetric, prev: BasicMetric, core_norm: float = 1) -> None:
        self._llc_hit_ratio = curr.llc_hit_ratio - prev.llc_hit_ratio

        if curr.llc_miss_ps == 0:
            if prev.llc_miss_ps == 0:
                self._llc_miss_ps = 0
            else:
                self._llc_miss_ps = prev.llc_miss_ps / self._MAX_MEM_BANDWIDTH_PS
        elif prev.llc_miss_ps == 0:
            # TODO: is it fair?
            self._llc_miss_ps = -curr.llc_miss_ps / self._MAX_MEM_BANDWIDTH_PS
        else:
            self._llc_miss_ps = curr.llc_miss_ps / (prev.llc_miss_ps * core_norm) - 1

        self._instruction_ps = curr.instruction_ps / (prev.instruction_ps * core_norm) - 1

    @property
    def llc_hit_ratio(self) -> float:
        return self._llc_hit_ratio

    @property
    def local_mem_util_ps(self) -> float:
        return self._llc_miss_ps

    @property
    def instruction_ps(self) -> float:
        return self._instruction_ps

    def verify(self) -> bool:
        return self._llc_miss_ps <= 1 and self._instruction_ps <= 1

    def __repr__(self) -> str:
        return f'L3 hit ratio diff: {self._llc_hit_ratio:>6.03f}, ' \
               f'Local Memory access diff: {self._llc_miss_ps:>6.03f}, ' \
               f'Instructions per sec. diff: {self._instruction_ps:>6.03f}'
