# coding: UTF-8

import subprocess
from pathlib import Path
from typing import ClassVar, Iterable

from libs.utils.cgroup import CpuSet
from libs.utils.machine_type import ArchType, MachineChecker


class CPUDVFS:
    # FREQ_RANGE_INDEX : 0 ~ 11
    CPU_TYPE = MachineChecker.get_cpu_arch_type()
    # ARCH = 'desktop'
    FREQ_RANGE = list()
    JETSONTX2_CPU_FREQ_RANGE = [345600, 499200, 652800, 806400, 960000, 1113600, 1267200, 1420800,
                                1574400, 1728000, 1881600, 2035200]
    DESKTOP_CPU_FREQ_RANGE = [345600, 499200, 652800, 806400, 960000, 1113600, 1267200, 1420800,
                              1574400, 1728000, 1881600, 2035200] # SDC nodes
    #XEON_CPU_FREQ_RANGE = [1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000, 2000000,
    #                       2100000, 2101000]  # bc5 (Xeon) nodes

    # FIXME: DESKTOP_CPU_FREQ_RANGE should be initialized and changed! (SDC Node freq_driver is now intel_pstate..)
    if CPU_TYPE == ArchType.AARCH64:
        MIN: ClassVar[int] = JETSONTX2_CPU_FREQ_RANGE[0]
        MAX: ClassVar[int] = JETSONTX2_CPU_FREQ_RANGE[11]
        FREQ_RANGE = JETSONTX2_CPU_FREQ_RANGE
        MIN_IDX: ClassVar[int] = 0
        STEP_IDX: ClassVar[int] = 1  # STEP is defined with its index
        MAX_IDX: ClassVar[int] = 11
    elif CPU_TYPE == ArchType.X86_64:
        #MIN: ClassVar[int] = DESKTOP_CPU_FREQ_RANGE[0]     # SDC
        #MAX: ClassVar[int] = DESKTOP_CPU_FREQ_RANGE[11]    # SDC
        MIN: ClassVar[int] = int(Path('/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq').read_text()) # XEON
        MAX: ClassVar[int] = int(Path('/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq').read_text()) # XEON
        FREQ_RANGE = list(map(int, sorted(Path('/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies')
                                          .read_text().split())))
        MIN_IDX: ClassVar[int] = 0
        STEP_IDX: ClassVar[int] = 1  # STEP is defined with its index
        MAX_IDX: ClassVar[int] = 10

    def __init__(self, group_name):
        self._group_name: str = group_name
        self._cur_cgroup = CpuSet(self._group_name)

    @staticmethod
    def get_freq_range():
        return CPUDVFS.FREQ_RANGE

    def set_freq_cgroup(self, target_freq: int):
        """
        Set the frequencies to current cgroup cpusets
        :param target_freq: freq. to set to cgroup cpuset
        :return:
        """
        CPUDVFS.set_freq(target_freq, self._cur_cgroup.read_cpus())

    @staticmethod
    def set_freq(freq: int, cores: Iterable[int]) -> None:
        """
        Set the freq. to the specified cores
        :param freq: freq. to set
        :param cores:
        :return:
        """
        for core in cores:
            subprocess.run(args=('sudo', 'tee', f'/sys/devices/system/cpu/cpu{core}/cpufreq/scaling_max_freq'),
                           check=True, input=f'{freq}\n', encoding='ASCII', stdout=subprocess.DEVNULL)
