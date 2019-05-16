# coding: UTF-8

import subprocess
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
    # FIXME: DESKTOP_CPU_FREQ_RANGE should be initialized and changed! (SDC Node freq_driver is now intel_pstate..)
    MIN_IDX: ClassVar[int] = 0
    STEP_IDX: ClassVar[int] = 1  # STEP is defined with its index
    MAX_IDX: ClassVar[int] = 11
    if CPU_TYPE == ArchType.AARCH64:
        MIN: ClassVar[int] = JETSONTX2_CPU_FREQ_RANGE[0]
        MAX: ClassVar[int] = JETSONTX2_CPU_FREQ_RANGE[11]
        FREQ_RANGE = JETSONTX2_CPU_FREQ_RANGE
    elif CPU_TYPE == ArchType.X86_64:
        MIN: ClassVar[int] = DESKTOP_CPU_FREQ_RANGE[0]
        MAX: ClassVar[int] = DESKTOP_CPU_FREQ_RANGE[11]
        FREQ_RANGE = DESKTOP_CPU_FREQ_RANGE

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
