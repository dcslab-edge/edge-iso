# coding: UTF-8

from enum import IntEnum
from pathlib import Path
from typing import Optional

from cpuinfo import cpuinfo


class NodeType(IntEnum):
    CPU = 0
    IntegratedGPU = 1
    DiscreteGPU = 2


class ArchType(IntEnum):
    X86_64 = 0
    AARCH64 = 1


class MachineChecker:
    @staticmethod
    def get_node_type() -> Optional[NodeType]:
        """

        :return: the node type which is either CPU or GPU node
        """
        node_type = None

        gpu_type = MachineChecker.get_gpu_type()
        if gpu_type is None:
            node_type = NodeType.CPU
        elif gpu_type == 'integrated':
            node_type = NodeType.IntegratedGPU
        elif gpu_type == 'discrete':
            node_type = NodeType.DiscreteGPU
        else:
            print("Unknown node type")

        return node_type

    @staticmethod
    def get_cpu_arch_type() -> Optional[ArchType]:
        arch_type = None

        cpu_type = MachineChecker._get_cpu_type()
        if cpu_type == 'x86_64':
            arch_type = ArchType.X86_64
        elif cpu_type == 'aarch64':
            arch_type = ArchType.AARCH64
        else:
            print("Unknown cpu arch type")

        return arch_type

    @staticmethod
    def _get_cpu_type() -> str:
        """

        :return: the CPU type which is either Intel (x86_64) or ARM (aarch64)
        """
        info = cpuinfo.get_cpu_info()
        cpu_arch = info['arch_string_raw']
        return cpu_arch

    @staticmethod
    def get_gpu_type() -> Optional[str]:
        """

        :return: the GPU type which is either integrated or discrete one
        """
        gpu_type = None
        tegra_path = Path('/proc/device-tree/compatible')
        if tegra_path.exists():
            with tegra_path.open() as fp:
                tegra_info = fp.readline()
                if 'tegra186' in tegra_info:
                    gpu_type = 'integrated'

        # TODO: Code for checking whether discrete GPU exist
        #gpu_type = 'discrete'

        return gpu_type
