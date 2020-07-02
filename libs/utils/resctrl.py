# coding: UTF-8

import logging

import re
import subprocess
from pathlib import Path
from typing import ClassVar, List, Pattern, Tuple, Dict


def len_of_mask(mask: str) -> int:
    cnt = 0
    num = int(mask, 16)
    while num is not 0:
        cnt += 1
        num >>= 1
    return cnt


def bits_to_mask(bits: int) -> str:
    return f'{bits:x}'


class ResCtrl:
    MOUNT_POINT: ClassVar[Path] = Path('/sys/fs/resctrl')
    MAX_MASK: ClassVar[str] = Path('/sys/fs/resctrl/info/L3/cbm_mask').read_text(encoding='ASCII').strip()
    MAX_BITS: ClassVar[int] = len_of_mask((MOUNT_POINT / 'info' / 'L3' / 'cbm_mask').read_text())
    MIN_BITS: ClassVar[int] = int((MOUNT_POINT / 'info' / 'L3' / 'min_cbm_bits').read_text())
    MIN_MASK: ClassVar[str] = bits_to_mask(MIN_BITS)
    STEP: ClassVar[int] = 1
    _read_regex: ClassVar[Pattern] = re.compile(r'L3:((\d+=[0-9a-fA-F]+;?)*)', re.MULTILINE)

    def __init__(self, group_name: str) -> None:
        self._group_name: str = group_name
        self._group_path: Path = ResCtrl.MOUNT_POINT / f'{group_name}'

    @property
    def group_name(self):
        return self._group_name

    @group_name.setter
    def group_name(self, new_name):
        self._group_name = new_name
        self._group_path: Path = ResCtrl.MOUNT_POINT / new_name

    def add_task(self, pid: int) -> None:
        subprocess.run(args=('sudo', 'tee', str(self._group_path / 'tasks')),
                       input=f'{pid}\n', check=True, encoding='ASCII', stdout=subprocess.DEVNULL)

    def assign_llc(self, *masks: str) -> None:
        logger = logging.getLogger(__name__)
        masks = (f'{i}={mask}' for i, mask in enumerate(masks))
        mask = ';'.join(masks)
        # subprocess.check_call('ls -ll /sys/fs/resctrl/', shell=True)
        logger.debug(f'[assign_llc] mask: {mask}')
        subprocess.run(args=('sudo', 'tee', str(self._group_path / 'schemata')),
                       input=f'L3:{mask}\n', check=True, encoding='ASCII', stdout=subprocess.DEVNULL)

    def read_assigned_llc(self) -> Tuple[int, ...]:
        schemata = self._group_path / 'schemata'
        if not schemata.is_file():
            raise ProcessLookupError()

        with schemata.open() as fp:
            content: str = fp.read().strip()

        l3_schemata = ResCtrl._read_regex.search(content).group(1)

        # example: [('0', '00fff'), ('1', 'fff00')]
        pairs: List[Tuple[str, str]] = sorted(tuple(pair.split('=')) for pair in l3_schemata.split(';'))
        return tuple(len_of_mask(mask) for socket, mask in pairs)

    @staticmethod
    def gen_mask(start: int, end: int = None) -> str:
        if end is None or end > ResCtrl.MAX_BITS:
            end = ResCtrl.MAX_BITS

        if start < 0:
            raise ValueError('start must be greater than 0')

        return format(((1 << (end - start)) - 1) << (ResCtrl.MAX_BITS - end), 'x')

    def remove_group(self) -> None:
        subprocess.check_call(args=('sudo', 'rmdir', str(self._group_path)))

    def get_llc_mask(self) -> List[str]:
        """
        :return: `socket_masks` which is the elements of list in hex_str
        """
        proc = subprocess.Popen(['cat', f'{ResCtrl.MOUNT_POINT}/{self._group_name}/schemata'],
                                stdout=subprocess.PIPE)
        line = proc.communicate()[0].decode().lstrip()
        striped_schema_line = line.lstrip('L3:').rstrip('\n').split(';')
        socket_masks = list()
        for i, item in enumerate(striped_schema_line):
            mask = item.lstrip(f'{i}=')
            socket_masks.append(mask)
        return socket_masks

    @staticmethod
    def get_llc_bits_from_mask(masks: List[str]) -> List[int]:
        """
        :param masks: Assuming the elements of list is hex_str such as "0xfffff"
        :return:
        """
        output_list = list()
        for mask in masks:
            hex_str = mask
            #print(f'hex_str: {hex_str}')
            hex_int = int(hex_str, 16)
            #print(f'hex_int: {hex_int}')
            bin_tmp = bin(hex_int)
            #print(f'bin_tmp: {bin_tmp}, type: {type(bin_tmp)}')
            llc_bits = len(bin_tmp.lstrip('0b'))
            #print(f'llc_bits: {llc_bits}')
            output_list.append(llc_bits)
        return output_list

    def read_llc_bits(self) -> int:
        socket_masks = self.get_llc_mask()
        llc_bits_list = ResCtrl.get_llc_bits_from_mask(socket_masks)
        ret_llc_bits = 0
        for llc_bits in llc_bits_list:
            ret_llc_bits += llc_bits
        return ret_llc_bits

    @staticmethod
    def get_llc_bit_ranges_from_mask(masks: List[str]) -> List[List[int]]:
        """
        :param masks: ["0xfffff","0xfffff"]
        :return: output_list: [[0,19], [0,19]] ; llc bit ranges for all sockets
        """
        logger = logging.getLogger(__name__)
        output_list = list()
        for mask in masks:
            hex_str = mask
            #print(f'hex_str: {hex_str}')
            hex_int = int(hex_str, 16)
            logger.debug(f'[get_llc_bit_ranges_from_mask] hex_int: {hex_int}')
            #print(f'hex_int: {hex_int}')
            #bin_tmp = bin(hex_int).lstrip('0b')
            bin_tmp = format(hex_int, '020b')
            logger.debug(f'[get_llc_bit_ranges_from_mask] bin_tmp: {bin_tmp}')
            s = bin_tmp.find('1')
            e = bin_tmp.rfind('1')
            llc_range = [s, e]
            logger.debug(f'[get_llc_bit_ranges_from mask] llc_range: {llc_range}')
            output_list.append(llc_range)
        return output_list

    @staticmethod
    def update_llc_ranges(in1: List[List[int]],
                          in2: List[List[int]],
                          op: str) -> List[List[int]]:
        """
        It performs operations for two llc_ranges
        :param in1: 1st llc_ranges
        :param in2: 2nd llc_ranges
        :param op:  operation type
        :return: 1st llc_ranges (op) 2nd llc_ranges
        """

        logger = logging.getLogger(__name__)

        out_list = []
        new_llc_range = int('1'*20)
        #logger.critical(f'in1: {in1}, in2: {in2}')

        for idx in range(len(in1)):
            r_in1 = in1[idx]
            r_in2 = in2[idx]

            s1 = r_in1[0]
            e1 = r_in1[1]
            len_in1 = e1 - s1 + 1
            llc_range_in1 = '0'*s1 + '1'*len_in1 + '0'*(19-e1)

            s2 = r_in2[0]
            e2 = r_in2[1]
            len_in2 = e2 - s2 + 1
            llc_range_in2 = '0'*s2 + '1'*len_in2 + '0'*(19-e2)

            #logger.critical(f'[update_llc_ranges] llc_range_in1: {llc_range_in1},'
            #                f'llc_range_in2: {llc_range_in2}, op: {op}')
            if op == '+':
                new_llc_range = int(llc_range_in1, 2) | int(llc_range_in2, 2)
            elif op == '-':
                overlapped_range = int(llc_range_in1, 2) & int(llc_range_in2, 2)
                new_llc_range = int(llc_range_in1, 2) ^ overlapped_range

            #logger.critical(f'[update_llc_ranges] new_llc_range: {hex(new_llc_range)}')

            new_s = format(new_llc_range, '020b').find('1')
            new_e = format(new_llc_range, '020b').rfind('1')
            #logger.critical(f'[update_llc_ranges] new_s: {new_s}, new_e: {new_e}')
            out_list.append([new_s, new_e])
        #logger.critical(f'[update_llc_ranges] out_list: {out_list}')
        #print(f'out_list: {out_list}')
        return out_list

    @staticmethod
    def get_llc_mask_from_ranges(ranges: List[List[int]]) -> List[str]:
        """
        It converts ranges to masks ([[0,19], [0,19]] -> ['fffff', 'fffff']
        :param ranges:
        :return:
        """
        logger = logging.getLogger(__name__)
        masks = []

        logger.debug(f'[get_llc_mask_from_ranges] ranges: {ranges}')
        for r in ranges:
            s = r[0]
            e = r[1]
            logger.debug(f'[get_llc_mask_from_ranges] r: {r}, s: {s}, e: {e}')
            if s == -1 and e == -1:
                mask = hex(int('0', 2))
                masks.append(mask)
                logger.debug(f'[get_llc_mask_from_ranges] r: {r}, mask: {mask}')
                continue
            len_r = e - s + 1
            mask = '0'*s+'1'*len_r+'0'*(19-e)
            logger.debug(f'[get_llc_mask_from_ranges] r: {r}, mask: {mask}')
            mask = hex(int(mask, 2))
            masks.append(mask)
        logger.debug(f'[get_llc_mask_from_ranges] masks: {masks}')
        return masks




