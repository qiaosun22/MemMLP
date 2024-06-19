import ctypes
import os
import time

import numpy as np
from pynq import MMIO
from pynq.lib.iic import AxiIIC
from pathlib import Path
import signal

BASE_ADDR = 0xA0000000
ADDRESS_RANGE = 0x4000

PULSE_UNIT_TIME = 100  # ns

REG0_ADDR = 0
REG1_ADDR = 1 * 4
REG2_ADDR = 2 * 4
REG3_ADDR = 3 * 4
REG4_ADDR = 4 * 4
REG5_ADDR = 5 * 4
REG6_ADDR = 6 * 4
REG7_ADDR = 7 * 4
REG8_ADDR = 8 * 4
REG9_ADDR = 9 * 4
REG10_ADDR = 10 * 4
REG11_ADDR = 11 * 4
REG12_ADDR = 12 * 4
REG13_ADDR = 13 * 4
REG14_ADDR = 14 * 4
REG15_ADDR = 15 * 4
REG51_ADDR = 51 * 4
REG52_ADDR = 52 * 4
REG55_ADDR = 55 * 4
REG56_ADDR = 56 * 4
REG57_ADDR = 57 * 4
REG58_ADDR = 58 * 4
REG59_ADDR = 59 * 4
REG65_ADDR = 65 * 4
REG66_ADDR = 66 * 4
REG67_ADDR = 67 * 4
REG75_ADDR = 75 * 4
REG76_ADDR = 76 * 4
REG77_ADDR = 77 * 4
REG80_ADDR = 80 * 4

VERSION_TOGGLE              = 0x0274
# Programming  State  Register
WRITE_WEIGHT_TOTAL_NUM      = 0x1100
WRITE_WEIGHT_RIGHT_NUM      = 0x1104
WRITE_WEIGHT_NOSET_NUM      = 0x1108
WRITE_WEIGHT_OVERTIME_NUM   = 0x110c
# Programming protection  Register
WRITE_WEIGHT_MAX_TOLERANCE      = 0x1088
WRITE_WEIGHT_SET_MAX_TIMES      = 0x108c
WRITE_WEIGHT_RESET_MAX_TIMES    = 0x1090
MAX_CLKNUM_WRITE_WEIGHT         = 0x1094
VERIFY_WEIGHT_MAX_TIMES         = 0x1098

TOTAL_ROW = 1152
TOTAL_CHANNEL = 8
ONE_CHANNEL_ROW = 144
TOTAL_COL = 128

OP_CFG_DAC = 1
OP_CALC = 2
OP_FORM = 3
OP_SET = 4
OP_RESET = 5
OP_READ = 6
OP_SELECT_CHIP = 9
OP_CFG_CSN = 0x0A
OP_POWER_ON = 0x0F
OP_POWER_OFF = 0x1F

POS_DIR = 1
NEG_DIR = 2

FILE_PATH = os.path.realpath('')
# 获取父目录
par_dir = Path(FILE_PATH).parent




CLIB_PATH = f'/root/Helium-100_Demo/sunqiao/c200_sdk/libBaseApi.so'  # f'{par_dir}/c200_sdk/libBaseApi.so'

def mySleep(delayTime):
    startTime = time.perf_counter()
    while time.perf_counter() - startTime < delayTime:
        pass


def DACVToReg(voltage):
    return int(voltage * 0xFFFF / 5)


# mA
def regToCurrent(regVal):
    current = 3125 * regVal // 0xFFFF
    # print('current:%d,val:%x'%(current, val))
    return current


def currentToReg(valmA):
    val = valmA * 0xFFFF // 3125
    return val


class DIN():
    def __init__(self, regAddr = 0, selectedBitMap = 0, actualBitMap = 0):
        self.cfgPara(regAddr, selectedBitMap, actualBitMap)

    def cfgPara(self, regAddr, selectedBitMap, actualBitMap = 0):
        self.regAddr = regAddr
        self.selectedBitMap = selectedBitMap
        self.actualBitMap = actualBitMap


class BaseAPI():
    version = "v2.0"

    def __init__(self):
        self.mmio = MMIO(BASE_ADDR, ADDRESS_RANGE)
        self.DIN0 = [DIN() for _ in range(ONE_CHANNEL_ROW)]
        self.DIN1 = [DIN() for _ in range(ONE_CHANNEL_ROW)]
        self.DIN2 = [DIN() for _ in range(ONE_CHANNEL_ROW)]
        self.DIN3 = [DIN() for _ in range(ONE_CHANNEL_ROW)]
        self.DIN4 = [DIN() for _ in range(ONE_CHANNEL_ROW)]
        self.DIN5 = [DIN() for _ in range(ONE_CHANNEL_ROW)]
        self.DIN6 = [DIN() for _ in range(ONE_CHANNEL_ROW)]
        self.DIN7 = [DIN() for _ in range(ONE_CHANNEL_ROW)]
        self.cfgDIN()
        self.DINArr = (self.DIN0, self.DIN1, self.DIN2, self.DIN3, self.DIN4,
                       self.DIN5, self.DIN6, self.DIN7)
        self.dictIic3 = {'phys_addr': 0xA0004000, 'addr_range': 0x1000}
        self.dictIic4 = {'phys_addr': 0xA0005000, 'addr_range': 0x1000}
        self.dictIic5 = {'phys_addr': 0xA0006000, 'addr_range': 0x1000}
        self.clib = ctypes.cdll.LoadLibrary(CLIB_PATH)

        # print(self.clib)
        # print("!!!!!!!!!!!!")
        self.clib.ElememDev_Init()
        self.clib.OpenMMIO()
        self.irq_flag = 0

    def __del__(self):
        self.clib.CloseMMIO()

    def irq_signal(self, signum, frame):
        self.irq_flag = 1

    def switch_hd_version(self, on):
        self.writeReg(VERSION_TOGGLE, on)

    def set_write_weight_config(self):
        self.writeReg(WRITE_WEIGHT_MAX_TOLERANCE, (1<<16))
        self.writeReg(WRITE_WEIGHT_SET_MAX_TIMES, 20)
        self.writeReg(WRITE_WEIGHT_RESET_MAX_TIMES, 80)
        self.writeReg(MAX_CLKNUM_WRITE_WEIGHT, 500000)
        self.writeReg(VERIFY_WEIGHT_MAX_TIMES, 0x0101)

    def clearAllActualBitMap(self):
        for dinNum in self.DINArr:
            for din in dinNum:
                din.actualBitMap = 0

    def cfgDIN(self):
        reg = REG51_ADDR
        for i in range(TOTAL_ROW // 32):
            self.DIN0[0 + 4 * i].cfgPara(reg, 0x01000000)
            self.DIN0[1 + 4 * i].cfgPara(reg, 0x00010000)
            self.DIN0[2 + 4 * i].cfgPara(reg, 0x00000100)
            self.DIN0[3 + 4 * i].cfgPara(reg, 0x00000001)

            self.DIN1[0 + 4 * i].cfgPara(reg, 0x02000000)
            self.DIN1[1 + 4 * i].cfgPara(reg, 0x00020000)
            self.DIN1[2 + 4 * i].cfgPara(reg, 0x00000200)
            self.DIN1[3 + 4 * i].cfgPara(reg, 0x00000002)

            self.DIN2[0 + 4 * i].cfgPara(reg, 0x04000000)
            self.DIN2[1 + 4 * i].cfgPara(reg, 0x00040000)
            self.DIN2[2 + 4 * i].cfgPara(reg, 0x00000400)
            self.DIN2[3 + 4 * i].cfgPara(reg, 0x00000004)

            self.DIN3[0 + 4 * i].cfgPara(reg, 0x08000000)
            self.DIN3[1 + 4 * i].cfgPara(reg, 0x00080000)
            self.DIN3[2 + 4 * i].cfgPara(reg, 0x00000800)
            self.DIN3[3 + 4 * i].cfgPara(reg, 0x00000008)

            self.DIN4[0 + 4 * i].cfgPara(reg, 0x10000000)
            self.DIN4[1 + 4 * i].cfgPara(reg, 0x00100000)
            self.DIN4[2 + 4 * i].cfgPara(reg, 0x00001000)
            self.DIN4[3 + 4 * i].cfgPara(reg, 0x00000010)

            self.DIN5[0 + 4 * i].cfgPara(reg, 0x20000000)
            self.DIN5[1 + 4 * i].cfgPara(reg, 0x00200000)
            self.DIN5[2 + 4 * i].cfgPara(reg, 0x00002000)
            self.DIN5[3 + 4 * i].cfgPara(reg, 0x00000020)

            self.DIN6[0 + 4 * i].cfgPara(reg, 0x40000000)
            self.DIN6[1 + 4 * i].cfgPara(reg, 0x00400000)
            self.DIN6[2 + 4 * i].cfgPara(reg, 0x00004000)
            self.DIN6[3 + 4 * i].cfgPara(reg, 0x00000040)

            self.DIN7[0 + 4 * i].cfgPara(reg, 0x80000000)
            self.DIN7[1 + 4 * i].cfgPara(reg, 0x00800000)
            self.DIN7[2 + 4 * i].cfgPara(reg, 0x00008000)
            self.DIN7[3 + 4 * i].cfgPara(reg, 0x00000080)
            reg = reg - 4

    def writeReg(self, addr, value):
        self.mmio.write(addr, value)

    def readReg(self, addr):
        return self.mmio.read(addr)

    # def getAdcValue(self, regVal, colIdx):
    #     idx = colIdx % 8
    #     adcVal = (regVal >> ((7 - idx) * 4)) & 0xF
    #     return adcVal

    # def selectOneCol(self, colIdx):
    #     assert 0 <= colIdx < TOTAL_COL
    #     self.writeReg(REG52_ADDR + 4 * 0, 0)
    #     self.writeReg(REG52_ADDR + 4 * 1, 0)
    #     self.writeReg(REG52_ADDR + 4 * 2, 0)
    #     self.writeReg(REG52_ADDR + 4 * 3, 0)
    #     regNum = colIdx // 32
    #     remainder = colIdx % 32
    #     regVal = 1 << remainder
    #     self.writeReg(REG52_ADDR + 4 * regNum, regVal)

    def getAdcValue(self, regVal, colIdx):
        idx = colIdx % 8
        adcVal = (regVal >> (idx * 4)) & 0xF
        return adcVal

    def selectOneCol(self, colIdx):
        assert 0 <= colIdx < TOTAL_COL
        self.writeReg(REG52_ADDR + 4 * 0, 0)
        self.writeReg(REG52_ADDR + 4 * 1, 0)
        self.writeReg(REG52_ADDR + 4 * 2, 0)
        self.writeReg(REG52_ADDR + 4 * 3, 0)
        regNum = colIdx // 32
        remainder = colIdx % 32
        regVal = 1 << (31 - remainder)
        self.writeReg(REG55_ADDR - 4 * regNum, regVal)

    def selectOneRow(self, rowIdx, POS_or_NEG):
        assert POS_or_NEG == 'POS' or POS_or_NEG == 'NEG'
        assert 0 <= rowIdx < (TOTAL_ROW // 2)
        regRow = 0
        channel = rowIdx * 2 // ONE_CHANNEL_ROW
        if POS_or_NEG == 'POS':
            regRow = rowIdx * 2 % ONE_CHANNEL_ROW + 1  # reg 行数从 1 开始
            self.writeReg(REG14_ADDR, channel)  # 奇数行
        else:
            regRow = rowIdx * 2 % ONE_CHANNEL_ROW + 2
            self.writeReg(REG14_ADDR, channel | 0x80000000)  # 偶数行
        self.writeReg(REG13_ADDR, regRow)

    def waitFlagBit(self, addr, bitPosition, delay, bitSet = True):
        # while delay > 0:
        #     ret = self.readReg(addr)
        #     if bitSet:
        #         # if ret & (1 << bitPosition) != 0:
        #         if ret & (1 << bitPosition) :
        #             return True
        #     else:
        #         if not (ret & (1 << bitPosition)):
        #             return True
        #     delay = delay - 1
        for i in range(delay):
            ret = self.readReg(addr)
            if bitSet:
                if ret & (1 << bitPosition):
                    return True
            else:
                if not (ret & (1 << bitPosition)):
                    return True
        return False

    def waitOpFinish(self, op):
        if (op == 0xf) or (op == 0x1f):
            bitPosition = 8
        else:
            bitPosition = op
        ret = self.waitFlagBit(REG0_ADDR, bitPosition, 100000, True)
        if not ret:
            raise TimeoutError('The completion flag is not set')

    def clearOp(self, op):
        v = 0
        if (op == 0xf) or (op == 0x1f):
            v = 0xFF
            bitPosition = 8
        else:
            v = op | 0x10
            bitPosition = op
        self.writeReg(REG3_ADDR, v)
        ret = self.waitFlagBit(REG0_ADDR, bitPosition, 100000, False)
        if not ret:
            raise TimeoutError('The completion flag is not reset')

    def opFlow(self, op):
        # print('opFlow1 : ', 1000 * time.time())
        self.writeReg(REG3_ADDR, op)
        # print('opFlow2 : ', 1000 * time.time())
        self.writeReg(REG1_ADDR, 1)
        # print('opFlow3 : ', 1000 * time.time())
        self.waitOpFinish(op)
        # print('opFlow4 : ', 1000 * time.time())
        self.clearOp(op)
        # print('opFlow5 : ', 1000 * time.time())
        self.writeReg(REG1_ADDR, 0)
        # print('opFlow6 : ', 1000 * time.time())

    def oneCellOp(self, op, rowIdx, colIdx, POS_or_NEG):
        # print('oneCellOp1 : ', 1000 * time.time())
        assert POS_or_NEG == 'POS' or POS_or_NEG == 'NEG'
        # print('oneCellOp2 : ', 1000 * time.time())
        self.selectOneRow(rowIdx, POS_or_NEG)
        # print('oneCellOp3 : ', 1000 * time.time())
        self.selectOneCol(colIdx)
        # print('oneCellOp4 : ', 1000 * time.time())

        if op == OP_CALC:
            self.writeReg(REG11_ADDR, 8 | POS_DIR | NEG_DIR)
        else:
            if POS_or_NEG == 'POS':
                self.writeReg(REG11_ADDR, 8 | POS_DIR)
            else:
                self.writeReg(REG11_ADDR, 8 | NEG_DIR)

        # print('oneCellOp5 : ', 1000 * time.time())
        if op == OP_RESET:
            self.writeReg(REG15_ADDR, 0x80000001)
        else:
            self.writeReg(REG15_ADDR, 1)
        # print('oneCellOp6 : ', 1000 * time.time())
        self.opFlow(op)
        # print('oneCellOp7 : ', 1000 * time.time())

    def cfgCSN(self, channel, value, isReg):
        assert 0 <= channel <= 15
        if not isReg:
            value = currentToReg(value)
        self.writeReg(REG66_ADDR, value)
        self.writeReg(REG65_ADDR, channel)
        self.opFlow(OP_CFG_CSN)

    def cfgDAC(self, channel, value, isReg):
        assert 0 <= channel <= 15
        if not isReg:
            value = DACVToReg(value)
        self.writeReg(REG5_ADDR, value)
        regVal = (1 << 4) | channel
        self.writeReg(REG4_ADDR, regVal)
        self.opFlow(OP_CFG_DAC)

    def cfgAD5254(self, channel, regVal):
        pass  # TODO

    def cfgVBLMid(self, value, isReg = False):
        self.cfgDAC(5, value, isReg)

    def cfgVBLNeg(self, value, isReg = False):
        self.cfgDAC(15, value, isReg)

    def cfgVBLPos(self, value, isReg = False):
        self.cfgDAC(14, value, isReg)

    def cfgVrefCompDn(self, value, isReg = False):
        self.cfgDAC(7, value, isReg)

    def cfgVrefCompUp(self, value, isReg = False):
        self.cfgDAC(4, value, isReg)

    def cfgVSLClamp(self, value, isReg = False):
        self.cfgDAC(0, value, isReg)

    def cfgVDDLV(self, value, isReg = False):
        self.cfgDAC(12, value, isReg)

    # def cfgIrefComp(self, value, isReg=False):
    #     self.cfgCSN(2, value, isReg)

    # def cfgIrefInteg(self, value, isReg=False):
    #     self.cfgCSN(1, value, isReg)

    # def cfgImpIdc(self, value, isReg=False):
    #     self.cfgCSN(0, value, isReg)

    # def cfgVdd0(self, val):
    #     self.cfgAD5254(3, val)

    # def cfgVdd1(self, val):
    #     self.cfgAD5254(0, val)

    # def cfgVdd2(self, val):
    #     self.cfgAD5254(2, val)

    def cfgFormVBL(self, value, isReg = False):
        self.cfgDAC(6, value, isReg)

    def cfgFormVWL(self, value, isReg = False):
        self.cfgDAC(1, value, isReg)

    def cfgSetVBL(self, value, isReg = False):
        self.cfgDAC(13, value, isReg)

    def cfgSetVWL(self, value, isReg = False):
        self.cfgDAC(3, value, isReg)

    def cfgResetVSL(self, value, isReg = False):
        self.cfgDAC(2, value, isReg)

    def cfgResetVWL(self, value, isReg = False):
        self.cfgDAC(11, value, isReg)

    def cfgFloatBl(self):
        val = 1 | (1 << 4)
        self.writeReg(REG12_ADDR, val)

    def cfgClockMhz(self, clockMhz):
        self.writeReg(REG10_ADDR, 100 // clockMhz)

    def cfgFRSPulse(self, ts = 0x50, tp = 0x3C, te = 0x3C):
        '''
            for form/reset/set
            ____|----------------|______  BLPulse
            ____|-ts-|--tp--|-te-|______  WLPulse
        '''
        regVal = tp | (ts << 8)
        self.writeReg(REG7_ADDR, regVal)
        self.writeReg(REG56_ADDR, te)

    def cfgReadPulse(self, ti = 0x20, ts = 0x28, tb = 0xFF, te = 0x3C):
        '''
            ____|--------------------|______  BLPulse
            ____|-ts-|----------|-te-|______  WLPulse
            ____|----|----tb--|_____________  CTRL_INTEG
            ____|-------------|-ti-|________  KEEP_INTEG
        '''
        regVal = tb | (ts << 8) | (ti << 16)
        self.writeReg(REG6_ADDR, regVal)
        self.writeReg(REG56_ADDR, te)

    def resetSystem(self):
        self.writeReg(REG2_ADDR, 1)
        time.sleep(0.01)
        self.writeReg(REG2_ADDR, 0)

    def powerOn(self):
        self.opFlow(OP_POWER_ON)

    def powerOff(self):
        self.opFlow(OP_POWER_OFF)

    def selectRows(self, rowData):
        assert len(rowData) == (TOTAL_ROW // 2)
        self.clearAllActualBitMap()
        for idx, data in enumerate(rowData):
            if data == 1:
                rowIndex = idx * 2
                channel = rowIndex // 144
                din = self.DINArr[channel]
                din[rowIndex % 144].actualBitMap = din[rowIndex %
                                                       144].selectedBitMap

        for i in range(TOTAL_ROW // 32):
            v = 0
            for j in range(TOTAL_CHANNEL):
                v = v | self.DINArr[j][0 + 4 * i].actualBitMap
                v = v | self.DINArr[j][1 + 4 * i].actualBitMap
                v = v | self.DINArr[j][2 + 4 * i].actualBitMap
                v = v | self.DINArr[j][3 + 4 * i].actualBitMap
                self.writeReg(self.DINArr[0][0 + 4 * i].regAddr, v)

    def selectInput(self, inputData):
        assert len(inputData) == (TOTAL_ROW // 2)
        self.clearAllActualBitMap()
        for idx, data in enumerate(inputData):
            if data == -1:
                rowIndex = idx * 2
                channel = rowIndex // 144
                din = self.DINArr[channel]
                din[rowIndex % 144].actualBitMap = din[rowIndex %
                                                       144].selectedBitMap
            elif data == 1:
                rowIndex = idx * 2 + 1
                channel = rowIndex // 144
                din = self.DINArr[channel]
                din[rowIndex % 144].actualBitMap = din[rowIndex %
                                                       144].selectedBitMap
            elif data == 0:
                pass
            else:
                raise ValueError('The element of inputData must be -1, 0, 1')

        for i in range(TOTAL_ROW // 32):
            v = 0
            for j in range(TOTAL_CHANNEL):
                v = v | self.DINArr[j][0 + 4 * i].actualBitMap
                v = v | self.DINArr[j][1 + 4 * i].actualBitMap
                v = v | self.DINArr[j][2 + 4 * i].actualBitMap
                v = v | self.DINArr[j][3 + 4 * i].actualBitMap
                self.writeReg(self.DINArr[0][0 + 4 * i].regAddr, v)

    # def calcArray(self, rowInput, colStart, colCount):
    #     """calculate Selected cells.

    #     Args:
    #         rowInput: list
    #                 The element of rowInput is -1, 0, 1.
    #         colStart: int
    #                 Start column
    #         colCount: int
    #                 The count of selected column

    #     Returns:
    #         result: list
    #                 ADC values of selected area
    #     """
    #     data = []
    #     self.writeReg(REG11_ADDR, 8 | POS_DIR | NEG_DIR)
    #     self.writeReg(REG15_ADDR, 0)
    #     self.selectInput(rowInput)
    #     for colIndex in range(colStart, colStart + colCount):
    #         self.selectOneCol(colIndex)
    #         self.opFlow(OP_CALC)
    #         value = self.readReg(REG76_ADDR)
    #         data.append(self.getAdcValue(value, colIndex))
    #     return data

    def map_single_device(self,
                          rowIdx,
                          colIdx,
                          poldirstr,
                          target_adc,
                          tolerance = 0,
                          try_limit = 500,
                          strategy = 0,
                          verbose = 1):
        succ_flag = 0
        # set parameter limits
        set_vol_lim = 3  # default 3 V
        set_gate_lim = 1.8  # default 1.8 V
        reset_vol_lim = 3.5  # default 3.5 V
        reset_gate_lim = 4.5  # default 4.5 V

        # set paramters
        if not strategy:
            set_vol_start = 0.8
            set_vol_step = 0.05
            set_gate_start = 1.500
            set_gate_step = 0
            set_pul_wid = 500
            reset_vol_start = 1.0
            reset_vol_step = 0.1
            reset_gate_start = 4
            reset_gate_step = 0
            reset_pul_wid = 4000
        else:
            print('Unknown strategy! exit mapping process')
            return 0

        # start operation
        if poldirstr == 'POS':
            poldir = 'POS'
        elif poldirstr == 'NEG':
            poldir = 'NEG'
        else:
            poldir = 'POS'
            print('Error! invalid poldir! (assuming POS)', flush = True)

        set_vol_now = set_vol_start
        set_gate_now = set_gate_start
        reset_vol_now = reset_vol_start
        reset_gate_now = reset_gate_start

        last_operation = 0  # set 1, reset -1, do nothing 0
        last_effective_operation_dir = 0  # set 1, reset -1, do nothing 0
        map_succ_count = 0
        strongest_op_combo_count = 0
        op_interval_time = 0.000
        for idx in range(try_limit):
            # print('1read s: ', 1000 * time.time(),' ms')
            result_adc_now = self.readOneCell(rowIdx, colIdx, poldirstr)
            # print('1read e: ', 1000 * time.time(),' ms')
            if verbose:
                print('[%03d %03d %s]' % (rowIdx, colIdx, poldirstr), end = ' ')
            if abs(target_adc -
                   result_adc_now) <= tolerance:  # need stay and watch
                map_succ_count = map_succ_count + 1
                if verbose:
                    print('[%d] TAR: %d, CURR: %d, CNT: %d.' %
                          (idx, target_adc, result_adc_now, map_succ_count),
                          flush = True)
                # time.sleep(op_interval_time)
                last_operation = 0
                strongest_op_combo_count = 0

            elif target_adc > result_adc_now:  # need to SET
                map_succ_count = 0
                if last_operation == 1:
                    set_vol_now = min(set_vol_now + set_vol_step, set_vol_lim)
                    set_gate_now = min(set_gate_now + set_gate_step,
                                       set_gate_lim)
                elif last_effective_operation_dir == -1:
                    set_vol_now = set_vol_start
                    set_gate_now = set_gate_start
                else:
                    pass
                if verbose:
                    print('[%d] TAR: %d, CURR: %d, SET: %.2f V, GATE: %.2f V.' %
                          (idx, target_adc, result_adc_now, set_vol_now,
                           set_gate_now),
                          flush = True)
                # print('1 set s: ', 1000 * time.time(),' ms')
                self.setOneCell(rowIdx, colIdx, poldirstr, set_vol_now,
                                set_gate_now, set_pul_wid)
                # print('1 set e: ', 1000 * time.time(),' ms')
                if set_vol_now == set_vol_lim:
                    strongest_op_combo_count = strongest_op_combo_count + 1
                else:
                    strongest_op_count = 0
                # time.sleep(op_interval_time)
                last_operation = 1
                last_effective_operation_dir = 1

            else:  # need to RESET
                map_succ_count = 0
                if last_operation == -1:  # RESET at last time
                    reset_vol_now = min(reset_vol_now + reset_vol_step,
                                        reset_vol_lim)
                    reset_gate_now = min(reset_gate_now + reset_gate_step,
                                         reset_gate_lim)
                elif last_effective_operation_dir == 1:  # SET at last time
                    reset_vol_now = reset_vol_start
                    reset_gate_now = reset_gate_start
                else:
                    pass
                if verbose:
                    print('[%d] TAR: %d, CURR: %d, RESET: %.2f V, GATE: %.2f V.' %
                          (idx, target_adc, result_adc_now, reset_vol_now,
                           reset_gate_now),
                          flush = True)
                # print('1rset s: ', 1000 * time.time(),' ms')
                self.resetOneCell(rowIdx, colIdx, poldirstr, reset_vol_now,
                                  reset_gate_now, reset_pul_wid)
                # print('1rset e: ', 1000 * time.time(),' ms')
                if reset_vol_now == reset_vol_lim:
                    strongest_op_combo_count = strongest_op_combo_count + 1
                else:
                    strongest_op_combo_count = 0
                # time.sleep(op_interval_time)
                last_operation = -1
                last_effective_operation_dir = -1

            if map_succ_count >= 1:  #####################
                succ_flag = 1
                break
            if strongest_op_combo_count >= 10:
                break
        return succ_flag

    def setOneCell_1(self, rowIdx, colIdx, vbl, vwl, pulseWidth):
        row = rowIdx // 2
        if rowIdx % 2 == 0:
            POS_or_NEG = 'NEG'
        else:
            POS_or_NEG = 'POS'
        self.setOneCell(row, colIdx, POS_or_NEG, vbl, vwl, pulseWidth)

    def resetOneCell_1(self, rowIdx, colIdx, vsl, vwl, pulseWidth):
        row = rowIdx // 2
        if rowIdx % 2 == 0:
            POS_or_NEG = 'NEG'
        else:
            POS_or_NEG = 'POS'
        self.resetOneCell(row, colIdx, POS_or_NEG, vsl, vwl, pulseWidth)

    def formOneCell_1(self, rowIdx, colIdx, vbl, vwl, pulseWidth):
        row = rowIdx // 2
        if rowIdx % 2 == 0:
            POS_or_NEG = 'NEG'
        else:
            POS_or_NEG = 'POS'
        self.formOneCell(row, colIdx, POS_or_NEG, vbl, vwl, pulseWidth)

    def calcOneCell_1(self, rowIdx, colIdx):
        row = rowIdx // 2
        if rowIdx % 2 == 0:
            POS_or_NEG = 'NEG'
        else:
            POS_or_NEG = 'POS'
        return self.calcOneCell(row, colIdx, POS_or_NEG)

    def readOneCell_1(self, rowIdx, colIdx):
        row = rowIdx // 2
        if rowIdx % 2 == 0:
            POS_or_NEG = 'NEG'
        else:
            POS_or_NEG = 'POS'
        return self.readOneCell(row, colIdx, POS_or_NEG)

    def writeRDAC(self, numAd5254, channel, value):
        assert numAd5254 < 5
        assert channel < 4
        assert value <= 0xff
        if numAd5254 < 4:
            dev = AxiIIC(self.dictIic3)
        else:
            dev = AxiIIC(self.dictIic4)

        slaveAddr = 0x2c | (numAd5254 & 3)
        data = bytes([channel, value])
        dev.send(slaveAddr, data, 2)

    def cfgMapTimePara(self):
        self.cfgReadPulse(ti = 63, ts = 40, tb = 255, te = 60)

    def cfgCalcTimePara(self):
        self.cfgReadPulse(ti = 5, ts = 10, tb = 10, te = 10)

    def devInit(self):
        """Power on, initialize the device voltage and so on.

        Args:
            None

        Returns:
            None
        """
        self.resetSystem()
        self.cfgFloatBl()
        self.cfgClockMhz(10)
        self.cfgFRSPulse(ts = 0x50, tp = 0x3C, te = 0x3C)
        self.cfgReadPulse(ti = 63, ts = 40, tb = 255, te = 60)
        self.powerOn()
        self.cfgVBLMid(2.5)
        self.cfgVBLNeg(2.598)
        self.cfgVBLPos(2.399)
        self.cfgVrefCompDn(0)
        self.cfgVrefCompUp(4.699)
        self.cfgVSLClamp(2.5)
        self.cfgVDDLV(1.799)
        self.cfgSetVBL(2.814)
        self.cfgSetVWL(1.5) #self.cfgSetVWL(2.01)
        self.cfgResetVSL(4.298)
        self.cfgResetVWL(4.0) #self.cfgResetVWL(3.968)
        self.cfgFormVBL(4.556)
        self.cfgFormVWL(1.871)
        self.set_write_weight_config()
        self.clib.ElememDev_WriteInitVol()

    def selectChip(self, chipNum):
        """Select the chip.

        Args:
            chipNum: int
                    Number of the selected chip, 0 <= chipNum < 12.

        Returns:
            None
        """
        assert chipNum < 12
        chipNum += 1
        self.writeReg(REG67_ADDR, chipNum)
        self.opFlow(OP_SELECT_CHIP)
        time.sleep(0.01)

    def readOneCell(self, rowIdx, colIdx, POS_or_NEG):
        """Read one cell.

        Args:
            rowIdx: int
                    Index of the selected chip, 0 <= rowIdx < 576.
            colIdx: int
                    Index of the selected chip, 0 <= colIdx < 128.
            POS_or_NEG: string
                    Select the location of the cell, 'POS' or 'NEG'.

        Returns:
            result: int
                    ADC value, 8 <= result < 16
        """
        # print('read0 : ', 1000 * time.time())
        # print('read1 : ', 1000 * time.time())
        self.oneCellOp(OP_READ, rowIdx, colIdx, POS_or_NEG)
        # print('read2 : ', 1000 * time.time())
        if POS_or_NEG == 'POS':
            value = self.readReg(REG76_ADDR)
        else:
            value = self.readReg(REG77_ADDR)

        # print('read3 : ', 1000 * time.time())
        adcVal = self.getAdcValue(value, colIdx)
        # print('read4 : ', 1000 * time.time())
        return adcVal

    def setOneCell(self, rowIdx, colIdx, POS_or_NEG, vbl, vwl, pulseWidth):
        """Set one cell.

        Args:
            rowIdx: int
                    Index of the selected chip, 0 <= rowIdx < 576.
            colIdx: int
                    Index of the selected chip, 0 <= colIdx < 128.
            POS_or_NEG: string
                    Select the location of the cell, 'POS' or 'NEG'.
            vbl: float
                    voltage of BL, unit(V)
            vwl: float
                    voltage of WL, unit(V)
            pulseWidth: int
                    Pulse width, unit(ns)

        Returns:
            None
        """
        self.cfgSetVBL(vbl)
        self.cfgSetVWL(vwl)
        self.writeReg(REG58_ADDR, pulseWidth // PULSE_UNIT_TIME)
        self.oneCellOp(OP_SET, rowIdx, colIdx, POS_or_NEG)

    def resetOneCell(self, rowIdx, colIdx, POS_or_NEG, vsl, vwl, pulseWidth):
        """Reset one cell.

        Args:
            rowIdx: int
                    Index of the selected chip, 0 <= rowIdx < 576.
            colIdx: int
                    Index of the selected chip, 0 <= colIdx < 128.
            POS_or_NEG: string
                    Select the location of the cell, 'POS' or 'NEG'.
            vsl: float
                    voltage of SL, unit(V)
            vwl: float
                    voltage of WL, unit(V)
            pulseWidth: int
                    Pulse width, unit(ns)

        Returns:
            None
        """

        # print('rst1 : ', 1000 * time.time())
        self.cfgResetVSL(vsl)
        # print('rst2 : ', 1000 * time.time())
        self.cfgResetVWL(vwl)
        # print('rst3 : ', 1000 * time.time())
        self.writeReg(REG59_ADDR, pulseWidth // PULSE_UNIT_TIME)
        # print('rst4 : ', 1000 * time.time())
        self.oneCellOp(OP_RESET, rowIdx, colIdx, POS_or_NEG)
        # print('rst5 : ', 1000 * time.time())

    def formOneCell(self, rowIdx, colIdx, POS_or_NEG, vbl, vwl, pulseWidth):
        """Form one cell.

        Args:
            rowIdx: int
                    Index of the selected chip, 0 <= rowIdx < 576.
            colIdx: int
                    Index of the selected chip, 0 <= colIdx < 128.
            POS_or_NEG: string
                    Select the location of the cell, 'POS' or 'NEG'.
            vbl: float
                    voltage of BL, unit(V)
            vwl: float
                    voltage of WL, unit(V)
            pulseWidth: int
                    Pulse width, unit(ns)

        Returns:
            None
        """
        self.cfgFormVBL(vbl)
        self.cfgFormVWL(vwl)
        self.writeReg(REG57_ADDR, pulseWidth // PULSE_UNIT_TIME)
        self.oneCellOp(OP_FORM, rowIdx, colIdx, POS_or_NEG)

    def calcOneCell(self, rowIdx, colIdx, POS_or_NEG = 'POS'):
        """calculate one cell.

        Args:
            rowIdx: int
                    Index of the selected chip, 0 <= rowIdx < 576.
            colIdx: int
                    Index of the selected chip, 0 <= colIdx < 128.
            POS_or_NEG: string
                    Select the location of the cell, 'POS' or 'NEG'.

        Returns:
            result: int
                    ADC value, 0 <= result < 16
        """
        self.oneCellOp(OP_CALC, rowIdx, colIdx, POS_or_NEG)
        value = self.readReg(REG76_ADDR)
        adcVal = self.getAdcValue(value, colIdx)
        return adcVal

    def calcArray(self, rowInput: np.ndarray, rowStart, colStart, colCount):
        """calculate Selected cells.

        Args:
            rowInput: numpy.ndarray, two-axis matrix
                    The element of rowInput is 0, 1.
            colStart: int
                    Start column
            colCount: int
                    The count of selected column

        Returns:
            result: numpy.ndarray, two-axis matrix
                    The element of result is ADC value, 0 <= result < 16
        """
        assert (rowInput.dtype == 'uint8') or (rowInput.dtype == 'int8')
        # self.cfgCalcTimePara()
        bRowInput = bytes(rowInput)
        calcCount = int(rowInput.shape[0])
        rowCount = int(rowInput.shape[1])
        output = bytes(colCount * calcCount * [0])
        ret = self.clib.CalcArray_2(bRowInput, rowStart, rowCount, colStart,
                                    colCount, output, calcCount)
        if ret != 0:
            raise Exception('CalcArray_2() return error')
        output = np.frombuffer(output, dtype = np.uint8)
        output.resize(calcCount, colCount)
        # self.cfgMapTimePara()
        return output

    def map_single_device_2T2R(self,
                               rowIdx,
                               colIdx,
                               target_adc,
                               tolerance = 0,
                               try_limit = 500,
                               strategy = 0,
                               with_form = 3,
                               verbose = 0):
        succ_flag = 0
        form_to_try_times = with_form

        # set parameter limits
        set_vol_lim = 3  # default 3 V
        set_gate_lim = 1.8  # default 1.8 V
        reset_vol_lim = 3.5  # default 3.5 V
        reset_gate_lim = 4.5  # default 4.5 V

        # set paramters
        op_interval_time = 0.000
        if not strategy:
            set_vol_start = 0.8
            set_vol_step = 0.05
            set_gate_start = 1.5
            set_gate_step = 0
            set_pul_wid = 500
            reset_vol_start = 0.6
            reset_vol_step = 0.05
            reset_gate_start = 4
            reset_gate_step = 0
            reset_pul_wid = 4000
        else:
            print('Unknown strategy! exit mapping process')
            return 0

        # Read current situation
        # sdk.device.set_integ_width(3200)
        # current_weight = self.calcOneCell(rowIdx, colIdx, 'POS')
        if verbose:
            print('\n\n')
            print('2T2R mapping start')
            print('target_adc: %d' % (target_adc))
            print('current state: %d' % (self.calcOneCell(rowIdx, colIdx, 'POS')), flush = True)

        # === POS or NEG? ==================================================
        if target_adc < 8:  # NEG

            # make R_POS at HRS
            reset_succ = self.map_single_device(rowIdx,
                                                colIdx,
                                                'POS',
                                                8,
                                                tolerance = 0,
                                                verbose = verbose)
            if verbose:
                if reset_succ:
                    print('reset R_POS to HRS successfully')
                else:
                    print('reset R_POS to HRS failed')
                    return 0

            # start mapping (adjust R_NEG)
            set_vol_now = set_vol_start
            set_gate_now = set_gate_start
            reset_vol_now = reset_vol_start
            reset_gate_now = reset_gate_start
            polstr = 'NEG'

            last_operation = 0  # set 1, reset -1, do nothing 0
            last_effective_operation_dir = 0  # set 1, reset -1, do nothing 0
            map_succ_count = 0
            strongest_op_combo_count = 0

            for idx in range(try_limit):
                # print('2read s: ', 1000 * time.time(),' ms')
                result_adc_now = self.calcOneCell(rowIdx, colIdx, 'POS')
                # print('2read e: ', 1000 * time.time(),' ms')
                if verbose:
                    print('[%03d %03d]' % (rowIdx, colIdx), end = ' ')
                    print('<R_NEG>', end = ' ')
                if abs(target_adc - result_adc_now) <= tolerance:  # need stay and watch
                    map_succ_count = map_succ_count + 1
                    if verbose:
                        print(
                            '[%d] TAR: %.1f, CURR: %d, CNT: %d.' %
                            (idx, target_adc, result_adc_now, map_succ_count),
                            flush = True)
                    # time.sleep(op_interval_time)
                    last_operation = 0
                    strongest_op_combo_count = 0

                elif target_adc < result_adc_now:  # need to SET R_NEG
                    map_succ_count = 0
                    if last_operation == 1:
                        set_vol_now = min(set_vol_now + set_vol_step,
                                          set_vol_lim)
                        set_gate_now = min(set_gate_now + set_gate_step,
                                           set_gate_lim)
                    elif last_effective_operation_dir == -1:
                        set_vol_now = set_vol_start
                        set_gate_now = set_gate_start
                    else:
                        pass
                    if verbose:
                        print('[%d] TAR: %.1f, CURR: %d, SET: %.2f V, GATE: %.2f V.'
                              % (idx, target_adc, result_adc_now, set_vol_now, set_gate_now),
                              flush = True)
                    # print('2 set s: ', 1000 * time.time(),' ms')
                    self.setOneCell(rowIdx, colIdx, polstr, set_vol_now,
                                    set_gate_now, set_pul_wid)
                    # print('2 set e: ', 1000 * time.time(),' ms')
                    if set_vol_now == set_vol_lim:
                        strongest_op_combo_count = strongest_op_combo_count + 1
                    else:
                        strongest_op_count = 0
                    # time.sleep(op_interval_time)
                    last_operation = 1
                    last_effective_operation_dir = 1

                else:  # need to RESET R_NEG
                    map_succ_count = 0
                    if last_operation == -1:  # RESET at last time
                        reset_vol_now = min(reset_vol_now + reset_vol_step,
                                            reset_vol_lim)
                        reset_gate_now = min(reset_gate_now + reset_gate_step,
                                             reset_gate_lim)
                    elif last_effective_operation_dir == 1:  # SET at last time
                        reset_vol_now = reset_vol_start
                        reset_gate_now = reset_gate_start
                    else:
                        pass
                    if verbose:
                        print(
                            '[%d] TAR: %.1f, CURR: %d, RESET: %.2f V, GATE: %.2f V.'
                            % (idx, target_adc, result_adc_now, reset_vol_now,
                               reset_gate_now),
                            flush = True)
                    # print('2rset s: ', 1000 * time.time(),' ms')
                    self.resetOneCell(rowIdx, colIdx, polstr, reset_vol_now,
                                      reset_gate_now, reset_pul_wid)
                    # print('2rset e: ', 1000 * time.time(),' ms')
                    if reset_vol_now == reset_vol_lim:
                        strongest_op_combo_count = strongest_op_combo_count + 1
                    else:
                        strongest_op_combo_count = 0
                    # time.sleep(op_interval_time)
                    last_operation = -1
                    last_effective_operation_dir = -1

                if map_succ_count >= 1:  ########################5
                    succ_flag = 1
                    break
                if strongest_op_combo_count >= 10:
                    if form_to_try_times > 0 and last_operation == 1:
                        form_to_try_times = form_to_try_times - 1
                        if verbose:
                            print('Try forming (%d times left)' %
                                  (form_to_try_times))
                        self.formOneCell(rowIdx, colIdx, 'NEG', 4.6, 2, 100000)
                        set_vol_now = set_vol_start
                        set_gate_now = set_gate_start
                        reset_vol_now = reset_vol_start
                        reset_gate_now = reset_gate_start
                        strongest_op_combo_count = 0
                        last_operation = 1
                        last_effective_operation_dir = 1
                    else:
                        break

        # === POS or NEG? ==================================================
        else:  # POS
            # make R_NEG at HRS
            reset_succ = self.map_single_device(rowIdx,
                                                colIdx,
                                                'NEG',
                                                8,
                                                tolerance = 0,
                                                verbose = verbose)
            if verbose:
                if reset_succ:
                    print('reset R_NEG to HRS successfully')
                else:
                    print('reset R_NEG to HRS failed')
                    return 0

            # start mapping (adjust R_POS)
            set_vol_now = set_vol_start
            set_gate_now = set_gate_start
            reset_vol_now = reset_vol_start
            reset_gate_now = reset_gate_start
            polstr = 'POS'

            last_operation = 0  # set 1, reset -1, do nothing 0
            last_effective_operation_dir = 0  # set 1, reset -1, do nothing 0
            map_succ_count = 0
            strongest_op_combo_count = 0

            for idx in range(try_limit):
                result_adc_now = self.calcOneCell(rowIdx, colIdx, 'POS')
                # print('2P')
                # print(1000 * time.time())
                if verbose:
                    print('[%03d %03d]' % (rowIdx, colIdx), end = ' ')
                    print('<R_POS>', end = ' ')
                if abs(target_adc -
                       result_adc_now) <= tolerance:  # need stay and watch
                    map_succ_count = map_succ_count + 1
                    if verbose:
                        print(
                            '[%d] TAR: %.1f, CURR: %d, CNT: %d.' %
                            (idx, target_adc, result_adc_now, map_succ_count),
                            flush = True)
                    # time.sleep(op_interval_time)
                    last_operation = 0
                    strongest_op_combo_count = 0

                elif target_adc > result_adc_now:  # need to SET R_POS
                    map_succ_count = 0
                    if last_operation == 1:
                        set_vol_now = min(set_vol_now + set_vol_step,
                                          set_vol_lim)
                        set_gate_now = min(set_gate_now + set_gate_step,
                                           set_gate_lim)
                    elif last_effective_operation_dir == -1:
                        set_vol_now = set_vol_start
                        set_gate_now = set_gate_start
                    else:
                        pass
                    if verbose:
                        print('[%d] TAR: %.1f, CURR: %d, SET: %.2f V, GATE: %.2f V.'
                              % (idx, target_adc, result_adc_now, set_vol_now, set_gate_now),
                              flush = True)
                    self.setOneCell(rowIdx, colIdx, polstr, set_vol_now,
                                    set_gate_now, set_pul_wid)
                    if set_vol_now == set_vol_lim:
                        strongest_op_combo_count = strongest_op_combo_count + 1
                    else:
                        strongest_op_count = 0
                    # time.sleep(op_interval_time)
                    last_operation = 1
                    last_effective_operation_dir = 1

                else:  # need to RESET R_POS
                    map_succ_count = 0
                    if last_operation == -1:  # RESET at last time
                        reset_vol_now = min(reset_vol_now + reset_vol_step,
                                            reset_vol_lim)
                        reset_gate_now = min(reset_gate_now + reset_gate_step,
                                             reset_gate_lim)
                    elif last_effective_operation_dir == 1:  # SET at last time
                        reset_vol_now = reset_vol_start
                        reset_gate_now = reset_gate_start
                    else:
                        pass
                    if verbose:
                        print(
                            '[%d] TAR: %.1f, CURR: %d, RESET: %.2f V, GATE: %.2f V.'
                            % (idx, target_adc, result_adc_now, reset_vol_now,
                               reset_gate_now),
                            flush = True)
                    self.resetOneCell(rowIdx, colIdx, polstr, reset_vol_now,
                                      reset_gate_now, reset_pul_wid)
                    if reset_vol_now == reset_vol_lim:
                        strongest_op_combo_count = strongest_op_combo_count + 1
                    else:
                        strongest_op_combo_count = 0
                    # time.sleep(op_interval_time)
                    last_operation = -1
                    last_effective_operation_dir = -1

                if map_succ_count >= 1:  ########################5
                    succ_flag = 1
                    break

                if strongest_op_combo_count >= 10:
                    if form_to_try_times > 0 and last_operation == 1:
                        form_to_try_times = form_to_try_times - 1
                        if verbose:
                            print('Try forming (%d times left)' %
                                  (form_to_try_times))
                        self.formOneCell(rowIdx, colIdx, 'POS', 4.6, 2, 100000)
                        set_vol_now = set_vol_start
                        set_gate_now = set_gate_start
                        reset_vol_now = reset_vol_start
                        reset_gate_now = reset_gate_start
                        strongest_op_combo_count = 0
                        last_operation = 1
                        last_effective_operation_dir = 1
                    else:
                        break

        if verbose:
            if succ_flag:
                print('mapping successful!', flush = True)
            else:
                print('mapping failed!', flush = True)
        return succ_flag

    def map_single_device_2T2R_POR(self,
                                   rowIdx,
                                   colIdx,
                                   target_adc,
                                   tolerance = 0,
                                   try_limit = 100,
                                   strategy = 0,
                                   with_form = 3,
                                   verbose = 0):
        succ_flag = 0
        form_to_try_times = with_form

        # set parameter limits
        set_vol_lim = 3  # default 3 V
        set_gate_lim = 1.8  # default 1.8 V
        reset_vol_lim = 3.5  # default 3.5 V
        reset_gate_lim = 4.5  # default 4.5 V

        # set paramters
        op_interval_time = 0.000
        if not strategy:
            set_vol_start = 0.8
            set_vol_step = 0.1
            set_gate_start = 1.5
            set_gate_step = 0
            set_pul_wid = 500
            reset_vol_start = 0.6
            reset_vol_step = 0.05
            orr_vol_start = 0.8
            orr_vol_step = 0.1
            reset_gate_start = 4
            reset_gate_step = 0
            orr_gate_step = 0
            reset_pul_wid = 4000
        else:
            print('Unknown strategy! exit mapping process')
            return 0

        # Read current situation
        # sdk.device.set_integ_width(3200)
        # current_weight = self.calcOneCell(rowIdx, colIdx, 'POS')
        if verbose:
            print('\n\n')
            print('2T2R mapping start')
            print('target_adc: %d' % (target_adc))
            print('current state: %d' % (self.calcOneCell(rowIdx, colIdx, 'POS')), flush = True)

        # === POS or NEG? ==================================================
        if target_adc < 8:  # NEG

            # make R_POS at HRS
            reset_succ = self.map_single_device(rowIdx,
                                                colIdx,
                                                'POS',
                                                8,
                                                tolerance = 0,
                                                verbose = verbose)
            if verbose:
                if reset_succ:
                    print('reset R_POS to HRS successfully')
                else:
                    print('reset R_POS to HRS failed')
                    return 0

            # start mapping (adjust R_NEG)
            set_vol_now = set_vol_start
            set_gate_now = set_gate_start
            reset_vol_now = reset_vol_start
            reset_gate_now = reset_gate_start
            polstr = 'NEG'

            last_operation = 0  # set 1, reset -1, over range reset -2, do nothing 0
            last_effective_operation_dir = 0  # set 1, reset -1, do nothing 0
            map_succ_count = 0
            strongest_op_combo_count = 0

            for idx in range(try_limit):
                result_adc_now = self.calcOneCell(rowIdx, colIdx, 'POS')
                if verbose:
                    print('[%03d %03d]' % (rowIdx, colIdx), end = ' ')
                    print('<R_NEG>', end = ' ')
                if (abs(target_adc - result_adc_now) <= tolerance) and (
                        last_operation == 0 or last_operation == -1):  # pass
                    map_succ_count = map_succ_count + 1
                    if verbose:
                        print(
                            '[%d] TAR: %d, CURR: %d, CNT: %d, LAST_OP: %d.' %
                            (idx, target_adc, result_adc_now, map_succ_count, last_operation),
                            flush = True)
                    # time.sleep(op_interval_time)
                    last_operation = 0
                    strongest_op_combo_count = 0
                    succ_flag = 1
                    break

                elif target_adc > result_adc_now + 1:  # need to ORR R_NEG
                    map_succ_count = 0
                    if last_operation == -2:  # ORR at last time
                        reset_vol_now = min(reset_vol_now + orr_vol_step,
                                            reset_vol_lim)
                        reset_gate_now = min(reset_gate_now + orr_gate_step,
                                             reset_gate_lim)
                    # elif last_effective_operation_dir == 1:  # SET at last time
                    else:
                        reset_vol_now = orr_vol_start
                        reset_gate_now = reset_gate_start
                    # else:
                    #     pass
                    if verbose:
                        print(
                            '[%d] TAR: %d, CURR: %d, RESET: %.2f V, GATE: %.2f V, LAST_OP: %d.'
                            % (idx, target_adc, result_adc_now, reset_vol_now,
                               reset_gate_now, last_operation),
                            flush = True)
                    self.resetOneCell(rowIdx, colIdx, polstr, reset_vol_now,
                                      reset_gate_now, reset_pul_wid)
                    if reset_vol_now == reset_vol_lim:
                        strongest_op_combo_count = strongest_op_combo_count + 1
                    else:
                        strongest_op_combo_count = 0
                    # time.sleep(op_interval_time)
                    last_operation = -2
                    last_effective_operation_dir = -2

                elif (target_adc > result_adc_now) and (
                        last_operation == 0 or last_operation == -1):  # need to RESET R_NEG
                    map_succ_count = 0
                    if last_operation == -1:  # RESET at last time
                        reset_vol_now = min(reset_vol_now + reset_vol_step,
                                            reset_vol_lim)
                        reset_gate_now = min(reset_gate_now + reset_gate_step,
                                             reset_gate_lim)
                    # elif last_effective_operation_dir == 1:  # SET at last time
                    else:
                        reset_vol_now = reset_vol_start
                        reset_gate_now = reset_gate_start
                    # else:
                    #     pass
                    if verbose:
                        print(
                            '[%d] TAR: %d, CURR: %d, RESET: %.2f V, GATE: %.2f V, LAST_OP: %d.'
                            % (idx, target_adc, result_adc_now, reset_vol_now,
                               reset_gate_now, last_operation),
                            flush = True)
                    self.resetOneCell(rowIdx, colIdx, polstr, reset_vol_now,
                                      reset_gate_now, reset_pul_wid)
                    if reset_vol_now == reset_vol_lim:
                        strongest_op_combo_count = strongest_op_combo_count + 1
                    else:
                        strongest_op_combo_count = 0
                    # time.sleep(op_interval_time)
                    last_operation = -1
                    last_effective_operation_dir = -1

                elif (target_adc > result_adc_now) and (
                        last_operation == 1 or last_operation == -2):  # need to Delay
                    if verbose:
                        print(
                            '[%d] TAR: %d, CURR: %d, CNT: %d, LAST_OP: %d.' %
                            (idx, target_adc, result_adc_now, map_succ_count, last_operation),
                            flush = True)
                    last_operation = 0
                    strongest_op_combo_count = 0
                    succ_flag = 0
                    break

                else:  # need to SET R_NEG
                    map_succ_count = 0
                    if last_operation == 1:
                        set_vol_now = min(set_vol_now + set_vol_step,
                                          set_vol_lim)
                        set_gate_now = min(set_gate_now + set_gate_step,
                                           set_gate_lim)
                    # elif last_effective_operation_dir == -1:
                    else:
                        set_vol_now = set_vol_start
                        set_gate_now = set_gate_start
                    # else:
                    #     pass
                    if verbose:
                        print('[%d] TAR: %d, CURR: %d, SET: %.2f V, GATE: %.2f V, LAST_OP: %d.'
                              % (idx, target_adc, result_adc_now, set_vol_now, set_gate_now, last_operation),
                              flush = True)
                    self.setOneCell(rowIdx, colIdx, polstr, set_vol_now,
                                    set_gate_now, set_pul_wid)
                    if set_vol_now == set_vol_lim:
                        strongest_op_combo_count = strongest_op_combo_count + 1
                    else:
                        strongest_op_count = 0
                    # time.sleep(op_interval_time)
                    last_operation = 1
                    last_effective_operation_dir = 1

                if strongest_op_combo_count >= 10:
                    if form_to_try_times > 0 and last_operation == 1:
                        form_to_try_times = form_to_try_times - 1
                        if verbose:
                            print('Try forming (%d times left)' %
                                  (form_to_try_times))
                        self.formOneCell(rowIdx, colIdx, 'NEG', 4.6, 2, 100000)
                        set_vol_now = set_vol_start
                        set_gate_now = set_gate_start
                        reset_vol_now = reset_vol_start
                        reset_gate_now = reset_gate_start
                        strongest_op_combo_count = 0
                        last_operation = 1
                        last_effective_operation_dir = 1
                    else:
                        break

        # === POS or NEG? ==================================================
        else:  # POS
            # make R_NEG at HRS
            reset_succ = self.map_single_device(rowIdx,
                                                colIdx,
                                                'NEG',
                                                8,
                                                tolerance = 0,
                                                verbose = verbose)
            if verbose:
                if reset_succ:
                    print('reset R_NEG to HRS successfully')
                else:
                    print('reset R_NEG to HRS failed')
                    return 0

            # start mapping (adjust R_POS)
            set_vol_now = set_vol_start
            set_gate_now = set_gate_start
            reset_vol_now = reset_vol_start
            reset_gate_now = reset_gate_start
            polstr = 'POS'

            last_operation = 0  # set 1, reset -1, do nothing 0
            last_effective_operation_dir = 0  # set 1, reset -1, do nothing 0
            map_succ_count = 0
            strongest_op_combo_count = 0

            for idx in range(try_limit):
                result_adc_now = self.calcOneCell(rowIdx, colIdx, 'POS')
                if verbose:
                    print('[%03d %03d]' % (rowIdx, colIdx), end = ' ')
                    print('<R_POS>', end = ' ')
                if (abs(target_adc - result_adc_now) <= tolerance) and (
                        last_operation == 0 or last_operation == -1):  # pass
                    map_succ_count = map_succ_count + 1
                    if verbose:
                        print(
                            '[%d] TAR: %d, CURR: %d, CNT: %d, LAST_OP: %d.' %
                            (idx, target_adc, result_adc_now, map_succ_count, last_operation),
                            flush = True)
                    # time.sleep(op_interval_time)
                    last_operation = 0
                    strongest_op_combo_count = 0
                    succ_flag = 1
                    break


                elif target_adc + 1 < result_adc_now:  # need to ORR R_POS
                    map_succ_count = 0
                    if last_operation == -2:  # ORR at last time
                        reset_vol_now = min(reset_vol_now + orr_vol_step,
                                            reset_vol_lim)
                        reset_gate_now = min(reset_gate_now + orr_gate_step,
                                             reset_gate_lim)
                    # elif last_effective_operation_dir == 1:  # SET at last time
                    else:
                        reset_vol_now = orr_vol_start
                        reset_gate_now = reset_gate_start
                    # else:
                    #     pass
                    if verbose:
                        print(
                            '[%d] TAR: %d, CURR: %d, RESET: %.2f V, GATE: %.2f V, LAST_OP: %d.'
                            % (idx, target_adc, result_adc_now, reset_vol_now,
                               reset_gate_now, last_operation),
                            flush = True)
                    self.resetOneCell(rowIdx, colIdx, polstr, reset_vol_now,
                                      reset_gate_now, reset_pul_wid)
                    if reset_vol_now == reset_vol_lim:
                        strongest_op_combo_count = strongest_op_combo_count + 1
                    else:
                        strongest_op_combo_count = 0
                    # time.sleep(op_interval_time)
                    last_operation = -2
                    last_effective_operation_dir = -1

                elif (target_adc < result_adc_now) and (
                        last_operation == 0 or last_operation == -1):  # need to RESET R_POS
                    map_succ_count = 0
                    if last_operation == -1:  # RESET at last time
                        reset_vol_now = min(reset_vol_now + reset_vol_step,
                                            reset_vol_lim)
                        reset_gate_now = min(reset_gate_now + reset_gate_step,
                                             reset_gate_lim)
                    # elif last_effective_operation_dir == 1:  # SET at last time
                    else:
                        reset_vol_now = reset_vol_start
                        reset_gate_now = reset_gate_start
                    # else:
                    #     pass
                    if verbose:
                        print(
                            '[%d] TAR: %d, CURR: %d, RESET: %.2f V, GATE: %.2f V, LAST_OP: %d.'
                            % (idx, target_adc, result_adc_now, reset_vol_now,
                               reset_gate_now, last_operation),
                            flush = True)
                    self.resetOneCell(rowIdx, colIdx, polstr, reset_vol_now,
                                      reset_gate_now, reset_pul_wid)
                    if reset_vol_now == reset_vol_lim:
                        strongest_op_combo_count = strongest_op_combo_count + 1
                    else:
                        strongest_op_combo_count = 0
                    # time.sleep(op_interval_time)
                    last_operation = -1
                    last_effective_operation_dir = -1

                elif (target_adc < result_adc_now) and (
                        last_operation == 1 or last_operation == -2):  # need to Delay
                    if verbose:
                        print(
                            '[%d] TAR: %d, CURR: %d, CNT: %d, LAST_OP: %d.' %
                            (idx, target_adc, result_adc_now, map_succ_count, last_operation),
                            flush = True)
                    last_operation = 0
                    strongest_op_combo_count = 0
                    succ_flag = 0
                    break

                # elif target_adc > result_adc_now:  # need to SET R_POS
                else:  # need to SET R_POS
                    map_succ_count = 0
                    if last_operation == 1:
                        set_vol_now = min(set_vol_now + set_vol_step,
                                          set_vol_lim)
                        set_gate_now = min(set_gate_now + set_gate_step,
                                           set_gate_lim)
                    # elif last_effective_operation_dir == -1:
                    else:
                        set_vol_now = set_vol_start
                        set_gate_now = set_gate_start
                    # else:
                    #     pass
                    if verbose:
                        print('[%d] TAR: %d, CURR: %d, SET: %.2f V, GATE: %.2f V, LAST_OP: %d.'
                              % (idx, target_adc, result_adc_now, set_vol_now, set_gate_now, last_operation),
                              flush = True)
                    self.setOneCell(rowIdx, colIdx, polstr, set_vol_now,
                                    set_gate_now, set_pul_wid)
                    if set_vol_now == set_vol_lim:
                        strongest_op_combo_count = strongest_op_combo_count + 1
                    else:
                        strongest_op_count = 0
                    # time.sleep(op_interval_time)
                    last_operation = 1
                    last_effective_operation_dir = 1

                # if map_succ_count >= 1:  ########################5
                #     succ_flag = 1
                #     break

                if strongest_op_combo_count >= 10:
                    if form_to_try_times > 0 and last_operation == 1:
                        form_to_try_times = form_to_try_times - 1
                        if verbose:
                            print('Try forming (%d times left)' %
                                  (form_to_try_times))
                        self.formOneCell(rowIdx, colIdx, 'POS', 4.6, 2, 100000)
                        set_vol_now = set_vol_start
                        set_gate_now = set_gate_start
                        reset_vol_now = reset_vol_start
                        reset_gate_now = reset_gate_start
                        strongest_op_combo_count = 0
                        last_operation = 1
                        last_effective_operation_dir = 1
                    else:
                        break

        if verbose:
            if succ_flag:
                print('mapping successful!', flush = True)
            else:
                print('mapping failed!', flush = True)
        return succ_flag

    # @calc function
    def calc_array(self, addr, input):
        """calculate Selected cells.

        Args:
            addr: Addrinfo
                    Selected area
            input:  input data for rowselect

        Returns:
            result: list
                    ADC values of selected area
        """
        # rowstart, rowcount, colstart, colcount = addr
        # data = []
        # self.writeReg(REG11_ADDR, 8 | POS_DIR | NEG_DIR)
        # self.writeReg(REG15_ADDR, 0)
        # # rowData = [0] * (TOTAL_ROW // 2)
        # num,rows = input.shape
        # time1 = time.time()
        # for j in range(num):
        #     data_ = []
        #     rowData = input[j][:].tolist()
        #     time3 = time.time()
        #     data_ = self.calcArray(rowData,colstart,colcount)
        #     time4 = time.time()
        #     print(f'python calc:{time4-time3}')
        #     data.append(data_)
        # time2 = time.time()
        # print(f'calc on chip time:{time2 - time1}')
        # output = np.array(data)
        # print(input.shape)
        rowstart, rowcount, colstart, colcount = addr
        time1 = time.time()
        output = self.calcArray(input, rowstart, colstart, colcount)
        time2 = time.time()
        # print(f'calc on chip time:{time2 - time1}')
        return output

    def elemem_read_weight(self, rowStart, rowCount, colStart, colCount, time_out_ms = 16*1000):
        """calculate Selected cells.

        Args:

        Returns:
            output: weight
        """
        type_is_2t2r = 0
        output = bytes(rowCount * colCount * [0])
        ret = self.clib.ElememDev_ReadWeight(rowStart,colStart,rowCount,colCount, output, type_is_2t2r, time_out_ms)
        if ret != 0:
            raise Exception('ElememDev_ReadWeight() return error')
        output = np.frombuffer(output, dtype = np.uint8)
        output.resize(rowCount, colCount)

        return output

    def elemem_write_weight(self, weightInput: np.ndarray, rowStart, colStart, time_out_s = 20*60):
        """calculate Selected cells.

        Args:
            rowInput: numpy.ndarray, two-axis matrix
                    The element of rowInput is 0, 1.
            colStart: int
                    Start column
            colCount: int
                    The count of selected column

        Returns:
            result: numpy.ndarray, two-axis matrix
                    The element of result is ADC value, 0 <= result < 16
        """
        # call this fun must check the weight value : 0 <= result < 16
        assert (weightInput.dtype == 'uint8') or (weightInput.dtype == 'int8')
        data_is_zero = 0
        bweightInput = bytes(weightInput)
        rowCount = int(weightInput.shape[0])
        colCount = int(weightInput.shape[1])
        #print("weight data shape:", rowCount,colCount)
        signal.signal(signal.SIGIO, self.irq_signal)
        ret = self.clib.ElememDev_WriteWeight(rowStart,colStart,rowCount,colCount, data_is_zero, bweightInput, time_out_s)
        if ret != 0:
            raise Exception('write_chip_weight() return error')
        return True

    def get_write_weight_status(self):
        cell_processed = self.readReg(WRITE_WEIGHT_TOTAL_NUM) #写过的个数
        cell_pass_num = self.readReg(WRITE_WEIGHT_RIGHT_NUM) #写正确的个数
        cell_unchanged_num = self.readReg(WRITE_WEIGHT_NOSET_NUM) #无需写操作的个数
        cell_timeout_num = self.readReg(WRITE_WEIGHT_OVERTIME_NUM) #写超时的个数
        cell_pass_num = cell_pass_num + cell_unchanged_num
        return cell_processed, cell_pass_num, cell_unchanged_num, cell_timeout_num

    def elemem_calc_array(self, rowInput: np.ndarray, rowStart, rowCount, colStart, colCount,
                          data_type = 0, split_mode = 0, time_out_ms = 1000):
        """calculate Selected cells.

        Args:
            rowInput: numpy.ndarray, two-axis matrix
            data_type: int
                    input data type, 0:int1.5; 1:int4; 2:int8
            split_mode: int
                    data expand mode, 0:全展开; 1:位展开

        Returns:
            result: numpy.ndarray, two-axis matrix
                    The element of result(int16)
        """
        assert (rowInput.dtype == 'uint8') or (rowInput.dtype == 'int8')
        bRowInput = bytes(rowInput)
        calcCount = int(rowInput.shape[0])
        rowCount = int(rowInput.shape[1])
        output = bytes(colCount * calcCount *2 * [0] )
        ret = self.clib.ElememDev_CalcArray(bRowInput, rowStart, rowCount, colStart, colCount,
                                            output, calcCount, data_type, split_mode, time_out_ms)
        if ret != 0:
            raise Exception('elemem_calc_array() return error')
        output = np.frombuffer(output, dtype = np.int16)
        output.resize(calcCount, colCount)

        return output

