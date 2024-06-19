import time


def mySleep(delayTime):
    startTime = time.perf_counter()
    while time.perf_counter() - startTime < delayTime:
        pass


def DACVToReg(voltage):
    return int(voltage * 0xFFFF / 5)


# mA
def regToCurrent(regVal):
        current = 3125 * regVal // 0xFFFF
        #print('current:%d,val:%x'%(current, val))
        return current


def currentToReg(valmA):
    val = valmA * 0xFFFF // 3125
    return val