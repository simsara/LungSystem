
import os

def getFreeId():  # 获取所有使用率小于70的GPU
    import pynvml 
    pynvml.nvmlInit()
    def getFreeRatio(id):
        """
        获取GPU使用率
        """
        handle = pynvml.nvmlDeviceGetHandleByIndex(id) # 获取到设备号为id的句柄
        use = pynvml.nvmlDeviceGetUtilizationRates(handle)   # 或者GPU 使用率
        ratio = 0.5*(float(use.gpu+float(use.memory)))
        return ratio

    deviceCount = pynvml.nvmlDeviceGetCount() # 一共有几个设备
    available = []
    for i in range(deviceCount):
        if getFreeRatio(i)<70:  # GPU使用率小于70
            available.append(i) # 就认为它是可用的
    gpus = ''
    for g in available:
        gpus = gpus+str(g)+','  # 拼接成字符串
    gpus = gpus[:-1]  # 去掉最后一个逗号
    return gpus


def setgpu(gpuinput):
    freeids = getFreeId()
    if gpuinput=='all':
        gpus = freeids
    else:
        gpus = gpuinput
        input_split = gpus.split(',')
        gpuid_list = freeids.split(',')
        for g in input_split:
            if g not in gpuid_list:         
                raise ValueError('gpu'+g+'is being used')
    print('using gpu '+gpus)
    os.environ['CUDA_VISIBLE_DEVICES']=gpus
    return len(gpus.split(','))
