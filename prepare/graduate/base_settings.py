import os
import datetime


WORKER_POOL_SIZE = 4 # 工作进程数
TARGET_VOXEL_MM = 1.00  # 也就是说分辨率是1:1的
MEAN_PIXEL_VALUE_NODULE = 41  # TODO 
LUNA_SUBSET_START_INDEX = 0  # luna 子数据集从什么子数据集开始
SEGMENTER_IMG_SIZE = 320  # TODO 分割图片大小


# create the folder name ndsb3 for saving the corresponding results
BASE_DIR_SSD = "D:/everyproject/NDSB/kaggle_ndsb2017-master/ndsb3/"
# create the folder name ndsb4 for placing the input data here
BASE_DIR = "D:/everyproject/NDSB/kaggle_ndsb2017-master/ndsb4/"
# place here extra data given by julian in his repository.  TODO 额外数据？
EXTRA_DATA_DIR = "D:/everyproject/NDSB/kaggle_ndsb2017-master/kaggle_ndsb2017-master/resources/"
# place here the kaggle data which will further（CT）
NDSB3_RAW_SRC_DIR = BASE_DIR + "ndsb_raw/stage12/"
# place here the LUNA16 database  TODO 只有raw数据集吗
LUNA16_RAW_SRC_DIR = BASE_DIR + "luna_raw/"


# all below directories are created for saving the corresponding results
#  of the preprocessing and nodule detector script
NDSB3_EXTRACTED_IMAGE_DIR = BASE_DIR_SSD + "ndsb3_extracted_images/"
LUNA16_EXTRACTED_IMAGE_DIR = BASE_DIR_SSD + "luna16_extracted_images/"
NDSB3_NODULE_DETECTION_DIR = BASE_DIR_SSD + "ndsb3_nodule_predictions/"
LUNA_NODULE_DETECTION_DIR = BASE_DIR_SSD + "luna16_nodule_predictions/"
LUNA_16_TRAIN_DIR = BASE_DIR_SSD+'luna_16_train_dir/'

# 各数据地址
BASE_DIR = 'D:/Mywork/'
DICOM_SRC_DIR = 'I:/src_dicom/'
DICOM_EXTRACT_DIR = 'D:/Mywork/data/extracted_image/'
TRAIN_COORD = 'D:/Mywork/coord/'
TRAIN_LABEL = 'D:/Mywork/data/generated_trainlabel/'
TRAIN_DATA = 'D:/Mywork/data/generated_traindata/'
PREDICT_CUBE = 'D:/Mywork/data/predict_cube/'

# 计时器
class Stopwatch(object):

    def start(self):
        self.start_time = Stopwatch.get_time()

    def get_elapsed_time(self):  # 计算已经走过的时间
        current_time = Stopwatch.get_time()
        res = current_time - self.start_time
        return res

    def get_elapsed_seconds(self): #计算已经走过的时间(按秒钟计算)
        elapsed_time = self.get_elapsed_time()
        res = elapsed_time.total_seconds()
        return res

    @staticmethod  # 静态方法
    def get_time():
        res = datetime.datetime.now()
        return res

    @staticmethod
    def start_new():  # 开始一个新的计时器
        res = Stopwatch()
        res.start()
        return res


