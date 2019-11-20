
ORIGIN_DICOM_ROOT = "H:/medical_data_for_huodong/Images"



config  = {
    "origin_dicom_root":ORIGIN_DICOM_ROOT,
    "classification_data_root"
    "black_list":[]
}






import os
import datetime

# 工作进程数
WORKER_POOL_SIZE = 4
TARGET_VOXEL_MM = 1.00 
MEAN_PIXEL_VALUE_NODULE = 41
LUNA_SUBSET_START_INDEX = 0
SEGMENTER_IMG_SIZE = 320


# create the folder name ndsb3 for saving the corresponding results
BASE_DIR_SSD = "D:/everyproject/NDSB/kaggle_ndsb2017-master/ndsb3/"
# create the folder name ndsb4 for placing the input data here
BASE_DIR = "D:/everyproject/NDSB/kaggle_ndsb2017-master/ndsb4/"
# place here extra data given by julian in his repository.
EXTRA_DATA_DIR = "D:/everyproject/NDSB/kaggle_ndsb2017-master/kaggle_ndsb2017-master/resources/"
# place here the kaggle data which will further（CT）
NDSB3_RAW_SRC_DIR = BASE_DIR + "ndsb_raw/stage12/"
# place here the LUNA16 database
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

    def get_elapsed_time(self):
        current_time = Stopwatch.get_time()
        res = current_time - self.start_time
        return res

    def get_elapsed_seconds(self):
        elapsed_time = self.get_elapsed_time()
        res = elapsed_time.total_seconds()
        return res

    @staticmethod
    def get_time():
        res = datetime.datetime.now()
        return res

    @staticmethod
    def start_new():
        res = Stopwatch()
        res.start()
        return res


