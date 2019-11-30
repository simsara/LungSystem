import os
import sys
sys.path.append(os.path.realpath("."))
print(sys.path)
import datetime
import glob
import random

import numpy
from keras import utils
from keras import backend as K
from sklearn.metrics import confusion_matrix, classification_report


# limit memory usage..
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import mingyue_classifier

import prepare.graduate.base_dicom_process as base_dicom_process

K.clear_session()

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

# 改变图像维度顺序为tensorflow维度顺序（height，width，channels）
K.set_image_dim_ordering("tf")
CUBE_SIZE = 64
MEAN_PIXEL_VALUE = 118
P_TH = 0.6
PREDICT_STEP = 12
USE_DROPOUT = False
CROP_SIZE = 64

# 生成训练和验证集图像地址列表并返回
def get_test_files(test_src):

    zero_samples = glob.glob(test_src+'zero/' + '*.png')
    one_samples = glob.glob(test_src + 'one/' + "*.png")
    two_samples = glob.glob(test_src + 'two/' + "*.png")
    three_samples = glob.glob(test_src + 'three/' + "*.png")
    four_samples = glob.glob(test_src + 'four/' + "*.png")
    samples = zero_samples[:50] + one_samples[:50] + two_samples[:50] + three_samples[:50] + four_samples[:50]

    print(samples)
    print("test samples count : ", len(samples))

    test_res = []
    for sample in samples:
        class_label = sample.split('_')[-2]
        if class_label == 'SCLC':
            test_res.append([sample, '0'])
        else:
            test_res.append([sample, class_label])

    print("Test count: ", len(test_res))
    return test_res

# 这是一个训练数据和验证数据生成器，每个batch返回一次数据
def data_generator(batch_size, record_list, train_set):
    batch_idx = 0
    while True:
        img_list = []
        # class_list 是“肺癌类型”的标签集合[1，2，3，4, 5]
        class_list = []
        CROP_SIZE = CUBE_SIZE       # 32

        # 对每张图片进行遍历
        for record_idx, record_item in enumerate(record_list):

            class_label = int(record_item[1])
            # cube_image : 64*64*64
            cube_image = base_dicom_process.load_cube_img(record_item[0], 8, 8, 64)

            current_cube_size = cube_image.shape[0]  # 64
            indent_x = (current_cube_size - CROP_SIZE) / 2  # 16
            indent_y = (current_cube_size - CROP_SIZE) / 2  # 16
            indent_z = (current_cube_size - CROP_SIZE) / 2  # 16
            wiggle_indent = 0
            wiggle = current_cube_size - CROP_SIZE - 1  # 31
            if wiggle > (CROP_SIZE / 2):
                wiggle_indent = CROP_SIZE / 4  # 8
                wiggle = current_cube_size - CROP_SIZE - CROP_SIZE / 2 - 1  # 15
            if train_set:
                indent_x = wiggle_indent + random.randint(0, wiggle)
                indent_y = wiggle_indent + random.randint(0, wiggle)
                indent_z = wiggle_indent + random.randint(0, wiggle)

            indent_x = int(indent_x)
            indent_y = int(indent_y)
            indent_z = int(indent_z)
            # 在64*64*64的立方体中，随机裁剪出32*32*32的小立方体（这里好像不太随机，小立方体像素范围是（8~54）
            cube_image = cube_image[indent_z:indent_z + CROP_SIZE, indent_y:indent_y + CROP_SIZE,
                         indent_x:indent_x + CROP_SIZE]
            assert cube_image.shape == (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)

            # cube_image.mean() 计算三维矩阵所有数的平均数，结果是一个数
            # means.append(cube_image.mean())
            # cube_image 是 32*32 *32
            # img3d 为 1*32*32*32*1
            img3d = prepare_image_for_net3D(cube_image)
            img_list.append(img3d)
            class_list.append(class_label)

            batch_idx += 1
            if batch_idx >= batch_size:
                x = numpy.vstack(img_list)
                y_class = numpy.vstack(class_list)
                # 将标签转换为分类的 one-hot 编码
                one_hot_labels = utils.to_categorical(y_class, num_classes=6)
                yield x, {"out_class": one_hot_labels}
                img_list = []
                class_list = []
                batch_idx = 0


# 对结果矩阵进行统计，得到不同结果数量
def sort_predict(test_list):
    a = list(test_list.flatten())
    a_list = list(set(a))
    re_list = []
    for i in a_list:
        counts = a.count(i)
        print(counts)
        re_list.append([i, counts])
    return re_list

def prepare_image_for_net3D(img):
    img = img.astype(numpy.float32)
    img -= MEAN_PIXEL_VALUE
    img /= 255.
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2], 1)
    return img

def load_testdata(record_list):

    img_list = []
    # class_list 是“肺癌类型”的标签集合[1，2，3，4, 5]
    class_list = []
    CROP_SIZE = CUBE_SIZE  # 32

    # 对每张图片进行遍历
    for record_idx, record_item in enumerate(record_list):

        class_label = int(record_item[1])
        # cube_image : 64*64*64
        cube_image = base_dicom_process.load_cube_img(record_item[0], 8, 8, 64)

        current_cube_size = cube_image.shape[0]  # 64
        indent_x = (current_cube_size - CROP_SIZE) / 2  # 16
        indent_y = (current_cube_size - CROP_SIZE) / 2  # 16
        indent_z = (current_cube_size - CROP_SIZE) / 2  # 16
        wiggle_indent = 0
        wiggle = current_cube_size - CROP_SIZE - 1  # 31
        if wiggle > (CROP_SIZE / 2):
            wiggle_indent = CROP_SIZE / 4  # 8
            wiggle = current_cube_size - CROP_SIZE - CROP_SIZE / 2 - 1  # 15

        indent_x = int(indent_x)
        indent_y = int(indent_y)
        indent_z = int(indent_z)
        # 在64*64*64的立方体中，随机裁剪出32*32*32的小立方体（这里好像不太随机，小立方体像素范围是（8~54）
        cube_image = cube_image[indent_z:indent_z + CROP_SIZE, indent_y:indent_y + CROP_SIZE,
                     indent_x:indent_x + CROP_SIZE]
        assert cube_image.shape == (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)

        # cube_image.mean() 计算三维矩阵所有数的平均数，结果是一个数
        # means.append(cube_image.mean())
        # cube_image 是 32*32 *32
        # img3d 为 1*32*32*32*1
        img3d = prepare_image_for_net3D(cube_image)
        img_list.append(img3d)
        class_list.append(class_label)
        x = numpy.vstack(img_list)
        y_class = numpy.vstack(class_list)
    return [x, class_list]


# # 生成器生成批量数据进行预测
# def predict_with_generator():
#     batch_size = 16
#     test_src = 'D:/Mywork/data/generated_testdata/'
#     model_path = "models/model_cancer_type__fs_best.hd5"
#     test_files = get_test_files(test_src)

#     # 开始计时
#     sw = base_settings.Stopwatch.start_new()
#     # 生成训练和验证batch
#     # train_gen 和 holdout_gen 结构：[x, {"out_class": y_class}]
#     test_gen = data_generator(batch_size, test_files, False)

#     # 导入模型
#     model = mingyue_classifier.get_net(input_shape=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 1),
#                                                load_weight_path=model_path)

#     predIdxs = model.predict_generator(test_gen, steps=int(len(test_files) / batch_size))
#     print('predIdxs:', predIdxs)
#     print(len(predIdxs))

#     # 测试花费时间
#     print("Done in : ", sw.get_elapsed_seconds(), " seconds")

# 不用生成器进行预测
def predict_without_generator():

    test_src = "/home/liubo/data/graduate/classification_dataset/test/"
    model_path = "/home/liubo/nn_project/LungSystem/workdir/cancer_classifier/model_cancer_classifier_best.hd5"
    test_files = get_test_files(test_src)

    test_data = load_testdata(test_files)

    # 导入模型
    model = mingyue_classifier.get_net(input_shape=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 1),load_weight_path=model_path)

    # 开始计时
    start_time = datetime.datetime.now()
    predIdxs = model.predict(test_data[0])
    print('predIdxs:', predIdxs)
    print(len(predIdxs))

    # 测试花费时间
    current_time = datetime.datetime.now()
    res = current_time - start_time
    print("Done in : ", res.total_seconds(), " seconds")

    print(test_data[1])
    predict_label = numpy.argmax(predIdxs, axis=1)
    print(predict_label)
    print(len(predict_label))

    # 计算预测的混淆矩阵
    confus_predict = confusion_matrix(test_data[1], predict_label)
    print(confus_predict)

    # 结果评估
    # target_names = ['class 1', 'class 2']
    target_names = ['class 1', 'class 2', 'class 3', 'class 4', 'class 5']
    classification_show = classification_report(test_data[1], predict_label, labels=None, target_names=target_names)
    print(classification_show)

if __name__ == "__main__":
    predict_without_generator()

