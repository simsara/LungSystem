import os
import sys
sys.path.append(os.path.realpath("."))
print(sys.path)
import shutil
import random
import numpy as np
import glob  # 官方文档https://docs.python.org/3/library/glob.html
import tensorflow as tf 
from keras import backend as K
from keras.models import Model
from keras.backend.tensorflow_backend import set_session
from keras.metrics import categorical_accuracy, binary_crossentropy, binary_accuracy
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.optimizers import Adam, SGD
from keras.losses import categorical_crossentropy
from keras.layers import Input, Convolution3D, MaxPooling3D, Flatten, Dropout, AveragePooling3D, Dense
from keras import utils

import prepare.graduate.base_dicom_process as base_dicom_process

import configs.graduate.config_training as config 

# 关闭掉session中的其他模型 https://blog.csdn.net/ssswill/article/details/90211223
K.clear_session()


# tf.ConfigProto()函数用在创建session的时候，用来对session进行参数配置
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9  # fraction 是分数的意思
set_session(tf.Session(config=config))

LEARN_RATE = 0.001
USE_DROPOUT = False
log_dir = './logs/000'  # TODO  稍后检查log 文件
mean_pixel_values = 118 # 像素平均数
BATCH_SIZE = 8
TRAIN_DATA_ROOT = "/home/liubo/data/graduate/classification_dataset/train/"

# 生成训练和验证集图像地址列表并返回
def get_train_holdout_files(train_percentage=0.8, full_luna_set=False):
    # 不同的标签分别加载
    src_dir = TRAIN_DATA_ROOT 
    one_samples = glob.glob(src_dir+'one/' + "*.png")
    two_samples = glob.glob(src_dir+'two/' + "*.png")
    three_samples = glob.glob(src_dir+'three/' + "*.png")
    four_samples = glob.glob(src_dir+'four/' + "*.png")
    five_samples = glob.glob(src_dir+'zero/' + "*.png")        

    # 分出训练集和测试集
    one_sample_train = int(len(one_samples) * train_percentage)
    two_sample_train = int(len(two_samples) * train_percentage)
    three_sample_train = int(len(three_samples) * train_percentage)
    four_sample_train = int(len(four_samples) * train_percentage)
    five_sample_train = int(len(five_samples) * train_percentage)                                            

    samples_train = one_samples[:one_sample_train] + two_samples[:two_sample_train] + three_samples[:three_sample_train] \
                    + four_samples[:four_sample_train] + five_samples[:five_sample_train]
    samples_holdout = one_samples[one_sample_train:] + two_samples[two_sample_train:] + three_samples[:three_sample_train:] \
                      + four_samples[four_sample_train:] + five_samples[five_sample_train:]
    random.shuffle(samples_train)
    random.shuffle(samples_holdout)


    # 如果测试集也加入到训练集
    if full_luna_set:
        # pos_samples_train 变为所有数据
        samples_train += samples_holdout
        print('samples_train:', len(samples_train))


    # 加标签
    # TODO 可以优化
    train_res = []
    holdout_res = []
    for sample in samples_train:
        class_label = sample.split('_')[-2]
        if class_label == 'SCLC':
            train_res.append([sample, '0'])
        else:
            train_res.append([sample, class_label])
    for h_sample in samples_holdout:
        h_class_label = h_sample.split('_')[-2]
        if h_class_label == 'SCLC':
            holdout_res.append([h_sample, '0'])
        else:
            holdout_res.append([h_sample, h_class_label])   
    
    print("Train count: ", len(train_res), ", holdout count: ", len(holdout_res))
    return train_res, holdout_res 



def prepare_image_for_net3D(img):
    """
    输入网络前进行类型转换 以及归一化和reshape
    """
    img = img.astype(np.float32)
    img -= mean_pixel_values
    img /= 255.
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2], 1)
    return img


def data_generator(batch_size, record_list, is_train_set):
    """
    数据生成器 每次返回一个batch的数据
    """
    batch_count = 0 # 当计数到batch_size的时候统一返回
    means = []
    while True:
        img_list = []
        class_list = []
        if is_train_set:
            random.shuffle(record_list)
        for record_idx, record_item in enumerate(record_list):
            class_label = int(record_item[1])
            cube_image = base_dicom_process.load_cube_img(record_item[0], 8, 8, 64)
        
        # 以下为四种随机翻转方式
        if is_train_set:  
            if random.randint(0, 100) > 50:
                cube_image = np.fliplr(cube_image)
            if random.randint(0, 100) > 50:
                cube_image = np.flipud(cube_image)
            if random.randint(0, 100) > 50:
                cube_image = cube_image[:, :, ::-1]
            if random.randint(0, 100) > 50:
                cube_image = cube_image[:, ::-1, :]

        img3d = prepare_image_for_net3D(cube_image)
        img_list.append(img3d)
        class_list.append(class_label)

        batch_count += 1
        if batch_count >= batch_size:
            x = np.vstack(img_list)
            y_class = np.vstack(class_list)
            # 将标签转换为分类的 one-hot 编码
            one_hot_labels = utils.to_categorical(y_class, num_classes=5)
            # yield 是返回部分，详见python生成器解释
            yield x, {"out_class": one_hot_labels}
            img_list = []
            class_list = []
            batch_count = 0



# -> 标注函数返回类型   https://blog.csdn.net/orangefly0214/article/details/91583506
def get_net(input_shape=(64, 64, 64, 1), load_weight_path=None, features=False) -> Model:
    inputs = Input(shape=input_shape, name="input_1")
    x = inputs
    x = AveragePooling3D(strides=(2, 1, 1), pool_size=(2, 1, 1), padding="same")(x)

    x = Convolution3D(64, (3, 3, 3), activation='relu', strides=(1, 1, 1), padding='same', name='conv1', )(x)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid', name='pool1')(x)


    x = Convolution3D(128, (3, 3, 3), activation='relu', padding='same', name='conv2', strides=(1, 1, 1))(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool2')(x)

    if USE_DROPOUT:
        x = Dropout(rate=0.3)(x)

    x = Convolution3D(256, (3, 3, 3), activation='relu', padding='same', name='conv3a', strides=(1, 1, 1))(x)
    x = Convolution3D(256, (3, 3, 3), activation='relu', padding='same', name='conv3b', strides=(1, 1, 1))(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool3')(x)

    if USE_DROPOUT:
        x = Dropout(rate=0.4)(x)

    x = Convolution3D(512, (3, 3, 3), activation='relu', padding='same', name='conv4a', strides=(1, 1, 1))(x)
    x = Convolution3D(512, (3, 3, 3), activation='relu', padding='same', name='conv4b', strides=(1, 1, 1),)(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool4')(x)

    if USE_DROPOUT:
        x = Dropout(rate=0.5)(x)

    # 新加部分
    x = Convolution3D(512, (3, 3, 3), activation='relu', padding='same', name='conv5a', strides=(1, 1, 1))(x)
    x = Convolution3D(512, (3, 3, 3), activation='relu', padding='same', name='conv5b', strides=(1, 1, 1), )(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool5')(x)

    last64 = Convolution3D(64, (2, 2, 2), activation="relu", name="last_64")(x)

    out_class = Convolution3D(5, (1, 1, 1), activation="softmax", name="out_class_last")(last64)
    out_class = Flatten(name="out_class")(out_class)
    # 1*1*1*1：一个像素点，即一个值
    # out_malignancy = Convolution3D(1, (1, 1, 1), activation=None, name="out_malignancy_last")(last64)
    # out_malignancy = Flatten(name="out_malignancy")(out_malignancy)

    # 定义一个有一个输入一个输出的模型
    model = Model(inputs=inputs, outputs=out_class)
    if load_weight_path is not None:
        model.load_weights(load_weight_path, by_name=False)
    # 定义损失函数、优化函数、和评测方法
    # optimzer:SGD()是随机梯度下降以及对应参数
    # loss:计算损失函数，这里指定了两个损失函数，分别对应两个输出结果，out_class:binary_crossentropy,  out_malignancy:mean_absolute_error
    # metris：性能评估函数,这里指定了两个性能评估函数
    # binary_accuracy: 对二分类问题,计算在所有预测值上的平均正确率，binary_crossentropy是对数损失
    model.compile(optimizer=SGD(lr=LEARN_RATE, momentum=0.9, nesterov=True),
                  loss=categorical_crossentropy, metrics=[categorical_crossentropy, categorical_accuracy])

    if features:
        model = Model(input=inputs, output=[last64])
    # 打印出模型概况
    model.summary(line_length=140)
    return model



# 以epoch为参数，得到一个新的学习率
def step_decay(epoch):
    """
    step_decay 是传入LearningRateScheduler的一个函数参数
    """
    res = 0.001
    if epoch > 5:
        res = 0.0001
    print("learnrate: ", res, " epoch: ", epoch)
    return res


def train(model_name,train_full_set=False,load_weights_path=None):
    batch_size = BATCH_SIZE

    # train_full_set 被Liubo去掉了，认为它不科学
    train_files, holdout_files = get_train_holdout_files(train_percentage=0.8)

    # 生成训练和验证batch
    # train_gen 和 holdout_gen 结构：[x, {"out_class": y_class}]
    train_gen = data_generator(batch_size, train_files, True)
    holdout_gen = data_generator(batch_size, holdout_files, False)

    # keras 的动态学习率调度，learnrate_scheduler是一个新的学习率
    learnrate_scheduler = LearningRateScheduler(step_decay)
    model = get_net(load_weight_path=load_weights_path)

    # 每隔1轮保存一次模型
    print("###############ModelCheckpoint")
    checkpoint = ModelCheckpoint("workdir/cancer_classifier/model_" + model_name + "_" + "_e" + "{epoch:02d}-{val_loss:.4f}.hd5",
                                 monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    # 每隔一轮且每当val_loss降低时保存一次模型
    checkpoint_fixed_name = ModelCheckpoint("workdir/cancer_classifier/model_" + model_name + "_best.hd5",
                                            monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)


    print("len(train_files)",len(train_files))
    print("len(holdout_files)",len(holdout_files))
    model.fit_generator(generator=train_gen, steps_per_epoch=int(len(train_files)/batch_size), epochs=1000, validation_data=holdout_gen,
                        validation_steps=int(len(holdout_files)/batch_size), callbacks=[checkpoint, checkpoint_fixed_name, learnrate_scheduler, TensorBoard(log_dir=log_dir)])

    # model.save("../../models/graduate/model_" + model_name + "_end.hd5")
    model.save("model_" + model_name + "_end.hd5")




if __name__ == "__main__":
    # 模型要保存得路径
    # model_path = "../../models/graduate/model_cancer_type__fs_best.hd5"
    model_path = "model_cancer_type__fs_best.hd5"
    if not os.path.exists("models/"):
        os.mkdir("models")
    train(model_name="cancer_classifier",train_full_set=False)
    # TODO 路径更改
    # shutil.copy("models/graduate/model_cancer_type_best.hd5", "models/model_cancer_type_best.hd5")
    