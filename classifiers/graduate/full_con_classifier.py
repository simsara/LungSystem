import os
import sys
sys.path.append(os.path.realpath("."))
print(sys.path)
from classifiers.graduate.resample_dataset import ClassificationDataset
from keras.models import Model 
from keras.layers import Input, Convolution3D, MaxPooling3D, Flatten, Dropout, AveragePooling3D, Dense
from keras.metrics import categorical_accuracy, binary_crossentropy
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.optimizers import Adam, SGD
from keras.losses import categorical_crossentropy



configs = {
    "train_dir":"/home/liubo/data/graduate/classification_dataset/train/",
    "test_dir" :"/home/liubo/data/graduate/classification_dataset/test/",
    "batch_size" : 8,
    "log_dir":"./logs/000",
    "model_name":"cancer_classifier",
    "model_save_path":"/home/liubo/nn_project/LungSystem/models/guaduate/model_cancer_classifier_best.hd5",
    "learn_rate":0.0001
}


def get_net(input_shape=(64, 64, 64, 1), load_weight_path=None, features=False) -> Model:
    """
    获取模型结构
    """
    inputs = Input(shape=input_shape, name="input_1")
    x = inputs
    x = AveragePooling3D(strides=(2, 1, 1), pool_size=(2, 1, 1), padding="same")(x)

    x = Convolution3D(64, (3, 3, 3), activation='relu', strides=(1, 1, 1), padding='same', name='conv1', )(x)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid', name='pool1')(x)

    x = Convolution3D(128, (3, 3, 3), activation='relu', padding='same', name='conv2', strides=(1, 1, 1))(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool2')(x)

    x = Convolution3D(256, (3, 3, 3), activation='relu', padding='same', name='conv3a', strides=(1, 1, 1))(x)
    x = Convolution3D(256, (3, 3, 3), activation='relu', padding='same', name='conv3b', strides=(1, 1, 1))(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool3')(x)

    x = Convolution3D(512, (3, 3, 3), activation='relu', padding='same', name='conv4a', strides=(1, 1, 1))(x)
    x = Convolution3D(512, (3, 3, 3), activation='relu', padding='same', name='conv4b', strides=(1, 1, 1),)(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool4')(x)

    x = Convolution3D(512, (3, 3, 3), activation='relu', padding='same', name='conv5a', strides=(1, 1, 1))(x)
    x = Convolution3D(512, (3, 3, 3), activation='relu', padding='same', name='conv5b', strides=(1, 1, 1), )(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool5')(x)

    last64 = Convolution3D(64, (2, 2, 2), activation="relu", name="last_64")(x)

    out_class = Convolution3D(5, (1, 1, 1), activation="softmax", name="out_class_last")(last64)
    out_class = Flatten(name="out_class")(out_class)
    
    model = Model(inputs=inputs, outputs=out_class)

    if load_weight_path is not None:  
        model.load_weights(load_weight_path, by_name=False)

    model.compile(optimizer=SGD(lr=configs["learn_rate"], 
                                momentum=0.9, 
                                nesterov=True),
                  loss=categorical_crossentropy, 
                  metrics=[categorical_crossentropy, categorical_accuracy])

    model.summary(line_length=140)
    return model


def train(model_name,load_weight_path=None):
    """
    训练模型 
    :param model_name: 模型名称
    :param load_weight_path:之前的预训练模型
    """
    train_dir = configs["train_dir"]
    test_dir = configs["test_dir"]
    batch_size = configs["batch_size"]
    dataset= ClassificationDataset(train_dir,test_dir,batch_size)
    dataset.prepare_train_val_dataset()
    train_resampled,val_resampled = dataset.get_resampled_train_val_dataset(train_n=80,val_n=20)
    train_gen = dataset.data_generator(batch_size=8,record_list=train_resampled,is_train_set=True)
    val_gen = dataset.data_generator(batch_size=8,record_list=val_resampled,is_train_set=False)

    model = get_net()
    # 每隔1轮保存一次模型
    checkpoint = ModelCheckpoint(filepath="/home/liubo/nn_project/LungSystem/workdir/cancer_classifier/model_" + model_name + "_" + "_e" + "{epoch:02d}-{val_loss:.4f}.hd5",
                                 monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    # 每隔一轮且每当val_loss降低时保存一次模型
    checkpoint_fixed_name = ModelCheckpoint("/home/liubo/nn_project/LungSystem/workdir/cancer_classifier/model_" + model_name + "_best.hd5",
                                            monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    model.fit_generator(generator=train_gen, 
                        steps_per_epoch=int(len(train_resampled)/batch_size), 
                        epochs=1000, 
                        validation_data=val_gen,
                        validation_steps=int(len(val_resampled)/batch_size), 
                        callbacks=[checkpoint, 
                                   checkpoint_fixed_name, 
                                   TensorBoard(log_dir=configs["log_dir"])])



if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    model_path_dir = "/home/liubo/nn_project/LungSystem/models/guaduate/"
    if not os.path.exists("models/guaduate/"):
        os.mkdir("models/guaduate/")
    model_name = "model_full_con_classifier_best.hd5"
    train(model_name=configs["model_name"])
    # TODO copy best model


    
    