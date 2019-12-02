import os
import sys
sys.path.append(os.path.realpath("."))
print(sys.path)
import datetime
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from classifiers.graduate.resample_dataset import ClassificationDataset
from classifiers.graduate.full_con_classifier import configs,get_net


def predict():
    """
    进行预测
    """
    model_save_path = configs["model_save_path"]
    train_dir = configs["train_dir"]
    test_dir = configs["test_dir"]
    batch_size = configs["batch_size"]

    # 数据集
    dataset= ClassificationDataset(train_dir,test_dir,batch_size)
    dataset.prepare_test_dataset()
    test_data = dataset.get_test_dataset()  # test_data : [test_img_list,test_label_list]

    # 模型
    model = get_net(load_weight_path=model_save_path)

    # 开始计时
    start_time = datetime.datetime.now()
    predIdxs = model.predict(test_data[0])
    # print('predIdxs:', predIdxs)
    # print(len(predIdxs))

    # 测试花费时间
    current_time = datetime.datetime.now()
    res = current_time - start_time
    print("Done in : ", res.total_seconds(), " seconds")

    print(test_data[1].flatten())
    predict_label = np.argmax(predIdxs, axis=1)
    print(predict_label)
    # print(len(predict_label))

    # 计算预测的混淆矩阵
    confus_predict = confusion_matrix(test_data[1], predict_label)
    print(confus_predict)

    # 结果评估
    # target_names = ['class 1', 'class 2']
    target_names = ['class 1', 'class 2', 'class 3', 'class 4', 'class 5']
    classification_show = classification_report(test_data[1], predict_label, labels=None, target_names=target_names)
    print(classification_show)

if __name__ == "__main__":
    predict()
