import configs.luna16.config_training as base_dicom_process
import os
import math
import numpy
import glob
import pandas
import ntpath
import cv2
import pydicom
import shutil
import skimage.transform
from scipy import ndimage


# 检查new_right_axis.txt文件中的ID是否存在对应CT源文件，不存在的没有意义
def inspect_dicom():
    without_list = []
    dicom_dir = 'I:/src_dicom/'
    inspect_dir = 'new_right_axis.txt'
    id_list = get_usecoord_withoutrepeat(inspect_dir)
    dicom_list = os.listdir(dicom_dir)
    for patientid in id_list:
        if patientid not in dicom_list:
            without_list.append(patientid)
            # with open('D:/Mywork/image_coord_regenerate/without_dicom.txt', "a", encoding="UTF-8") as target:
            #     target.write(patientid + '\n')
    print('没有CT源文件的病人共有：', len(without_list))
    print('具体ID如下：')
    print(without_list)

# 得到CT加载sort后某张图像具体的Z坐标
def get_ct_older(patientid, dicom_name):
    number_dicom = -1
    dicom_path = 'I:/src_dicom/'+patientid+'/'
    slices = load_patient(dicom_path)
    # print(slices[0].pixel_array)
    # print(slices[0].SOPInstanceUID)
    for i in range(0, len(slices)):
        my_dicom = str(slices[i].SOPInstanceUID)
        if dicom_name == my_dicom:
            number_dicom = i+1
    return number_dicom

# 针对单个坐标转换:改变Z坐标为 len（ct）-z
def filter_one_coord(axis_z, patientID):
    ct_src_dir = 'I:/src_dicom/'
    # ct_list = os.listdir(ct_src_dir)
    number_ct = os.listdir(ct_src_dir + patientID + '/')
    # print('number of ct :', len(number_ct))
    new_coord_z = len(number_ct) - int(axis_z)
    return new_coord_z

# 提取指定文件坐标（x,y,z,lung_type）:可能存在多个
def get_xyz(patientID):
    coord_dir = 'D:/Mywork/image_coord_regenerate/coord/last_train_coord.txt'
    f = open(coord_dir, 'rb')
    line = f.readline().decode('UTF-8')

    coord_zxyt = []
    while line:
        patient = str(line.split(':')[0])
        if patientID == patient:
            coord_x = str(line.split(':')[1])
            coord_y = str(line.split(':')[2])
            coord_z = str(line.split(':')[3])
            lung_type = str(line.split(':')[4])
            noudle_r = int(float(str(line.split(':')[5]).replace('\r', '').replace('\n', '')))
            if lung_type in ['1', '2', '3', '4', 'SCLC']:
                coord_zxyt.append([coord_x, coord_y, coord_z, lung_type, noudle_r])
        line = f.readline().decode('UTF-8')
    f.close()
    return coord_zxyt

# 得到无重复的有坐标提取信息的病人id
def get_usecoord_withoutrepeat(txt_name):
    src_dir = 'D:/Mywork/image_coord_regenerate/new_coord/'
    src_dir = src_dir+txt_name
    id_list = []
    f = open(src_dir, 'rb')
    line = f.readline().decode('UTF-8')
    while line:
        patient_id = str(line.split(':')[0])

        if patient_id not in id_list:
            id_list.append(patient_id)
        line = f.readline().decode('UTF-8')
    f.close()
    return id_list

# 由序号得到具体dicom图像的名称
def get_dicom_name(patientid, dicom_number):
    ct_path = 'I:/src_dicom/'+patientid+'/'
    ct_list = os.listdir(ct_path)
    ct_list.sort(key=lambda x: int(str(x.split('.')[-3])+str(x.split('.')[-2]).rjust(3, '0')))
    # print(ct_list)
    return ct_list[dicom_number-1]

# 从原坐标文件里筛选出z坐标对应的文件名，转存成新文件
# 该函数使用new_right_axis.txt
# 生成first_coord.txt
def filter_coord_one():

    src_dir = 'D:/Mywork/image_coord_regenerate/coord/new_right_axis.txt'
    f = open(src_dir, 'rb')
    line = f.readline().decode('UTF-8')
    while line:
        patientid = str(line.split(':')[0])
        old_z = int(line.split(':')[1].split('-')[-1].split('.')[0])
        dicom_name = get_dicom_name(patientid, old_z)
        dicom_name = str(dicom_name).replace('.dcm', '')
        print(dicom_name)
        lung_type = str(line.split(':')[1].split('-')[1])
        coord_x = int(line.split(':')[-1].split('-')[0])
        coord_y = int(line.split(':')[-1].split('-')[1])
        noudle_r = int(line.split(':')[-1].split('-')[2])
        new_line = patientid + ':' + dicom_name + ':' + str(coord_x) + ':' + str(coord_y)+':'+lung_type+':'+str(noudle_r)
        with open('D:/Mywork/image_coord_regenerate/coord/first_coord.txt', "a", encoding="UTF-8") as target:
            target.write(new_line+'\n')
        line = f.readline().decode('UTF-8')

    f.close()

# 读取load，然后sort的CT目录，依据first_coord.txt中的文件名，对应到实际Z坐标
# 该函数使用first_coord.txt
# 生成second_coord.txt
def filter_coord_two():

    src_dir = 'D:/Mywork/image_coord_regenerate/coord/first_coord.txt'
    f = open(src_dir, 'rb')
    line = f.readline().decode('UTF-8')
    while line:
        patientid = str(line.split(':')[0])
        dicom_name = str(line.split(':')[1])
        print(dicom_name)
        dicom_name = dicom_name.replace(dicom_name.split('.')[0], '1')
        lung_type = str(line.split(':')[-2])
        coord_x = int(line.split(':')[2])
        coord_y = int(line.split(':')[3])
        noudle_r = int(line.split(':')[-1])

        coord_z = get_ct_older(patientid, dicom_name)
        if coord_z != -1:
            new_line = patientid + ':' + str(coord_x) + ':' + str(coord_y)+':'+str(coord_z)+':'+lung_type+':'+str(noudle_r)
            with open('D:/Mywork/image_coord_regenerate/coord/second_coord.txt', "a", encoding="UTF-8") as target:
                target.write(new_line+'\n')
        else:
            print(patientid+' do not have dicom!')
        line = f.readline().decode('UTF-8')

    f.close()

# 遍历
def extract_dicom_axis(clean_targetdir_first=False, only_patient_id=None):
    print("Extracting noudle axis and type for train")
    # CT noudle坐标和类型存储目录
    target_dir = 'D:/Mywork/image_coord_regenerate/coord/'
    # 如果target_dir目录下存在文件，则进入if,删除该目录下所有文件
    if clean_targetdir_first and only_patient_id is None:
        print("Cleaning train_coord.txt")
        # 获取target_dir目录下的所有格式文件
        if os.path.exists(target_dir + "train_coord.txt"):
            # 删除该文件
            os.remove(target_dir + "train_coord.txt")

    if only_patient_id is None:
        # 要遍历的id目录
        txt_dir = 'sign_label.txt'
        dirs = get_usecoord_withoutrepeat(txt_dir)
        print('dirs count is : ', len(dirs))
        if dirs is not None:
            for ct_dir in dirs:
                # print(ct_dir)
                extract_dicom_axis_patient(ct_dir)
    else:
        # print('haha')
        extract_dicom_axis_patient(only_patient_id)

# 得到重采样后的坐标并保存
# train_coord.txt
def extract_dicom_axis_patient(src_dir):
    dir_path = 'I:/src_dicom/' + src_dir + "/"
    patient_id = src_dir
    slices = load_patient(dir_path)
    # print(len(slices), "\t", slices[0].SliceThickness, "\t", slices[0].PixelSpacing)
    # print("Orientation: ", slices[0].ImageOrientationPatient)
    cos_value = (slices[0].ImageOrientationPatient[0])

    # round 函数是返回浮点数的四舍五入值
    # acos() 函数返回一个数的反余弦值，单位是弧度
    # degrees() 函数将弧度转换为角度
    cos_degree = round(math.degrees(math.acos(cos_value)), 2)
    if cos_degree > 0:
        print(patient_id, ': cos_degree > 0')
        assert False, '坐标有角度偏转'
    pixels = base_dicom_process.get_pixels_hu(slices)
    image = pixels
    # img_zxy_shape = image.shape
    # print(img_zxy_shape)

    # true 为没有倒置， false为倒置
    invert_order = slices[1].ImagePositionPatient[2] > slices[0].ImagePositionPatient[2]
    # 图像是否倒置
    print("Invert order: ", invert_order, " - ", slices[1].ImagePositionPatient[2], ",", slices[0].ImagePositionPatient[2])

    pixel_spacing = slices[0].PixelSpacing
    pixel_spacing.append(slices[0].SliceThickness)


    # 得到缩放后的病灶坐标以及肺癌类型（z,x,y,type）,四舍五入取整
    coord_xyzt = get_xyz(patient_id)
    if coord_xyzt is not None:
        for coord in coord_xyzt:
            coord[0] = str(int(pixel_spacing[0] * int(coord[0]) + 0.5))
            coord[1] = str(int(pixel_spacing[1] * int(coord[1]) + 0.5))
            coord[2] = str(int(pixel_spacing[2] * int(coord[2]) + 0.5))

            if not invert_order:
                # 图像反转
                image = numpy.flipud(image)
                print(patient_id + ' 倒置！')
                coord[2] = filter_one_coord(coord[2], patient_id)

            # print(coord_zxy)
            # 保存用于训练用的（z,x,y）
            with open('D:/Mywork/image_coord_regenerate/coord/train_coord.txt', "a", encoding="UTF-8") as target:
                save_str = str(patient_id)+':'+str(coord[0])+':'\
                           + str(coord[1])+':'+str(coord[2])+':'+str(coord[3]).replace('\r', '').replace('\n', '')
                target.write(save_str+'\n')
        print(patient_id, 'is over!')
    else:
        print(patient_id, 'can not found!')
    return None


# 将txt格式存储的训练坐标存储为csv格式
def extract_train_csv(clean_targetdir_first=True, only_patient_id=None):
    print("Extracting noudle axis and type for train")
    # CT noudle坐标和类型存储目录
    target_dir = 'D:/Mywork/image_coord_regenerate/newest_img_coord/img_csv/'
    # 如果target_dir目录下存在文件，则进入if,删除该目录下所有文件
    if clean_targetdir_first and only_patient_id is None:
        print("Cleaning patientID_annotation.csv")
        # 获取target_dir目录下的所有格式文件
        for f_path in glob.glob(target_dir+'*.*'):
            # 删除该文件
            os.remove(f_path)

    if only_patient_id is None:
        # CT 文件夹
        dirs = get_usecoord_withoutrepeat('last_train_coord.txt')
        print('dirs:', dirs)
        if dirs is not None:
            for ct_dir in dirs:
                txt_to_csv(ct_dir)
    else:
        txt_to_csv(only_patient_id)

# 将坐标存储为csv文件
def txt_to_csv(patientID):

    coord_dir = 'D:/Mywork/image_coord_regenerate/newest_img_coord/last_train_coord.txt'
    f = open(coord_dir, 'rb')
    line = f.readline().decode('UTF-8')
    all_lines = []
    # print('first patientID is:', patientID, len(patientID))
    # print('patientID length is:', len(patientID))
    while line:
        patient = str(line.split(':')[0]).replace(' ', '')
        # print('patient:', patient, ':', len(patient))
        if patientID == patient:
            coord_x = line.split(':')[1]
            coord_y = line.split(':')[2]
            coord_z = line.split(':')[3]
            lung_type = str(line.split(':')[4])
            lung_r = int(float(str(line.split(':')[5]).replace('\r', '').replace('\n', '')))
            # print('patientID:', patientID)
            all_lines.append([patientID, lung_type, coord_x, coord_y, coord_z, lung_r])
            print(patientID+':'+lung_type)
            line = f.readline().decode('UTF-8')
        else:
            line = f.readline().decode('UTF-8')
    f.close()
    df_annos = pandas.DataFrame(all_lines, columns=["patient_id", "luna_type", "coord_x", "coord_y", "coord_z", "luna_r"])
    df_annos.to_csv('D:/Mywork/image_coord_regenerate/newest_img_coord/img_csv/' + patientID + "_annotations.csv", index=False, encoding='UTF-8')

# 依据肿瘤的下，(x,y,z)坐标，以该座标为中心，裁剪出64*64*64的cube
# x,y,z可分别用图片的第（x,y）像素点，第z张图表示（前提是把原CT图像素间距调整好）
def get_cube_from_img(img3d, center_x, center_y, center_z, block_size):
    start_x = max(center_x - block_size / 2, 0)
    if start_x + block_size > img3d.shape[2]:
        start_x = img3d.shape[2] - block_size

    start_y = max(center_y - block_size / 2, 0)
    start_z = max(center_z - block_size / 2, 0)
    if start_z + block_size > img3d.shape[0]:
        start_z = img3d.shape[0] - block_size
    start_z = int(start_z)
    start_y = int(start_y)
    start_x = int(start_x)
    res = img3d[start_z:start_z + block_size, start_y:start_y + block_size, start_x:start_x + block_size]
    return res

# 将cube逐个排列在一起保存为图片
def save_cube_img(target_path, cube_img, rows, cols):
    assert rows * cols == cube_img.shape[0]
    img_height = cube_img.shape[1]
    img_width = cube_img.shape[1]
    res_img = numpy.zeros((rows * img_height, cols * img_width), dtype=numpy.uint8)

    for row in range(rows):
        for col in range(cols):
            target_y = row * img_height
            target_x = col * img_width
            res_img[target_y:target_y + img_height, target_x:target_x + img_width] = cube_img[row * cols + col]

    cv2.imwrite(target_path, res_img)

def make_annotation_images():
    src_dir = 'D:/Mywork/image_coord_regenerate/newest_img_coord/img_csv/'
    dst_dir = 'D:/Mywork/image_coord_regenerate/newest_img_coord/crop_img/'
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    # 清空该目录
    for file_path in glob.glob(dst_dir + "*.*"):
        os.remove(file_path)

    for patient_index, csv_file in enumerate(glob.glob(src_dir + "*_annotations.csv")):
        patient_id = ntpath.basename(csv_file).replace("_annotations.csv", "")
        df_annos = pandas.read_csv(csv_file)
        print(patient_id, 'len(df_annos): ', len(df_annos))
        if len(df_annos) == 0:
            continue
        # 从 extracted_image 读取图像并将一个病人的图像合在一起，例如：349张330*330的图像，返回一个（349，330，330）的三维矩阵（z,y,x）
        images = base_dicom_process.load_patient_images(patient_id, 'D:/Mywork/image_coord_regenerate/extract_img/', "*" + '_i' + ".png")
        if images is None:
            continue
        print('images shape is:', images.shape)
        for index, row in df_annos.iterrows():
            coord_x = int(row["coord_x"])
            coord_y = int(row["coord_y"])
            coord_z = int(row["coord_z"])
            luna_type = str(row["luna_type"])
            if luna_type == 'SCLC':
                luna_type = '0'
            luna_r = int(row["luna_r"])
            print('coord_x:', coord_x)
            print('coord_y:', coord_y)
            print('coord_z:', coord_z)
            print('luna_type:', luna_type)
            print('luna_r:', luna_r)
            cube_img = get_cube_from_img(images, coord_x, coord_y, coord_z, 3*luna_r)
            if cube_img.sum() < 5:
                print(" ***** Skipping ", coord_x, coord_y, coord_z)
                continue

            if cube_img.mean() < 10:
                print(" ***** Suspicious ", coord_x, coord_y, coord_z)

            if cube_img.shape != (3*luna_r, 3*luna_r, 3*luna_r):
                print(" ***** incorrect shape !!! ", str(luna_type), " - ",(coord_x, coord_y, coord_z))
                continue

            save_cube_img(dst_dir + patient_id + "_" + str(int(float(luna_type))) + '_' + str(index) + ".png", cube_img, rows=3, cols=luna_r)

# 得到CT扫描切片厚度，加入到切片信息中
def load_patient(src_dir):
    # src_dir是病人CT文件夹地址
    slices = [pydicom.read_file(src_dir + '/' + s) for s in os.listdir(src_dir)]
    # slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = numpy.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = numpy.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices

def load_patient2(src_dir):
    # src_dir是病人CT文件夹地址
    slices = [pydicom.read_file(src_dir + '/' + s) for s in os.listdir(src_dir)]
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
    try:
        slice_thickness = numpy.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = numpy.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices

# 按比例分离训练集和测试集
def depart_train_test(src_dir, percentage):

    dst_dir = 'D:/Mywork/data/generated_testdata/'
    count_list = glob.glob(src_dir+'*.png')
    quantity = len(count_list)
    print(quantity)
    test_quantity = int(quantity * percentage)
    for i in count_list[0:test_quantity]:
        sample = i.split('\\')[-1]
        print(sample)
        shutil.copyfile(i, dst_dir+sample)
    print('test data saved!')

# 不同种类数据集分开存储
def depart_save_type(src_dir):
    samples = os.listdir(src_dir)
    for name in samples:
        sample_type = str(name.split('_')[1])
        if sample_type in ['1', '2', '3', '4', 'SCLC', '0']:
            if sample_type == '1':
                shutil.copyfile(src_dir+name, 'D:/Mywork/image_coord_regenerate/newest_img_coord/different_type/one/'+name)
            elif sample_type == '2':
                shutil.copyfile(src_dir + name, 'D:/Mywork/image_coord_regenerate/newest_img_coord/different_type/two/' + name)
            elif sample_type == '3':
                shutil.copyfile(src_dir + name, 'D:/Mywork/image_coord_regenerate/newest_img_coord/different_type/three/' + name)
            elif sample_type == '4':
                shutil.copyfile(src_dir + name, 'D:/Mywork/image_coord_regenerate/newest_img_coord/different_type/four/' + name)
            elif sample_type == 'SCLC' or sample_type == '0':
                shutil.copyfile(src_dir + name, 'D:/Mywork/image_coord_regenerate/newest_img_coord/different_type/zero/' + name)
            else:
                print('It is Unqualified!')
        print(name, 'is over!')

# 读取train_coord.txt文件信息
def get_name_list():
    all_lines = []
    coord_dir = 'D:/Mywork/image_coord_regenerate/coord/train_coord.txt'
    f = open(coord_dir, 'rb')
    line = f.readline().decode('UTF-8')

    while line:
        patient = str(line.split(':')[0])
        coord_x = str(line.split(':')[1])
        coord_y = str(line.split(':')[2])
        coord_z = str(line.split(':')[3])
        lung_type = str(line.split(':')[4].replace('\r', '').replace('\n', ''))
        all_lines.append([patient, coord_x, coord_y, coord_z, lung_type])
        # print(patient)
        line = f.readline().decode('UTF-8')

    f.close()
    return all_lines

# 读取get_t.txt文件信息
def get_r_list():
    all_lines = []
    coord_dir = 'D:/Mywork/image_coord_regenerate/coord/get_r.txt'
    f = open(coord_dir, 'rb')
    line = f.readline().decode('UTF-8')

    while line:
        patient = str(line.split(':')[0])
        luna_r = str(line.split(':')[1].replace('\r', '').replace('\n', ''))
        all_lines.append([patient, luna_r])
        # print(patient)
        line = f.readline().decode('UTF-8')

    f.close()
    return all_lines

# 得到重采样后的病灶半径
def get_r():
    id_list = get_usecoord_withoutrepeat('train_coord.txt')
    dicom_path = 'I:/src_dicom/'
    reasmple_path = 'D:/Mywork/data/extracted_image/img_0000_i.png'
    r_list = []
    for patientid in id_list:
        # 读取CT缩放比
        slices_path = glob.glob(dicom_path+patientid+'/*.*')
        slice = pydicom.read_file(slices_path[0])
        img_scale = slice.PixelSpacing
        print(img_scale)
        # 读取缩放前半径
        imgs_r = get_xyz(patientid)
        for img_r in imgs_r:
            print(patientid+':'+img_r[-1])
            new_img_r = numpy.round(int(img_r[-1])*img_scale[0])
            new_line = patientid+':'+str(new_img_r)
            with open('D:/Mywork/image_coord_regenerate/coord/get_r.txt', "a", encoding="UTF-8") as target:
                target.write(new_line+'\n')
            r_list.append([patientid, str(new_img_r)])
    print(r_list)
    print(len(r_list))

# 把半径加到对应训练txt中
def add_r():
    id_list = get_usecoord_withoutrepeat('train_coord.txt')
    traintxt_list = get_name_list()
    print(traintxt_list)
    print(len(traintxt_list))
    r_list = get_r_list()
    print(r_list)
    print(len(r_list))
    count = 0
    for i in range(0, len(r_list)):
        traintxt_list[i].append(r_list[i][1])
        new_line = str(traintxt_list[i][0]+':'+traintxt_list[i][1]+':'+traintxt_list[i][2]+':'+traintxt_list[i][3]+':'+traintxt_list[i][4]+':'+r_list[i][1])
        with open('D:/Mywork/image_coord_regenerate/coord/last_train_coord.txt', "a", encoding="UTF-8") as target:
            target.write(new_line + '\n')

    print(traintxt_list)
    print(len(traintxt_list))

# 读取裁剪图像（尺寸不统一）
def load_cube_img(src_path, rows=3):
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    print(img.shape)
    size = int(img.shape[0]/rows)
    cols = int(size/3)
    res = numpy.zeros((rows * cols, size, size))

    img_height = size
    img_width = size

    for row in range(rows):
        for col in range(cols):
            src_y = row * img_height
            src_x = col * img_width
            # res[0] = img[0:48,0:48], res[1] = img[0:48, 48:96], res[7] = [0:48, 336:384]
            res[row * cols + col] = img[src_y:src_y + img_height, src_x:src_x + img_width]

    return res

def uniform_size():
    imgs_path = 'D:/Mywork/image_coord_regenerate/newest_img_coord/crop_img/'
    targets_path = 'D:/Mywork/image_coord_regenerate/newest_img_coord/resize_img/'
    name_list = os.listdir(imgs_path)
    for name in name_list:
        img = load_cube_img(src_path=imgs_path+name, rows=3)
        new_img = skimage.transform.resize(img, (64,64,64))
        save_cube_img(target_path=targets_path+name, cube_img=new_img, rows=8, cols=8)
        print(name+' has saved!')


# 数据增强
def data_augmentation():
    src_dir = 'D:/Mywork/image_coord_regenerate/newest_img_coord/different_type/'
    one_samples = glob.glob(src_dir + 'two/' + "*.png")
    for record_number, record_item in enumerate(one_samples):

        cube_image = base_dicom_process.load_cube_img(record_item, 8, 8, 64)
        cube_image1 = numpy.fliplr(cube_image)
        save_cube_img(target_path=src_dir+'two/'+str(record_number)+'_2_0.png', cube_img=cube_image1, cols=8, rows=8)
        cube_image2 = numpy.flipud(cube_image)
        save_cube_img(target_path=src_dir+'two/'+str(record_number)+'_2_1.png', cube_img=cube_image2, cols=8, rows=8)
        cube_image3 = cube_image[:, :, ::-1]
        save_cube_img(target_path=src_dir+'two/'+str(record_number)+'_2_2.png', cube_img=cube_image3, cols=8, rows=8)
        cube_image4 = cube_image[:, ::-1, :]
        save_cube_img(target_path=src_dir+'two/'+str(record_number)+'_2_3.png', cube_img=cube_image4, cols=8, rows=8)
        # cube_image5 = ndimage.rotate(cube_image, 180, axes=(1, 2), reshape=True, mode='mirror')
        # save_cube_img(target_path=src_dir+'one/'+str(record_number)+'_1_4.png', cube_img=cube_image5, cols=8, rows=8)
        # cube_image6 = ndimage.rotate(cube_image, 180, axes=(1, 0), reshape=True, mode='mirror')
        # save_cube_img(target_path=src_dir+'one/'+str(record_number)+'_1_5.png', cube_img=cube_image6, cols=8, rows=8)

def relative_position():
    img_path = 'D:/Mywork/image_coord_regenerate/extract_img/'
    txt_path = 'D:/Mywork/image_coord_regenerate/coord/last_train_coord.txt'
    patient_list = get_usecoord_withoutrepeat('last_train_coord.txt')
    for patientid in patient_list:
        img = cv2.imread(img_path+patientid+'/img_0000_i.png')
        message_list = get_xyz(patientid)
        for mess in message_list:
            relative_x = int(mess[0])/img.shape[0]
            relative_y = int(mess[1]) / img.shape[0]
            print(patientid, relative_x, relative_y)
            new_line = patientid+':'+str(mess[0])+':'+str(mess[1])+':'+str(mess[2])+':'+str(mess[3])+':'+str(mess[4])+':'+\
                       str(round(relative_x, 2))+':'+str(round(relative_y, 2))
            with open('D:/Mywork/image_coord_regenerate/coord/relative_position.txt', "a", encoding="UTF-8") as target:
                target.write(new_line + '\n')




if __name__ == '__main__':
    print('Hello,zmy')
    # 所有切割数据存储地址
    src_dir = 'D:/Mywork/image_coord_regenerate/newest_img_coord/resize_img/'
    # 种类文件夹地址
    type_dir = 'D:/Mywork/image_coord_regenerate/newest_img_coord/different_type/'
    # 步骤一，检查dicom源文件是否存在
    # inspect_dicom()
    # 步骤二，更新z坐标
    # filter_coord_one()
    # filter_coord_two()
    # 步骤三，统一化肺转移、淋巴等类型,该步骤省略(转移类型数据无法判断肺癌类型)，只使用标准类型的数据
    # 步骤四，得到重采样后的坐标信息（排除了非标准类型数据）
    extract_dicom_axis()
    # 步骤五，得到重采样后的病灶半径
    # get_r()
    # 步骤六，将重采样后半径添加到train.txt中
    # add_r()
    # 步骤七，生成csv文件
    # extract_train_csv()
    # 步骤八，裁剪图像并保存
    # make_annotation_images()
    # 步骤九，统一裁剪图像尺寸
    # uniform_size()
    # 步骤十，数据集不同种类分开存储
    # depart_save_type(src_dir)
    # 步骤十一，按比例划分训练集和测试集,废弃，人工划分训练集和测试集
    # depart_train_test(src_dir=type_dir, percentage=0.2)
    # 数据增强
    # data_augmentation()
    # 计算病灶相对位置
    # relative_position()
    # slices = load_patient('I:/2012/20120525/12703/')
    # slice = pydicom.read_file('I:/2012/20120525/12703/PT_1.2.840.113619.2.124.11362.103613198.1337981643.392043700.dcm')
    # slices2 = load_patient2('I:/src_dicom/13957/')
    # print(slice)
    # for x in range(0, len(slices)):
    #     print(slices[x].InstanceNumber, ':', slices2[x].InstanceNumber)
