import os
import shutil
import numpy as np
import sys
from configs.config_training import config
from scipy.io import loadmat
import numpy as np
import h5py
import pandas
# The h5py package is a Pythonic interface to the HDF5 binary data format.
import h5py
import scipy
from scipy.ndimage.interpolation import zoom
from skimage import measure
import SimpleITK as sitk
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure
from skimage.morphology import convex_hull_image
import pandas
from multiprocessing import Pool  # 多线程操作
from functools import partial
import warnings


def load_itk_image(filename):
    """
    加载图像
    :param filename: 以mhd结尾的文件路径字符串
    """
    # 第一步，通过读取源文件判断是否为转换文件 isflip
    with open(filename) as f:
        contents = f.readlines()
        line = [k for k in contents if k.startswith('TransformMatrix')][0]
        transformM = np.array(line.split(' = ')[1].split(' ')).astype('float')
        transformM = np.round(transformM)
        if np.any(transformM != np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])):
            isflip = True
        else:
            isflip = False

    # 使用GetSize()的输出为：(Width, Height, Depth)，也即原始SimpleITK数据的存储形式
    # 使用GetArrayFromImage()方法后，X轴与Z轴发生了对调，输出形状为：(Depth, Height, Width)
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)  # 源文件x,z 轴是调换的
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(
        list(reversed(itkimage.GetSpacing())))  # spacing是两个像素之间的间隔
    # numpy 原图片 原点坐标 两个像素之间的距离 是否翻转
    return numpyImage, numpyOrigin, numpySpacing, isflip


def process_mask(mask):
    """
    Mask处理：先求凸包，再求膨胀
    :param mask: True Flase 矩阵，表示（左/右）肺掩码
    """
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):  # 对每一层进行处理
        # Numpy中，随机初始化的数组默认都是C连续的，经过不规则的slice操作，则会改变连续性
        # 将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快
        mask1 = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1) > 0:
            # skimage 中求凸包的操作 Compute the convex hull image of a binary image
            #  convex_hull_image的解释  https://blog.csdn.net/wuguangbin1230/article/details/80083572
            mask2 = convex_hull_image(mask1)  # 求凸包
            if np.sum(mask2) > 1.5*np.sum(mask1):
                mask2 = mask1  # 求出来的凸包大于1.5倍原图像的时候用原图
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    # 数学形态学内容 https://blog.csdn.net/SHU15121856/article/details/76349846
    struct = generate_binary_structure(3, 1)  # 求三维连通性
    dilatedMask = binary_dilation(
        convex_mask, structure=struct, iterations=10)  # 膨胀运算
    return dilatedMask


def lumTrans(img):
    """
    亮度转换,将数据归一化至0~255，使之可视化
    :param img:原来的图像，阈值可能不在0-255
    """
    lungwin = np.array([-1200., 600.])  # 按照 -1200 到 600 进行裁剪
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])  # 归一化
    newimg[newimg < 0] = 0  # 小于-1200的置为0
    newimg[newimg > 1] = 1  # 大于600的置为1
    newimg = (newimg*255).astype('uint8')
    return newimg  # 返回可视化图片格式


def resample(imgs, spacing, resolution, order=2):
    """
    原始CT分辨率往往不一致，为便于应用网络，需要统一分辨率
    :param imgs: voxel coord 图像（经过了亮度可视化变换）
    :param spacing: 原文件 spacing 
    :param resolution: 采用的新分辨率
    """
    if len(imgs.shape) == 3:
        print("resample,shape=3")
        new_shape = np.round(imgs.shape * spacing / resolution)
        true_spacing = spacing * imgs.shape / new_shape  # 变成真实坐标值的spacing值
        resize_factor = new_shape / imgs.shape
        # zoom 使用双线性插值法进行处理
        imgs = zoom(imgs, resize_factor, mode='nearest', order=order)
        return imgs, true_spacing  # 返回变换之后的真实图像以及 转换spacing
    elif len(imgs.shape) == 4:  # TODO 这里还不太理解
        print("resample,shape=4")
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:, :, :, i]
            newslice, true_spacing = resample(slice, spacing, resolution)
            newimg.append(newslice)
        newimg = np.transpose(np.array(newimg), [1, 2, 3, 0])
        return newimg, true_spacing
    else:
        raise ValueError('wrong shape')


def worldToVoxelCoord(worldCoord, origin, spacing):
    """
    坐标转换：给定的标签是世界坐标，单位是mm，需要转换为体素坐标，也就是在像素体内的坐标(现实世界的坐标)
    """
    stretchedVoxelCoord = np.absolute(worldCoord - origin)  # 还原坐标原点
    voxelCoord = stretchedVoxelCoord / spacing  # 除以像素间隔
    return voxelCoord


# 转换成numpy 格式 在处理的过程中这里使用了多线程
def savenpy_luna(id, annos, filelist, luna_segment, luna_data, savepath):
    """
    主要分为以下几步
    1. 加载原始数据和掩码，用的是load_itk_image函数
    2. 求取掩码的边界，即非零部分的边缘，求出一个box，然后对其应用新的分辨率，也就是重采样，将分辨率统一
    3. 将数据clip至-1200~600，此范围外的数据置为-1200或600，然后再将数据归一化至0~255，采用的是lum_trans函数
    4. 掩码进行一下膨胀操作，去除肺部的小空洞，采用的函数是process_mask，然后对原始数据应用新掩码，并将掩码外的数据值为170（水的HU值经过归一化后的新数值）
    5. 将原始数据重采样，再截取box内的数据即可
    6. 读取标签，将其转换为体素坐标，采用的函数是worldToVoxelCoord，再对其应用新的分辨率，最后注意，数据是box内的数据，所以坐标是相对box的坐标。
    7. 将预处理后的数据和标签用npy格式存储 
    :param id: file_list中的索引
    :param annos: annotations.csv
    :param filelist: 每个subset的待处理文件
    :param luna_segment: 分割好肺部的mhd文件
    """
    islabel = True
    isClean = True
    resolution = np.array([1, 1, 1])  # 分辨率
    name = filelist[id]  # 文件名

    # 加载原图和掩码
    slice_im, slice_origin, slice_spacing, slice_isflip = load_itk_image(
        os.path.join(luna_data, name+'.mhd'))  # 加载原始数据
    mask_im, mask_origin, mask_spacing, mask_isflip = load_itk_image(
        os.path.join(luna_segment, name+'.mhd'))  # 加载相应的掩码

    # 处理翻转
    if slice_isflip:
        slice_im = slice_im[:, ::-1, ::-1]
        print('flip!')
    if mask_isflip:  # 掩码顺序调换
        mask_im = mask_im[:, ::-1, ::-1]

    m1 = mask_im == 3  # LUNA16的掩码有两种值，3和4 ，3 左肺 4 右肺
    m2 = mask_im == 4
    mask_im = m1 + m2  # 将两种掩码合并

    # extendbox 用于在真实坐标系下框出肺部区域
    xx, yy, zz = np.where(mask_im)  # 肺部所在的位置坐标
    # 6个值 确定一个立方体 为肺部组织的框
    box = np.array([[np.min(xx), np.max(xx)], [np.min(yy),
                                               np.max(yy)], [np.min(zz), np.max(zz)]])
    # expand_dims https://www.jianshu.com/p/da10840660cb
    # 这里乘以spacing相当于还原成原图
    box = box*np.expand_dims(mask_spacing, 1) / \
        np.expand_dims(resolution, 1)  # 对边界即掩码的最小外部长方体应用新分辨率
    # np.floor   Return the floor of the input, element-wise.
    box = np.floor(box).astype('int')
    margin = 5
    newshape = np.round(np.array(mask_im.shape)*mask_spacing /
                        resolution).astype('int')  # 计算Mask的边界坐标
    # np.vstack资料 https://www.jianshu.com/p/2469e0e2a1cf
    # TODO tobe debugged
    extendbox = np.vstack([np.max([[0, 0, 0], box[:, 0]-margin], 0),
                           np.min([newshape, box[:, 1]+2*margin], axis=0).T]).T

    # 找到该病例对应标签
    this_annos = np.copy(annos[annos[:, 0] == (name)])

    if isClean:
        dm1 = process_mask(m1)  # 处理左肺 凸包 膨胀
        dm2 = process_mask(m2)  # 处理右肺 凸包 膨胀
        dilated_mask = dm1+dm2  # 膨胀之后的掩码
        mask_im = m1+m2   # 原来的掩码

        extra_mask = dilated_mask ^ mask_im  # ^是按位异或逻辑运算符
        bone_thresh = 210  # 骨头的HU值经过归一化后的新数值
        pad_value = 170    # 170（水的HU值经过归一化后的新数值）

        slice_im = lumTrans(slice_im)  # 亮度调整使之可视化
        slice_im = slice_im*dilated_mask + pad_value * \
            (1-dilated_mask).astype('uint8')  # 掩码不覆盖的区域都填充为水
        # extramask区域可能有骨头 210对应归一化后的骨头，凡是大于骨头的区域都填充为水
        bones = (slice_im*extra_mask) > bone_thresh
        slice_im[bones] = pad_value

        slice_im1, _ = resample(slice_im, slice_spacing,
                                resolution, order=1)  # 采用新分辨率
        slice_im2 = slice_im1[extendbox[0, 0]:extendbox[0, 1],  # 将extendbox内数据取出作为最后结果
                              extendbox[1, 0]:extendbox[1, 1],
                              extendbox[2, 0]:extendbox[2, 1]]
        slice_im = slice_im2[np.newaxis, ...]
        np.save(os.path.join(savepath, name+'_clean.npy'),
                slice_im)  # 调整亮度 去除骨头和肺泡 以及其他区域
        np.save(os.path.join(savepath, name+'_spacing.npy'),
                slice_spacing)  # 源文件 slice_spacing
        np.save(os.path.join(savepath, name+'_extendbox.npy'),
                extendbox)  # 框出肺部的框框
        np.save(os.path.join(savepath, name+'_origin.npy'), slice_origin)  # 坐标原点
        np.save(os.path.join(savepath, name+'_mask.npy'), mask_im)  # mask 图像数据

    if islabel:
        # 一行代表一个结节，所以一个病例可能对应多行标签
        this_annos = np.copy(annos[annos[:, 0] == (name)])
        label = []
        if len(this_annos) > 0:
            for c in this_annos:  # 对于这一套CT 的每一个标注
                # position 坐标
                pos = worldToVoxelCoord(
                    c[1:4][::-1], origin=slice_origin, spacing=slice_spacing)  # 将世界坐标转换为体素坐标
                if slice_isflip:
                    pos[1:] = mask_im.shape[1:3]-pos[1:]
                # c[4]/slice_spacing[1] 表示在现实世界中的直径
                label.append(np.concatenate([pos, [c[4]/slice_spacing[1]]]))

        label = np.array(label)
        if len(label) == 0:  # 若没有结节则设为全0
            label2 = np.array([[0, 0, 0, 0]])
        else:
            label2 = np.copy(label).T
            # 对标签应用新的分辨率
            label2[:3] = label2[:3] * \
                np.expand_dims(slice_spacing, 1)/np.expand_dims(resolution, 1)
            label2[3] = label2[3]*slice_spacing[1]/resolution[1]  # 对直径应用新的分辨率
            # 将box外的长度砍掉，也就是相对于box的坐标
            label2[:3] = label2[:3]-np.expand_dims(extendbox[:, 0], 1)
            label2 = label2[:4].T
        np.save(os.path.join(savepath, name+'_label.npy'), label2)

    print(name, " convert to numpy image and label complete")


def preprocess_luna():  # 预处理
    """
    针对luna 数据集的每一个subset，进行处理
    每个subset都进行多线程处理
    """
    print(config)  # 是一个字典
    luna_segment = config['luna_segment']
    savepath = config['preprocess_result_path']  # 应该是肺部切割后的结果 一共 888*2 个结果
    luna_root = config['luna_root']  # 原始图片根目录
    luna_label = config['luna_label']  # annotations.csv
    finished_flag = '.flag_preprocessluna'  # 处理完成后可以用flag来表示
    print('starting preprocessing luna')   # 开始处理提示
    if not os.path.exists(finished_flag):
        annos = np.array(pandas.read_csv(luna_label))  # 读取annotations.csv
        pool = Pool()
        # 创建处理结果保存环境
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        for setidx in range(10):
            print('process subset', setidx)
            # Each .mhd file is stored with a separate .raw binary file for the pixeldata
            idx_data_dir = luna_root+'subset'+str(setidx)  # 第idx下subset的个数
            filelist = [f.split('.mhd')[0] for f in os.listdir(
                idx_data_dir) if f.endswith('.mhd')]  # filelist eg： subset0中所有的.mhd文件
            # 处理完成的数据保存文件夹
            if not os.path.exists(savepath+'subset'+str(setidx)):
                os.mkdir(savepath+'subset'+str(setidx))
            # 多线程处理之前定义偏函数
            # 把一个函数的某些参数给固定住（也就是设置默认值），返回一个新的函数，调用这个新函数会更简单
            # Python中的多线程适合IO密集型任务，而不适合计算密集型任务
            partial_savenpy_luna = partial(savenpy_luna, annos=annos, filelist=filelist,
                                           luna_segment=luna_segment, luna_data=idx_data_dir+'/',
                                           savepath=savepath+'subset'+str(setidx)+'/')
            N = len(filelist)  # 多少个文件开多少个线程
            _ = pool.map(partial_savenpy_luna, range(N))
        pool.close()
        pool.join()
    print('end preprocessing luna')
    f = open(finished_flag, "w+")


if __name__ == "__main__":
    preprocess_luna()  # 进行处理的入口
