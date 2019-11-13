import os 
import shutil
import numpy as np 
import sys
from configs.config_training import config
from scipy.io import loadmat
import numpy as np
import h5py
import pandas
import h5py # The h5py package is a Pythonic interface to the HDF5 binary data format.
import scipy
from scipy.ndimage.interpolation import zoom
from skimage import measure
import SimpleITK as sitk
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
from skimage.morphology import convex_hull_image
import pandas
from multiprocessing import Pool  # 多线程操作
from functools import partial
import warnings

def load_itk_image(filename):  # 加载图片
    """
    加载图像：
    :param filename: 以mhd结尾的文件路径字符串
    """
    with open(filename) as f:
        contents = f.readlines()
        line = [k for k in contents if k.startswith('TransformMatrix')][0]
        transformM = np.array(line.split(' = ')[1].split(' ')).astype('float')
        transformM = np.round(transformM)
        if np.any( transformM!=np.array([1,0,0, 0, 1, 0, 0, 0, 1])):
            isflip = True
        else:
            isflip = False

    # 使用GetSize()的输出为：(Width, Height, Depth)，也即原始SimpleITK数据的存储形式
    # 使用GetArrayFromImage()方法后，X轴与Z轴发生了对调，输出形状为：(Depth, Height, Width)
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage) # x ,z 轴调换
     
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))  # spacing是两个像素之间的间隔
     
    return numpyImage, numpyOrigin, numpySpacing,isflip  # numpy 图片 转换之前的原图 两个像素之间的距离 是否翻转



def savenpy_luna(id, annos, filelist, luna_segment, luna_data,savepath): # 转换成numpy 格式 在处理的过程中这里使用了多线程
    """
    主要分为以下几步
    1. 加载原始数据和掩码，用的是load_itk_image函数
    2. 求取掩码的边界，即非零部分的边缘，求出一个box，然后对其应用新的分辨率，也就是重采样，将分辨率统一，采用的函数是resample
    3. 将数据clip至-1200~600，此范围外的数据置为-1200或600，然后再将数据归一化至0~255，采用的是lum_trans函数
    4. 掩码进行一下膨胀操作，去除肺部的小空洞，采用的函数是process_mask，然后对原始数据应用新掩码，并将掩码外的数据值为170（水的HU值经过归一化后的新数值）
    5. 将原始数据重采样，再截取box内的数据即可
    6. 读取标签，将其转换为体素坐标，采用的函数是worldToVoxelCoord，再对其应用新的分辨率，最后注意，数据是box内的数据，所以坐标是相对box的坐标。
    7. 将预处理后的数据和标签用npy格式存储 
    :param id:
    :param annos: annotations.csv
    :param filelist: 每个subset的待处理文件
    :param luna_segment: 分割好肺部的mhd文件
    """
    islabel = True  
    isClean = True  
    resolution = np.array([1,1,1])  # 分辨率
    name = filelist[id]  # 文件名数组中的第id个 
    
    sliceim,origin,spacing,isflip = load_itk_image(os.path.join(luna_data,name+'.mhd')) #加载原始数据

    Mask,origin,spacing,isflip = load_itk_image(os.path.join(luna_segment,name+'.mhd')) #加载相应的掩码
    if isflip:
        Mask = Mask[:,::-1,::-1]
    newshape = np.round(np.array(Mask.shape)*spacing/resolution).astype('int')
    m1 = Mask==3  #LUNA16的掩码有两种值，3和4
    m2 = Mask==4
    Mask = m1+m2  #将两种掩码合并
    
    xx,yy,zz= np.where(Mask)   #确定掩码的边界
    box = np.array([[np.min(xx),np.max(xx)],[np.min(yy),np.max(yy)],[np.min(zz),np.max(zz)]]) # 6个值 确定一个立方体
    box = box*np.expand_dims(spacing,1)/np.expand_dims(resolution,1) #对边界即掩码的最小外部长方体应用新分辨率
    box = np.floor(box).astype('int')
    margin = 5
    extendbox = np.vstack([np.max([[0,0,0],box[:,0]-margin],0),np.min([newshape,box[:,1]+2*margin],axis=0).T]).T #对box留置一定空白

    this_annos = np.copy(annos[annos[:,0]==(name)])           #读取该病例对应标签

    if isClean:
        convex_mask = m1
        dm1 = process_mask(m1)  #对掩码采取膨胀操作，去除肺部黑洞
        dm2 = process_mask(m2)
        dilatedMask = dm1+dm2
        Mask = m1+m2

        extramask = dilatedMask ^ Mask
        bone_thresh = 210
        pad_value = 170

        if isflip:
            sliceim = sliceim[:,::-1,::-1]
            print('flip!')
        sliceim = lumTrans(sliceim)  #对掩码采取膨胀操作，去除肺部黑洞
        sliceim = sliceim*dilatedMask+pad_value*(1-dilatedMask).astype('uint8') #170对应归一化话后的水，掩码外的区域补充为水
        bones = (sliceim*extramask)>bone_thresh   #210对应归一化后的骨头，凡是大于骨头的区域都填充为水
        sliceim[bones] = pad_value
        
        sliceim1,_ = resample(sliceim,spacing,resolution,order=1)  #对原始数据重采样，即采用新分辨率
        sliceim2 = sliceim1[extendbox[0,0]:extendbox[0,1],    #将extendbox内数据取出作为最后结果
                    extendbox[1,0]:extendbox[1,1],
                    extendbox[2,0]:extendbox[2,1]]
        sliceim = sliceim2[np.newaxis,...]
        np.save(os.path.join(savepath, name+'_clean.npy'), sliceim)
        np.save(os.path.join(savepath, name+'_spacing.npy'), spacing)
        np.save(os.path.join(savepath, name+'_extendbox.npy'), extendbox)
        np.save(os.path.join(savepath, name+'_origin.npy'), origin)
        np.save(os.path.join(savepath, name+'_mask.npy'), Mask)

    if islabel:
        this_annos = np.copy(annos[annos[:,0]==(name)])  #一行代表一个结节，所以一个病例可能对应多行标签
        label = []
        if len(this_annos)>0:
            
            for c in this_annos:
                pos = worldToVoxelCoord(c[1:4][::-1],origin=origin,spacing=spacing) #将世界坐标转换为体素坐标
                if isflip:
                    pos[1:] = Mask.shape[1:3]-pos[1:]
                label.append(np.concatenate([pos,[c[4]/spacing[1]]]))
            
        label = np.array(label)
        if len(label)==0:
            label2 = np.array([[0,0,0,0]])  #若没有结节则设为全0
        else:
            label2 = np.copy(label).T
            label2[:3] = label2[:3]*np.expand_dims(spacing,1)/np.expand_dims(resolution,1) #对标签应用新的分辨率
            label2[3] = label2[3]*spacing[1]/resolution[1]     #对直径应用新的分辨率
            label2[:3] = label2[:3]-np.expand_dims(extendbox[:,0],1)  #将box外的长度砍掉，也就是相对于box的坐标
            label2 = label2[:4].T
        np.save(os.path.join(savepath,name+'_label.npy'), label2)
        
    print(name)

def preprocess_luna(): # 预处理
    """
    针对luna 数据集的每一个subset，进行处理
    每个subset都进行多线程处理
    """
    print(config) # 是一个字典
    luna_segment = config['luna_segment']
    savepath = config['preprocess_result_path']  # 应该是肺部切割后的结果 一共 888*2 个结果
    luna_root = config['luna_root']  # 原始图片根目录
    luna_label = config['luna_label']  # annotations.csv
    finished_flag = '.flag_preprocessluna' # 处理完成后可以用flag来表示
    print('starting preprocessing luna')   # 开始处理提示
    if not os.path.exists(finished_flag):
        annos = np.array(pandas.read_csv(luna_label)) # 读取annotations.csv
        pool = Pool()
        # 创建处理结果保存环境
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        for setidx in range(10):
            print('process subset',setidx)
            # Each .mhd file is stored with a separate .raw binary file for the pixeldata
            idx_data_dir = luna_root+'subset'+str(setidx) # 第idx下subset的个数
            filelist = [f.split('.mhd')[0] for f in os.listdir(idx_data_dir) if f.endswith('.mhd') ] # filelist eg： subset0中所有的.mhd文件
            # 处理完成的数据保存文件夹
            if not os.path.exists(savepath+'subset'+str(setidx)):
                os.mkdir(savepath+'subset'+str(setidx))  
            # 多线程处理之前定义偏函数 
            # 把一个函数的某些参数给固定住（也就是设置默认值），返回一个新的函数，调用这个新函数会更简单
            # Python中的多线程适合IO密集型任务，而不适合计算密集型任务
            partial_savenpy_luna = partial(savenpy_luna, annos=annos, filelist=filelist,
                                       luna_segment=luna_segment, luna_data=idx_data_dir+'/', 
                                       savepath=savepath+'subset'+str(setidx)+'/')
            N = len(filelist) # 多少个文件开多少个线程
            _=pool.map(partial_savenpy_luna,range(N))
        pool.close()
        pool.join()
    print('end preprocessing luna')
    f= open(finished_flag,"w+")

if __name__ == "__main__":
    preprocess_luna() # 进行处理的入口
    
