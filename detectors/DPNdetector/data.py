import numpy as np
import torch
from torch.utils.data import Dataset # 引入数据
import os
import time
import collections
import random
from layers import iou

from scipy.ndimage import zoom # 双线性插值法


class LUNA16Dataset(Dataset):

    def __init__(self, data_dir, split_path, config, phase='train', split_comber=None):    
        """
        LUNA16数据集
        :param data_dir: 数据目录
        :param split_path:分割目录
        :param config: 配置文件
        :param phase: 训练阶段 train val test
        :param split_comber: 分割合并函数
        """
        assert(phase == 'train' or phase == 'val' or phase == 'test') 
        self.phase = phase  # 训练阶段
        # 下面这些配置都是跟随模型的，得上模型里面找
        self.max_stride = config['max_stride']       
        self.stride = config['stride']       
        sizelim = config['sizelim']/config['reso'] # reso = resolution 分辨率 res18内默认是1
        sizelim2 = config['sizelim2']/config['reso']
        sizelim3 = config['sizelim3']/config['reso']
        self.blacklist = config['blacklist']  # 黑名单
        self.isScale = config['aug_scale']
        self.r_rand = config['r_rand_crop']
        self.augtype = config['augtype'] # 数据增强类型 目前res18 目前只有对称变换和旋转
        self.pad_value = config['pad_value']  # 170
        self.split_comber = split_comber  # None
        idcs = split_path # np.load(split_path)
        if phase!='test':
            idcs = [f for f in idcs if (f not in self.blacklist)] # 训练验证转换为文件id的list

        # 每个处理过后的idx对应都有 clean extendbox label mask origin spacing 六个文件
        self.filenames = [os.path.join(data_dir, '%s_clean.npy' % idx) for idx in idcs]
        # print self.filenames
        self.lunanames = [f for f in self.filenames]     
        labels = []
        
        print("刨除黑名单后，还有%s个文件"%len(idcs))  # 打印文件个数
        for idx in idcs:
            # print data_dir, idx
            print data_dir,idx
            l = np.load(data_dir+idx+'_label.npy',allow_pickle=True)  # 这里的AP更新了，老版本默认是allow_pickle=FALSE
            # print l, os.path.join(data_dir, '%s_label.npy' %idx)
            if np.all(l==0):
                l=np.array([])
            labels.append(l)

        self.sample_bboxes = labels
        if self.phase != 'test':
            self.bboxes = []
            for i, l in enumerate(labels):
                # print l
                if len(l) > 0 :
                    for t in l:
                        if t[3]>sizelim:
                            self.bboxes.append([np.concatenate([[i],t])])
                        if t[3]>sizelim2:
                            self.bboxes+=[[np.concatenate([[i],t])]]*2
                        if t[3]>sizelim3:
                            self.bboxes+=[[np.concatenate([[i],t])]]*4
            self.bboxes = np.concatenate(self.bboxes,axis = 0)

        self.crop = Crop(config)
        self.label_mapping = LabelMapping(config, self.phase)


    def __getitem__(self, idx,split=None):
        t = time.time()
        np.random.seed(int(str(t%1)[2:7]))#seed according to time

        isRandomImg  = False
        if self.phase !='test':
            if idx>=len(self.bboxes):
                isRandom = True
                idx = idx%len(self.bboxes)
                isRandomImg = np.random.randint(2)
            else:
                isRandom = False
        else:
            isRandom = False
        
        if self.phase != 'test':
            if not isRandomImg:
                bbox = self.bboxes[idx]
                filename = self.filenames[int(bbox[0])]
                imgs = np.load(filename)
                bboxes = self.sample_bboxes[int(bbox[0])]
                isScale = self.augtype['scale'] and (self.phase=='train')
                sample, target, bboxes, coord = self.crop(imgs, bbox[1:], bboxes,isScale,isRandom)
                if self.phase=='train' and not isRandom:
                     sample, target, bboxes, coord = augment(sample, target, bboxes, coord,
                        ifflip = self.augtype['flip'], ifrotate=self.augtype['rotate'], ifswap = self.augtype['swap'])
            else:
                randimid = np.random.randint(len(self.kagglenames))
                filename = self.kagglenames[randimid]
                imgs = np.load(filename)
                bboxes = self.sample_bboxes[randimid]
                isScale = self.augtype['scale'] and (self.phase=='train')
                sample, target, bboxes, coord = self.crop(imgs, [], bboxes,isScale=False,isRand=True)
            # print sample.shape, target.shape, bboxes.shape
            label = self.label_mapping(sample.shape[1:], target, bboxes, filename)
            sample = (sample.astype(np.float32)-128)/128
            #if filename in self.kagglenames and self.phase=='train':
            #    label[label==-1]=0
            return torch.from_numpy(sample), torch.from_numpy(label), coord
        else:
            imgs = np.load(self.filenames[idx])
            bboxes = self.sample_bboxes[idx]
            nz, nh, nw = imgs.shape[1:]
            pz = int(np.ceil(float(nz) / self.stride)) * self.stride
            ph = int(np.ceil(float(nh) / self.stride)) * self.stride
            pw = int(np.ceil(float(nw) / self.stride)) * self.stride
            imgs = np.pad(imgs, [[0,0],[0, pz - nz], [0, ph - nh], [0, pw - nw]], 'constant',constant_values = self.pad_value)
            
            xx,yy,zz = np.meshgrid(np.linspace(-0.5,0.5,imgs.shape[1]/self.stride),
                                   np.linspace(-0.5,0.5,imgs.shape[2]/self.stride),
                                   np.linspace(-0.5,0.5,imgs.shape[3]/self.stride),indexing ='ij')
            coord = np.concatenate([xx[np.newaxis,...], yy[np.newaxis,...],zz[np.newaxis,:]],0).astype('float32')
            imgs, nzhw = self.split_comber.split(imgs)
            coord2, nzhw2 = self.split_comber.split(coord,
                                                   side_len = self.split_comber.side_len/self.stride,
                                                   max_stride = self.split_comber.max_stride/self.stride,
                                                   margin = self.split_comber.margin/self.stride)
            assert np.all(nzhw==nzhw2)
            imgs = (imgs.astype(np.float32)-128)/128
            return torch.from_numpy(imgs), bboxes, torch.from_numpy(coord2), np.array(nzhw)

    def __len__(self):
        if self.phase == 'train':
            return len(self.bboxes)/(1-self.r_rand)  # 猜测是训练的时候只取1-0.3=70% 算是最终结果
        elif self.phase =='val':
            return len(self.bboxes)
        else:
            return len(self.sample_bboxes)
        





class Crop(object):
    def __init__(self, config):
        # TODO 这里并没有指明配置
        self.crop_size = config['crop_size']
        self.bound_size = config['bound_size']
        self.stride = config['stride']
        self.pad_value = config['pad_value']
    def __call__(self, imgs, target, bboxes,isScale=False,isRand=False):
        if isScale:
            radiusLim = [8.,120.]
            scaleLim = [0.75,1.25]
            scaleRange = [np.min([np.max([(radiusLim[0]/target[3]),scaleLim[0]]),1])
                         ,np.max([np.min([(radiusLim[1]/target[3]),scaleLim[1]]),1])]
            scale = np.random.rand()*(scaleRange[1]-scaleRange[0])+scaleRange[0]
            crop_size = (np.array(self.crop_size).astype('float')/scale).astype('int')
        else:
            crop_size=self.crop_size
        bound_size = self.bound_size
        target = np.copy(target)
        bboxes = np.copy(bboxes)
        
        start = []
        for i in range(3):
            if not isRand:
                r = target[3] / 2
                s = np.floor(target[i] - r)+ 1 - bound_size
                e = np.ceil (target[i] + r)+ 1 + bound_size - crop_size[i] 
            else:
                s = np.max([imgs.shape[i+1]-crop_size[i]/2,imgs.shape[i+1]/2+bound_size])
                e = np.min([crop_size[i]/2,              imgs.shape[i+1]/2-bound_size])
                target = np.array([np.nan,np.nan,np.nan,np.nan])
            if s>e:
                start.append(np.random.randint(e,s))#!
            else:
                start.append(int(target[i])-crop_size[i]/2+np.random.randint(-bound_size/2,bound_size/2))
                
                
        normstart = np.array(start).astype('float32')/np.array(imgs.shape[1:])-0.5
        normsize = np.array(crop_size).astype('float32')/np.array(imgs.shape[1:])
        xx,yy,zz = np.meshgrid(np.linspace(normstart[0],normstart[0]+normsize[0],self.crop_size[0]/self.stride),
                           np.linspace(normstart[1],normstart[1]+normsize[1],self.crop_size[1]/self.stride),
                           np.linspace(normstart[2],normstart[2]+normsize[2],self.crop_size[2]/self.stride),indexing ='ij')
        coord = np.concatenate([xx[np.newaxis,...], yy[np.newaxis,...],zz[np.newaxis,:]],0).astype('float32')

        pad = []
        pad.append([0,0])
        for i in range(3):
            leftpad = max(0,-start[i])
            rightpad = max(0,start[i]+crop_size[i]-imgs.shape[i+1])
            pad.append([leftpad,rightpad])
        crop = imgs[:,
            max(start[0],0):min(start[0] + crop_size[0],imgs.shape[1]),
            max(start[1],0):min(start[1] + crop_size[1],imgs.shape[2]),
            max(start[2],0):min(start[2] + crop_size[2],imgs.shape[3])]
        crop = np.pad(crop,pad,'constant',constant_values =self.pad_value)
        for i in range(3):
            target[i] = target[i] - start[i] 
        for i in range(len(bboxes)):
            for j in range(3):
                bboxes[i][j] = bboxes[i][j] - start[j] 
                
        if isScale:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                crop = zoom(crop,[1,scale,scale,scale],order=1)
            newpad = self.crop_size[0]-crop.shape[1:][0]
            if newpad<0:
                crop = crop[:,:-newpad,:-newpad,:-newpad]
            elif newpad>0:
                pad2 = [[0,0],[0,newpad],[0,newpad],[0,newpad]]
                crop = np.pad(crop, pad2, 'constant', constant_values=self.pad_value)
            for i in range(4):
                target[i] = target[i]*scale
            for i in range(len(bboxes)):
                for j in range(4):
                    bboxes[i][j] = bboxes[i][j]*scale
        return crop, target, bboxes, coord
