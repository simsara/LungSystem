import argparse
import os
import time
import numpy as np
from importlib import import_module
from utils import setgpu

import torch
from torch.nn import DataParallel

# 解析参数
parser = argparse.ArgumentParser(description="DPN Detector")

# 模型的backbone 可以是res18,可以用import_module动态引入
parser.add_argument('--model', '-m', metavar='MODEL',
                    default='base', help='model')
# 配置文件，在run_training.sh文件中使用默认配置
parser.add_argument('--config', '-c', default='config_training', type=str)
# worker 是每个dataload附加的工作线程数量
parser.add_argument('-j', '--workers', default=32, type=int,
                    metavar='N', help='number of data loading workers (default: 32)')
# epoch总数
parser.add_argument('--epochs', default=100, type=int,
                    metavar='N', help='number of total epochs to run')
# 起始epoch数量 默认是从第0个epoch 开始的
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
# 默认的batch size  默认16 ，感觉可能有点大
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
# 初始化的学习率，感觉可能有点大
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
# 使用momentun优化器时候的超参数
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
# 权重衰退
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# 保存频率
parser.add_argument('--save-freq', default='1', type=int, metavar='S',
                    help='save frequency')
# 开始训练的checkpoint
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# 存储checkpoint的路径
parser.add_argument('--save-dir', default='', type=str, metavar='SAVE',
                    help='directory to save checkpoint (default: none)')
# 测试选项，是否进行
parser.add_argument('--test', default=0, type=int, metavar='TEST',
                    help='1 do test evaluation, 0 not (default:0)')
# pbb阈值
parser.add_argument('--testthresh', default=-3, type=float,
                    help='threshod for get pbb')
# 在测试阶段要把图片分成8分
parser.add_argument('--split', default=8, type=int, metavar='SPLIT',
                    help='In the test phase, split the image to 8 parts')
# 使用多少个GPU ,默认是全都使用的
parser.add_argument('--gpu', default='all', type=str,
                    metavar='N',  help='use gpu')
# 使用对少个gpu进行测试
parser.add_argument('--n_test', default=4, type=int, metavar='N',
                    help='number of gpu for test')


def main():
    global args   # python 全局变量是模块比别的
    args = parser.parse_args()  # 命令行参数解析
    config_training = import_module(args.config)  # 动态导入模块
    config_training = config_training.config   # 配置字典
    torch.manual_seed(0)   # 随机数生成种子
    torch.cuda.set_device(0)  # 设置cuda 设备号

    model = import_module(args.model)   # 动态的导入模块文件
    # get_pbb 是一个函数 pdd 是 predicted probability的缩写
    config, net, loss, get_pbb = model.get_model()
    start_epoch = args.start_epoch   # 开始epoch
    save_dir = args.save_dir      # 保存路径

    if args.resume:  # 从之前的检查点开始训练
        checkpoint = torch.load(args.resume)  # 如果有之前训练好的与预训练模型，可以进行预训练
        net.load_state_dict(checkpoint['state_dict'])
    # 开始的epoch值
    if start_epoch == 0:
        start_epoch = 1

    # save dir
    if not save_dir:
        exp_id = time.strftime('%Y%m%d-%H%M%S', time.localtime())  # 目录为时间
        save_dir = os.path.join('results', args.model + '-' + exp_id)
    else:  # 如果有savedir参数 ，就保存在参数路径下
        save_dir = os.path.join('results', save_dir)
    if not os.path.exists(save_dir):  # 创建结果保存目录
        os.makedirs(save_dir)

    # log
    logfile = os.path.join(save_dir, 'log')  # 日志文件
    if args.test == 0:  # 训练
        sys.stdout = Logger(logfile)
        pyfiles = [f for f in os.listdir('./') if f.endswith('.py')]
        for f in pyfiles:
            shutil.copy(f, os.path.join(save_dir, f))  # shell 拷贝文件到目标文件夹

    n_gpu = setgpu(args.gpu)   # 返回的是GPU的数量
    args.n_gpu = n_gpu  # 设置GPU实际 使用数量为参数

    net = net.cuda() # 网络使用CUDA 进行运算
    loss = loss.cuda()
    # cudnn.benchmark = True这个 flag 可以让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法
    # 1.如果网络的输入数据维度或类型上变化不大，设置  torch.backends.cudnn.benchmark = true  可以增加运行效率；
    # 2.如果网络的输入数据在每次 iteration 都变化的话，会导致 cnDNN 每次都会去寻找一遍最优配置，这样反而会降低运行效率
    cudnn.benchmark = False  # True
    # DataParallel splits your data automatically and sends job orders to multiple models on several GPUs. After each model finishes their job, DataParallel collects and merges the results before returning it to you.
    net = DataParallel(net)
    # TODO
    traindatadir = config_training['train_preprocess_result_path']  # 处理完的训练数据
    valdatadir = config_training['val_preprocess_result_path']  # 处理完的验证数据
    testdatadir = config_training['test_preprocess_result_path']  # 处理完的测试数据
    trainfilelist = []  # 训练集文件
    for folder in config_training['train_data_path']:
        print(folder)  # 打印原始训练集的文件夹
        for f in os.listdir(folder):
            # 不在黑名单内
            if f.endswith('.mhd') and f[:-4] not in config_training['black_list']:  # 结尾是.mhd 并且id不在黑名单中的文件
                trainfilelist.append(folder.split('/')[-2]+'/'+f[:-4]) # subset0/id
    valfilelist = []    # 验证集
    for folder in config_training['val_data_path']:
        for f in os.listdir(folder):
            if f.endswith('.mhd') and f[:-4] not in config_training['black_list']:
                valfilelist.append(folder.split('/')[-2]+'/'+f[:-4])
    testfilelist = []   # 测试集
    for folder in config_training['test_data_path']:
        for f in os.listdir(folder):
            if f.endswith('.mhd') and f[:-4] not in config_training['black_list']:
                testfilelist.append(folder.split('/')[-2]+'/'+f[:-4])

    if args.test == 1:
        margin = 32 # matgin 边缘
        sidelen = 144 # 侧边长 ，为了传参
        import data
        split_comber = SplitComb(
            sidelen, config['max_stride'], config['stride'], margin, config['pad_value'])
        dataset = data.LUNA16Dataset(   # 数据集
            testdatadir,
            testfilelist,
            config,
            phase='test',
            split_comber=split_comber)
        test_loader = DataLoader(    # 数据加载
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=data.collate,
            pin_memory=False)

        for i, (data, target, coord, nzhw) in enumerate(test_loader):  # check data consistency
            if i >= len(testfilelist)/args.batch_size:
                break

        test(test_loader, net, get_pbb, save_dir, config)   # 参数是测试的时候就进行测试
        return
    #net = DataParallel(net)
    import data
    print(len(trainfilelist))  # 训练文件集合
    dataset = data.LUNA16Dataset(
        traindatadir,
        trainfilelist,
        config,
        phase='train')
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)

    dataset = data.LUNA16Dataset(
        valdatadir,
        valfilelist,
        config,
        phase='val')
    val_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    for i, (data, target, coord) in enumerate(train_loader):  # check data consistency
        if i >= len(trainfilelist)/args.batch_size:
            break

    for i, (data, target, coord) in enumerate(val_loader):  # check data consistency
        if i >= len(valfilelist)/args.batch_size:
            break

    optimizer = torch.optim.SGD(  # 参数为0.9的优化器
        net.parameters(),
        args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay)

    def get_lr(epoch):    # 参数逐渐衰减
        if epoch <= args.epochs * 1/3:  # 0.5:
            lr = args.lr
        elif epoch <= args.epochs * 2/3:  # 0.8:
            lr = 0.1 * args.lr
        elif epoch <= args.epochs * 0.8:
            lr = 0.05 * args.lr
        else:
            lr = 0.01 * args.lr
        return lr

    for epoch in range(start_epoch, args.epochs + 1):  # 每一个epoch 都进行训练和验证
        train(train_loader, net, loss, epoch, optimizer,
              get_lr, args.save_freq, save_dir)
        validate(val_loader, net, loss)


def train(data_loader, net, loss, epoch, optimizer, get_lr, save_freq, save_dir):   # 训练的函数
    start_time = time.time()

    net.train()
    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    metrics = []

    for i, (data, target, coord) in enumerate(data_loader):
        data = Variable(data.cuda(async=True))
        target = Variable(target.cuda(async=True))
        coord = Variable(coord.cuda(async=True))

        output = net(data, coord)
        loss_output = loss(output, target)
        optimizer.zero_grad()
        loss_output[0].backward()
        optimizer.step()

        loss_output[0] = loss_output[0].item()  # 更新api
        metrics.append(loss_output)

    if epoch % args.save_freq == 0:
        state_dict = net.module.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()

        torch.save({
            'epoch': epoch,
            'save_dir': save_dir,
            'state_dict': state_dict,
            'args': args},
            os.path.join(save_dir, '%03d.ckpt' % epoch))

    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)
    print('Epoch %03d (lr %.5f)' % (epoch, lr))
    print('Train:      tpr %3.2f, tnr %3.2f, total pos %d, total neg %d, time %3.2f' % (
        100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
        100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
        np.sum(metrics[:, 7]),
        np.sum(metrics[:, 9]),
        end_time - start_time))
    print('loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' % (
        np.mean(metrics[:, 0]),
        np.mean(metrics[:, 1]),
        np.mean(metrics[:, 2]),
        np.mean(metrics[:, 3]),
        np.mean(metrics[:, 4]),
        np.mean(metrics[:, 5])))
    print


def validate(data_loader, net, loss):    # 验证的函数
    start_time = time.time()

    net.eval()

    metrics = []
    for i, (data, target, coord) in enumerate(data_loader):
        data = Variable(data.cuda(async=True), volatile=True)
        target = Variable(target.cuda(async=True), volatile=True)
        coord = Variable(coord.cuda(async=True), volatile=True)

        output = net(data, coord)
        loss_output = loss(output, target, train=False)

        loss_output[0] = loss_output[0].item()  # 更新api
        metrics.append(loss_output)
    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)
    print('Validation: tpr %3.2f, tnr %3.8f, total pos %d, total neg %d, time %3.2f' % (
        100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
        100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
        np.sum(metrics[:, 7]),
        np.sum(metrics[:, 9]),
        end_time - start_time))
    print('loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' % (
        np.mean(metrics[:, 0]),
        np.mean(metrics[:, 1]),
        np.mean(metrics[:, 2]),
        np.mean(metrics[:, 3]),
        np.mean(metrics[:, 4]),
        np.mean(metrics[:, 5])))
    print
    print


def test(data_loader, net, get_pbb, save_dir, config):    # 测试的函数
    start_time = time.time()
    save_dir = os.path.join(save_dir, 'bbox')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(save_dir)
    net.eval()
    namelist = []
    split_comber = data_loader.dataset.split_comber
    for i_name, (data, target, coord, nzhw) in enumerate(data_loader):
        s = time.time()
        target = [np.asarray(t, np.float32) for t in target]
        lbb = target[0]
        nzhw = nzhw[0]
        name = data_loader.dataset.filenames[i_name].split(
            '/')[-1].split('_clean')[0]  # .split('-')[0]  wentao change
        data = data[0][0]
        coord = coord[0][0]
        isfeat = False
        if 'output_feature' in config:
            if config['output_feature']:
                isfeat = True
        n_per_run = args.n_test
        print(data.size())
        splitlist = range(0, len(data)+1, n_per_run)
        if splitlist[-1] != len(data):
            splitlist.append(len(data))
        outputlist = []
        featurelist = []

        for i in range(len(splitlist)-1):
            input = Variable(
                data[splitlist[i]:splitlist[i+1]], volatile=True).cuda()
            inputcoord = Variable(
                coord[splitlist[i]:splitlist[i+1]], volatile=True).cuda()
            if isfeat:
                output, feature = net(input, inputcoord)
                featurelist.append(feature.data.cpu().numpy())
            else:
                output = net(input, inputcoord)
            outputlist.append(output.data.cpu().numpy())
        output = np.concatenate(outputlist, 0)
        output = split_comber.combine(output, nzhw=nzhw)
        if isfeat:
            feature = np.concatenate(featurelist, 0).transpose(
                [0, 2, 3, 4, 1])[:, :, :, :, :, np.newaxis]
            feature = split_comber.combine(feature, sidelen)[..., 0]

        thresh = args.testthresh  # -8 #-3
        print 'pbb thresh', thresh
        pbb, mask = get_pbb(output, thresh, ismask=True)
        if isfeat:
            feature_selected = feature[mask[0], mask[1], mask[2]]
            np.save(os.path.join(save_dir, name+'_feature.npy'), feature_selected)
        #tp,fp,fn,_ = acc(pbb,lbb,0,0.1,0.1)
        # print([len(tp),len(fp),len(fn)])
        print([i_name, name])
        e = time.time()
        np.save(os.path.join(save_dir, name+'_pbb.npy'), pbb)
        np.save(os.path.join(save_dir, name+'_lbb.npy'), lbb)
    np.save(os.path.join(save_dir, 'namelist.npy'), namelist)
    end_time = time.time()

    print('elapsed time is %3.2f seconds' % (end_time - start_time))
    print
    print


def singletest(data, net, config, splitfun, combinefun, n_per_run, margin=64, isfeat=False):  # 测试
    z, h, w = data.size(2), data.size(3), data.size(4)
    print(data.size())
    data = splitfun(data, config['max_stride'], margin)
    data = Variable(data.cuda(async=True), volatile=True, requires_grad=False)
    splitlist = range(0, args.split+1, n_per_run)
    outputlist = []
    featurelist = []
    for i in range(len(splitlist)-1):
        if isfeat:
            output, feature = net(data[splitlist[i]:splitlist[i+1]])
            featurelist.append(feature)
        else:
            output = net(data[splitlist[i]:splitlist[i+1]])
        output = output.data.cpu().numpy()
        outputlist.append(output)

    output = np.concatenate(outputlist, 0)
    output = combinefun(
        output, z / config['stride'], h / config['stride'], w / config['stride'])
    if isfeat:
        feature = np.concatenate(featurelist, 0).transpose([0, 2, 3, 4, 1])
        feature = combinefun(
            feature, z / config['stride'], h / config['stride'], w / config['stride'])
        return output, feature
    else:
        return output


if __name__ == '__main__':  # 主函数入口
    main()
