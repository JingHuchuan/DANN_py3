import random
import os
import sys
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
from model import CNNModel
import torch.multiprocessing as multiprocessing
from sklearn.model_selection import train_test_split
from GetData import GetLoader


def getLoader(data, label):
    # 归一化处理
    norm_data = np.zeros((data.shape[0], data.shape[1], data.shape[2]))

    for j in range(data.shape[0]):
        for k in range(data.shape[2]):
            dataChannel = data[j, :, k]
            mean = dataChannel.mean()
            sigma = dataChannel.std()
            norm_data[j, :, k] = (dataChannel - mean) / sigma

    # 将数据分为参与训练的和不参与训练的
    x_train, x_test, y_train, y_test = train_test_split(norm_data, label, test_size=0.2,
                                                        random_state=None)

    dataset_train = GetLoader(x_train, y_train)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                                                   num_workers=2)

    dataset_test = GetLoader(x_test, y_test)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True,
                                                  num_workers=2)

    return dataloader_train, dataloader_test


def test(dataloader):
    test_net = torch.load(os.path.join(model_root, 'current.pth'))
    test_net = test_net.eval()

    if cuda:
        test_net = test_net.to(device)

    len_test_dataloader = len(dataloader)
    data_test_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0

    while i < len_test_dataloader:
        data_test = next(data_test_iter)
        data, label = data_test

        batch_size = len(label)

        if cuda:
            data = data.to(device)
            label = label.to(device)

        # 维度的转换，要扩充一个维度，表示训练数据的维度
        data = torch.reshape(data, (data.shape[0], -1, data.shape[1], data.shape[2]))
        data = data.to(torch.float32)  # 需要进行数据类型的转换
        label = label.long()

        class_output, _ = test_net(input_data=data, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1

    accu = n_correct.data.numpy() * 1.0 / n_total
    return accu


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    model_root = 'models'
    cuda = True
    cudnn.benchmark = True
    lr = 1e-3
    batch_size = 16
    n_epoch = 100
    routing_iterations = 3
    num_classes = 2
    device = torch.device("cuda:0")

    manual_seed = random.randint(1, 10000)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    # 数据读取
    source_data = np.load('dataset/EEG/DEAP/eachSub/data/s3.npy')
    source_label = np.load('dataset/EEG/DEAP/eachSub/label/s1.npy')[:, 0]

    target_data = np.load('dataset/EEG/DEAP/eachSub/data/s2.npy')
    target_label = np.load('dataset/EEG/DEAP/eachSub/label/s2.npy')[:, 0]

    dataloader_source_train, dataloader_source_test = getLoader(source_data, source_label)
    dataloader_target_train, dataloader_target_test = getLoader(target_data, target_label)

    # load model
    my_net = CNNModel(routing_iterations, num_classes)

    # setup optimizer
    optimizer = optim.Adam(my_net.parameters(), lr=lr)

    loss_class = torch.nn.NLLLoss()
    loss_domain = torch.nn.NLLLoss()

    if cuda:
        my_net = my_net.to(device)
        loss_class = loss_class.to(device)
        loss_domain = loss_domain.to(device)

    for p in my_net.parameters():
        p.requires_grad = True

    # training
    best_accu_t = 0.0
    for epoch in range(n_epoch):
        len_dataloader = min(len(dataloader_source_train), len(dataloader_target_train))
        data_source_iter = iter(dataloader_source_train)
        data_target_iter = iter(dataloader_target_train)

        for i in range(len_dataloader):
            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # training model using source data
            data_source = next(data_source_iter)
            s_data, s_label = data_source

            my_net.zero_grad()
            batch_size = len(s_label)

            # 源域的label设置为0
            domain_label = torch.zeros(batch_size).long()

            # 维度的转换，要扩充一个维度，表示训练数据的维度
            s_data = torch.reshape(s_data, (s_data.shape[0], -1, s_data.shape[1], s_data.shape[2]))
            s_data = s_data.to(torch.float32)  # 需要进行数据类型的转换
            s_label = s_label.long()

            if cuda:
                s_data = s_data.to(device)
                s_label = s_label.to(device)
                domain_label = domain_label.to(device)

            class_output, domain_output = my_net(input_data=s_data, alpha=alpha)
            err_s_label = loss_class(class_output, s_label)
            err_s_domain = loss_domain(domain_output, domain_label)

            # training model using target data
            data_target = next(data_target_iter)
            t_data, _ = data_target

            batch_size = len(t_data)

            # 目标域的label设置为1
            domain_label = torch.ones(batch_size).long()

            # 维度的转换，要扩充一个维度，表示训练数据的维度
            t_data = torch.reshape(t_data, (t_data.shape[0], -1, t_data.shape[1], t_data.shape[2]))
            t_data = t_data.to(torch.float32)

            if cuda:
                t_data = t_data.to(device)
                domain_label = domain_label.to(device)

            _, domain_output = my_net(input_data=t_data, alpha=alpha)
            err_t_domain = loss_domain(domain_output, domain_label)
            err = err_t_domain + err_s_domain + err_s_label
            err.backward()
            optimizer.step()

            sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
                             % (epoch, i + 1, len_dataloader, err_s_label.data.cpu().numpy(),
                                err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().item()))
            sys.stdout.flush()
            torch.save(my_net, '{0}/current.pth'.format(model_root))

        # test
        print('\n')
        accu_s = test(dataloader_source_test)
        print('Accuracy of source_domain: %f' % accu_s)
        accu_t = test(dataloader_target_test)
        print('Accuracy of target_domain: %f\n' % accu_t)
        if accu_t > best_accu_t:
            best_accu_s = accu_s
            best_accu_t = accu_t
            torch.save(my_net, '{0}/best.pth'.format(model_root))

    print('============ Summary ============= \n')
    print('Accuracy of source_domain: %f' % best_accu_s)
    print('Accuracy of target_domain: %f' % best_accu_t)
    print('Corresponding model was save in ' + model_root + '/best.pth')
