import random
import sys
import torch.optim as optim
import torch.utils.data
import numpy as np
from model import CNNModel
import torch.multiprocessing as multiprocessing
from sklearn.model_selection import train_test_split
from GetData import GetLoader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nUser', default=32, type=int, help='the number of users')
parser.add_argument('--lr', default=1e-3, type=float, help='the initial learning rate')
parser.add_argument('--epochs', default=500, type=int, help='the epochs of training process')
parser.add_argument('--num_classes', default=2, type=int, help='the number of classification categories')
parser.add_argument('--batch_size', default=16, type=int, help='the batch size of DataLoader')
parser.add_argument('--num_workers', default=0, type=int, help='the num_workers of DataLoader')
parser.add_argument('--device', default='cuda:0', type=str, help='the number of gpu device')
parser.add_argument('--father_path', default='../dataset/EEG/DEAP/eachSub/', type=str, help='the path of data')
# parser.add_argument('--data_path', default='../data/EEG/DREAMER/eachSub/data/', type=str, help='the path of data')
# parser.add_argument('--label_path', default='../data/EEG/DREAMER/eachSub/label/', type=str, help='the path of label')
parser.add_argument('--save_model_path', default='./result/models/', type=str, help='the path of save model')
parser.add_argument('--save_record_path', default='./result/record/', type=str, help='the path of save train val test')
parser.add_argument('--routing_iterations', type=int, default=3)


def getLoader(data, label):
    # 归一化处理

    # mean = np.mean(data, axis=1)
    # std = np.std(data, axis=1)
    # norm_data = (data - mean[:, np.newaxis, :]) / std[:, np.newaxis, :]

    min_val = np.min(data, axis=1)
    max_val = np.max(data, axis=1)
    norm_data = (data - min_val[:, np.newaxis, :]) / (max_val[:, np.newaxis, :] - min_val[:, np.newaxis, :])

    dataset = GetLoader(norm_data, label)
    # dataset = GetLoader(data, label)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                             num_workers=args.num_workers)

    return dataloader


def test(dataloader):
    load_path = args.save_model_path + 's{}tos{}@currentEpoch.pth'
    test_net = torch.load(load_path.format(souSub + 1, tarSub + 1))
    test_net = test_net.eval()

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
    args = parser.parse_args(sys.argv[1:])
    device = args.device

    model_root = 'models'

    manual_seed = random.randint(1, 10000)
    random.seed(manual_seed)
    torch.manual_seed(42)

    nUser = 32
    userList = [i for i in range(nUser)]

    best_accu_s_array = np.zeros((args.nUser, nUser))
    best_accu_t_array = np.zeros((args.nUser, nUser))

    for tarSub in range(args.nUser):
        print("target domain subject is sub{}".format(tarSub + 1))
        trainUserList = [item for item in userList if item != tarSub]
        target_train_data = np.load(args.father_path + '/train_data/s{}.npy'.format(tarSub + 1))
        target_train_label = np.load(args.father_path + '/train_label/s{}.npy'.format(tarSub + 1))[:, 0]
        target_test_data = np.load(args.father_path + '/test_data/s{}.npy'.format(tarSub + 1))
        target_test_label = np.load(args.father_path + '/test_label/s{}.npy'.format(tarSub + 1))[:, 0]

        dataloader_target_train = getLoader(target_train_data, target_train_label)
        dataloader_target_test = getLoader(target_test_data, target_test_label)

        for souSub in trainUserList:
            print("source domain subject is sub{}".format(souSub + 1))
            source_train_data = np.load(args.father_path + '/train_data/s{}.npy'.format(souSub + 1))
            source_train_label = np.load(args.father_path + '/train_label/s{}.npy'.format(souSub + 1))[:, 0]
            source_test_data = np.load(args.father_path + '/test_data/s{}.npy'.format(souSub + 1))
            source_test_label = np.load(args.father_path + '/test_label/s{}.npy'.format(souSub + 1))[:, 0]

            dataloader_source_train = getLoader(source_train_data, source_train_label)
            dataloader_source_test = getLoader(source_test_data, source_test_label)

            # load model
            my_net = CNNModel(args.routing_iterations, args.num_classes)

            # setup optimizer
            optimizer = optim.Adam(my_net.parameters(), lr=args.lr)

            loss_class = torch.nn.NLLLoss()
            loss_domain = torch.nn.NLLLoss()

            my_net = my_net.to(device)
            loss_class = loss_class.to(device)
            loss_domain = loss_domain.to(device)

            for p in my_net.parameters():
                p.requires_grad = True

            # training
            best_accu_t = 0.0
            for epoch in range(args.epochs):
                len_dataloader = min(len(dataloader_source_train), len(dataloader_target_train))
                data_source_iter = iter(dataloader_source_train)
                data_target_iter = iter(dataloader_target_train)

                for i in range(len_dataloader):
                    p = float(i + epoch * len_dataloader) / args.epochs / len_dataloader
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

                    t_data = t_data.to(device)
                    domain_label = domain_label.to(device)

                    _, domain_output = my_net(input_data=t_data, alpha=alpha)
                    err_t_domain = loss_domain(domain_output, domain_label)
                    err = err_t_domain + err_s_domain + err_s_label
                    err.backward()
                    optimizer.step()

                    sys.stdout.write(
                        '\r epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
                        % (epoch, i + 1, len_dataloader, err_s_label.data.cpu().numpy(),
                           err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().item()))
                    sys.stdout.flush()
                    save_path = args.save_model_path + 's{}tos{}@currentEpoch.pth'
                    torch.save(my_net, save_path.format(souSub + 1, tarSub + 1))

                # test
                print('\n')
                accu_s = test(dataloader_source_test)
                print('Accuracy of source_domain: %f' % accu_s)
                accu_t = test(dataloader_target_test)
                print('Accuracy of target_domain: %f\n' % accu_t)
                if accu_t > best_accu_t:
                    best_accu_s = accu_s
                    best_accu_t = accu_t
                    save_path = args.save_model_path + 's{}tos{}@bestEpoch.pth'
                    torch.save(my_net, save_path.format(souSub + 1, tarSub + 1))

            best_accu_s_array[tarSub][souSub] = best_accu_s
            best_accu_t_array[tarSub][souSub] = best_accu_t
            print('============ Summary =============')
            print('Accuracy of source_domain: %f' % best_accu_s)
            print('Accuracy of target_domain: %f' % best_accu_t)
            print('\n')

        print('============ The Summary sub{} as target subject============= \n'.format(tarSub + 1))

        np.save(args.save_record_path + 'best_accu_s_array.npy', best_accu_s_array)
        np.save(args.save_record_path + 'best_accu_t_array.npy', best_accu_t_array)

    np.save(args.save_record_path + 'best_accu_s_array.npy', best_accu_s_array)
    np.save(args.save_record_path + 'best_accu_t_array.npy', best_accu_t_array)
