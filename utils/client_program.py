import copy
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from utils import globalval
from utils.server_program import evaluate_server_cal, train_server_forward, train_server_backward, FedAvg
from utils.add_func import calculate_accuracy


KL_Loss = nn.KLDivLoss(reduction='batchmean')
Softmax = nn.Softmax(dim=1)
LogSoftmax = nn.LogSoftmax(dim=1)
CE_Loss = nn.CrossEntropyLoss()


def js_div(p_output, q_output, get_softmax=True):
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    if get_softmax:
        p_output = F.softmax(p_output)
        q_output = F.softmax(q_output)
    log_mean_output = ((p_output + q_output) / 2).log()
    return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output)) / 2


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class Client(object):
    def __init__(self, idx, args, dataset_train, dataset_test, idxs_train, idxs_test):
        self.idx = idx
        self.device = args.device
        self.lr = args.lr
        self.local_ep = args.local_ep
        self.args = args
        self.alpha = args.alpha
        self.beta = args.beta
        self.num_classes = args.num_classes
        self.local_train = DataLoader(DatasetSplit(dataset_train, idxs_train), batch_size=50, shuffle=True)
        self.local_test = DataLoader(DatasetSplit(dataset_test, idxs_test), batch_size=50, shuffle=True)
        self.global_test = DataLoader(dataset=dataset_test, batch_size=50, shuffle=True)

    def train(self, net_p1, net_p2, local_net):
        net_p1.train()
        net_p2.train()
        local_net.train()
        optimizer_client_p1 = torch.optim.Adam(net_p1.parameters(), lr=self.lr)
        optimizer_client_p2 = torch.optim.Adam(net_p2.parameters(), lr=self.lr)
        optimizer_client_local = torch.optim.Adam(local_net.parameters(), lr=self.lr)

        for iter in range(self.local_ep):
            len_batch = len(self.local_train)
            for batch_idx, (images, labels) in enumerate(self.local_train):
                images = images.to(self.device)
                labels = labels.to(self.device)
                optimizer_client_p1.zero_grad()
                optimizer_client_p2.zero_grad()
                optimizer_client_local.zero_grad()

                fx = net_p1(images)
                client_fx = fx.clone().detach().requires_grad_(True)
                fx_ser, fx_s, fx_c = train_server_forward(client_fx, self.idx, self.args)
                server_fx = fx_ser.to(self.args.device)
                output_global = net_p2(server_fx)
                output_local = local_net(images)

                ce_local = CE_Loss(output_local, labels)
                # kl_local = KL_Loss(LogSoftmax(output_local), Softmax(output_global.detach()))
                ce_global = CE_Loss(output_global, labels)
                # kl_global = KL_Loss(LogSoftmax(output_global), Softmax(output_local.detach()))
                djs = js_div(output_global, output_local)

                loss_local = self.alpha * ce_local + (1 - self.alpha) * djs
                loss_global = self.beta * ce_global + (1 - self.beta) * djs
                loss_local.backward(retain_graph=True)
                loss_global.backward(retain_graph=True)
                acc_global = calculate_accuracy(output_global, labels)

                dfx_server = server_fx.grad.clone().detach()
                dfx_client = train_server_backward(dfx_server, fx_s, fx_c)
                fx.backward(dfx_client)
                optimizer_client_p1.step()
                optimizer_client_p2.step()
                optimizer_client_local.step()

                globalval.batch_loss_train.append(loss_global.item())
                globalval.batch_acc_train.append(acc_global.item())
                globalval.net_model_server[self.idx] = copy.deepcopy(globalval.net_server)
                globalval.count1 += 1
                if globalval.count1 == len_batch:
                    acc_avg_train = sum(globalval.batch_acc_train) / len(globalval.batch_acc_train)
                    loss_avg_train = sum(globalval.batch_loss_train) / len(globalval.batch_loss_train)
                    globalval.batch_acc_train = []
                    globalval.batch_loss_train = []
                    globalval.count1 = 0
                    w_server = globalval.net_server.state_dict()
                    if iter == self.local_ep - 1:
                        globalval.l_epoch_check = True
                        globalval.w_locals_server.append(copy.deepcopy(w_server))
                        acc_avg_train_all = acc_avg_train
                        loss_avg_train_all = loss_avg_train
                        globalval.loss_train_collect_user.append(loss_avg_train_all)
                        globalval.acc_train_collect_user.append(acc_avg_train_all)
                        if self.idx not in globalval.idx_collect:
                            globalval.idx_collect.append(self.idx)
                    if len(globalval.idx_collect) == self.args.num_users:
                        globalval.fed_check = True
                        globalval.w_glob_server = FedAvg(globalval.w_locals_server)
                        globalval.net_glob_server_middle.load_state_dict(globalval.w_glob_server)
                        globalval.net_model_server = [globalval.net_glob_server_middle for i in range(self.args.num_users)]
                        globalval.w_locals_server = []
                        globalval.idx_collect = []
                        globalval.acc_avg_all_user_train = sum(globalval.acc_train_collect_user) / len(globalval.acc_train_collect_user)
                        globalval.loss_avg_all_user_train = sum(globalval.loss_train_collect_user) / len(globalval.loss_train_collect_user)
                        globalval.loss_train_collect.append(globalval.loss_avg_all_user_train)
                        globalval.acc_train_collect.append(globalval.acc_avg_all_user_train)
                        globalval.acc_train_collect_user = []
                        globalval.loss_train_collect_user = []
        return net_p1.state_dict(), net_p2.state_dict(), local_net.state_dict()

    def evaluate(self, net_p1, net_p2, ell):
        net_p1.eval()
        net_p2.eval()

        target_num = torch.zeros((1, self.num_classes))
        predict_num = torch.zeros((1, self.num_classes))
        acc_num = torch.zeros((1, self.num_classes))

        with torch.no_grad():
            len_batch = len(self.global_test)
            for batch_idx, (images, labels) in enumerate(self.global_test):
                images = images.to(self.device)
                labels = labels.to(self.device)
                fx_client = net_p1(images)
                fx_middle = evaluate_server_cal(fx_client, self.idx, self.args)
                output_global = net_p2(fx_middle)
                loss = globalval.criterion(output_global, labels)
                acc = calculate_accuracy(output_global, labels)
                globalval.batch_loss_test.append(loss.item())
                globalval.batch_acc_test.append(acc.item())
                _, predicted = output_global.max(1)
                pre_mask = torch.zeros(output_global.size()).scatter_(1, predicted.cpu().view(-1, 1), 1.)
                predict_num += pre_mask.sum(0)
                tar_mask = torch.zeros(output_global.size()).scatter_(1, labels.data.cpu().view(-1, 1), 1.)
                target_num += tar_mask.sum(0)
                acc_mask = pre_mask * tar_mask
                acc_num += acc_mask.sum(0)
                globalval.count2 += 1
                if globalval.count2 == len_batch:
                    acc_avg_test = sum(globalval.batch_acc_test) / len(globalval.batch_acc_test)
                    loss_avg_test = sum(globalval.batch_loss_test) / len(globalval.batch_loss_test)
                    globalval.batch_acc_test = []
                    globalval.batch_loss_test = []
                    globalval.count2 = 0
                    if globalval.l_epoch_check:
                        globalval.l_epoch_check = False
                        acc_avg_test_all = acc_avg_test
                        loss_avg_test_all = loss_avg_test

                        globalval.loss_test_collect_user.append(loss_avg_test_all)
                        globalval.acc_test_collect_user.append(acc_avg_test_all)

                    if globalval.fed_check:
                        globalval.fed_check = False

                        acc_avg_all_user = sum(globalval.acc_test_collect_user) / len(globalval.acc_test_collect_user)
                        loss_avg_all_user = sum(globalval.loss_test_collect_user) / len(globalval.loss_test_collect_user)

                        globalval.loss_test_collect.append(loss_avg_all_user)
                        globalval.acc_test_collect.append(acc_avg_all_user)
                        globalval.acc_test_collect_user = []
                        globalval.loss_test_collect_user = []

                        print('===== server: round {:3d}, model_pub: {}'.format(ell + 1, self.args.model_pub))
                        print('===== train: accuracy {:.3f} | loss {:.3f}'.format(globalval.acc_avg_all_user_train, globalval.loss_avg_all_user_train))
                        print('===== test:  accuracy {:.3f} | loss {:.3f}'.format(acc_avg_all_user, loss_avg_all_user))
        return
