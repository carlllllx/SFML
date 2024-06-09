import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import copy

from utils.server_program import FedAvg
from utils.add_func import dataset_iid
from utils.client_program import Client
from utils import globalval
from options import args_parser


if __name__ == '__main__':
    SEED = 123
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(args.device)

    num_users = args.num_users
    epochs = args.epochs
    frac = args.frac
    lr = args.lr

    print(globalval.net_glob_client_part1)
    print(globalval.net_glob_client_part2)
    print(globalval.net_glob_server_middle)

    # Dataset stored in mnist format
    train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=False)
    test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=False)

    global_train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True)
    global_test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=True)

    dict_users_train = dataset_iid(train_dataset, args.num_users)
    dict_users_test = dataset_iid(test_dataset, args.num_users)

    globalval.net_glob_client_part1.train()
    globalval.net_glob_client_part2.train()
    for i in range(args.num_users):
        globalval.net_local_client[i].train()

    w_glob_client_part1 = globalval.net_glob_client_part1.state_dict()
    w_glob_client_part2 = globalval.net_glob_client_part2.state_dict()

    for iter in range(epochs):
        print('\n-------- round {:3d} --------'.format(iter + 1))

        m = max(int(frac * num_users), 1)
        idxs_users = np.random.choice(range(num_users), m, replace=False)
        w_locals_client_p1 = []
        w_locals_client_p2 = []
        w_locals_model = []

        for idx in idxs_users:
            print("client: {}, model_priv: {}".format(idx, args.model_priv))

            local = Client(idx, args, dataset_train=train_dataset, dataset_test=test_dataset, idxs_train=dict_users_train[idx], idxs_test=dict_users_test[idx])
            w_p1, w_p2, local_net = local.train(net_p1=copy.deepcopy(globalval.net_glob_client_part1).to(args.device),
                                   net_p2=copy.deepcopy(globalval.net_glob_client_part2).to(args.device),
                                   local_net=copy.deepcopy(globalval.net_local_client[idx]).to(args.device))
            w_locals_client_p1.append(copy.deepcopy(w_p1))
            w_locals_client_p2.append(copy.deepcopy(w_p2))
            globalval.net_local_client[idx].load_state_dict(local_net)

            local.evaluate(net_p1=copy.deepcopy(globalval.net_glob_client_part1).to(args.device),
                           net_p2=copy.deepcopy(globalval.net_glob_client_part2).to(args.device), ell=iter)
            #local.evaluate_user(net=globalval.net_local_client[idx], idx=idx)
        w_glob_client_p1 = FedAvg(w_locals_client_p1)
        w_glob_client_p2 = FedAvg(w_locals_client_p2)

        globalval.net_glob_client_part1.load_state_dict(w_glob_client_p1)
        globalval.net_glob_client_part2.load_state_dict(w_glob_client_p2)

    print("Training and Evaluation completed!")