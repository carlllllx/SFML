import copy
import torch
from torch import nn
from options import args_parser

from models.model_resnet18 import resnet18, resnet18_client_side_part1, resnet18_client_side_part2, resnet18_server_side_middle
from models.model_cnn1d import CNN1D, CNN1D_client_side_part1, CNN1D_client_side_part2, CNN1D_server_side_middle
from models.model_mobilenetv3 import MobileNetV3_Small, MobileNetV3_Large, MobileNetV3_Small_client_side_part1, MobileNetV3_Small_client_side_part2, MobileNetV3_Small_server_side_middle, MobileNetV3_Large_client_side_part1, MobileNetV3_Large_client_side_part2, MobileNetV3_Large_server_side_middle
from models.model_squeezenet import SqueezeNet, SqueezeNet_client_side_part1, SqueezeNet_client_side_part2, SqueezeNet_server_side_middle
from models.model_shufflenetv2 import ShuffleNetV2, ShuffleNetV2_client_side_part1, ShuffleNetV2_client_side_part2, ShuffleNetV2_server_side_middle
from models.model_mnasnet import MnasNet, MnasNet_client_side_part1, MnasNet_client_side_part2, MnasNet_server_side_middle
from models.model_ghostnetv2 import ghostnetv2, ghostnetv2_client_side_part1, ghostnetv2_client_side_part2, ghostnetv2_server_side_middle
from models.model_efficientv2 import effnetv2_s, effnetv2_m, effnetv2_s_client_side_part1, effnetv2_s_client_side_part2, effnetv2_s_server_side_middle, effnetv2_m_client_side_part1, effnetv2_m_client_side_part2, effnetv2_m_server_side_middle


args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

if args.model_pub == 'resnet18':
    net_glob_client_part1 = resnet18_client_side_part1(args)
    net_glob_client_part2 = resnet18_client_side_part2(args)
    net_glob_server_middle = resnet18_server_side_middle(args)
elif args.model_pub == 'cnn1d':
    net_glob_client_part1 = CNN1D_client_side_part1(args)
    net_glob_client_part2 = CNN1D_client_side_part2(args)
    net_glob_server_middle = CNN1D_server_side_middle(args)
elif args.model_pub == 'mobilenetv3_s':
    net_glob_client_part1 = MobileNetV3_Small_client_side_part1(args)
    net_glob_client_part2 = MobileNetV3_Small_client_side_part2(args)
    net_glob_server_middle = MobileNetV3_Small_server_side_middle(args)
elif args.model_pub == 'mobilenetv3_l':
    net_glob_client_part1 = MobileNetV3_Large_client_side_part1(args)
    net_glob_client_part2 = MobileNetV3_Large_client_side_part2(args)
    net_glob_server_middle = MobileNetV3_Large_server_side_middle(args)
elif args.model_pub == 'squeezenet':
    net_glob_client_part1 = SqueezeNet_client_side_part1(args)
    net_glob_client_part2 = SqueezeNet_client_side_part2(args)
    net_glob_server_middle = SqueezeNet_server_side_middle(args)
elif args.model_pub == 'shufflenetv2':
    net_glob_client_part1 = ShuffleNetV2_client_side_part1(args)
    net_glob_client_part2 = ShuffleNetV2_client_side_part2(args)
    net_glob_server_middle = ShuffleNetV2_server_side_middle(args)
elif args.model_pub == 'mnasnet':
    net_glob_client_part1 = MnasNet_client_side_part1(args)
    net_glob_client_part2 = MnasNet_client_side_part2(args)
    net_glob_server_middle = MnasNet_server_side_middle(args)
elif args.model_pub == 'ghostnetv2':
    net_glob_client_part1 = ghostnetv2_client_side_part1(args)
    net_glob_client_part2 = ghostnetv2_client_side_part2(args)
    net_glob_server_middle = ghostnetv2_server_side_middle(args)
elif args.model_pub == 'efficientv2_s':
    net_glob_client_part1 = effnetv2_s_client_side_part1(args)
    net_glob_client_part2 = effnetv2_s_client_side_part2(args)
    net_glob_server_middle = effnetv2_s_server_side_middle(args)
elif args.model_pub == 'efficientv2_m':
    net_glob_client_part1 = effnetv2_m_client_side_part1(args)
    net_glob_client_part2 = effnetv2_m_client_side_part2(args)
    net_glob_server_middle = effnetv2_m_server_side_middle(args)

if args.model_priv == 'resnet18':
    net_local_client = [resnet18(args).to(args.device) for i in range(args.num_classes)]
elif args.model_priv == 'cnn1d':
    net_local_client = [CNN1D(args).to(args.device) for i in range(args.num_classes)]
elif args.model_priv == 'mobilenetv3_s':
    net_local_client = [MobileNetV3_Small(args).to(args.device) for i in range(args.num_classes)]
elif args.model_priv == 'mobilenetv3_l':
    net_local_client = [MobileNetV3_Large(args).to(args.device) for i in range(args.num_classes)]
elif args.model_priv == 'squeezenet':
    net_local_client = [SqueezeNet(args).to(args.device) for i in range(args.num_classes)]
elif args.model_priv == 'shufflenetv2':
    net_local_client = [ShuffleNetV2(args).to(args.device) for i in range(args.num_classes)]
elif args.model_priv == 'mnasnet':
    net_local_client = [MnasNet(args).to(args.device) for i in range(args.num_classes)]
elif args.model_priv == 'ghostnetv2':
    net_local_client = [ghostnetv2(args).to(args.device) for i in range(args.num_classes)]
elif args.model_priv == 'efficientv2_s':
    net_local_client = [effnetv2_s(args).to(args.device) for i in range(args.num_classes)]
elif args.model_priv == 'efficientv2_m':
    net_local_client = [effnetv2_m(args).to(args.device) for i in range(args.num_classes)]

net_glob_client_part1 = net_glob_client_part1.to(args.device)
net_glob_client_part2 = net_glob_client_part2.to(args.device)
net_glob_server_middle = net_glob_server_middle.to(args.device)

loss_train_collect = []
acc_train_collect = []
loss_test_collect = []
acc_test_collect = []
batch_acc_train = []
batch_loss_train = []
batch_acc_test = []
batch_loss_test = []

criterion = nn.CrossEntropyLoss()
count1 = 0
count2 = 0
count2_user = 0
count1_user = 0
acc_avg_all_user_train = 0
loss_avg_all_user_train = 0
loss_train_collect_user = []
acc_train_collect_user = []
loss_test_collect_user = []
acc_test_collect_user = []

w_glob_server = net_glob_server_middle.state_dict()
w_locals_server = []
idx_collect = []
l_epoch_check = False
fed_check = False
net_model_server = [net_glob_server_middle for i in range(args.num_users)]
net_server = copy.deepcopy(net_model_server[0]).to(args.device)
optimizer_server = torch.optim.Adam(net_server.parameters(), lr=args.lr)
