import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_pub', type=str, default='resnet18', help='model name')
    parser.add_argument('--model_priv', type=str, default='resnet18', help='model name')
    # model name: cnn1d, resnet18, mobilenetv3_s, mobilenetv3_l, squeezenet, shufflenetv2, mnasnet, ghostnetv2, efficientv2_s, efficientv2_m
    parser.add_argument('--epochs', type=int, default=100, help="rounds of training")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--frac', type=float, default=1, help="the fraction of clients")
    parser.add_argument('--bs', type=int, default=50, help="test batch size")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs")
    parser.add_argument('--num_classes', type=int, default=8, help="number of classes")
    parser.add_argument('--num_users', type=int, default=5, help="number of users")
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")

    args = parser.parse_args()
    return args