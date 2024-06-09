import copy
import torch
from utils import globalval


def train_server_forward(fx_client, idx, args):
    globalval.net_server = copy.deepcopy(globalval.net_model_server[idx]).to(args.device)
    globalval.net_server.train()
    globalval.optimizer_server = torch.optim.Adam(globalval.net_server.parameters(), lr=args.lr)
    globalval.optimizer_server.zero_grad()
    fx_client = fx_client.to(args.device)
    fx_server = globalval.net_server(fx_client)
    fx_ser = fx_server.clone().detach().requires_grad_(True)
    return fx_ser, fx_server, fx_client

def train_server_backward(dfx_server, fx_server, fx_client):
    fx_server.backward(dfx_server)
    dfx_client = fx_client.grad.clone().detach()
    globalval.optimizer_server.step()
    return dfx_client

def evaluate_server_cal(fx_client, idx, args):
    net_middle = copy.deepcopy(globalval.net_model_server[idx]).to(args.device)
    net_middle.eval()
    with torch.no_grad():
        fx_client = fx_client.to(args.device)
        fx_server = net_middle(fx_client)
    return fx_server

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg