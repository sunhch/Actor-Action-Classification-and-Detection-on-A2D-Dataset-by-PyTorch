from loader import a2d_dataset
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # GPU ID
from torch.utils.data import Dataset, DataLoader
from cfg.deeplab_pretrain_a2d import train as train_cfg
from cfg.deeplab_pretrain_a2d import val as val_cfg
from network import resnet152, ResNet, Bottleneck
import time
from utils.eval_metrics import Precision, Recall, F1
import torchvision.models as models
# from network import ResNet, Bottleneck

# use gpu if cuda can be detected
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SigmoidLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SigmoidLinear, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

        linear_params = list(self.linear.parameters())
        linear_params[0].data.normal_(0, 0.01)
        linear_params[0].data.fill_(0)

    def forward(self, x):
        return self.sigmoid(self.linear(x))


def main(args):
    # Create model directory for saving trained models
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    val_dataset = a2d_dataset.A2DDataset(val_cfg, args.dataset_path)
    data_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

    model = resnet152(pretrained=True)
    model.fc = torch.nn.Linear(2048, 43)
    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join(args.model_path, 'net.ckpt')))
    
    X = np.zeros((data_loader.__len__(), args.num_cls))
    Y = np.zeros((data_loader.__len__(), args.num_cls))
    print(data_loader.__len__())
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            # mini-batch
            images = data[0].to(device)
            labels = data[1].type(torch.FloatTensor).to(device)
            output = model(images).cpu().detach().numpy()
            target = labels.cpu().detach().numpy()
            output[output >= 0.1] = 1
            output[output < 0.1] = 0
            X[batch_idx, :] = output
            Y[batch_idx, :] = target
        
    P = Precision(X, Y)
    R = Recall(X, Y)
    F = F1(X, Y)
    print('Precision: {:.1f} Recall: {:.1f} F1: {:.1f}'.format(100 * P, 100 * R, 100 * F))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument('--dataset_path', type=str, default='../A2D', help='a2d dataset')
    parser.add_argument('--log_step', type=int, default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained models')
    parser.add_argument('--num_cls', type=int, default=43)
    args = parser.parse_args()

main(args)
