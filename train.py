import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch.nn
import torchvision.models as models
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from model import *
import DataLoader as dalo
import Extract
from utils import *
import argparse

parser = argparse.ArgumentParser(description='PyTorch ImageNet Attack Sparse')

parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                    help='input batch size for adversarial attack train')
parser.add_argument('--hidden_size', type=int, default=30,
                    help='hidden size of kl')
parser.add_argument('--epoch', type=int, default=100,)
parser.add_argument('--retrain', default=0, type=int,help='Whether to retrain the model')
parser.add_argument('--w_name', default='', type=str,help='The name of the model saved')
parser.add_argument('--l_name', default='',type=str, help='The name of the model load')
parser.add_argument('--train_path', default='',type=str, help='Path to the train set')
parser.add_argument('--test_path', default='',type=str, help='Path to the test set')
parser.add_argument('--c_noise', required=True, type=float)
parser.add_argument('--c_edge', required=True, type=float)
parser.add_argument('--gamma', default=0.5, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=0.0, type=float)
parser.add_argument('--momentum', default=0.0, type=float)
parser.add_argument('--lr', default=0.01, type=float)

args = parser.parse_args()

def KL_devergence(p, q):
    """
    Calculate the KL-divergence of (p,q)
    :param p:
    :param q:
    :return:
    """
    q = torch.nn.functional.softmax(q, dim=0)
    q = torch.sum(q, dim=0)/args.batch_size
    s1 = torch.sum(p*torch.log(p/q))
    s2 = torch.sum((1-p)*torch.log((1-p)/(1-q)))
    return s1+s2

device = torch.device(0)
train_loader, test_loader, mean, std = dalo.imagenet(batch_size=args.batch_size,
                                                     train_path=args.train_path,
                                                     test_path=args.test_path)

net = models.resnet50(pretrained=True)
net2=ld_encoder()
net.to(device)
net = torch.nn.DataParallel(net, device_ids=[0])
net2.to(device)
net2 = torch.nn.DataParallel(net2, device_ids=[0])

tho_tensor = torch.FloatTensor([args.expect_tho for _ in range(args.hidden_size)])
if torch.cuda.is_available():
    tho_tensor = tho_tensor.to(device)
_beta = 3

def train(retrain,weight_load_name,weight_save_name,c_noise,c_edge):


    if retrain == True:
        net2.load_state_dict(torch.load('./weights/ours/' + weight_load_name))
    net.eval()
    net2.train()
    hingeloss = MarginLoss(margin=10)
    optimizer_G = torch.optim.SGD(net2.parameters(), lr=args.lr, momentum=args.momentum,
                                  weight_decay=args.weight_decay, nesterov=True)
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=args.epoch // 10,
                                                  gamma=args.gamma)

    accuracy_global = 100.0
    advsuccess_global = 0.0
    for epoch in range(args.epoch):
        scheduler_G.step()
        correct = 0.
        total = 0.
        sumcorr = 0.
        sumsuccess = 0.
        for i, (data, label) in enumerate(train_loader):
            cv_all = Extract.extract(data.data,c_edge)
            if torch.cuda.is_available():
                data = data.to(device)
                label = label.to(device)
                cv_all = cv_all.float().to(device)
                cv_all[:, 2:, :, :] = 1
            out = net(data)
            _, predicted2 = torch.max(out.data, 1)
            truelabel = (predicted2 == label)
            encoder_out,noise_out,kl_out = net2(data)
            data = torch.clamp(data + c_noise * noise_out.mul(cv_all > 0), 0, 1)
            out = net(data)#
            optimizer_G.zero_grad()
            loss = hingeloss(out, label)
            _kl = KL_devergence(tho_tensor, kl_out)
            loss+= _beta * _kl
            loss.backward()
            optimizer_G.step()
            _, predicted = torch.max(out.data, 1)
            sumcorr += truelabel.sum()
            falselabel = (predicted == label)
            sumsuccess += (truelabel * (~falselabel)).sum()
            total += label.size(0)
            print("total:", total)
            correct += (predicted == label).sum()
            print("correct:", correct)
            accuracy = correct / total
            print("accuracy:", accuracy)
            print("sumsuccess:", sumsuccess)
            advsuccess = sumsuccess / sumcorr
            print("advsuccess:", advsuccess)
        advsuccess = sumsuccess / sumcorr
        accuracy = correct / total
        if  advsuccess>advsuccess_global:
            torch.save(net2.state_dict(), './weights/ours/'+weight_save_name)
            print("准确率由：", accuracy_global, "下降至：", accuracy, "已更新并保存权值")
            print("攻击率由：", advsuccess_global, "上升至：", advsuccess, "已更新并保存权值")
            accuracy_global = accuracy
            advsuccess_global = advsuccess
        print('第%d个epoch的识别准确率为：%f%%' % (epoch + 1, 100 * accuracy))

if __name__ == '__main__':
    train(retrain=args.retrain,weight_load_name=args.l_name,weight_save_name=args.w_name,c_noise=args.c_noise,c_edge=args.c_edge)


