import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import Nets


def read_img(root, filedir, transform=None):
    # Data loading
    with open(filedir, 'r') as f:
        lines = f.readlines()  
    output = []    
    for line in lines:
        linesplit = line.split('\n')[0].split(' ')
        addr = linesplit[0]
        target = torch.Tensor([float(linesplit[1])])
        img = Image.open(os.path.join(root, addr)).convert('RGB')

        if transform is not None:
            img = transform(img)
        
        output.append([img, target])

    return output


def load_model(pretrained_dict, new):
    model_dict = new.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict['state_dict'].items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    new.load_state_dict(model_dict)


def main():
    # net definition 
    net = Nets.AlexNet().cuda()
    # net = Nets.ResNet(block = Nets.BasicBlock, layers = [2, 2, 2, 2], num_classes = 1).cuda()

    # load pretrained model
    load_model(torch.load('./models/alexnet.pth'), net)
    # load_model(torch.load('./models/resnet18.pth'), net)

    # evaluate
    net.eval()

    # loading data...
    root = '../data/faces'
    valdir = '../data/1/test_1.txt'
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])  
    val_dataset = read_img(root, valdir, transform=transform)

    with torch.no_grad():
        label = []
        pred = []

        for i, (img, target) in enumerate(val_dataset):
            img = img.unsqueeze(0).cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = net(img).squeeze(1)
            label.append(target.cpu()[0])
            pred.append(output.cpu()[0])
            print i

        # measurements
        label = np.array(label)
        pred = np.array(pred)
        correlation = np.corrcoef(label, pred)[0][1]
        mae = np.mean(np.abs(label - pred))
        rmse = np.sqrt(np.mean(np.square(label - pred)))

    print('Correlation:{correlation:.4f}\t'
          'Mae:{mae:.4f}\t'
          'Rmse:{rmse:.4f}\t'.format(
            correlation=correlation, mae=mae, rmse=rmse))


if __name__ == '__main__':
    main()
