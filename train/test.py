import numpy as np
import torch
import os
import importlib
from torchvision import transforms
from core.utils.distributed import *
from PIL import Image
from argparse import ArgumentParser
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage
from torchvision import transforms
from lednet import Net
import torch.utils.data as data
from core.utils.visualize import get_color_pallete
import visdom

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

NUM_CHANNELS = 3
NUM_CLASSES = 150

def main(args):

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    model = Net(NUM_CLASSES)
  
    model = torch.nn.DataParallel(model)
    if (not args.cpu):
        model = model.cuda()

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath))
    print ("Model and weights LOADED successfully")

    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image = Image.open(args.test_img).convert('RGB')
    image = image.resize((480, 480))  #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = transform(image).unsqueeze(0).to(device)

    output = model(images)
    label = output[0].max(0)[1].byte().cpu().data
    label = label.numpy()

    mask = get_color_pallete(label, 'ade20k')
    outname = args.test_img.split('.')[0] + '.png'
    mask.save(os.path.join('./', outname))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--loadDir',default="../save/logs/")
    parser.add_argument('--loadWeights', default="model_best.pth")
    parser.add_argument('--loadModel', default="lednet.py")
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--test-img', default="./01.jpg")
    main(parser.parse_args())
