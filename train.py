import cv2
import torch.nn as nn
import numpy as np
import os
from torch.autograd import Variable
from models import Autoencoder,toTensor,tensor_to_np
from utils import stack_images,get_transpose_axes,load_images
from training_data import get_training_data
import argparse
import torch.utils.data
from torch.nn import functional as F
import torch.backends.cudnn as cudnn

parser=argparse.ArgumentParser(description="pytorch-deepfakes")
parser.add_argument('--batch-size',type=int,default=64,metavar='N',help="")
parser.add_argument('--epochs',type=int,default=100000,metavar='N',help="")
parser.add_argument('--no-cuda',action='store_true',default=False,help="")
parser.add_argument('--seed',type=int,default=1,metavar='S',help="")
parser.add_argument('--log',type=int,default=100,metavar='N',help="")

args=parser.parse_args()
args.cuda=not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    print('gpu training')
else:
    print('cpu training')

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print("=======loading images=====")
images_a=[x.path for x in os.scandir('data/person1') if x.name.endswith(".jpg") or x.name.endswith(".png")]
images_b=[x.path for x in os.scandir('data/person2') if x.name.endswith(".jpg") or x.name.endswith(".png")]
images_a=load_images(images_a)/255.0
images_b=load_images(images_b)/255.0
images_a += images_b.mean(axis=(0, 1, 2)) - images_a.mean(axis=(0, 1, 2))
model=Autoencoder()
if args.cuda:
    model.cuda()
    cudnn.benchmark=True
cirterion=nn.L1Loss()
optimizer_1=torch.optim.Adam([{'params':model.encoder.parameters()},
                                {'params':model.decoder_a.parameters()}],
                        lr=5e-5,betas=(0.5,0.999))
optimizer_2=torch.optim.Adam([{'params':model.encoder.parameters()},
                            {'params':model.decoder_b.parameters()}],
                        lr=5e-5,betas=(0.5,0.999))                        
if __name__=="__main__":
    files=open('log.txt','a+')
    batch_size=args.batch_size
    start=0
    for epoch in range(start,args.epochs):
        wrap_a,target_a=get_training_data(images_a,batch_size)
        wrap_b,target_b=get_training_data(images_b,batch_size)
        
        wrap_a,target_a=toTensor(wrap_a),toTensor(target_a)
        wrap_b,target_b=toTensor(wrap_b),toTensor(target_b)

        if args.cuda:
            wrap_a=wrap_a.cuda()
            wrap_b=wrap_b.cuda()
            target_a=target_a.cuda()
            target_b=target_b.cuda()

        wrap_a,target_a=Variable(wrap_a.float()),Variable(target_a.float())
        wrap_b,target_b=Variable(wrap_b.float()),Variable(target_b.float())

        optimizer_1.zero_grad()
        optimizer_2.zero_grad()

        wrap_a=model(wrap_a,'A')
        wrap_b=model(wrap_b,'B')

        loss1=cirterion(wrap_a,target_a)
        loss2=cirterion(wrap_b,target_b)
        loss1.backward()
        loss2.backward()
        optimizer_1.step()
        optimizer_2.step()
        files.write(str(epoch)+" "+str(loss1.item())+" "+str(loss2.item())+"\n")
        print(epoch,loss1.item(),loss2.item())
        if epoch % args.log == 0:

            test_a_ = target_a[0:14]
            test_b_ = target_b[0:14]
            test_a = tensor_to_np(target_a[0:14])
            test_b = tensor_to_np(target_b[0:14])
            print('===> Saving models...')
            state = {
                'state': model.state_dict(),
                'epoch': epoch
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/autoencoder.t7')

        figure_A = np.stack([
            test_a,
            tensor_to_np(model(test_a_, 'A')),
            tensor_to_np(model(test_a_, 'B')),
        ], axis=1)
        figure_B = np.stack([
            test_b,
            tensor_to_np(model(test_b_, 'B')),
            tensor_to_np(model(test_b_, 'A')),
        ], axis=1)
        print(test_a.shape,tensor_to_np(model(test_a_,'A')).shape,figure_A.shape)
        figure = np.concatenate([figure_A, figure_B], axis=0)
        print(figure.shape)
        figure = figure.transpose((0, 1, 3, 4, 2))
        figure = figure.reshape((4, 7) + figure.shape[1:])
        figure = stack_images(figure)
        print(figure.shape)
        figure = np.clip(figure * 255, 0, 255).astype('uint8')
        if epoch%200==0:
            cv2.imwrite("result/{}.jpg".format(str(epoch)), figure)
    files.close()
