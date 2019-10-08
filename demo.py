'''
@Author: xxxmy
@Github: github.com/VectXmy
@Date: 2019-09-26
@Email: xxxmy@foxmail.com
'''

import cv2
from model.fcos import FCOSDetector
import torch
from torchvision import transforms
import numpy as np
from dataloader.VOC_dataset import VOCDataset
import time

def preprocess_img(image,input_ksize):
    ih, iw    = input_ksize
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0,dtype=np.float32)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    return image_paded

def convertSyncBNtoBN(module):
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module_output = torch.nn.BatchNorm2d(module.num_features,
                                               module.eps, module.momentum,
                                               module.affine,
                                               module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
    for name, child in module.named_children():
        module_output.add_module(name,convertSyncBNtoBN(child))
    del module
    return module_output
if __name__=="__main__":
    model=FCOSDetector(mode="inference")
    model=torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    print("INFO===>success convert BN to SyncBN")
    model.load_state_dict(torch.load("./logs/voc2012_multigpu_800x1333_epoch30_loss0.5746.pth",map_location=torch.device('cpu')))
    model=convertSyncBNtoBN(model)
    print("INFO===>success convert SyncBN to BN")
    model=model.cuda().eval()
    print("===>success loading model")

    import os
    root="./test_images/"
    names=os.listdir(root)
    for name in names:
        img_bgr=cv2.imread(root+name)
        img_pad=preprocess_img(img_bgr,[800,1024])
        img=cv2.cvtColor(img_pad.copy(),cv2.COLOR_BGR2RGB)
        img1=transforms.ToTensor()(img.copy())
        img1= transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225),inplace=True)(img1)
        img1=img1.cuda()
        

        start_t=time.time()
        with torch.no_grad():
            out=model(img1.unsqueeze_(dim=0))
        end_t=time.time()
        cost_t=1000*(end_t-start_t)
        print("===>success processing img, cost time %.2f ms"%cost_t)
        # print(out)
        scores,classes,boxes=out

        boxes=boxes[0].cpu().numpy().tolist()
        classes=classes[0].cpu().numpy().tolist()
        scores=scores[0].cpu().numpy().tolist()

        for i,box in enumerate(boxes):
            pt1=(int(box[0]),int(box[1]))
            pt2=(int(box[2]),int(box[3]))
            img_pad=cv2.rectangle(img_pad,pt1,pt2,(0,255,0))
            img_pad=cv2.putText(img_pad,"%s %.3f"%(VOCDataset.CLASSES_NAME[int(classes[i])],scores[i]),(int(box[0]),int(box[1])+10),cv2.FONT_HERSHEY_SIMPLEX,0.5,[0,200,20],2)
        cv2.imwrite("./out_images/"+name,img_pad)





