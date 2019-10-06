'''
@Author: xxxmy
@Github: github.com/VectXmy
@Date: 2019-09-26
@Email: xxxmy@foxmail.com
'''

import torch
import xml.etree.ElementTree as ET
import os
import cv2
import numpy as np
from torchvision import transforms

class VOCDataset(torch.utils.data.Dataset):
    CLASSES_NAME = (
        "__background__ ",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )
    def __init__(self,root_dir,resize_size=[800,1024],split='train',use_difficult=False):
        self.root=root_dir
        self.use_difficult=use_difficult
        self.imgset=split

        self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
        self._imgpath = os.path.join(self.root, "JPEGImages", "%s.jpg")
        self._imgsetpath = os.path.join(self.root, "ImageSets", "Main", "%s.txt")

        with open(self._imgsetpath%self.imgset) as f:
            self.img_ids=f.readlines()
        self.img_ids=[x.strip() for x in self.img_ids]
        self.name2id=dict(zip(VOCDataset.CLASSES_NAME,range(len(VOCDataset.CLASSES_NAME))))
        self.resize_size=resize_size
        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]
        print("INFO=====>voc dataset init finished  ! !")

    def __len__(self):
        return len(self.img_ids)

    def _read_img_rgb(self,path):
        return cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)

    def __getitem__(self,index):
        
        img_id=self.img_ids[index]
        img=self._read_img_rgb(self._imgpath%img_id)

        anno=ET.parse(self._annopath%img_id).getroot()
        boxes=[]
        classes=[]
        for obj in anno.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if not self.use_difficult and difficult:
                continue
            _box=obj.find("bndbox")
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            box=[
                _box.find("xmin").text, 
                _box.find("ymin").text, 
                _box.find("xmax").text, 
                _box.find("ymax").text,
            ]
            TO_REMOVE=1
            box = tuple(
                map(lambda x: x - TO_REMOVE, list(map(float, box)))
            )
            boxes.append(box)

            name=obj.find("name").text.lower().strip()
            classes.append(self.name2id[name])
        
        boxes=np.array(boxes,dtype=np.float32)
        img,boxes=self.preprocess_img_boxes(img,boxes,self.resize_size)

        img=transforms.ToTensor()(img)
        img= transforms.Normalize(self.mean, self.std,inplace=True)(img)
        boxes=torch.from_numpy(boxes)
        classes=torch.LongTensor(classes)

        return img,boxes,classes
        

    def preprocess_img_boxes(self,image,boxes,input_ksize):
        '''
        resize image and bboxes 
        Returns
        image_paded: input_ksize  
        bboxes: [None,4]
        '''
        ih, iw    = input_ksize
        h,  w, _  = image.shape

        scale = min(iw/w, ih/h)
        nw, nh  = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))

        image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0,dtype=np.float32)
        dw, dh = (iw - nw) // 2, (ih-nh) // 2
        image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized

        if boxes is None:
            return image_paded
        else:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + dw
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + dh
            return image_paded, boxes
    def collate_fn(self,data):
        imgs_list,boxes_list,classes_list=zip(*data)
        
        max_num=0
        batch_size=len(boxes_list)
        for i in range(batch_size):
            n=boxes_list[i].shape[0]
            if n>max_num:max_num=n
            
        pad_boxes_list=[]
        pad_classes_list=[]
        for i in range(batch_size):
            pad_boxes_list.append(torch.nn.functional.pad(boxes_list[i],(0,0,0,max_num-boxes_list[i].shape[0]),value=-1))
            pad_classes_list.append(torch.nn.functional.pad(classes_list[i],(0,max_num-classes_list[i].shape[0]),value=-1))
        

        batch_boxes=torch.stack(pad_boxes_list)
        batch_classes=torch.stack(pad_classes_list)
        batch_imgs=torch.stack(imgs_list)

        return batch_imgs,batch_boxes,batch_classes


if __name__=="__main__":
    import cv2
    
    dataset=VOCDataset("/home/xht/voc2007/VOCdevkit/VOC2007")
    # img,boxes,classes=dataset[0]
    # print(boxes,classes,"\n",img.shape,boxes.shape,classes.shape,boxes.dtype,classes.dtype,img.dtype)
    # cv2.imwrite("./123.jpg",img)
    img,boxes,classes=dataset.collate_fn([dataset[0],dataset[1],dataset[2]])
    print(boxes,classes,"\n",img.shape,boxes.shape,classes.shape,boxes.dtype,classes.dtype,img.dtype)






