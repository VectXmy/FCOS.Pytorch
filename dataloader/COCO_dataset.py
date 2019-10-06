'''
@Author: xxxmy
@Github: github.com/VectXmy
@Date: 2019-09-26
@Email: xxxmy@foxmail.com
'''

from torchvision.datasets import CocoDetection
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

class COCODataset(CocoDetection):
    CLASS_NAME = (
    'back_ground', 'person', 'bicycle', 'car', 'motorcycle',
    'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush')
    def __init__(self,imgs_path,anno_path,resize_size=[800,1024],transform=None,target_transform=None):
        super().__init__(imgs_path,anno_path)

        print("INFO====>check annos, filtering invalid data......")
        ids=[]
        for id in self.ids:
            ann_id=self.coco.getAnnIds(imgIds=id,iscrowd=None)
            ann=self.coco.loadAnns(ann_id)
            if self._has_valid_annotation(ann):
                ids.append(id)
        self.ids=ids
        self.category2id = {v: i + 1 for i, v in enumerate(self.coco.getCatIds())}
        self.id2category = {v: k for k, v in self.category2id.items()}

        self.transform=transform
        self.target_transform=target_transform
        self.resize_size=resize_size
        self.mean=[0.40789654, 0.44719302, 0.47026115]
        self.std=[0.28863828, 0.27408164, 0.27809835]

    def __getitem__(self,index):
        
        img,ann=super().__getitem__(index)

        ann = [o for o in ann if o['iscrowd'] == 0]
        boxes = [o['bbox'] for o in ann]
        boxes=np.array(boxes,dtype=np.float32)
        #xywh-->xyxy
        boxes[...,2:]=boxes[...,2:]+boxes[...,:2]
        img=np.array(img)
        
        img,boxes=self.preprocess_img_boxes(img,boxes,self.resize_size)
        # img=draw_bboxes(img,boxes)
        

        classes = [o['category_id'] for o in ann]
        classes = [self.category2id[c] for c in classes]
        


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



    def _has_only_empty_bbox(self,annot):
        return all(any(o <= 1 for o in obj['bbox'][2:]) for obj in annot)


    def _has_valid_annotation(self,annot):
        if len(annot) == 0:
            return False

        if self._has_only_empty_bbox(annot):
            return False

        return True
    
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
    
    dataset=COCODataset("/home/xht/coco2017/train2017","/home/xht/coco2017/instances_train2017.json")
    # img,boxes,classes=dataset[0]
    # print(boxes,classes,"\n",img.shape,boxes.shape,classes.shape,boxes.dtype,classes.dtype,img.dtype)
    # cv2.imwrite("./123.jpg",img)
    img,boxes,classes=dataset.collate_fn([dataset[0],dataset[1],dataset[2]])
    print(boxes,classes,"\n",img.shape,boxes.shape,classes.shape,boxes.dtype,classes.dtype,img.dtype)




        
