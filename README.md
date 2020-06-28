## FCOS: Fully Convolutional One-Stage Object Detection  

####  implemented by pytorch1.0  

### Updates   
*  ctr. on reg
*  giou loss
*  ctr. sampling

### TODO  
* normalizing the regression targets   

### Requirements  
* opencv-python  
* pytorch >= 1.2  
* torchvision >= 0.4. 

### Anchor Points  
Let's say the white boxes are the gt boxes, the points of different colors represent the sampling points of different feature layers while applying ctr-sampling.  
![](assets/4.jpg)  
### Results  
Train on 2 1080Ti, 3 imgs for each gpu, init lr=1e-5 cosine decays to 1e-6, but performance is not good on VOC07test. Maybe should remove centerness head while applying central sampling.  
![test1](assets/tensorboard.jpg)   
![test2](assets/2007_000793.jpg)      
![test3](assets/2007_001430.jpg)    

### Pretrained weights   
Due to computational resource constraints, I was unable to fully train the model on the COCO dataset. I have converted the [official pre-training model weights FCOS_R_50_FPN_1x](https://cloudstor.aarnet.edu.au/plus/s/dDeDPBLEAt19Xrl/download) into my own.  
The converted weights is avaliable [Baidu driver link](https://pan.baidu.com/s/14KbDMisTksx_m91uMt-LIA), password: rpni     
The official implementation of preprocessing(pixel is not  normalized to 0-1 and input img follows BGR fomat ) is a little different from mine.   

### Other  
some excellent work based on this repo:  
[FCOS-Pytorch-37.2AP](https://github.com/ChingHo97/FCOS-PyTorch-37.2AP)  
[FCOS_DET_MASK](https://github.com/2anchao/FCOS_DET_MASK)  
