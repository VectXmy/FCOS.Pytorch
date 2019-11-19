## FCOS: Fully Convolutional One-Stage Object Detection  

####  implemented by pytorch1.0  

### Updates   
*  ctr. on reg
*  giou loss
*  ctr. sampling

### TODO  
* normalizing the regression targets  

### Train  
```python    
python train_voc.py  
```   
### Results  
Train on 2 1080Ti, 3 imgs for each gpu, init lr=1e-5 cosine decays to 1e-6, but performance is not good on VOC07test. Maybe should remove centerness head while applying central sampling.  
![test2](https://github.com/VectXmy/FCOS.Pytorch/blob/master/assets/tensorboard.jpg?raw=true)   
![test1](https://github.com/VectXmy/FCOS.Pytorch/blob/master/assets/2007_000793.jpg?raw=true)    
![test2](https://github.com/VectXmy/FCOS.Pytorch/blob/master/assets/2007_001430.jpg?raw=true)    
