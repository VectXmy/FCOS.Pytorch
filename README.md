## FCOS: Fully Convolutional One-Stage Object Detection  

####  implemented by pytorch1.0  

### Updates   
*  ctr. on reg
*  giou loss

### TODO  
* ctr. sampling  
* normalizing the regression targets  

### Train  
```python    
python train_voc.py  
```   
### Results  
I trained the model on voc2007 train+voc2012 trainval by using 2 1080Ti and evaluated on voc2007 val. The results were bad, and losses fell very slowly. I haven't figured out the reason yet.  
all classes AP =====>  
 {1: 0.4589222755841319, 2: 0.16750686606534348, 3: 0.23386165318528654, 4: 0.07443572314624947, 5: 0.02686479254228414, 6: 0.31909952283462445, 7: 0.22307122275788488, 8: 0.49809191931290303, 9: 0.005313204275314465, 10: 0.1834014494807389, 11: 0.12792439703153988, 12: 0.38767288756656426, 13: 0.28302314827432107, 14: 0.2384053718985165, 15: 0.11849104609356342, 16: 0.026424833584631057, 17: 0.08134400901722834, 18: 0.11621394524237763, 19: 0.5631170179843081, 20: 0.05373971210178918, 21: 0.0}
