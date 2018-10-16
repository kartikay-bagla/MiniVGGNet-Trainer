# MiniVGGNet Trainer

The dataset-folder should have the following structure:  

``` bash
dataset  
| class_1  
| | 1.png  
| | 2.png  
| | ..  
| class_2  
| | 1.png  
| | 2.png  
| | ..  
| ..  
```

Usage:  

``` bash
python train_model.py -d cross_extraction -m detector.model
python test_nn.py -i Cross.webm -m detector.model -o CrossIdentify.webm
```
