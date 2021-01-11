# CLAE
Code for contrastive learning with adversarial examples

# Usage
##  Preprocess
1. Clone the project to directory
```
git clone https://github.com/chihhuiho/CLAE.git
```
2. Initiate the conda environment
```
cd CLAE
conda env create -f environment.yml -n CLAE
conda activate CLAE
```
3. Download the tinyImagenet dataset.
```
cd datasets
sh download_tinyImagenet.sh
```

##  Run Plain contrastive learning methods
1. Enter to Plain folder
```
cd Plain
```
2. Run contrastive learning baseline (use cifar100 [cifar10, tinyImagenet] for example)
```
python main.py --dataset cifar100 --batch-size 128 --gpu 0 --trial 1 
python eval.py --dataset cifar100 --batch-size 128 --gpu 0 --trial 1 
```
3. Run contrastive learning with adversarial training (use cifar100 [cifar10, tinyImagenet] for example)
```
python main.py --dataset cifar100 --batch-size 128 --gpu 0 --trial 1 --adv --bn_adv_momentum 0.01 --eps 0.03 
python eval.py --dataset cifar100 --batch-size 128 --gpu 0 --trial 1 --adv --bn_adv_momentum 0.01 --eps 0.03 
```

##  Run UEL contrastive learning methods
1. Enter to UEL folder
```
cd UEL
```
2. Run contrastive learning baseline (use cifar100 [cifar10, tinyImagenet] for example)
```
python main.py --dataset cifar100 --batch-size 128 --gpu 0 --trial 1 
python eval.py --dataset cifar100 --batch-size 128 --gpu 0 --trial 1 
```
3. Run contrastive learning with adversarial training (use cifar100 [cifar10, tinyImagenet] for example)
```
python main.py --dataset cifar100 --batch-size 128 --gpu 0 --trial 1 --adv --bn_adv_momentum 0.01 --eps 0.03 
python eval.py --dataset cifar100 --batch-size 128 --gpu 0 --trial 1 --adv --bn_adv_momentum 0.01 --eps 0.03 
```

##  Run SimCLR contrastive learning methods
1. Enter to SimCLR folder
```
cd SimCLR
```
2. Run contrastive learning baseline (use cifar100 [cifar10, tinyImagenet] for example)
```
python main.py  --trial 1 --gpu 0  --dataset CIFAR100 
python eval_lr.py  --trial 1  --gpu 0   --dataset CIFAR100 
python eval_knn.py  --trial 1  --gpu 0  --dataset CIFAR100
```
3. Run contrastive learning with adversarial training (use cifar100 [cifar10, tinyImagenet] for example)
```
python main.py --alpha 1.0 --trial 1 --gpu 0 --adv  --eps 0.03 --bn_adv_momentum 0.01 --dataset CIFAR100  
python eval_lr.py --alpha 1.0 --trial 1 --adv --gpu 0  --eps 0.03 --bn_adv_momentum 0.01 --dataset CIFAR100 
python eval_knn.py --alpha 1.0 --trial 1 --adv --gpu 0  --eps 0.03 --bn_adv_momentum 0.01 --dataset CIFAR100
```
