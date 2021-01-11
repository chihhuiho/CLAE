python main.py  --trial 1 --gpu 0  --dataset CIFAR10
python eval_lr.py  --trial 1  --gpu 0   --dataset CIFAR10  
python eval_knn.py  --trial 1  --gpu 0   --dataset CIFAR10

python main.py --alpha 1.0 --trial 1 --gpu 0 --adv  --eps 0.03 --bn_adv_momentum 0.01 --dataset CIFAR10  
python eval_lr.py --alpha 1.0 --trial 1 --adv --gpu 0  --eps 0.03 --bn_adv_momentum 0.01 --dataset CIFAR10   
python eval_knn.py --alpha 1.0 --trial 1 --adv --gpu 0  --eps 0.03 --bn_adv_momentum 0.01 --dataset CIFAR10 
