python main_imagenet.py --dataset tinyImagenet --batch-size 128 --gpu 0 --trial 1  
python eval_imagenet.py --dataset tinyImagenet --batch-size 128 --gpu 0 --trial 1 

python main_imagenet.py --dataset tinyImagenet --batch-size 128 --gpu 0 --trial 1 --adv --bn_adv_momentum 0.01 --eps 0.03  
python eval_imagenet.py --dataset tinyImagenet  --batch-size 128 --gpu 0 --trial 1 --adv --bn_adv_momentum 0.01 --eps 0.03 
