#### general ####
device=0
embedder=SGGSR

dataset=pubmed #cora citeseer pubmed polblogs computers
epochs=3000
pos_ratio=0.5
for attack in noisy_str_meta # clean noisy_str_meta real_world
do
for knn in 5 
do
for lr in 0.05 0.01 0.005 0.001
do
for z in 0.0 0.1 0.3 0.5 0.7 0.9
do
for x in 0.0 0.1 0.3 0.5 0.7 0.9
do
for alpha in 0.2 0.5 1 2 3 4 5
do
for dropout in 0.0 0.2 0.4 0.6 0.8
do
for topk in 0.1 0.3 0.5 0.7 0.9
do
python main.py --cut_z $cut_z --cut_x $cut_x --topk $topk --knn $knn --h_knn $knn --lr $lr --dropout $dropout --alpha $alpha --pos_ratio $pos_ratio --dataset $dataset --embedder $embedder --attack $attack --epochs $epochs --device $device --layers 16 --task transfer --no_debug
done
done
done
done
done
done
done
done
done
done
done
done


