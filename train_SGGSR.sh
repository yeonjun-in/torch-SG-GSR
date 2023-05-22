device=0
embedder=SGGSR
epochs=3000

###CORA###
dataset=cora
attack=clean
lr=0.005
alpha=2
dropout=0.6
cut_z=0.0
cut_x=0.0
topk=0.5
knn=5
h_knn=5
pos_ratio=1
python main.py --cut_z $cut_z --cut_x $cut_x --topk $topk --knn $knn --h_knn $h_knn --lr $lr --dropout $dropout --alpha $alpha --pos_ratio $pos_ratio --dataset $dataset --embedder $embedder --attack $attack --epochs $epochs --device $device --layers 16 --task transfer --no_debug

attack=noisy_str_meta
lr=0.01
alpha=5
dropout=0.6
cut_z=0.0
cut_x=0.5
topk=0.9
knn=5
h_knn=5
pos_ratio=1
python main.py --cut_z $cut_z --cut_x $cut_x --topk $topk --knn $knn --h_knn $h_knn --lr $lr --dropout $dropout --alpha $alpha --pos_ratio $pos_ratio --dataset $dataset --embedder $embedder --attack $attack --epochs $epochs --device $device --layers 16 --task transfer --no_debug

attack=real_world
lr=0.01
alpha=3
dropout=0.4
cut_z=0.0
cut_x=0.3
topk=0.9
knn=5
h_knn=5
pos_ratio=1
python main.py --cut_z $cut_z --cut_x $cut_x --topk $topk --knn $knn --h_knn $h_knn --lr $lr --dropout $dropout --alpha $alpha --pos_ratio $pos_ratio --dataset $dataset --embedder $embedder --attack $attack --epochs $epochs --device $device --layers 16 --task transfer --no_debug

####CITESEER####
dataset=citeseer
attack=clean
lr=0.001
alpha=1
dropout=0.6
cut_z=0.1
cut_x=0.1
topk=0.3
knn=5
h_knn=5
pos_ratio=1
python main.py --cut_z $cut_z --cut_x $cut_x --topk $topk --knn $knn --h_knn $h_knn --lr $lr --dropout $dropout --alpha $alpha --pos_ratio $pos_ratio --dataset $dataset --embedder $embedder --attack $attack --epochs $epochs --device $device --layers 16 --task transfer --no_debug

attack=noisy_str_meta
lr=0.001
alpha=2
dropout=0.6
cut_z=0.0
cut_x=0.5
topk=0.3
knn=5
h_knn=5
pos_ratio=1
python main.py --cut_z $cut_z --cut_x $cut_x --topk $topk --knn $knn --h_knn $h_knn --lr $lr --dropout $dropout --alpha $alpha --pos_ratio $pos_ratio --dataset $dataset --embedder $embedder --attack $attack --epochs $epochs --device $device --layers 16 --task transfer --no_debug

attack=real_world
lr=0.005
alpha=5
dropout=0.6
cut_z=0.0
cut_x=0.3
topk=0.7
knn=5
h_knn=5
pos_ratio=1
python main.py --cut_z $cut_z --cut_x $cut_x --topk $topk --knn $knn --h_knn $h_knn --lr $lr --dropout $dropout --alpha $alpha --pos_ratio $pos_ratio --dataset $dataset --embedder $embedder --attack $attack --epochs $epochs --device $device --layers 16 --task transfer --no_debug

####PUBMED####
dataset=pubmed
attack=clean
lr=0.05
alpha=4
dropout=0.2
cut_z=0.0
cut_x=0.1
topk=0.7
knn=5
h_knn=5
pos_ratio=0.5
python main.py --cut_z $cut_z --cut_x $cut_x --topk $topk --knn $knn --h_knn $h_knn --lr $lr --dropout $dropout --alpha $alpha --pos_ratio $pos_ratio --dataset $dataset --embedder $embedder --attack $attack --epochs $epochs --device $device --layers 16 --task transfer --no_debug

attack=noisy_str_meta
lr=0.05
alpha=2
dropout=0.2
cut_z=0.1
cut_x=0.3
topk=0.9
knn=5
h_knn=5
pos_ratio=0.5
python main.py --cut_z $cut_z --cut_x $cut_x --topk $topk --knn $knn --h_knn $h_knn --lr $lr --dropout $dropout --alpha $alpha --pos_ratio $pos_ratio --dataset $dataset --embedder $embedder --attack $attack --epochs $epochs --device $device --layers 16 --task transfer --no_debug

attack=real_world
lr=0.01
alpha=4
dropout=0.0
cut_z=0.5
cut_x=0.0
topk=0.9
knn=5
h_knn=5
pos_ratio=0.5
python main.py --cut_z $cut_z --cut_x $cut_x --topk $topk --knn $knn --h_knn $h_knn --lr $lr --dropout $dropout --alpha $alpha --pos_ratio $pos_ratio --dataset $dataset --embedder $embedder --attack $attack --epochs $epochs --device $device --layers 16 --task transfer --no_debug

####POLBLOGs####
dataset=polblogs
attack=clean
lr=0.01
alpha=3
dropout=0.2
cut_z=0.0
cut_x=0.0
topk=0.5
knn=50
h_knn=50
pos_ratio=1
python main.py --cut_z $cut_z --cut_x $cut_x --topk $topk --knn $knn --h_knn $h_knn --lr $lr --dropout $dropout --alpha $alpha --pos_ratio $pos_ratio --dataset $dataset --embedder $embedder --attack $attack --epochs $epochs --device $device --layers 16 --task transfer --no_debug

attack=noisy_str_meta
lr=0.05
alpha=3
dropout=0.8
cut_z=0.7
cut_x=0.0
topk=0.9
knn=50
h_knn=50
pos_ratio=1
python main.py --cut_z $cut_z --cut_x $cut_x --topk $topk --knn $knn --h_knn $h_knn --lr $lr --dropout $dropout --alpha $alpha --pos_ratio $pos_ratio --dataset $dataset --embedder $embedder --attack $attack --epochs $epochs --device $device --layers 16 --task transfer --no_debug

####COMPUTERS####
dataset=computers
attack=clean
lr=0.005
alpha=0.5
dropout=0.2
cut_z=0.5
cut_x=0.0
topk=0.3
knn=30
h_knn=30
pos_ratio=0.5
python main.py --cut_z $cut_z --cut_x $cut_x --topk $topk --knn $knn --h_knn $h_knn --lr $lr --dropout $dropout --alpha $alpha --pos_ratio $pos_ratio --dataset $dataset --embedder $embedder --attack $attack --epochs $epochs --device $device --layers 16 --task transfer --no_debug

attack=noisy_str_meta
lr=0.005
alpha=0.5
dropout=0.2
cut_z=0.5
cut_x=0.0
topk=0.7
knn=30
h_knn=30
pos_ratio=0.5
python main.py --cut_z $cut_z --cut_x $cut_x --topk $topk --knn $knn --h_knn $h_knn --lr $lr --dropout $dropout --alpha $alpha --pos_ratio $pos_ratio --dataset $dataset --embedder $embedder --attack $attack --epochs $epochs --device $device --layers 16 --task transfer --no_debug

attack=real_world
lr=0.005
alpha=0.5
dropout=0.2
cut_z=0.3 
cut_x=0.3
topk=0.7
knn=30
h_knn=30
pos_ratio=0.5
python main.py --cut_z $cut_z --cut_x $cut_x --topk $topk --knn $knn --h_knn $h_knn --lr $lr --dropout $dropout --alpha $alpha --pos_ratio $pos_ratio --dataset $dataset --embedder $embedder --attack $attack --epochs $epochs --device $device --layers 16 --task transfer --no_debug