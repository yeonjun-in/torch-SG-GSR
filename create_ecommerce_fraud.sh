device=0

dataset_str=Patio_Lawn_and_Garden
min_user=1
min_item=1
num_fraud=100
edge_thre=2
n_category=5
num_attack=0
for num_fraud in 100
do
for num_attack in 0 100
do
python graph-gen.py --save --dataset_str $dataset_str --text summary --n_category $n_category --min_user $min_user --min_item $min_item --num_fraud $num_fraud --num_attack $num_attack --edge_thre $edge_thre --device $device 
done
done


dataset_str=Pet_Supplies
min_user=5
min_item=5
num_fraud=100
edge_thre=2
n_category=5
num_attack=0
for num_fraud in 100
do
for num_attack in 0 200 
do
python graph-gen.py --save --dataset_str $dataset_str --text summary --n_category $n_category --min_user $min_user --min_item $min_item --num_fraud $num_fraud --num_attack $num_attack --edge_thre $edge_thre --device $device 
done
done



