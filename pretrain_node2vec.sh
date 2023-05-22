device=0

# detecting
dataset=cora
epochs=200
detector=node2vec
for dataset in cora citeseer pubmed polblogs computers
do
for attack in clean noisy_str_meta
do
python main.py --task detect --dataset $dataset --detector $detector --attack $attack --epochs $epochs --device $device --layers 128 --lr 0.001
done
done

cd results/summary_result/detecting/node2vec/bypass

cp -r ./cora/noisy_str_meta ./cora/real_world
cp -r ./citeseer/noisy_str_meta ./citeseer/real_world
cp -r ./pubmed/noisy_str_meta ./pubmed/real_world
cp -r ./computers/noisy_str_meta ./computers/real_world