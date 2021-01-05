#! /bin/bash

random_seed=41
### SINGLE CASE
#gpunum=1
#model=vbert
#fold=f1
#exp=""
#MAX_EPOCH=40

### DUET CASE
gpunum=1
model=duetbert
submodel1=vbert
submodel2=twinrbert
fold=f1
exp="prefix_v.t" ## duet fix weight
MAX_EPOCH=20
freeze_bert=1

if [ ! -z "$1" ]; then
    fold=$1
fi

# 1. make ./models/$model/weights.p (weights file) in ./models.
echo "training"
outdir="$model"_"$fold""$exp"
echo $outdir

if [ ! -z "$1" ]; then
    fold=$1
fi

if [[ "$exp" == "pre"* ]]; then
    echo $model
    sdir1=${submodel1}_"$fold"
    sdir2=${submodel2}_"$fold"
    echo $sdir1
    echo $sdir2
    python train.py \
      --model $model \
      --submodel1 $submodel1 \
      --submodel2 $submodel2 \
      --datafiles ../data/robust/queries.tsv ../data/robust/documents.tsv \
      --qrels ../data/robust/qrels \
      --train_pairs ../data/robust/$fold.train.pairs \
      --valid_run ../data/robust/$fold.valid.run \
      --initial_bert_weights models/$sdir1/weights.p,models/$sdir2/weights.p \
      --model_out_dir models/$outdir \
      --max_epoch $MAX_EPOCH \
      --gpunum $gpunum \
      --random_seed $random_seed \
      --freeze_bert $freeze_bert
else
   echo $model
   python train.py \
      --model $model \
      --datafiles ../data/robust/queries.tsv ../data/robust/documents.tsv \
      --qrels ../data/robust/qrels \
      --train_pairs ../data/robust/$fold.train.pairs \
      --valid_run ../data/robust/$fold.valid.run \
      --model_out_dir models/$outdir \
      --max_epoch $MAX_EPOCH \
      --gpunum $gpunum \
      --random_seed $random_seed 
fi

# 2. load model weights from ./models/$model/weights.p, run tests, and ./models/$model/test.run
echo "testing"
if [[ "$exp" == "pre"* ]]; then
    python rerank.py \
      --model $model \
      --submodel1 $submodel1 \
      --submodel2 $submodel2 \
      --datafiles ../data/robust/queries.tsv ../data/robust/documents.tsv \
      --run ../data/robust/$fold.test.run \
      --model_weights models/$outdir/weights.p \
      --out_path models/$outdir/test.run \
      --gpunum $gpunum
else
    python rerank.py \
      --model $model \
      --datafiles ../data/robust/queries.tsv ../data/robust/documents.tsv \
      --run ../data/robust/$fold.test.run \
      --model_weights models/$outdir/weights.p \
      --out_path models/$outdir/test.run \
      --gpunum $gpunum 
fi


#3. read ./models/$model/test.run, calculate scores using various metrics and save the result to ./models/$model/eval.result
echo "evaluating"
../bin/trec_eval -m all_trec ../data/robust/qrels models/$outdir/test.run > models/$outdir/eval.result




