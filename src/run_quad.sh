#! /bin/bash

##gpunum, model, fold, exp
## quad case
gpunum=1,2,3,4
model=quadbert
submodel1=vbert
submodel2=knrmvbert
submodel3=vbert
submodel4=knrmbert
fold=f1
exp="t2_v.kv.v.k" ## duet fix weight
MAX_EPOCH=20
freeze_bert=0

if [ ! -z "$1" ]; then
    fold=$1
fi

# 1. make ./models/$model/weights.p (weights file) in ./models.
echo "training"
outdir="$model"_"$fold""$exp"
echo $outdir

    sdir1=${submodel1}_"$fold"
    sdir2=${submodel2}_"$fold"
    sdir3=${submodel3}_"$fold"p
    sdir4=${submodel4}_"$fold"
    echo $sdir1
    echo $sdir2
    echo $sdir3
    echo $sdir4
if [[ "$exp" != "notrain"* ]]; then
    python train.py \
      --model $model \
      --submodel1 $submodel1 \
      --submodel2 $submodel2 \
      --submodel3 $submodel3 \
      --submodel4 $submodel4 \
      --datafiles ../data/robust/queries.tsv ../data/robust/documents.tsv \
      --qrels ../data/robust/qrels \
      --train_pairs ../data/robust/$fold.train.pairs \
      --valid_run ../data/robust/$fold.valid.run \
      --initial_bert_weights models/$sdir1/weights.p,models/$sdir2/weights.p,models/$sdir3/weights.p,models/$sdir4/weights.p \
      --model_out_dir models/$outdir \
      --max_epoch $MAX_EPOCH \
      --gpunum $gpunum
      --freeze_bert $freeze_bert

# 2. load model weights from ./models/$model/weights.p, run tests, and ./models/$model/test.run
echo "testing"
    python rerank.py \
      --model $model \
      --submodel1 $submodel1 \
      --submodel2 $submodel2 \
      --submodel3 $submodel3 \
      --submodel4 $submodel4 \
      --datafiles ../data/robust/queries.tsv ../data/robust/documents.tsv \
      --run ../data/robust/$fold.test.run \
      --model_weights models/$outdir/weights.p \
      --out_path models/$outdir/test.run \
      --gpunum $gpunum
else
mkdir models/$outdir
echo "testing"
    python rerank.py \
      --model $model \
      --submodel1 $submodel1 \
      --submodel2 $submodel2 \
      --submodel3 $submodel3 \
      --submodel4 $submodel4 \
      --datafiles ../data/robust/queries.tsv ../data/robust/documents.tsv \
      --run ../data/robust/$fold.test.run \
      --model_weights models/$sdir1/weights.p,models/$sdir2/weights.p,models/$sdir3/weights.p,models/$sdir4/weights.p \
      --out_path models/$outdir/test.run \
      --gpunum $gpunum
fi


#3. read ./models/$model/test.run, calculate scores using various metrics and save the result to ./models/$model/eval.result
echo "evaluating"
../bin/trec_eval -m all_trec ../data/robust/qrels models/$outdir/test.run > models/$outdir/eval.result




