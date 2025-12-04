#!/bin/bash

cd /home/yk449/python_projects/GSCBM
conda init
conda deactivate
conda activate scbm2


data='waterbirds'
model='GMM'
err_only=True

for k in 10
do
  for batch_size in 8
  do
    for lambda_c in 10
    do
      for seed in 42 58 73 123 666 2021
      do
    ### Real data
        printf "Running experiment with: \nseed: $seed \nnumber of clusters: $k \nlambda_c2: $lambda_c \n"

        /home/yk449/.conda/envs/scbm2/bin/python train_gmm.py +model=$model +data=$data logging.experiment_name="${data}_Sequential_valDistAsTrainV3_${model}_ErrOnly_k:${k}_lambda_c:${lambda_c}_seed:${seed}" seed=$seed model.train_batch_size=$batch_size model.val_batch_size=$batch_size model.gmm_params.loss.lambda_c1=$lambda_c model.gmm_params.loss.lambda_c2=$lambda_c model.gmm_params.n_clusters=$k model.gmm_params.train_on_erroneous_only=$err_only
      done;
    done;
  done; 
done;

# err_only=False

# for k in 5 10 15 20
# do
#   for batch_size in 8
#   do
#     for lambda_c in 5
#     do
#       for seed in 42
#       do
#     ### Real data
#         printf "Running experiment with: \nseed: $seed \nnumber of clusters: $k \nlambda_c2: $lambda_c \n"

#         /home/yk449/.conda/envs/scbm2/bin/python train_gmm.py +model=$model +data=$data logging.experiment_name="${data}_${model}_AllData_ECTPSelec_noGMMWarmup_k:${k}_lambda_c:${lambda_c}_decrease:40_sgd_seed:${seed}" seed=$seed model.train_batch_size=$batch_size model.val_batch_size=$batch_size model.gmm_params.loss.lambda_c1=$lambda_c model.gmm_params.loss.lambda_c2=$lambda_c model.gmm_params.n_clusters=$k model.gmm_params.train_on_erroneous_only=$err_only
#       done;
#     done;
#   done; 
# done;
