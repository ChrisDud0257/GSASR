# sh make_setup.sh

###########################################################
### Paper Version ###
###########################################################

### GSASR_EDSR-Baseline_paper_bicubic_x1_4
## Single GPU training
# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/train.py -opt ./options/train/paper/train_GSASR_EDSR-Baseline_paper_bicubic_x1_4.yml --auto_resume

## DDP training
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 ./basicsr/train.py -opt ./options/train/paper/train_GSASR_EDSR-Baseline_paper_bicubic_x1_4.yml --launcher pytorch --auto_resume

## Testing
# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/test.py -opt ./options/test/paper/test_GSASR_EDSR-Baseline_paper_bicubic_x1_4_x2.yml

# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/test.py -opt ./options/test/paper/test_GSASR_EDSR-Baseline_paper_bicubic_x1_4_x3.yml

# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/test.py -opt ./options/test/paper/test_GSASR_EDSR-Baseline_paper_bicubic_x1_4_x4.yml

# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/test.py -opt ./options/test/paper/test_GSASR_EDSR-Baseline_paper_bicubic_x1_4_x6.yml

# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/test.py -opt ./options/test/paper/test_GSASR_EDSR-Baseline_paper_bicubic_x1_4_x8.yml

# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/test.py -opt ./options/test/paper/test_GSASR_EDSR-Baseline_paper_bicubic_x1_4_x12.yml

# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/test.py -opt ./options/test/paper/test_GSASR_EDSR-Baseline_paper_bicubic_x1_4_x16.yml

# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/test.py -opt ./options/test/paper/test_GSASR_EDSR-Baseline_paper_bicubic_x1_4_x18.yml

# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/test.py -opt ./options/test/paper/test_GSASR_EDSR-Baseline_paper_bicubic_x1_4_x24.yml

# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/test.py -opt ./options/test/paper/test_GSASR_EDSR-Baseline_paper_bicubic_x1_4_x30.yml




### GSASR_RDN_paper_bicubic_x1-4
## Single GPU training
# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/train.py -opt ./options/train/paper/train_GSASR_RDN_paper_bicubic_x1_4.yml --auto_resume

## DDP training
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 ./basicsr/train.py -opt ./options/train/paper/train_GSASR_RDN_paper_bicubic_x1_4.yml --launcher pytorch --auto_resume

## Testing
# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/test.py -opt ./options/test/paper/test_GSASR_RDN_paper_bicubic_x1_4_x2.yml

# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/test.py -opt ./options/test/paper/test_GSASR_RDN_paper_bicubic_x1_4_x3.yml

# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/test.py -opt ./options/test/paper/test_GSASR_RDN_paper_bicubic_x1_4_x4.yml

# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/test.py -opt ./options/test/paper/test_GSASR_RDN_paper_bicubic_x1_4_x6.yml

# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/test.py -opt ./options/test/paper/test_GSASR_RDN_paper_bicubic_x1_4_x8.yml

# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/test.py -opt ./options/test/paper/test_GSASR_RDN_paper_bicubic_x1_4_x12.yml

# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/test.py -opt ./options/test/paper/test_GSASR_RDN_paper_bicubic_x1_4_x16.yml

# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/test.py -opt ./options/test/paper/test_GSASR_RDN_paper_bicubic_x1_4_x18.yml

# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/test.py -opt ./options/test/paper/test_GSASR_RDN_paper_bicubic_x1_4_x24.yml

# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/test.py -opt ./options/test/paper/test_GSASR_RDN_paper_bicubic_x1_4_x30.yml




### GSASR_SwinIR_paper_bicubic_x1-4
## Single GPU training
# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/train.py -opt ./options/train/paper/train_GSASR_SwinIR_paper_bicubic_x1_4.yml --auto_resume

## DDP training
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 ./basicsr/train.py -opt ./options/train/paper/train_GSASR_SwinIR_paper_bicubic_x1_4.yml --launcher pytorch --auto_resume

## Testing
# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/test.py -opt ./options/test/paper/test_GSASR_SwinIR_paper_bicubic_x1_4_x2.yml

# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/test.py -opt ./options/test/paper/test_GSASR_SwinIR_paper_bicubic_x1_4_x3.yml

# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/test.py -opt ./options/test/paper/test_GSASR_SwinIR_paper_bicubic_x1_4_x4.yml

# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/test.py -opt ./options/test/paper/test_GSASR_SwinIR_paper_bicubic_x1_4_x6.yml

# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/test.py -opt ./options/test/paper/test_GSASR_SwinIR_paper_bicubic_x1_4_x8.yml

# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/test.py -opt ./options/test/paper/test_GSASR_SwinIR_paper_bicubic_x1_4_x12.yml

# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/test.py -opt ./options/test/paper/test_GSASR_SwinIR_paper_bicubic_x1_4_x16.yml

# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/test.py -opt ./options/test/paper/test_GSASR_SwinIR_paper_bicubic_x1_4_x18.yml

# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/test.py -opt ./options/test/paper/test_GSASR_SwinIR_paper_bicubic_x1_4_x24.yml

# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/test.py -opt ./options/test/paper/test_GSASR_SwinIR_paper_bicubic_x1_4_x30.yml




###########################################################
### Enhancer Version- Automatic Mixed Precision (AMP) + Flash Attention ###
###########################################################



### GSASR_EDSR-Baseline_AMP_DIV2K_bicubic_x1-4
## Single GPU training
# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/train.py -opt ./options/train/AMP/train_GSASR_EDSR-Baseline_amp_DIV2K_bicubic_x1_4.yml --auto_resume

## DDP training
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 ./basicsr/train.py -opt ./options/train/AMP/train_GSASR_EDSR-Baseline_amp_DIV2K_bicubic_x1_4.yml --launcher pytorch --auto_resume

## Testing
# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/test.py -opt ./options/test/AMP/test_GSASR_EDSR-Baseline_amp_DIV2K_bicubic_x1_4_x4.yml




### GSASR_EDSR-Baseline_AMP_DF2K_bicubic_x1-4
## Single GPU training
# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/train.py -opt ./options/train/AMP/train_GSASR_EDSR-Baseline_amp_DF2K_bicubic_x1_4.yml --auto_resume

## DDP training
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 ./basicsr/train.py -opt ./options/train/AMP/train_GSASR_EDSR-Baseline_amp_DF2K_bicubic_x1_4.yml --launcher pytorch --auto_resume

## Testing
# CUDA_VISIBLE_DEVICES=1 \
# python ./basicsr/test.py -opt ./options/test/AMP/test_GSASR_EDSR-Baseline_amp_DF2K_bicubic_x1_4_x4.yml




### GSASR_RDN_AMP_DIV2K_bicubic_x1-4
## Single GPU training
# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/train.py -opt ./options/train/AMP/train_GSASR_RDN_amp_DIV2K_bicubic_x1_4.yml --auto_resume

## DDP training
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 ./basicsr/train.py -opt ./options/train/AMP/train_GSASR_RDN_amp_DIV2K_bicubic_x1_4.yml --launcher pytorch --auto_resume

## Testing
# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/test.py -opt ./options/test/AMP/test_GSASR_RDN_amp_DIV2K_bicubic_x1_4_x4.yml




### GSASR_RDN_AMP_DF2K_bicubic_x1-4
## Single GPU training
# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/train.py -opt ./options/train/AMP/train_GSASR_RDN_amp_DF2K_bicubic_x1_4.yml --auto_resume

## DDP training
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 ./basicsr/train.py -opt ./options/train/AMP/train_GSASR_RDN_amp_DF2K_bicubic_x1_4.yml --launcher pytorch --auto_resume

## Testing
# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/test.py -opt ./options/test/AMP/test_GSASR_RDN_amp_DF2K_bicubic_x1_4_x4.yml




### GSASR_SwinIR_AMP_DIV2K_bicubic_x1-4
## Single GPU training
# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/train.py -opt ./options/train/AMP/train_GSASR_SwinIR_amp_DIV2K_bicubic_x1_4.yml --auto_resume

## DDP training
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 ./basicsr/train.py -opt ./options/train/AMP/train_GSASR_SwinIR_amp_DIV2K_bicubic_x1_4.yml --launcher pytorch --auto_resume

## Testing
# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/test.py -opt ./options/test/AMP/test_GSASR_SwinIR_amp_DIV2K_bicubic_x1_4_x4.yml




### GSASR_SwinIR_AMP_DF2K_bicubic_x1-4
## Single GPU training
# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/train.py -opt ./options/train/AMP/train_GSASR_SwinIR_amp_DF2K_bicubic_x1_4.yml --auto_resume

## DDP training
# CUDA_VISIBLE_DEVICES=0,1 \
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=1234 ./basicsr/train.py -opt ./options/train/AMP/train_GSASR_SwinIR_amp_DF2K_bicubic_x1_4.yml --launcher pytorch --auto_resume

## Testing
# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/test.py -opt ./options/test/AMP/test_GSASR_SwinIR_amp_DF2K_bicubic_x1_4_x4.yml


###########################################################
### Ultra Performance Version ###
###########################################################


### GSASR_HATL_AMP_SA1B_bicubic_x1-16
## Single GPU training
# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/train.py -opt ./options/train/UltraPerformance/train_GSASR_HATL_amp_SA1B_bicubic_x1_16.yml --auto_resume

## DDP training
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 ./basicsr/train.py -opt ./options/train/UltraPerformance/train_GSASR_HATL_amp_SA1B_bicubic_x1_16.yml --launcher pytorch --auto_resume

## Testing
# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/test.py -opt ./options/test/UltraPerformance/test_GSASR_HATL_amp_SA1B_bicubic_x1_16_x4.yml



### Multiple Nodes training
### Please note that, we train GSASR_HATL_AMP_SA1B_bicubic_x1-16 on 16 NVIDIA A100 GPUs, on two Nodes, for 30 days, we use the following instructions for multiple Nodes training.
### However, the following commands might not work well in your environment. If you want to train on multiple nodes, please set up the configuration files, environment, and training commands on your own.

# export OMP_NUM_THREADS=8
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO

# NUM_GPUS=8
# NNODES=2
# NODE_RANK=${RANK:-0}
# PORT=${PORT:-29500}
# ADDR=${MASTER_ADDR:-"127.0.0.1"}

# python -m torch.distributed.launch --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" ./basicsr/train.py -opt ./options/train/UltraPerformance/train_GSASR_HATL_amp_SA1B_bicubic_x1_16.yml --launcher pytorch --auto_resume