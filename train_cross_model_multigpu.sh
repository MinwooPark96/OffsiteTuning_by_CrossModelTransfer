

mkdir PLM
mkdir PLM/BertMediumForMaskedLM
mkdir PLM/BertForMaskedLM
mkdir PLM/BertLargeForMaskedLM

mkdir PLM/RobertaForMaskedLM
mkdir PLM/RobertaLargeForMaskedLM

mkdir PLM/T5SmallForMaskedLM
mkdir PLM/T5ForMaskedLM
mkdir PLM/T5LargeForMaskedLM
mkdir PLM/T53BForMaskedLM
mkdir PLM/T511BForMaskedLM

gpus="0,1,2,3,4,5"




# PROMPT_EMB="sst2PromptT5"

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=$gpus torchrun --nnodes 1 --nproc-per-node 6 train_cross.py \
    --config config/develop_for_MTL_sourceBert.config \
    --gpu $gpus \
    --source_model Bert\
    # --prompt_emb ${PROMPT_EMB}\

    
    # --seed 28\
        
    
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=$gpus torchrun --nnodes 1 --nproc-per-node 6 train_cross.py \
    --config config/develop_for_MTL_sourceRoberta.config \
    --gpu $gpus \
    --source_model Roberta\
    # --prompt_emb ${PROMPT_EMB}\

    
    # --seed 28\

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=$gpus torchrun --nnodes 1 --nproc-per-node 6 train_cross.py \
    --config config/develop_for_MTL_sourceT5.config \
    --gpu $gpus \
    --source_model T5\
    # --prompt_emb ${PROMPT_EMB}\

    
    # --seed 28\
    
    
    
    