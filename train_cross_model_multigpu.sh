
# gpus="0"

# DATASET="laptop"

# MODEL_PROMPT="Roberta-base"
# SOURCE_MODEL="RobertaBase"
# TARGET_MODEL="RobertaLarge"

# CUDA_VISIBLE_DEVICES=$gpus python3 train_cross.py \
#     --config config/crossPrompt${TARGET_MODEL}_${DATASET}_100.config \
#     --gpu $gpus \
#     --model_prompt ${MODEL_PROMPT}

# mkdir RobertaForMaskedLM
# mkdir RobertaLargeForMaskedLM

mkdir PLM
mkdir PLM/BertMediumForMaskedLM
mkdir PLM/BertForMaskedLM

mkdir PLM/RobertaForMaskedLM
mkdir PLM/RobertaLargeForMaskedLM

mkdir PLM/T5SmallForMaskedLM
mkdir PLM/T5ForMaskedLM
mkdir PLM/T5LargeForMaskedLM
mkdir PLM/T53BForMaskedLM
mkdir PLM/T511BForMaskedLM

gpus="0,1,2,3,4,5"


# SOURCE_MODEL="RobertaBase"



# TARGET_MODEL="Roberta"
# --config config/crossPrompt${TARGET_MODEL}_${DATASET}_100.config \

# MODEL_PROMPT="Roberta-base"
# TARGET_MODEL="RobertaLarge"
#--config config/crossPrompt${TARGET_MODEL}_nli_100.config \

# PROMPT_EMB="sst2PromptBert"

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=$gpus torchrun --nnodes 1 --nproc-per-node 6 train_cross.py \
    --config config/develop_for_MTL.config \
    --gpu $gpus \
    --source_model Bert\
    # --seed 28\
        
    # --prompt_emb ${PROMPT_EMB}\

    
    
    
    