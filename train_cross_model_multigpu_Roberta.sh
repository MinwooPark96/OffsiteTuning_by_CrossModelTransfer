

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

gpus="1,2,3,4,5"




# # PROMPT_EMB="sst2PromptT5"

TARGET=Roberta

for SOURCE in Bert T5Small
    do
    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=$gpus torchrun --nnodes 1 --nproc-per-node 5 train_cross.py \
        --config config/develop_for_MTL_${SOURCE}_${TARGET}_distance.config \
        --gpu $gpus \
        --source_model ${SOURCE}\
        # --prompt_emb ${PROMPT_EMB}\
    done
    
