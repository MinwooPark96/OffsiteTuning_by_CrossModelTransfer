
# Bert: BertMedium, Bert (BertBase)
# Roberta: Roberta, RobertaBase (BertBase)
# T5: T5Small, T5 (T5Base), T5Large

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

gpus="0"

SOURCE=Bert
TARGET=Roberta
OPTION=distance_mask
DATAS="imdb_sst2"

mkdir valid_result
mkdir valid_result/${DATAS}_${SOURCE}_${TARGET}_${OPTION}

for (( EPOCH=2; EPOCH<=10; EPOCH+=2))
do
    for DATASET in imdb sst2 laptop restaurant movierationales tweetevalsentiment mnli qnli snli ethicsdeontology ethicsjustice qqp mrpc
    do
    
    CUDA_VISIBLE_DEVICES=$gpus python3 valid.py \
        --config config/valid_configs_${SOURCE}_${TARGET}_${OPTION}/${DATASET}.config \
        --gpu $gpus \
        --prompt_emb ${DATASET}Prompt${SOURCE}  \
        --projector model/${DATAS}_${SOURCE}_${TARGET}_${OPTION}/${EPOCH}_model_cross.pkl\
        --output_name valid_result/${DATAS}_${SOURCE}_${TARGET}_${OPTION}/${EPOCH}
    done 
done
