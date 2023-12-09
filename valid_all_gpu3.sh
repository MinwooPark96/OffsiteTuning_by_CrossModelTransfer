
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

gpus="3"

for DATASET in imdb sst2 laptop restaurant movierationales tweetevalsentiment mnli qnli snli ethicsdeontology ethicsjustice qqp mrpc
do
    CUDA_VISIBLE_DEVICES=$gpus python3 valid.py \
        --config config/valid_configs_Roberta_Large/${DATASET}.config \
        --gpu $gpus \
        --prompt_emb ${DATASET}PromptRoberta\
        --projector model/crossPromptTraining/20_model_cross_0.8423.pkl\
        --output_name '20ep'
done 

