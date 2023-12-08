
# Bert: BertMedium, Bert (BertBase)
# Roberta: Roberta, RobertaBase (BertBase)
# T5: T5Small, T5 (T5Base), T5Large


gpus="4"

for DATASET in imdb sst2 laptop restaurant mnli qnli snli qqp mrpc
do
    CUDA_VISIBLE_DEVICES=$gpus python3 valid.py \
        --config config/valid_configs_Roberta_Base/${DATASET}.config \
        --gpu $gpus \
        --prompt_emb ${DATASET}PromptBert\
        --seed 28 \
        --projector projector_hub/15_model_cross_0.8896.pkl
done 

    # --projector model/crossPromptTraining/50_model_cross_0.7117.pkl\
    
    
    # --checkpoint model/${DATASET}Prompt${BACKBONE} \
    # --model_transfer_projector \
    

# Bert: BertMedium, Bert (BertBase)
# Roberta: Roberta(=RobertBase), RobertaLarge
# T5: T5Small, T5 (T5Base), T5Large

# replacing_prompt: 가져올 prompt
# config: valid score 계산할 data

# gpus="0"
# SOURCE="Bert"     # 세번째 나오는 모델 (source model)
# TARGET="Roberta"  # 네번째 나오는 모델 (target model)
# FOLDER="BertRoberta_sst2_mnli_qqp"
# FILE="17_model_cross_0.9291.pkl"

# for DATASET in imdbPrompt sst2Prompt laptopPrompt restaurantPrompt mnliPrompt qnliPrompt snliPrompt qqpPrompt mrpcPrompt
# do    
#     CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/${DATASET}${TARGET}.config \
#         --gpu $gpus \
#         --prompt_emb ${DATASET}${SOURCE}\
#         --projector model/${FOLDER}/${FILE}
# done