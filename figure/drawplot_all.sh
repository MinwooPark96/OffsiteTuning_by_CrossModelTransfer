#!/bin/bash\

# python3 drawplot_all.py -n mnli_sst2_qqp_Bert_RobertaBase_lr0.005 --nbins 4
# python3 drawplot_all.py -n sst2_qqp_Bert_RobertaBase --nbins 8
# python3 drawplot_all.py -n sst2_mnli_qqp_Bert_RobertaBase_lr0.0001 --nbins 8
# python3 drawplot_all.py -n sst2_mnli_qqp_Bert_RobertaBase --nbins 8
# python3 drawplot_all.py -n sst2_mnli_qqp_Bert_RobertaBase --nbins 8
# python3 drawplot_all.py -n sst2_mnli_qqp_Bert_RobertaBase_AE_1_layer --nbins 8
# python3 drawplot_all.py -n sst2_mnli_qqp_Bert_RobertaBase_AE_1_layer_tokenwise_hidden128 --nbins 8
# python3 drawplot_all.py -n sst2_mnli_qqp_each_hidden256_nonflat_distance_lambda0.1 --nbins 8
# python3 drawplot_all.py -n sst2_mnli_qqp_Bert_RobertaBase_nonflat_pronorm --nbins 8
python3 drawplot_all.py -n sst2_mnli_qqp_Bert_RobertaBase_nonflat_lambda0.05 --nbins 8
