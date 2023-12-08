if __name__ == "__main__":
    import sys
    import os
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(current_directory)
    if parent_directory != sys.path[-1]:
        sys.path.append(parent_directory)
        print("Append parent path : ",parent_directory)
    else :
        print("existing")

    for path in sys.path:
        print(path)
    
    from output_init import init_output_function

else :
    try:
        from .output_init import init_output_function
    except ImportError:
        from output_init import init_output_function

    

import logging
import torch
from reader.reader import init_dataset, init_formatter, init_test_dataset
from model import get_model
from model.optimizer import init_optimizer
# from .output_init import init_output_function
from torch import nn
from transformers import AutoTokenizer
import string
import os
from tools.projector import AE_0_layer, AE_1_layer_mutiple_100, AE_1_layer, AE_auto_layer, AE_1_layer_tokenwise

from transformers import AutoConfig,AutoModelForMaskedLM,AutoTokenizer


logger = logging.getLogger(__name__)


def recover_model_transfer_prompt(prompt_emb,projector,config):
    # prompt_emb = recover_model_transfer_prompt(prompt_emb,params["args"].projector,config)
    
    # print("Applied Projector weight file :",prompt_emb)
    
    model_parameters = torch.load(projector, map_location=lambda storage, loc:storage)

    if config.get("projector","projector") == 'AE_1' : 
        if config.getboolean("projector","flatten") : 
            model = AE_1_layer_mutiple_100(dim_0=int(model_parameters["encoder.weight"].shape[1]),dim_1=int(model_parameters["encoder.weight"].shape[0]),dim_2=int(model_parameters["decoder.weight"].shape[0])).to("cuda")
        else :
            dim_0,dim_1,dim_2 = config.getint("projector","dim_0"),config.getint("projector","dim_1"),config.getint("projector","dim_2")
            model = AE_1_layer(dim_0 = dim_0, dim_1 = dim_1, dim_2 = dim_2).to("cuda")
    
    else :
        print("check projector in your config")
        NotImplementedError        
    
    if projector == "Random" or projector=="random":
        pass
    else:
        model.load_state_dict(model_parameters)
    
    #projector weight freeze.
    model.eval()
    if config.getboolean("projector","flatten") : 
        prompt_emb_ = prompt_emb.reshape(1,int(prompt_emb.shape[0])*int(prompt_emb.shape[1]))
        prompt_emb_ = model(prompt_emb_.to("cuda"))
        dim_out = int(int(model.decoder.weight.shape[0])/int(prompt_emb.shape[0]))
        prompt_emb_ = prompt_emb_.reshape(int(prompt_emb.shape[0]),dim_out)
    
    else :
        prompt_emb_ = torch.unsqueeze(prompt_emb,0)
        prompt_emb_ = model(prompt_emb_.to("cuda"))
        prompt_emb_ = torch.squeeze(prompt_emb_)
        
    return prompt_emb_

    
    

def init_all(config, gpu_list, mode, *args, **params): #->dictionary
    
    result = {} 
    local_rank = params['local_rank']
    
    # Using reader 
    if mode=="test":
        result["test_dataset"] = init_test_dataset(config, *args, **params)
        
    elif mode=="train" :
        result["train_dataset"], result["valid_dataset"] = init_dataset(config, *args, **params)
    
    elif mode == 'valid':
        result["valid_dataset"] = init_dataset(config, only_valid = True,*args, **params)
    else:
        logger.warning("Check your mode! mode = <{}> in your config file".format(mode))
    
    
    model = get_model(config.get("target_model", "model_name"))(config, gpu_list, *args, **params)
    
    if local_rank <= 0:
        logger.info(" model = <{}> ".format(config.get("target_model", "model_name")))
        
    optimizer = init_optimizer(model, config, *args, **params)
    trained_epoch = 0
    global_step = 0
    
    #Cross model training : 두 model 복원을 위한 부분인듯?
    if os.path.isdir("model/"+config.get("output", "model_name")) and "cross" in config.get("target_model", "model_name"):
        
        if local_rank <= 0:
            logger.info("searching in model/{} ".format(config.get("output", "model_name")))
        
        all_checkpoints = os.listdir("model/"+config.get("output", "model_name"))
        max_checkpoint_epoch = 0
        for checkpoint_epoch in all_checkpoints: 
            if int(checkpoint_epoch.split("_")[0]) > max_checkpoint_epoch:
                max_checkpoint_epoch = int(checkpoint_epoch.split("_")[0])
        trained_epoch = max_checkpoint_epoch
        
        if local_rank <= 0:
            logger.info("   => cross model! trained_epoch = <{}> is setted from <model/{}> ".format(trained_epoch,config.get("output", "model_name")))                
    
    else:
        if local_rank <= 0 :
            logger.info("cross model! There is no pretrained info.")
        pass
    
    

    ###########################< 이 곳은 valid, test mode 일 경우 작동합니다. >#############################
    
    if mode=="valid" or mode=="Valid" or mode=="test" or mode=="Test":        
        
        ###Replace or not
        if "Random" in params["args"].prompt_emb or "random" in params["args"].prompt_emb and params["args"].prompt_emb!="randomPromptRobertaLarge":
            
            model_size = config.get("target_model", "model_size")
            
            if 'large' in model_size.lower():
                prompt_emb = torch.rand(config.getint("prompt","prompt_num"),1024).to("cuda")
            if "small" in model_size.lower():
                prompt_emb = torch.rand(config.getint("prompt","prompt_num"),512).to("cuda")
            else:
                prompt_emb = torch.rand(config.getint("prompt","prompt_num"),768).to("cuda")
        
        else:
            
            load_prompt_dir = params["args"].prompt_emb
            
            load_prompt_dir = "task_prompt_emb/"+load_prompt_dir.replace(" ","")+"/task_prompt"
            
            prompt_emb = torch.load(load_prompt_dir, map_location=lambda storage, loc: storage)
            
            if local_rank<=0 :
                print("load sourece prompt_emb... from <{}>".format(load_prompt_dir))
        
        if params["args"].projector :    
            prompt_emb = recover_model_transfer_prompt(prompt_emb,params["args"].projector,config)
            
            if local_rank<=0 :
                print("source prompt success to pass projector!")

        else :
            if local_rank<=0 :
                print("There is no projector!")
            

        if prompt_emb != None: 
            prompt_emb = torch.nn.Parameter(prompt_emb).to("cuda")
            
            ##Put prompt emb back to model
            if "roberta" in config.get("target_model", "model_base").lower():
                model.encoder.roberta.embeddings.prompt_embeddings.weight.data = prompt_emb
                
            elif "bert" in config.get("target_model", "model_base").lower():
                model.encoder.bert.embeddings.prompt_embeddings.weight.data = prompt_emb
                
            elif "t5" in config.get("target_model", "model_base").lower():
                #model.encoder.t5.embeddings.prompt_embeddings.weight.data = prompt_emb
                model.encoder.prompt_embeddings.weight.data = prompt_emb
                model.encoder.encoder.prompt_tokens.weight.data = prompt_emb
                model.encoder.decoder.prompt_tokens.weight.data = prompt_emb
            else:
                
                logger.error("Wrong!!! -> prompt_emb can't attach to projector")
                exit()
            
            if local_rank <= 0 :
                logger.info("success to attach final embedding weight!")
            
        else:
            pass

    else:
        pass
    
    ##############################################< 다시 모든 mode 에 대하여 일괄적으로 적용됩니다.>#########################

    #병렬화 부분
    ############
    if len(gpu_list) > 0: #GPU 가 있다면,
        if params['local_rank'] < 0: #local rank = -1 이라면 (single machine)
            model = model.cuda()
        else:
            model = model.to(gpu_list[params['local_rank']])

        try:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[params['local_rank']], output_device=params['local_rank'], find_unused_parameters = True)
            if local_rank <= 0 :
                logger.info("nn.parallel.DistributedDataParallel run...")

        except Exception as e:
            logger.warning("do not use nn.parallel.DistributedDataParallel")
            

    result["model"] = model
    if mode == "train" or mode == "valid":
        
        result["optimizer"] = optimizer
        result["trained_epoch"] = trained_epoch
        result["output_function"] = init_output_function(config)
        result["global_step"] = global_step

    return result
