#1e-3 distance_mask lam 1

import logging
import os
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
import shutil
from timeit import default_timer as timer
import random
import numpy as np
#from tools.eval_tool_projector import valid, gen_time_str, output_value #저자주석
from tools.eval_tool import valid, gen_time_str, output_value
from tools.init_tool import init_test_dataset, init_formatter
from reader.reader import init_dataset, init_formatter, init_test_dataset
import torch.nn as nn
import torch.optim as optim
#from model.optimizer import init_optimizer #저자주석
import transformers
from tools.projector import AE_0_layer, AE_1_layer_mutiple_100, AE_1_layer, AE_1_layer_mutiple_100_paper,AE_transformer_layer,AE_1_layer_tokenwise,AE_auto_layer

from model.modelling_roberta import RobertaEmbeddings
from model.modelling_bert import BertEmbeddings
from model.modeling_t5 import T5EncoderModel
from transformers import AutoConfig,AutoModelForMaskedLM,AutoTokenizer

#minwoo
import json
from collections import defaultdict

logger = logging.getLogger(__name__)

# for save projector weight
def checkpoint(filename, model, optimizer, trained_epoch, config, global_step, model_AE, **kwargs):
    
    filename = filename.strip().replace(".pkl","")
    filename = filename+"_model_cross.pkl"
    try:
        torch.save(model_AE.state_dict(), filename)
        logger.info("save projector ... <{}> .".format(filename))
    
    except Exception as e:
        logger.warning("Fail to save projector")
        

def train(parameters, config, gpu_list, do_test=False, local_rank=-1, **params):
    
    epoch = config.getint("train", "epoch")
    batch_size = config.getint("train", "batch_size")
    
    train_valid_info = defaultdict(dict) #minwoo to make json file (training infomation)

    output_time = config.getint("output", "output_time")
    test_time = config.getint("output", "test_time")

    output_path = os.path.join(config.get("output", "model_path"), config.get("output", "model_name"))
    
    if os.path.exists(output_path):
        if local_rank <= 0 :
            logger.warning("Output path exists, check whether need to change a name of model")
    
    os.makedirs(output_path, exist_ok=True)

    trained_epoch = parameters["trained_epoch"] + 1
    model = parameters["model"]
    
    if local_rank <= 0 :
        logger.info("Read source code from <model/{}.py> ".format(config.get("target_model","model_name")))
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # <select projector> 
    projector = config.get("projector","projector")
    if 'AE_1' in projector:
        if config.getboolean("projector","flatten") : 
            dim_0,dim_1,dim_2 = config.getint("projector","dim_0"),config.getint("projector","dim_1"),config.getint("projector","dim_2")
            model_AE = AE_1_layer_mutiple_100(dim_0=dim_0,dim_1=dim_1,dim_2=dim_2).to(device)
            
        else :
            
            if 'auto' in projector:
                    values = list(map(int,config.get('projector','dims').strip().split(',')))
                    keys = ["dim_"+str(idx) for idx in range(len(values))]
                    dims = dict(zip(keys,values))
                    model_AE = AE_auto_layer(**dims).to("cuda")

            elif 'transformer' in projector:
                    dim_0,dim_1= config.getint("projector","dim_0"),config.getint("projector","dim_1")
                    model_AE = AE_transformer_layer(dim_0=dim_0,dim_1=dim_1).to("cuda")
            else:
                dim_0,dim_1,dim_2 = config.getint("projector","dim_0"),config.getint("projector","dim_1"),config.getint("projector","dim_2")
                model_AE = AE_1_layer(dim_0 = dim_0, dim_1 = dim_1, dim_2 = dim_2).to("cuda")
            
    
    else:
        logger.warning("Fail to select projector. check tools/projector.py")
        NotImplementedError
        
    if local_rank <=0 :
        logger.info("selected projector class is <{}>".format(projector))
    # <\select projector> 
    
    # if local_rank <= 0 :
        # logger.info("projector model is {}".format(model_AE))
    
    for module in model_AE.modules():
        if isinstance(module, torch.nn.Linear):
            if "roberta" in config.get("target_model","model_base").lower():
                pass
            elif "t5" in config.get("target_model","model_base").lower():
                torch.nn.init.normal_(module.weight, mean=0, std=1)
            else:
                torch.nn.init.normal_(module.weight, mean=0, std=1)

    checkpoint_dir= "model/"+config.get("output", "model_name") #사실상 output_path 랑 다를바가 없음.
    
    record_train_epoch = 0
    
    if os.path.isdir(checkpoint_dir):
        checkpoints = os.listdir(checkpoint_dir)
        if len(checkpoints) > 0:
            last_checkpoint = checkpoints[0] #가장 최신의 checkpoint.
            for checkpoint_name in checkpoints:
                checkpoint_epoch = int(checkpoint_name.split("_")[0])
                last_checkpoint_epoch = int(last_checkpoint.split("_")[0])
                if checkpoint_epoch >= last_checkpoint_epoch:
                    last_checkpoint = checkpoint_name
                    record_train_epoch = last_checkpoint_epoch
            
            model_AE.load_state_dict(torch.load(checkpoint_dir+"/"+last_checkpoint, map_location=lambda storage, loc:storage))
            
            if local_rank <= 0 :
                logger.info("Load pretrained 'projector' from <{}>".format(checkpoint_dir+"/"+last_checkpoint))
    
        else:
            pass
    else:    
        pass
    
    # optimizer_AE = transformers.AdamW(model_AE.parameters(), eps=1e-06, lr=0.01, weight_decay=0.0, correct_bias=True)
    # optimizer_AE = transformers.AdamW(model_AE.parameters(), eps=1e-06, lr=0.0001, weight_decay=0.0, correct_bias=True)
    # optimizer_AE = transformers.AdamW(model_AE.parameters(), eps=1e-06, lr=1e-5, weight_decay=0.0, correct_bias=True)
    # optimizer_AE = transformers.AdamW(model_AE.parameters(), eps=1e-06, lr=1e-2, weight_decay=0.0, correct_bias=True)
    optimizer_AE = transformers.AdamW(model_AE.parameters(), eps=1e-06, lr=1e-3, weight_decay=0.0, correct_bias=True)
    global_step = parameters["global_step"]
    
    output_function = parameters["output_function"]

    if do_test:
        init_formatter(config, ["test"])
    
    os.makedirs(os.path.join(config.get("output", "tensorboard_path"), config.get("output", "model_name")),
                exist_ok=True)
    
    writer = SummaryWriter(os.path.join(config.get("output", "tensorboard_path"), config.get("output", "model_name")),
                           config.get("output", "model_name"))

    step_size = config.getint("train", "step_size")
    gamma = config.getfloat("train", "lr_multiplier")
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_AE, step_size=step_size, gamma=gamma)
        
    exp_lr_scheduler.step(trained_epoch)
    
    source_encoder = load_model(config,target_model=False).to("cuda")
    target_encoder = load_model(config,target_model=True).to("cuda")
    
        
    for epoch_num in range(trained_epoch, epoch):
        
        
        # dataset, parameters["valid_dataset"] = init_dataset(config, **params) #왜 2번하는지..?
        
        total_len = min([len(dataloader) for dataloader in parameters['train_dataset']])
        dataloader_1, dataloader_2, dataloader_3 = parameters['train_dataset']
        datasloader_zipped = zip(dataloader_1,dataloader_2,dataloader_3)
        
        #for shuffle
        for dataloader in (dataloader_1,dataloader_2,dataloader_3):
            dataloader.sampler.set_epoch(epoch_num)
            
        # datasets_zipped = zip(parameters['train_dataset'])
        if local_rank <=0:
            print("Epoch  Stage  Iterations  Time Usage    Loss    Output Information")
        
        if total_len < 10000 and epoch_num==trained_epoch:
            more = "\t"

        start_time = timer()
        current_epoch = epoch_num
        
        model.eval() #model 부분은 건드리지 않기로함.
        exp_lr_scheduler.step(current_epoch)

        
        total_loss = 0
        valid_total_loss = 0
        
        acc_result = None
        
        performance = 0
        
        MTLoss = 0
        lossList = len(parameters['train_dataset'])*[0]
        totallossList = len(parameters['train_dataset'])*[0]

        output_info = ""
        step = -1
        
        distanceList = len(parameters['train_dataset'])*[0]
        totaldistanceList = len(parameters['train_dataset'])*[0]
        distance_loss = torch.nn.CosineSimilarity(dim=2)
        # distance_loss = torch.nn.PairwiseDistance()
        
        total_distance = 0                     
        
        
        #각 batch 에 대하여 
        for step, (dataloader_1, dataloader_2, dataloader_3) in enumerate(datasloader_zipped):
            # tensor to cuda
            for dataloader in (dataloader_1, dataloader_2, dataloader_3): #datas = [source,target]
                for dataset in dataloader:
                    for key in dataset.keys():
                        if isinstance(dataset[key], torch.Tensor):
                            if len(gpu_list) > 0:
                                dataset[key] = Variable(dataset[key].cuda())
                            else:
                                dataset[key] = Variable(dataset[key])
                
            
            model_AE.zero_grad() 
            
            if "T5" in config.get("target_model","model_base"):
                for idx,(source_dataset,target_dataset) in enumerate([dataloader_1,dataloader_2,dataloader_3]):
                    results = model(target_dataset, config, gpu_list, acc_result, "train", args=params, step=step, performance=performance, AE=model_AE)
                    loss, performance = results["loss"], results["performance"]
                    lossList[idx] = loss
                    
            else:
                for idx,(source_dataset,target_dataset) in enumerate([dataloader_1,dataloader_2,dataloader_3]):
                    results = model(target_dataset, config, gpu_list, acc_result, "train", AE=model_AE)
                    loss, acc_result = results["loss"], results["acc_result"]
                    lossList[idx] = loss
                    
                    assert config.getboolean('projector','flatten') == False
                    
                    with torch.no_grad():
                        source_embedding = source_encoder.bert.embeddings(input_ids = source_dataset['inputx'][:,100:],attention_mask=source_dataset['mask'][:,100:])                        
                        source_mask = source_dataset['mask'].unsqueeze(2).expand(source_dataset['inputx'].size(0),source_dataset['inputx'].size(1),source_embedding.size(2))[:,100:,:]
                        source_masked_embedding = source_mask * source_embedding
                        
                    
                        target_embedding = target_encoder.roberta.embeddings(input_ids = target_dataset['inputx'][:,100:],attention_mask=target_dataset['mask'][:,100:])
                        target_mask = target_dataset['mask'].unsqueeze(2).expand(target_dataset['inputx'].size(0),target_dataset['inputx'].size(1),target_embedding.size(2))[:,100:,:]
                        target_masked_embedding = target_mask * target_embedding
                        
                    if config.getboolean('projector','flatten'):
                        source_module = source_masked_embedding @ torch.transpose(source_masked_embedding,1,2)
                        target_module = target_masked_embedding @ torch.transpose(target_masked_embedding,1,2)
                        lambda_ = 1e-5
                        distance = torch.norm(source_module-target_module,p='fro')*lambda_
                        # print(source_module.size())
                    
                    else :
                        source_module = model_AE(source_masked_embedding)
                        target_module = target_masked_embedding
                        lambda_ = 1
                        
                        distance = distance_loss(source_module,target_module)
                        distance = torch.mean(1-distance)*lambda_
                        
                        
                        # distance = torch.mean(1-distance_loss(source_module,target_module))*lambda_
                        
                    distanceList[idx] = distance
            
                    totallossList[idx] += loss
                    totaldistanceList[idx] += distance
                    
            MTLoss = sum(lossList) + sum(distanceList)
            total_loss += float(MTLoss)
            
            MTLoss.backward()
            optimizer_AE.step()

            if step % output_time == 0 and local_rank <= 0:
                if "T5" in config.get("target_model","model_base"):
                    delta_t = timer() - start_time
                    output_value(current_epoch, "train", "%d/%d" % (step + 1, total_len), "%s/%s" % (
                        gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                                 "%.3lf" % (total_loss / (step + 1)), "\t", '\r', config)
                else:
                    output_info = output_function(acc_result, config)
                    delta_t = timer() - start_time
                    output_value(current_epoch, "train", "%d/%d" % (step + 1, total_len), "%s/%s" % (
                        gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                                 "%.3lf" % (total_loss / (step + 1)), output_info, '\r', config)



            if "T5" in config.get("target_model","model_base") and int(step%10) == 0 and local_rank <=0 : 
                print("\t \t \t \t \t \t \t","Performance:", performance) #현재 여기에서 모든 local_rank 에서 performance 를 출력중이다. -> localrank 설정 추가

            global_step += 1
            writer.add_scalar(config.get("output", "model_name") + "_train_iter", float(MTLoss), global_step)

        try:
            model.module.lower_temp(0.8)
        except:
            pass

        if local_rank <= 0:
            if "T5" in config.get("target_model","model_base"):
                pass
            else:
                output_info = output_function(acc_result, config)
                #output_info_target = output_function(acc_result_target, config)
                delta_t = timer() - start_time
                output_value(current_epoch, "train", "%d/%d" % (step + 1, total_len), "%s/%s" % (
                    gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                            "%.3lf" % (total_loss / (step + 1)), output_info, None, config)

        if step == -1:
            logger.error("There is no data given to the model in this epoch, check your data.")
            raise NotImplementedError

        if local_rank <= 0:
            checkpoint(os.path.join(output_path, "%d.pkl" % current_epoch), model, optimizer_AE, current_epoch, config, global_step, model_AE)
            
        writer.add_scalar(config.get("output", "model_name") + "_train_epoch_total_loss", float(total_loss) / (step + 1), current_epoch)
        
        train_total_loss = float(total_loss)
        train_epoch_loss = float(sum(lossList)) #last step loss in a epoch
        
        
        if "T5" in config.get("target_model","model_base"):
            pass 
        else:
            writer.add_scalar(config.get("output", "model_name") + "_train_epoch_acc", round(float(acc_result['right']/acc_result['total']),4), current_epoch)
            
        if current_epoch % test_time == 0 :
            with torch.no_grad():
                
                valid_epoch_loss_list = len(parameters['valid_dataset'])*[0]
                acc_result_eval_epoch = {'total': 0, 'right': 0}
                acc_result_eval_list = len(parameters['valid_dataset'])*[None]
                
                if "T5" in config.get("target_model","model_base"):
                    for idx,valid_dataset in enumerate(parameters['valid_dataset']):
                        acc_result_eval = valid(model, valid_dataset, current_epoch, writer, config, gpu_list, output_function, AE=model_AE)
                        
                        acc_result_eval_list[idx] = acc_result_eval
                        
                        acc_result_eval_epoch['total'] += acc_result_eval['total']
                        acc_result_eval_epoch['right'] += acc_result_eval['right']

                else:
                    for idx,valid_dataset in enumerate(parameters['valid_dataset']):
                        
                        valid_loss, acc_result_eval = valid(model, valid_dataset, current_epoch, writer, config, gpu_list, output_function, AE=model_AE)
                        
                        acc_result_eval_list[idx] = acc_result_eval
                        
                        acc_result_eval_epoch['total'] += acc_result_eval['total']
                        acc_result_eval_epoch['right'] += acc_result_eval['right']
                        
                        valid_epoch_loss_list[idx] = valid_loss

                    valid_total_loss += float(sum(valid_epoch_loss_list))
                    
                # if local_rank <=0 :
                #     root_dir = "model/"+config.get("output", "model_name")
                #     src_checkpoint_name = root_dir+"/"+str(current_epoch)+"_model_cross.pkl"
                #     targ_checkpoint_name = root_dir+"/"+str(current_epoch)+"_model_cross_"+str(round(float(acc_result_eval_epoch['right']/acc_result_eval_epoch['total']),4))+".pkl"
                #     os.rename(src_checkpoint_name, targ_checkpoint_name)

                    
        writer.add_scalar(config.get("output", "model_name") + "_valid_epoch_acc",round(float(acc_result_eval['right']/acc_result_eval['total']),4), current_epoch)
        
        
        if local_rank <=0 and not "T5" in config.get("target_model","model_base"):            
            
            if config.get("train","source_model"):
                json_path = "result/" + config.get("data","train_dataset_type").replace(',','_') + '_' + config.get("train","source_model")+'_'+config.get("target_model","model_base") + config.get("target_model","model_size")
            elif config.get("train","prompt_emb"):
                json_path = "result/" + config.get("data","train_dataset_type").replace(',','_') + '_' + config.get("train","prompt_emb")+'_'+config.get("target_model","model_base") + config.get("target_model","model_size")
            else :
                json_path = "result/" + config.get("data","train_dataset_type").replace(',','_') + '_' + 'NAN'+'_'+config.get("target_model","model_base") + config.get("target_model","model_size")
            if not os.path.exists('result'):
                os.mkdir('result')
            elif os.path.exists(json_path):
                with open(json_path,'r',encoding='utf-8') as file:
                    train_valid_info = json.load(file)
                    
            #each epoch average loss
            train_valid_info["train_average_loss"][current_epoch] = round(float(train_total_loss) / (step + 1),6)
            train_valid_info["valid_average_loss"][current_epoch] = round(float(valid_total_loss),6)
            
            #each epoch sample loss
            train_valid_info["train_epoch_loss"][current_epoch] = round(float(train_total_loss) / (step + 1),6)
            train_valid_info["valid_epoch_loss"][current_epoch] = round(float(sum(valid_epoch_loss_list)) ,4)
            train_valid_info["distance_epoch_loss"][current_epoch] = round(float(sum(totaldistanceList)/ (step + 1)),6)
            
            
            #each epoch sample acc
            train_valid_info["train_epoch_acc"][current_epoch] = round(float(acc_result['right']/acc_result['total']),4)
            train_valid_info["valid_epoch_acc"][current_epoch] = round(float(acc_result_eval_epoch['right']/acc_result_eval_epoch['total']),4)
            
            train_data_list = config.get("data","train_dataset_type").split(',')
            valid_data_list = config.get("data","valid_dataset_type").split(',')
            
            
            for idx,data in enumerate(train_data_list):
                train_loss = data + "_train_loss"
                train_valid_info[train_loss][current_epoch] = round(float(totallossList[idx]/(step+1)),4)
                
            for idx,data in enumerate(valid_data_list):    
                valid_loss = data + "_valid_loss"
                valid_acc = data + "_valid_acc"
                train_valid_info[valid_loss][current_epoch] = round(float(valid_epoch_loss_list[idx]),4)
                train_valid_info[valid_acc][current_epoch] = round(float(acc_result_eval_list[idx]['right']/acc_result_eval_list[idx]['total']),4)
                
            with open(json_path,'w',encoding='utf-8') as make_file:
                json.dump(train_valid_info,make_file,indent = "\t")
        
        if local_rank >= 0:
            torch.distributed.barrier()


def load_model(config,target_model = True):
    
    local_map = config.getboolean("train","local_map")
    
    assert local_map == False
    

    
    if target_model:    
        model_name = config.get("target_model","model_base").lower() 
    else :
        model_name = config.get("train","source_model").lower()
    
    if "roberta" in model_name:
        try:
            if "large" in model_name:
                model = "roberta-large"
                ckp = "PLM/RobertaLargeForMaskedLM"
                hidden_size = 1024
            else:
                model = "roberta-base"
                ckp = "PLM/RobertaForMaskedLM"
                hidden_size = 768
            
        except:
            model = "roberta-base"
            ckp = "PLM/RobertaForMaskedLM"
            hidden_size = 768
    
    elif "bert" in model_name:
        try:
            if "large" in model_name:
                model = "bert-large"
                ckp = "PLM/BertLargeForMaskedLM"
                hidden_size = 1024
            elif "medium" in model_name:
                model = "prajjwal1/bert-medium"
                ckp = "PLM/BertMediumForMaskedLM"
                hidden_size = 512
            else :
                model = "bert-base-uncased"
                ckp = "PLM/BertForMaskedLM"
                hidden_size = 768
            
        except:
            model = "bert-base-uncased"
            ckp = "PLM/BertForMaskedLM"
            hidden_size = 768
    elif 't5' in model_name :
        
        try:
            if "small" in model_name:
                model = "t5-small"
                ckp = "PLM/T5SmallForMaskedLM"
                hidden_size = 512
            elif "large" in model_name:
                model = "t5-large"
                ckp = "PLM/T5LargeForMaskedLM"
                hidden_size = 1024
            elif "b3" in model_name:
                model = "t5-b3"
                ckp = "PLM/T5B3ForMaskedLM"
                hidden_size = 1024
            
            else:
                model = "t5-base"
                ckp = "PLM/T5ForMaskedLM"
                hidden_size = 768
        except:
            model = "t5-base"
            ckp = "PLM/T5ForMaskedLM"
            hidden_size = 768

    
    else:
        
        print("load_model function error! <{}>".format(model_name))
        exit()
    
    if "bert-medium" in model: 
        model = "bert-medium"

    plmconfig = AutoConfig.from_pretrained(model)
    plmconfig.prompt_num = config.getint("prompt", "prompt_num")
    plmconfig.prompt_len = config.getint("prompt", "prompt_len")

    
    if "large" in model_name:
        init_model_path = str(ckp)+"/"+"Prompt"+str(model.split("-")[0].capitalize())+"Large"+"_init_params"
    
    elif "medium" in model_name:
        init_model_path = str(ckp)+"/"+"Prompt"+str(model.split("-")[0].capitalize())+"Medium"+"_init_params"
    
    else:
        init_model_path = str(ckp)+"/"+"Prompt"+str(model.split("-")[0].capitalize())+"_init_params"

    if os.path.exists(init_model_path+"/pytorch_model.bin"):
        if "roberta" in model_name:
            from model.modelling_roberta import RobertaForMaskedLM
            encoder = RobertaForMaskedLM.from_pretrained(init_model_path, config=plmconfig)
        elif "bert" in model_name:
            from model.modelling_bert import BertForMaskedLM
            encoder = BertForMaskedLM.from_pretrained(init_model_path, config=plmconfig)
        
        elif 't5' in model_name:
            from model.modeling_t5 import T5ForConditionalGeneration
            encoder = T5ForConditionalGeneration.from_pretrained(init_model_path, config=plmconfig)
            
        else:
            exit()
    else:
        if "roberta" in model_name:
            from model.modelling_roberta import RobertaForMaskedLM
            encoder = RobertaForMaskedLM.from_pretrained(model, config=plmconfig)
            os.mkdir(init_model_path)
            torch.save(encoder.state_dict(), str(init_model_path)+"/pytorch_model.bin")
            print("Save Done")
            encoder = RobertaForMaskedLM.from_pretrained(init_model_path, config=plmconfig)
        elif "bert" in model_name:
            from model.modelling_bert import BertForMaskedLM
            encoder = BertForMaskedLM.from_pretrained(model, config=plmconfig)
            os.mkdir(init_model_path)
            torch.save(encoder.state_dict(), str(init_model_path)+"/pytorch_model.bin")
            print("Save Done")
            encoder = BertForMaskedLM.from_pretrained(init_model_path, config=plmconfig)
        
        elif 't5' in model_name :
            encoder = T5ForConditionalGeneration.from_pretrained(model, config=plmconfig)

            os.mkdir(init_model_path)
            torch.save(encoder.state_dict(), str(init_model_path)+"/pytorch_model.bin")
            print("Save Done")

            encoder = T5ForConditionalGeneration.from_pretrained(init_model_path, config=plmconfig)

        
        else:
            exit()
    if config.getint('distributed','local_rank') <= 0:
        if target_model :
            print("load target model = <{}> for distance compute.".format(model))
        else :
            print("load source model = <{}> for distance compute.".format(model))
            
    return encoder
    ##############