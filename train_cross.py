import argparse
import os
import torch
import logging
import random
import numpy as np

from tools.init_tool import init_all
from config_parser import create_config

from tools.train_tool_cross_single import train as train_single
from tools.train_tool_cross_mtl2 import train as train_mtl2
from tools.train_tool_cross_mtl3 import train as train_mtl3
from tools.train_tool_cross_mtl4 import train as train_mtl4
from tools.train_tool_cross_mtl import train as train

#log 파일의 형식을 설정 
# format='%(asctime)s - %(levelname)s - %(name)s - %(message)s - %(lineno)d'

logging.basicConfig(format='%(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    # filename="./log/train_cross.log",
                    # filemode='w')
)
logger = logging.getLogger(__name__)


def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 필수 argument
    parser.add_argument('--config', '-c', help="specific config file", default='config/default.config') 
    parser.add_argument('--gpu', '-g', help="gpu id list", default='0') # -> "0"
    
    
    parser.add_argument("--prompt_emb",type=str,default=False) #projector 에 들어갈 사전 학습된 prompt_emb 를 불러올 argument.
    parser.add_argument("--source_model",type=str,default=False) #projector 에 들어갈 사전 학습된 prompt_emb 를 불러올 argument.
    
    
    
    parser.add_argument('--do_test', help="do test while training or not", action="store_true",default=False)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model_transfer", type=str, default=False)
    
    
    # parser.add_argument('--checkpoint', help="checkpoint file path", type=str, default=None)
    
    #parser.add_argument("--source_model", type=str, default=None) # -> "Roberta-base"
    # parser.add_argument("--task_transfer", type=str, default=False) #이걸 사용을 할지?    
    #parser.add_argument("--pre_train_mlm", type=bool, default=False)  #이걸 사용을 할지?
    #parser.add_argument('--comment', help="checkpoint file path", default=None) # checkpoint 랑 겹치는 것 같음.
    #parser.add_argument("--prompt_emb_output", type=bool, default=False) # 학습한 softprompt 를 task_prompt_emb 에 저장할지 말지?
    #parser.add_argument("--save_name", type=str, default=None) #-> emb extract mode 일 경우 task_prompt_emb/save_name
    #parser.add_argument('--local_rank', type=int, help='local rank', default=-1) -> 우리 코드에선 필요 없어보임.
    
    os.system("clear")
    
    args = parser.parse_args()
    configFilePath = args.config
    config = create_config(configFilePath)

     #minwoo - prompt_emb or source_model
    if args.prompt_emb and args.source_model :
        logger.warning("set only one argument <prompt_emb> or <source_model>")        
        exit()
    elif (not args.prompt_emb) and (not args.source_model):
        logger.warning("set at least one argument <prompt_emb> or <source_model>")        
        exit()
    
    config.set("train","prompt_emb",args.prompt_emb)
    config.set("train","source_model",args.source_model)

    print("prompt_emb",config.get("train","prompt_emb"))
    print("source_model",config.get("train","source_model"))
    
    use_gpu = True
    gpu_list = []
    if args.gpu is None:
        use_gpu = False
    
    else:
        use_gpu = True
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        device_list = args.gpu.split(",")
        for a in range(0, len(device_list)):
            gpu_list.append(int(a))

    if len(gpu_list)>=2 :
        args.local_rank = int(os.environ["LOCAL_RANK"])
        config.set('distributed', 'local_rank', args.local_rank)
    
    else :
        config.set('distributed', 'local_rank', -1)

    local_rank = config.getint('distributed', 'local_rank')
    
    if config.getboolean("distributed", "use") and len(gpu_list)>1:
        torch.distributed.init_process_group(backend=config.get("distributed", "backend"))
        torch.cuda.set_device(gpu_list[args.local_rank])
        config.set('distributed', 'gpu_num', len(gpu_list))
    else:
        config.set("distributed", "use", False)

    if local_rank >=0:
        torch.distributed.barrier()

    if local_rank <= 0:
        logger.info("config file = <{}>".format(configFilePath))
    
    cuda = torch.cuda.is_available()
    cuda_available = str(cuda)
    
    if not cuda and len(gpu_list) > 0:
        logger.error("CUDA is not available but specific gpu id")
        raise NotImplementedError
    
    set_random_seed(args.seed)

    
    parameters = init_all(config, gpu_list, "train", local_rank = local_rank, args=args) #model_prompt=args.source_model 제거
    
    do_test = False
    if args.do_test:
        do_test = True
    
    if local_rank >=0:
        torch.distributed.barrier()
    
    train_dataset_list = config.get("data","train_dataset_type").lower().split(',')
    
    # if len(train_dataset_list) == 1:
    #     train_single(parameters, config, gpu_list, do_test,local_rank, args=args)
    # elif len(train_dataset_list) == 2:
    #     train_mtl2(parameters, config, gpu_list, do_test,local_rank, args=args)
    # elif len(train_dataset_list) == 3:
    #     train_mtl3(parameters, config, gpu_list, do_test,local_rank, args=args)
    # elif len(train_dataset_list) == 4:
    #     train_mtl4(parameters, config, gpu_list, do_test,local_rank, args=args)
    # else :
    #     logger.warning("can not find train_tool!")

    train(parameters, config, gpu_list, do_test,local_rank, args=args)