import argparse
import os
import torch
import logging
import random
import numpy as np

from tools.init_tool import init_all
from config_parser import create_config
from tools.valid_tool import valid

# format='%(asctime)s - %(levelname)s - %(name)s - %(message)s - %(lineno)d'

logging.basicConfig(format='%(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    # filename="./log/train_cross.log",
                    # filemode='w')
)
logger = logging.getLogger(__name__)


#random_seed 설정
def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


    ###
    #model_prompt= "Bert-base", "Roberta-base", "Random"
    ###

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help="specific config file", required=True)
    parser.add_argument('--gpu', '-g', help="gpu id list")
    parser.add_argument('--local_rank', type=int, help='local rank', default=-1)
    
    parser.add_argument("--prompt_emb", type=str, default=None)
    parser.add_argument("--projector", type=str, default=None)
    
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mode", type=str, default="valid")
    
    parser.add_argument("--output_name", type=str, default=None)
    
    args = parser.parse_args()
    configFilePath = args.config
    
    config = create_config(configFilePath)

    if args.prompt_emb :
        config.set("train","prompt_emb",args.prompt_emb)
    else :
        logger.warning("please set prompt_emb! e.g. SST2PromptBert")
        exit()
    
    
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
    
    if config.getboolean("distributed", "use") and len(gpu_list)>1:
        torch.cuda.set_device(gpu_list[args.local_rank])
        torch.distributed.init_process_group(backend=config.get("distributed", "backend"))
        config.set('distributed', 'gpu_num', len(gpu_list))
    else :
        config.set("distributed", "use", False)
    
    os.system("clear")
    
    if args.local_rank <= 0 :
        print("config file path = <{}>".format(configFilePath))
        
    if args.local_rank >=0:
        torch.distributed.barrier()
    
    cuda = torch.cuda.is_available()
    cuda_available = str(cuda)
    
    if not cuda and len(gpu_list) > 0:
        logger.error("CUDA is not available but specific gpu id")
        raise NotImplementedError
    set_random_seed(args.seed)


    assert len(config.get("data","valid_dataset_type").lower().split(',')) == 1
    
    parameters = init_all(config, gpu_list, args.mode, local_rank = args.local_rank, args=args)
    
    model = parameters["model"]

    if args.local_rank >=0:
        torch.distributed.barrier()
    
    valid(model, parameters["valid_dataset"], 1, None, config, gpu_list, parameters["output_function"], mode=args.mode, args=args)
    