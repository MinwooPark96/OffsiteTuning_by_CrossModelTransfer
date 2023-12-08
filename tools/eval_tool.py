import logging
import os
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from timeit import default_timer as timer

logger = logging.getLogger(__name__)


def gen_time_str(t):
    t = int(t)
    minute = t // 60
    second = t % 60
    return '%2d:%02d' % (minute, second)


def output_value(epoch, mode, step, time, loss, info, end, config):
    try:
        delimiter = config.get("output", "delimiter")
    except Exception as e:
        delimiter = " "
    s = ""
    s = s + str(epoch) + " "
    while len(s) < 7:
        s += " "
    s = s + str(mode) + " "
    while len(s) < 14:
        s += " "
    s = s + str(step) + " "
    while len(s) < 25:
        s += " "
    s += str(time)
    while len(s) < 40:
        s += " "
    s += str(loss)
    while len(s) < 48:
        s += " "
    s += str(info)
    s = s.replace(" ", delimiter)
    if not (end is None):
        print(s, end=end)
    else:
        print(s)


def valid(model, dataset, epoch, writer, config, gpu_list, output_function, mode="valid", **kwargs):


    if "args" in kwargs:
        kwargs = kwargs["args"]

    model.eval()
    local_rank = config.getint('distributed', 'local_rank')

    acc_result = None
    #acc_result_target = None
    total_loss = 0
    total_loss_target = 0
    cnt = 0
    total_len = len(dataset)
    start_time = timer()
    output_info = ""

    output_time = config.getint("output", "output_time")
    step = -1
    more = ""
    if total_len < 10000:
        more = "\t"
    for step, data in enumerate(dataset):
        try:
            data_name = data['name']
        except :
            print(data)
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                if len(gpu_list) > 0:
                    data[key] = Variable(data[key].cuda())
                else:
                    data[key] = Variable(data[key])

        if "AE" in kwargs:
            results = model(data, config, gpu_list, acc_result, "valid", AE=kwargs["AE"])
        else:
            #실험중
            results = model(data, config, gpu_list, acc_result, "valid", args=kwargs)
            # results = model(data, config, gpu_list, acc_result, mode , args=kwargs)
        if "T5" in config.get("target_model","model_base"):
            acc_result = results["acc_result"]
        else:
            loss, acc_result = results["loss"], results["acc_result"]
        #if "AE" in kwargs:
        #    loss, loss_target, acc_result, acc_result_target = results["loss_total"], results["loss_target"], results["acc_result"], results["acc_result_target"]


        if "T5" in config.get("target_model","model_base"):
            pass
        else:
            total_loss += float(loss)
            #total_loss_target += float(loss_target)
            cnt += 1

            if step % output_time == 0 and local_rank <= 0:
                delta_t = timer() - start_time
                output_value(epoch, mode, "%d/%d" % (step + 1, total_len), "%s/%s" % (
                    gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                             "%.3lf" % (total_loss / (step + 1)), output_info, '\r', config)


    # if local_rank ==0 :
    #     print(acc_result)
    #     for i in acc_result:
    #         print(i)
    # # exit()
    
    if step == -1:
        logger.error("There is no data given to the model in this epoch, check your data.")
        raise NotImplementedError

    if config.getboolean("distributed", "use"):
        shape = (len(acc_result), 1) # 4 -> 1 (machie 개수 인듯?) 
        mytensor = torch.LongTensor([acc_result[key] for key in acc_result]).to(gpu_list[local_rank])
        
        # print(local_rank)
        # mytensor = torch.LongTensor([[key["TP"], key["FN"], key["FP"], key["TN"]] for key in acc_result]).to(gpu_list[local_rank])
        mylist = [torch.LongTensor(shape[0], shape[1]).to(gpu_list[local_rank]) for i in range(config.getint('distributed', 'gpu_num'))]
        # print('shape', shape)
        # print('mytensor', mytensor.shape)
        # print(mylist)
        
        torch.distributed.all_gather(mylist, mytensor)#, 0)
        
        if local_rank == 0:
            mytensor = sum(mylist)
            index = 0
            # for i in range(len(acc_result)):
            #     acc_result[i]['TP'], acc_result[i]['FN'], acc_result[i]['FP'], acc_result[i]['TN'] = int(mytensor[i][0]), int(mytensor[i][1]), int(mytensor[i][2]), int(mytensor[i][3])
            for key in acc_result:
                acc_result[key] = int(mytensor[index])
                index += 1
    
    if local_rank <= 0:
        delta_t = timer() - start_time
        output_info = output_function(acc_result, config)
        #if "AE" in kwargs:
        #    output_info_target = output_function(acc_result_target, config)

        if "T5" in config.get("target_model","model_base"):
            output_value(epoch, mode+"/"+data_name, "%d/%d" % (step + 1, total_len), "%s/%s" % (
                gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),"\t", output_info, None, config)

        else:
            output_value(epoch, mode+"/"+data_name, "%d/%d" % (step + 1, total_len), "%s/%s" % (
                gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                        "%.3lf" % (total_loss / (step + 1)), output_info, None, config)

        #if "AE" in kwargs:
        #    output_value(epoch, mode, "%d/%d" % (step + 1, total_len), "%s/%s" % (
        #        gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
        #                "%.3lf" % (total_loss_target / (step + 1)), output_info_target, None, config)

    if "T5" in config.get("target_model","model_base"):
        pass
    else:
        writer.add_scalar(config.get("output", "model_name") + "_eval_epoch", float(total_loss) / (step + 1), epoch)
    ######Acc
    writer.add_scalar(config.get("output", "model_name") + "_eval_epoch_acc", float(acc_result['right']/acc_result['total']), epoch)
        ######

    model.train()

    ###Add
    if "T5" in config.get("target_model","model_base"):
        return acc_result
    elif "AE" in kwargs:
        return round(total_loss/(step+1),3), acc_result
    else:
        return round(total_loss/(step+1),3)
    ###
