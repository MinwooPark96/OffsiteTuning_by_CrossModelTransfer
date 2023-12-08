# from .BinaryClsBERT import BinaryClsBERT,PromptClsBERT
# from .REBERT import REBERT
# from .REPromptRoberta import REPromptRoberta
from .PromptRoberta import PromptRoberta
# from .projectPromptRoberta import projectPromptRoberta
# from .projectPromptRoberta_prompt import projectPromptRoberta_prompt
# from .SQuADPromptRoberta import SQuADPromptRoberta
from .PromptBert import PromptBert
from .PromptT5 import PromptT5
from .crossPrompt import crossPrompt
from .crossPromptLocalmap import crossPrompt as crossPromptLocalmap
# from .mlmPrompt import mlmPrompt
from .crossPromptT5 import crossPromptT5

model_list = {
    # "BinaryClsBERT": BinaryClsBERT,
    # "RTEPrompt": PromptClsBERT,
    # "SST2Prompt": PromptClsBERT,
    # "RE": REBERT,
    # "REPrompt": REPromptRoberta,
    # "projectPromptRoberta": projectPromptRoberta,
    "PromptRoberta": PromptRoberta,
    # "projectPromptRoberta_prompt": projectPromptRoberta_prompt,
    "crossPrompt": crossPrompt,
    # "SQuADPromptRoberta": SQuADPromptRoberta
    "PromptBert": PromptBert,
    # "mlmPrompt": mlmPrompt,
    "PromptT5": PromptT5,
    "crossPromptT5": crossPromptT5,
    "crossPromptLocalmap" : crossPromptLocalmap
}

def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        print("There is no model called <{}> in model_list.".format(model_name))
        raise NotImplementedError

