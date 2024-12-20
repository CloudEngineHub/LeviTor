import math
from typing import Optional, Callable

import xformers
from models_diffusers.attention_processor import Attention


def classify_blocks(block_list, name):
    is_correct_block = False
    for block in block_list:
        if block in name:
            is_correct_block = True
            break
    return is_correct_block


def get_self_attn_feat(unet, tta_config=None):

    key_dict = dict()
    hidden_state_dict = dict()
    # query_dict = dict()
    # value_dict = dict()

    w_divide_h = 512 / 320.
    for name, module in unet.named_modules():

        module_name = type(module).__name__
        # if module_name == "Attention" and 'attn1' in name and classify_blocks(injection_config.blocks, name=name):
        # print(module_name)
        # if module_name == "Attention" and 'attn1' in name:
        if (module_name == "Attention") and ('temporal_transformer_blocks' not in name) and ('attn1' in name) and classify_blocks(tta_config.blocks, name=name):
            # print(name, module_name)
            # print(module.processor.hidden_state.shape)
            # print(module.processor.query.shape)
            # print(module.processor.key.shape)
            # print(module.processor.value.shape)

            res_h = int(math.sqrt(module.processor.hidden_state.shape[1] / w_divide_h))
            res_w = int(res_h * w_divide_h)
            bs = module.processor.hidden_state.shape[0]
            hidden_state_dict[name] = module.processor.hidden_state.permute(0, 2, 1).reshape(bs, -1, res_h, res_w)

            res_h = int(math.sqrt(module.processor.key.shape[1] / w_divide_h))
            res_w = int(res_h * w_divide_h)
            # query_dict[name] = module.processor.query.permute(0, 2, 1).reshape(bs, -1, res_h, res_w)
            key_dict[name] = module.processor.key.permute(0, 2, 1).reshape(bs, -1, res_h, res_w)
            # value_dict[name] = module.processor.value.permute(0, 2, 1).reshape(bs, -1, res_h, res_w)

    # return hidden_state_dict, query_dict, key_dict, value_dict
    return hidden_state_dict, key_dict
