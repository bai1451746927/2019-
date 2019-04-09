# -*- coding: utf-8 -*-
########################################################
# Copyright (c) 2019, Baidu Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# imitations under the License.
########################################################
"""
此模块用于读取配置文件
"""

import json
import os
import sys
import argparse
import ConfigParser


def load_conf(conf_filename):
    """
    加载配置文件
    :参数字符串conf_文件名：conf文件
    :rtype dict param_conf_dict: conf_dict
    """
    param_conf_dict={}#建立字典
    cf = ConfigParser.ConfigParser()
    cf.read(conf_filename)
    int_conf_keys = {
                'model_params': ["cost_threshold", "mark_dict_len", "word_dim",
                    "mark_dim", "postag_dim", "hidden_dim", "depth",
                    "pass_num", "batch_size", "class_dim"]
                }#读取的是字典，用于模型，为数字
    for session_key in int_conf_keys:
        for option_key in int_conf_keys[session_key]:#每个字典进行读取
            try:
                option_value = cf.get(session_key, option_key)
                param_conf_dict[option_key] = int(option_value)#成为字典
            except:
                raise ValueError("%s--%s is not a integer" % (session_key, option_key))
    str_conf_keys = {
                'model_params': ['is_sparse', "use_gpu", "emb_name",
                    "is_local", "word_emb_fixed", "mix_hidden_lr"],
                'p_model_dir': ["test_data_path", "train_data_path",
                    "p_model_save_dir"],
                'spo_model_dir': ['spo_test_data_path', 'spo_train_data_path', 
                    'spo_model_save_dir'],
                'dict_path': ["so_label_dict_path", "label_dict_path",
                    "postag_dict_path", "word_idx_path"]
                }#读取的是字典，用于模型，为字符串

    for session_key in str_conf_keys:
        for option_key in str_conf_keys[session_key]:
            try:
                param_conf_dict[option_key] = cf.get(session_key, option_key)
            except:
                raise ValueError("%s no such option %s" % (session_key, option_key))
    return param_conf_dict
