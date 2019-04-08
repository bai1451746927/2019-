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
这个模块用P分类模型来推断T.T
"""

import json
import os
import sys
import argparse
import ConfigParser
import math

import numpy as np
import paddle
import paddle.fluid as fluid#类似于pytorch和tensorflow

import p_data_reader#上一个文件

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../lib")))
import conf_lib


def predict_infer(conf_dict, data_reader, predict_data_path, \
        predict_result_path, model_path):
    """
    用训练模型预测
    """
    if len(predict_result_path) > 0:
        result_writer = open(predict_result_path, 'w')
    else:
        result_writer = sys.stdout#等于print "%VALUE%"

    np.set_printoptions(precision=3)#设置打印时显示方式，这里设置准确率为三位
    if len(model_path) == 0:
        return

    place = fluid.CPUPlace()
    word = fluid.layers.data(
        name='word_data', shape=[1], dtype='int64', lod_level=1)#加入worddata
    postag = fluid.layers.data(
        name='token_pos', shape=[1], dtype='int64', lod_level=1)#加入pos符号
    feeder = fluid.DataFeeder(feed_list=[word, postag], place=place)
    exe = fluid.Executor(place)#运行器

    test_batch_reader = paddle.batch(
        paddle.reader.buffered(data_reader.get_predict_reader\
                (predict_data_path, need_input=True, need_label=False),
                size=8192),
        batch_size=conf_dict["batch_size"])#对此行存疑
    inference_scope = fluid.core.Scope()#对此行存疑，应该是设定一个领域的意思
    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names, fetch_targets] = \
            fluid.io.load_inference_model(
                model_path, exe, params_filename='params')#param：参数

        # 批处理
        batch_id = 0
        for data in test_batch_reader():
            feeder_data = []#反馈数据
            input_data = []#输入数据
            for item in data:
                input_dic = json.loads(item[0])#加载item0维
                input_data.append(input_dic)#加入输入的dic
                feeder_data.append(item[1:])#加入item的后面
            results = exe.run(inference_program, feed=feeder.feed(feeder_data),
                              fetch_list=fetch_targets, return_numpy=False)
            label_scores = np.array(results[0]).tolist()#把结果的0维捞出转成列表
            #从一次批处理中推导
            infer_a_batch(label_scores, input_data, result_writer, data_reader)
            
            batch_id += 1


def infer_a_batch(label_scores, input_data, result_writer, data_reader):
    """推导出一次批处理的结果"""
    for sent_idx, label in enumerate(label_scores):#第一个里面是数，第二个里面是内容
        p_label = []#定义个列表
        label = map(float, label)#将列表里面的全部变为float形式
        for p_idx, p_score in enumerate(label):
            if sigmoid(p_score) > 0.5:#大于0.5便加入到列表中
                p_label.append(data_reader.get_label_output(p_idx))
        for p in p_label:
            output_fields = [json.dumps(input_data[sent_idx], ensure_ascii=False), p]#加入进去
            result_writer.write('\t'.join(output_fields).encode('utf-8'))
            result_writer.write('\n')


def sigmoid(x):
    """sigmode 函数"""
    return math.exp(x) / (1 + math.exp(x))


def main(conf_dict, model_path, predict_data_path, 
            predict_result_path, use_cuda=False):
    """预测主函数"""
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    data_generator = p_data_reader.RcDataReader(
        wordemb_dict_path=conf_dict['word_idx_path'],
        postag_dict_path=conf_dict['postag_dict_path'],
        label_dict_path=conf_dict['label_dict_path'],
        train_data_list_path=conf_dict['train_data_path'],
        test_data_list_path=conf_dict['test_data_path'])
    
    predict_infer(conf_dict, data_generator, predict_data_path, \
            predict_result_path, model_path)


if __name__ == '__main__':
    # 

Load configuration file
#加载配置文件
    parser = argparse.ArgumentParser()#打印
    parser.add_argument("--conf_path", type=str,
            help="conf_file_path_for_model. (default: %(default)s)",
            required=True)
    parser.add_argument("--model_path", type=str,
            help="model_path", required=True)
    parser.add_argument("--predict_file", type=str,
            help="the_file_to_be_predicted", required=True)
    parser.add_argument("--result_file", type=str,
            default='', help="the_file_of_predicted_results")
    args = parser.parse_args()
    conf_dict = conf_lib.load_conf(args.conf_path)#加载运行（？）路径
    model_path = args.model_path
    predict_data_path = args.predict_file
    predict_result_path = args.result_file
    for input_path in [model_path, predict_data_path]:
        if not os.path.exists(input_path):
            raise ValueError("%s not found." % (input_path))
    main(conf_dict, model_path, predict_data_path, predict_result_path)#运行主函数
