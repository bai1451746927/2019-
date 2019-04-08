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
此模块用于训练关系分类模型
"""

import json
import os
import sys
import time
import argparse
import ConfigParser

import paddle
import paddle.fluid as fluid
import six

import p_data_reader
import p_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../lib")))
import conf_lib


def train(conf_dict, data_reader, use_cuda=False):
    """
    P分类模型训练

    """
    label_dict_len = data_reader.get_dict_size('label_dict')
    # 输入层
    word = fluid.layers.data(
        name='word_data', shape=[1], dtype='int64', lod_level=1)
    postag = fluid.layers.data(
        name='token_pos', shape=[1], dtype='int64', lod_level=1)
    # 标签
    target = fluid.layers.data(
        name='target', shape=[label_dict_len], dtype='float32', lod_level=0)
    # NN:词嵌入+ lstm + 池化
    feature_out = p_model.db_lstm(data_reader, word, postag, conf_dict)
    # 多标签分类的损失函数
    class_cost = fluid.layers.sigmoid_cross_entropy_with_logits(x=feature_out, \
        label=target)#sigmoid逻辑交叉熵损失
    avg_cost = fluid.layers.mean(class_cost)#平均交叉熵损失
    #优化方法（sgd，adam）
    sgd_optimizer = fluid.optimizer.AdamOptimizer(
        learning_rate=2e-3, )

    sgd_optimizer.minimize(avg_cost)

    train_batch_reader = paddle.batch(
        paddle.reader.shuffle(data_reader.get_train_reader(), buf_size=8192),
        batch_size=conf_dict['batch_size'])

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()#c/gpu
    feeder = fluid.DataFeeder(feed_list=[word, postag, target], place=place)#数据反馈
    exe = fluid.Executor(place)

    save_dirname = conf_dict['p_model_save_dir']

    def train_loop(main_program, trainer_id=0):
        """开始训练"""
        exe.run(fluid.default_startup_program())#开始

        start_time = time.time()#计时
        batch_id = 0#统计开始
        for pass_id in six.moves.xrange(conf_dict['pass_num']):#xrange返回的是一个个数据
            pass_start_time = time.time()#计时
            cost_sum, cost_counter = 0, 0
            for data in train_batch_reader():
                cost = exe.run(main_program, feed=feeder.feed(data), fetch_list=[avg_cost])#损失值
                cost = cost[0]
                cost_sum += cost#损失值结果统计
                cost_counter += 1#损失值次数统计
                if batch_id % 10 == 0 and batch_id != 0:
                    print >> sys.stderr, "batch %d finished, second per batch: %02f" % (
                        batch_id, (time.time() - start_time) / batch_id)

                # 根据损失值大小决定要不要结束训练
                if float(cost) < 0.01:
                    pass_avg_cost = cost_sum / cost_counter if cost_counter > 0 else 0.0
                    print >> sys.stderr, "%d pass end, cost time: %02f, avg_cost: %f" % (
                        pass_id, time.time() - pass_start_time, pass_avg_cost)
                    save_path = os.path.join(save_dirname, 'final')
                    fluid.io.save_inference_model(save_path, ['word_data', 'token_pos'],
                                                  [feature_out], exe, params_filename='params')
                    return
                batch_id = batch_id + 1

            # 每次传递结束后保存模型
            pass_avg_cost = cost_sum / cost_counter if cost_counter > 0 else 0.0
            print >> sys.stderr, "%d pass end, cost time: %02f, avg_cost: %f" % (
                pass_id, time.time() - pass_start_time, pass_avg_cost)
            save_path = os.path.join(save_dirname, 'pass_%04d-%f' %
                                    (pass_id, pass_avg_cost))
            fluid.io.save_inference_model(save_path, ['word_data', 'token_pos'],
                                          [feature_out], exe, params_filename='params')

        else:
            # 通过时间结束，训练结束，保存模型
            save_path = os.path.join(save_dirname, 'final')
            fluid.io.save_inference_model(save_path, ['word_data', 'token_pos'],
                                          [feature_out], exe, params_filename='params')
        return

    train_loop(fluid.default_main_program())


def main(conf_dict, use_cuda=False):
    """训练主函数"""
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        print >> sys.stderr, 'No GPU'
        return
    data_generator = p_data_reader.RcDataReader(
        wordemb_dict_path=conf_dict['word_idx_path'],
        postag_dict_path=conf_dict['postag_dict_path'],
        label_dict_path=conf_dict['label_dict_path'],
        train_data_list_path=conf_dict['train_data_path'],
        test_data_list_path=conf_dict['test_data_path'])
    
    train(conf_dict, data_generator, use_cuda=use_cuda)


if __name__ == '__main__':
    # 加载配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_path", type=str,
        help="conf_file_path_for_model. (default: %(default)s)",
        required=True)
    args = parser.parse_args()
    conf_dict = conf_lib.load_conf(args.conf_path)
    use_gpu = True if conf_dict.get('use_gpu', 'False') == 'True' else False
    main(conf_dict, use_cuda=use_gpu)
