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
这个模块定义了一个类，该类是p分类的数据阅读器
"""

import json
import os
import codecs
import sys


class RcDataReader(object):
    """
    该类是p分类的数据阅读器
    """
    def __init__(self,
                wordemb_dict_path,
                postag_dict_path,
                label_dict_path,
                train_data_list_path='',
                test_data_list_path=''):
        self._wordemb_dict_path = wordemb_dict_path
        self._postag_dict_path = postag_dict_path
        self._label_dict_path = label_dict_path
        self.train_data_list_path = train_data_list_path
        self.test_data_list_path = test_data_list_path
        self._p_map_eng_dict = {}#定义字典
        # 加载字典
        self._dict_path_dict = {'wordemb_dict': self._wordemb_dict_path,
                                'postag_dict': self._postag_dict_path,
                                'label_dict': self._label_dict_path}
        #检查文件是否存在
        for input_dict in [wordemb_dict_path, postag_dict_path, \
                label_dict_path, train_data_list_path, test_data_list_path]:
            if not os.path.exists(input_dict):
                raise ValueError("%s not found." % (input_dict))
                return

        self._feature_dict = {}
        self._feature_dict['postag_dict'] = \
                self._load_dict_from_file(self._dict_path_dict['postag_dict'])#情感标签文件导入
        self._feature_dict['wordemb_dict'] = \
                self._load_dict_from_file(self._dict_path_dict['wordemb_dict'])#词嵌入文件导入
        self._feature_dict['label_dict'] = \
                self._load_label_dict(self._dict_path_dict['label_dict'])#标签文件导入
        self._reverse_dict = {name: self._get_reverse_dict(name) for name in
                              self._dict_path_dict.keys()}#对字典内容和key进行互换
        self._reverse_dict['eng_map_p_dict'] = self._reverse_p_eng(self._p_map_eng_dict)#对里面的内容和key进行转置
        self._UNK_IDX = 0

    def _load_label_dict(self, dict_name):
        """从文件中加载标签字典"""
        label_dict = {}
        with codecs.open(dict_name, 'r', 'utf-8') as fr:
            for idx, line in enumerate(fr):
                p, p_eng = line.strip().split('\t')
                label_dict[p_eng] = idx
                self._p_map_eng_dict[p] = p_eng
        return label_dict

    def _load_dict_from_file(self, dict_name, bias=0):
        """
        从文件中加载词汇表
        """
        dict_result = {}
        with codecs.open(dict_name, 'r', 'utf-8') as f_dict:
            for idx, line in enumerate(f_dict):
                line = line.strip()
                dict_result[line] = idx + bias
        return dict_result

    def _cal_mark_slot(self, spo_list, sentence):
        """
        计算标签的value
        """
        mark_list = [0] * len(self._feature_dict['label_dict'])#生成一个标签文件长度的0列表（0*50）
        for spo in spo_list:
            predicate = spo['predicate']
            p_idx = self._feature_dict['label_dict'][self._p_map_eng_dict[predicate]]#从字典中查找预测的结果对应的值
            mark_list[p_idx] = 1#对该值进行计数
        return mark_list

    def _is_valid_input_data(self, input_data):
        """输入数据是否有效"""
        try:
            dic = json.loads(input_data)
        except:
            return False
        if "text" not in dic or "postag" not in dic or \
                type(dic["postag"]) is not list:
            return False
        for item in dic['postag']:
            if "word" not in item or "pos" not in item:
                return False
        return True
    
    def _get_feed_iterator(self, line, need_input=False, need_label=True):
        # 验证每行的输入格式是否符合格式
        if not self._is_valid_input_data(line):
            print >> sys.stderr, 'Format is error'
            return None
        dic = json.loads(line)
        sentence = dic['text']
        sentence_term_list = [item['word'] for item in dic['postag']]
        sentence_pos_list = [item['pos'] for item in dic['postag']]
        sentence_emb_slot = [self._feature_dict['wordemb_dict'].get(w, self._UNK_IDX) \
                for w in sentence_term_list]
        sentence_pos_slot = [self._feature_dict['postag_dict'].get(pos, self._UNK_IDX) \
                for pos in sentence_pos_list]
        if 'spo_list' not in dic:
            label_slot = [0] * len(self._feature_dict['label_dict'])
        else:
            label_slot = self._cal_mark_slot(dic['spo_list'], sentence)
        # 验证特征是否有效
        if len(sentence_emb_slot) == 0 or len(sentence_pos_slot) == 0 \
                or len(label_slot) == 0:
            return None
        feature_slot = [sentence_emb_slot, sentence_pos_slot]
        input_fields = json.dumps(dic, ensure_ascii=False)
        output_slot = feature_slot
        if need_input:
            output_slot = [input_fields] + output_slot
        if need_label:
            output_slot = output_slot + [label_slot]
        return output_slot

    def path_reader(self, data_path, need_input=False, need_label=True):
        """从数据路径读取数据"""
        self._feature_dict['data_keylist'] = []#建立列表

        def reader():
            """生成器"""
            if os.path.isdir(data_path):
                input_files = os.listdir(data_path)#导入文件路径
                for data_file in input_files:
                    data_file_path = os.path.join(data_path, data_file)#成为文件
                    for line in open(data_file_path.strip()):
                        sample_result = self._get_feed_iterator(line.strip(), need_input, need_label)
                        if sample_result is None:
                            continue
                        yield tuple(sample_result)
            elif os.path.isfile(data_path):
                for line in open(data_path.strip()):
                    sample_result = self._get_feed_iterator(line.strip(), need_input, need_label)# 验证每行的输入格式是否符合格式
                    if sample_result is None:
                        continue
                    yield tuple(sample_result)

        return reader

    def get_train_reader(self, need_input=False, need_label=True):
        """训练期间的数据阅读器"""
        return self.path_reader(self.train_data_list_path, need_input, need_label)

    def get_test_reader(self, need_input=True, need_label=True):
        """测试期间的数据阅读器"""
        return self.path_reader(self.test_data_list_path, need_input, need_label)
    
    def get_predict_reader(self, predict_file_path='', need_input=True, need_label=False):
        """预测期间的数据阅读器"""
        return self.path_reader(predict_file_path, need_input, need_label)

    def get_dict(self, dict_name):
        """返回字典"""
        if dict_name not in self._feature_dict:
            raise ValueError("dict name %s not found." % (dict_name))#没有输入的key，报错
        return self._feature_dict[dict_name]#输出value

    def get_all_dict_name(self):
        """获取所有dict的名称"""
        return self._feature_dict.keys()

    def get_dict_size(self, dict_name):
        """返回dict长度"""
        if dict_name not in self._feature_dict:
            raise ValueError("dict name %s not found." % (dict_name))
        return len(self._feature_dict[dict_name])

    def _get_reverse_dict(self, dict_name):#反转key和value
        dict_reverse = {}
        for key, value in self._feature_dict[dict_name].iteritems():
            dict_reverse[value] = key
        return dict_reverse
    
    def _reverse_p_eng(self, dic):#反转key和value
        dict_reverse = {}
        for key, value in dic.iteritems():
            dict_reverse[value] = key
        return dict_reverse

    def get_label_output(self, label_idx):
        """输出最终标签，在预测和测试期间使用"""
        dict_name = 'label_dict'#暂时没找到相关，对此存疑
        if len(self._reverse_dict[dict_name]) == 0:
            self._get_reverse_dict(dict_name)
        p_eng = self._reverse_dict[dict_name][label_idx]
        return self._reverse_dict['eng_map_p_dict'][p_eng]


if __name__ == '__main__':
    # 初始化数据生成器
    data_generator = RcDataReader(
        wordemb_dict_path='./dict/word_idx',
        postag_dict_path='./dict/postag_dict',
        label_dict_path='./dict/p_eng',
        train_data_list_path='./data/train_data.json',
        test_data_list_path='./data/dev_data.json')#前三个没找到

    #准备数据读取器
    ttt = data_generator.get_test_reader()
    for index, features in enumerate(ttt()):
        input_sent, word_idx_list, postag_list, label_list = features
        print(input_sent.encode('utf-8'))
        print('第一个特征:', len(word_idx_list), word_idx_list)
        print('第二个特征:', len(postag_list), postag_list)
        print('第三个特征:', len(label_list), '\t', label_list)
