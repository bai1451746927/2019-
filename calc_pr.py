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
This module to calculate precision, recall and f1-value 
of the predicated results.
"""
import sys
import json
import os
import zipfile
import traceback
import argparse
import ConfigParser


SUCCESS = 0
FILE_ERROR = 1
ENCODING_ERROR = 2
JSON_ERROR = 3
SCHEMA_ERROR = 4
TEXT_ERROR = 5
CODE_INFO = ['success', 'file_reading_error', 'encoding_error', 'json_parse_error', \
        'schema_error', 'input_text_not_in_dataset']


def del_bookname(entity_name):
    """删除书名号"""
    if entity_name.startswith(u'《') and entity_name.endswith(u'》'):
        entity_name = entity_name[1:-1]
    return entity_name


def load_predict_result(predict_filename):
    """加载预测值"""
    predict_result = {}#建立字典
    ret_code = SUCCESS
    try:
        predict_file_zip = zipfile.ZipFile(predict_filename)#解压文件
    except:
        ret_code = FILE_ERROR
        return predict_result, ret_code
    for predict_file in predict_file_zip.namelist():#从压缩包中读文件
        for line in predict_file_zip.open(predict_file):#从文件中读取内容
            try:
                line = line.decode('utf8').strip()#定义为utf-8并且去除空格
            except:
                ret_code = ENCODING_ERROR
                return predict_result, ret_code
            try:
                json_info = json.loads(line)#加载json文件
            except:
                ret_code = JSON_ERROR
                return predict_result, ret_code
            if 'text' not in json_info or 'spo_list' not in json_info:#如果text或spo_list没在行里面，报错信息
                ret_code = SCHEMA_ERROR
                return predict_result, ret_code
            sent = json_info['text']#抽取字典中text对应的文本
            spo_set = set()#定义集合
            for spo_item in json_info['spo_list']:#抽取字典中spo_list对应的文本
                if type(spo_item) is not dict or 'subject' not in spo_item \#这几个条件为数据是否正常
                        or 'predicate' not in spo_item \
                        or 'object' not in spo_item or \
                        not isinstance(spo_item['subject'], basestring) or \
                        not isinstance(spo_item['object'], basestring):
                    ret_code = SCHEMA_ERROR
                    return predict_result, ret_code
                s = del_bookname(spo_item['subject'].lower())#把主语提取出来
                o = del_bookname(spo_item['object'].lower())#把宾语提取出来
                spo_set.add((s, spo_item['predicate'], o))#把主谓宾加入到集合里面
            predict_result[sent] = spo_set#文本对应着主谓宾的字典
    return predict_result, ret_code


def load_test_dataset(golden_filename):
    """加载黄金档案T.T（其实是测试集）"""
    golden_dict = {}#建立字典
    ret_code = SUCCESS
    with open(golden_filename) as gf:
        for line in gf:
            try:
                line = line.decode('utf8').strip()#去除空格
            except:
                ret_code = ENCODING_ERROR
                return golden_dict, ret_code
            try:
                json_info = json.loads(line)#加载json文件
            except:
                ret_code = JSON_ERROR
                return golden_dict, ret_code
            try:
                sent = json_info['text']#行里面text的内容赋值给sent
                spo_list = json_info['spo_list']#spo_list的内容赋值给spo_list
            except:
                ret_code = SCHEMA_ERROR
                return golden_dict, ret_code

            spo_result = []#定义列表
            for item in spo_list:
                o = del_bookname(item['object'].lower())#提取主语
                s = del_bookname(item['subject'].lower())#提取宾语
                spo_result.append((s, item['predicate'], o))#提取三元组
            spo_result = set(spo_result)#成为集合
            golden_dict[sent] = spo_result#文本对应着主谓宾的字典
    return golden_dict, ret_code


def load_dict(dict_filename):
    """加载alias字典？.？"""
    alias_dict = {}#定义字典
    ret_code = SUCCESS
    if dict_filename == "":#如果为空，返回{}
        return alias_dict, ret_code
    try:
        with open(dict_filename) as af:
            for line in af:
                line = line.decode().strip()#去除空格
                words = line.split('\t')#去除换行符
                alias_dict[words[0].lower()] = set()#第0个元素对应一个集合
                for alias_word in words[1:]:#把后面元素加入集合里面
                    alias_dict[words[0].lower()].add(alias_word.lower())
    except:
        ret_code = FILE_ERROR
    return alias_dict, ret_code


def is_spo_correct(spo, golden_spo_set, alias_dict, loc_dict):
    """如果三元组是正确的"""
    if spo in golden_spo_set:
        return True
    (s, p, o) = spo
    #alias 字典
    s_alias_set = alias_dict.get(s, set())#get（）返回指定键的值，如果值不在字典中返回默认值None。
    s_alias_set.add(s)#加入（存在可能主宾相反的问题，所以这样写）
    o_alias_set = alias_dict.get(o, set())#get（）返回指定键的值，如果值不在字典中返回默认值None。
    o_alias_set.add(o)#加入
    for s_a in s_alias_set:
        for o_a in o_alias_set:
            if (s_a, p, o_a) in golden_spo_set:#如果加入的元素有和golden字典相对应
                return True
    for golden_spo in golden_spo_set:
        (golden_s, golden_p, golden_o) = golden_spo
        golden_o_set = loc_dict.get(golden_o, set())#在本地字典中查找
        for g_o in golden_o_set:
            if s == golden_s and p == golden_p and o == g_o:
                return True
    return False


def calc_pr(predict_filename, alias_filename, location_filename, \
        golden_filename):
    """计算准确率，召回率, f1值"""
    ret_info = {}
    #加载本地字典
    loc_dict, ret_code = load_dict(location_filename)"""加载alias字典？.？"""
    if ret_code != SUCCESS:
        ret_info['errorCode'] = ret_code
        ret_info['errorMsg'] = CODE_INFO[ret_code]
        print >> sys.stderr, 'loc file is error'
        return ret_info

    #加载alias字典
    alias_dict, ret_code = load_dict(alias_filename)
    if ret_code != SUCCESS:
        ret_info['errorCode'] = ret_code
        ret_info['errorMsg'] = CODE_INFO[ret_code]
        print >> sys.stderr, 'alias file is error'
        return ret_info
    #加载测试数据集
    golden_dict, ret_code = load_test_dataset(golden_filename)
    if ret_code != SUCCESS:
        ret_info['errorCode'] = ret_code
        ret_info['errorMsg'] = CODE_INFO[ret_code]
        print >> sys.stderr, 'golden file is error'
        return ret_info
    #加载预测结果
    predict_result, ret_code = load_predict_result(predict_filename)
    if ret_code != SUCCESS:
        ret_info['errorCode'] = ret_code
        ret_info['errorMsg'] = CODE_INFO[ret_code]
        print >> sys.stderr, 'predict file is error'
        return ret_info
    
    #评估（下面代码自己理解）
    correct_sum, predict_sum, recall_sum = 0.0, 0.0, 0.0
    for sent in golden_dict:
        golden_spo_set = golden_dict[sent]
        predict_spo_set = predict_result.get(sent, set())
        
        recall_sum += len(golden_spo_set)
        predict_sum += len(predict_spo_set)
        for spo in predict_spo_set:
            if is_spo_correct(spo, golden_spo_set, alias_dict, loc_dict):
                correct_sum += 1
    print >> sys.stderr, 'correct spo num = ', correct_sum
    print >> sys.stderr, 'submitted spo num = ', predict_sum
    print >> sys.stderr, 'golden set spo num = ', recall_sum
    precision = correct_sum / predict_sum if predict_sum > 0 else 0.0
    recall = correct_sum / recall_sum if recall_sum > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) \
            if precision + recall > 0 else 0.0
    precision = round(precision, 4)
    recall = round(recall, 4)
    f1 = round(f1, 4)
    ret_info['errorCode'] = SUCCESS
    ret_info['errorMsg'] = CODE_INFO[SUCCESS]
    ret_info['data'] = []
    ret_info['data'].append({'name': 'precision', 'value': precision})
    ret_info['data'].append({'name': 'recall', 'value': recall})
    ret_info['data'].append({'name': 'f1-score', 'value': f1})
    return ret_info       


if __name__ == '__main__':
    reload(sys)#这是py2的写法
    sys.setdefaultencoding('utf-8')#这是py2的写法
    parser = argparse.ArgumentParser()#功能是把你的输入参数打印到屏幕 
    parser.add_argument("--golden_file", type=str,
        help="true spo results", required=True)#add_argument()方法，用来指定程序需要接受的命令参数
    parser.add_argument("--predict_file", type=str,
        help="spo results predicted", required=True)
    parser.add_argument("--loc_file", type=str,
        default='', help="location entities of various granularity")
    parser.add_argument("--alias_file", type=str,
        default='', help="entities alias dictionary")
    args = parser.parse_args()#这里没搞懂怎么输入的值
    golden_filename = args.golden_file
    predict_filename = args.predict_file
    location_filename = args.loc_file
    alias_filename = args.alias_file
    ret_info = calc_pr(predict_filename, alias_filename, location_filename, \
            golden_filename)
    print json.dumps(ret_info)
