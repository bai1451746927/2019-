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
此模块用于生成训练数据用于训练一个so-labeling模型
"""

import random
import codecs
import sys
import json
reload(sys)
sys.setdefaultencoding('utf-8')

def get_p(input_file):
    """
    为so labeling模型生成训练数据
    """
    with codecs.open(input_file, 'r', 'utf-8') as fr:
        for line in fr:
            try:
                dic = json.loads(line.decode('utf-8').strip())
            except:
                continue
            spo_list = dic['spo_list']
            p_list = [item['predicate'] for item in spo_list]
            for p in p_list:
                print "\t".join([json.dumps(dic, ensure_ascii=False), p]).encode('utf-8')#得到所有的预测值


if __name__ == '__main__':
    input_file = sys.argv[1]
    get_p(input_file)
