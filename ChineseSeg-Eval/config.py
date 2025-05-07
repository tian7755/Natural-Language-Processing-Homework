# 路径配置
# -*- coding: utf-8 -*-
import os

# 自动获取当前路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 语料路径配置
CORPUS_PATHS = {
    "news_a": os.path.join(BASE_DIR, "data/news_a.txt"),
    "news_o": os.path.join(BASE_DIR, "data/news_o.txt"),
    "dianshang_reviews": os.path.join(BASE_DIR, "data/dianshang_reviews.txt"),
    "bilbil_reviews": os.path.join(BASE_DIR, "data/bilbil_reviews.txt")
}

# 自定义词典路径
CUSTOM_DICT = os.path.join(BASE_DIR, "custom_dict.txt")
