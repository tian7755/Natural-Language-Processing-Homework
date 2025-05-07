# 评估逻辑
# -*- coding: utf-8 -*-
import time
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def timing(func):
    """改进版计时装饰器，记录耗时和文本长度"""

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        # 获取文本长度（args[1]是text参数）
        text_length = len(args[1]) if len(args) > 1 else 0

        # 存储结果
        wrapper.last_time = end - start
        wrapper.last_speed = text_length / wrapper.last_time / 10000 \
            if wrapper.last_time > 0 else 0  # 万字/秒

        print(
            f"[{func.__name__}] 耗时: {wrapper.last_time:.2f}s | "
            f"速度: {wrapper.last_speed:.2f}万字/秒")
        return result

    # 为 wrapper.last_time 和 wrapper.last_speed 初始化默认值为 0
    # 确保在未执行函数前这些属性存在
    wrapper.last_time = 0
    wrapper.last_speed = 0
    return wrapper


def evaluate_segmentation(pred, gold):
    """综合评估分词结果"""
    # 词语级别的评估
    word_precision, word_recall, word_f_score, word_accuracy = (
        calculate_metrics(pred, gold))

    # 字级别的评
    char_precision, char_recall, char_f_score, char_accuracy = (
        calculate_word_level_metrics(pred, gold))

    # 粒度差异评估
    granularity_diff = calculate_granularity_metrics(pred, gold)

    # 计算分词结果和标注数据的长度差异
    length_diff = abs(len(pred) - len(gold))

    return {
        "word_level": {
            "precision": word_precision,
            "recall": word_recall,
            "f_score": word_f_score,
            "accuracy": word_accuracy  # 新增准确率
        },
        "char_level": {
            "precision": char_precision,
            "recall": char_recall,
            "f_score": char_f_score,
            "accuracy": char_accuracy  # 新增准确率
        },
        "granularity_difference": granularity_diff,
        "length_difference": length_diff
    }


def calculate_metrics(pred, gold):
    """计算分词的精确率、召回率和F分数，考虑部分匹配和粒度差异"""
    # 用于统计正确切分的词语数，包括部分匹配的情况
    correct_pred = 0

    # 遍历预测的词语列表，检查每个词语是否在标注列表中
    for word in pred:
        if word in gold:
            correct_pred += 1
        else:
            # 检查是否存在部分匹配
            for gold_word in gold:
                if word in gold_word:
                    correct_pred += len(word) / len(gold_word)
                    break

    precision = correct_pred / len(pred) if len(pred) > 0 else 0
    recall = correct_pred / len(gold) if len(gold) > 0 else 0
    f_score = 2 * (precision * recall) / (precision + recall) if (
            (precision + recall) > 0) else 0
    # 正确切分的词语数（完全匹配）
    exact_correct_pred = sum(1 for word in pred if word in gold)
    accuracy = exact_correct_pred / len(pred) if len(pred) > 0 else 0

    return precision, recall, f_score, accuracy


def calculate_word_level_metrics(pred, gold):
    """计算字级别的精确率、召回率和F分数"""
    # 将分词结果和标注数据转换为字列表
    pred_chars = [char for word in pred for char in word]
    gold_chars = [char for word in gold for char in word]

    # 计算正确切分的字数
    correct_chars = sum(1 for p, g in zip(pred_chars, gold_chars) if p == g)
    # 计算精确率、召回率和F分数
    precision = correct_chars / len(pred_chars) if len(pred_chars) > 0 else 0
    recall = correct_chars / len(gold_chars) if len(gold_chars) > 0 else 0
    f_score = 2 * (precision * recall) / (precision + recall) if (
            (precision + recall) > 0) else 0

    # 正确切分的字数与预测字数的比值
    accuracy = correct_chars / len(pred_chars) if len(pred_chars) > 0 else 0

    return precision, recall, f_score, accuracy


def calculate_granularity_metrics(pred, gold):
    """计算分词粒度相关的评估指标"""
    # 计算平均分词长度
    avg_pred_length = sum(len(word) for word in pred) / len(pred) \
        if len(pred) > 0 else 0
    avg_gold_length = sum(len(word) for word in gold) / len(gold) \
        if len(gold) > 0 else 0

    # 计算粒度差异,两者之间的绝对差值
    granularity_difference = abs(avg_pred_length - avg_gold_length)

    return granularity_difference


def evaluate_statistics(segmented_text):
    """计算统计指标"""
    word_counts = Counter(segmented_text)

    # 词表多样性 = 唯一词数 / 总词数
    vocab_size = len(word_counts)
    total_words = len(segmented_text)
    diversity = vocab_size / total_words

    # 长尾词比例（词频<=2）
    rare_words = sum(1 for cnt in word_counts.values() if cnt <= 2)
    rare_ratio = rare_words / vocab_size

    return {
        "词表多样性": round(diversity, 4),
        "长尾词比例": round(rare_ratio, 4),
        "平均词长": round(sum(len(w) for w in segmented_text) / total_words, 2)
    }


def visualize_wordcloud(segmented_text, tool_name):
    text = " ".join(segmented_text)
    wc = WordCloud(font_path="SimHei.ttf", width=800, height=400).generate(
        text)

    plt.imshow(wc)
    plt.title(f"{tool_name} 分词词云")
    plt.axis("off")
    plt.show()


def check_special_cases(segmented):
    """检查特殊案例处理"""
    errors = 0

    # 检查标点分割
    for word in segmented:
        if len(word) == 1 and word in "！？，。；":
            errors += 1

    # 检查数字英文混合
    for word in segmented:
        if any(c.isdigit() for c in word) and any(c.isalpha() for c in word):
            errors += 1

    return errors
