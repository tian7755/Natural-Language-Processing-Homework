# -*- coding: utf-8 -*-
import os
import re
import chardet
import jieba
import pkuseg
import thulac
import hanlp
from config import CORPUS_PATHS, CUSTOM_DICT
from evaluator import (timing, evaluate_segmentation, evaluate_statistics,
                       visualize_wordcloud, check_special_cases)


class SegmenterTester:
    def __init__(self):
        # 初始化分词器
        self.jieba = jieba
        # self.jieba.load_userdict(CUSTOM_DICT)  # 加载自定义词典

        self.pkuseg_news = pkuseg.pkuseg(model_name="news")  # 新闻模型
        self.pkuseg_web = pkuseg.pkuseg(model_name="web")  # 网络模型

        self.thulac = thulac.thulac(seg_only=True)  # 仅分词模式

        self.hanlp = hanlp.load('LARGE_ALBERT_BASE')  # 加载HanLP分词器

    def pretreatment(self, corpus_type):
        """将文件内容截断为前5000个字符，并覆盖原文件。"""
        try:
            # 检查并创建输出目录
            output_dir = os.path.dirname(CORPUS_PATHS[corpus_type])
            os.makedirs(output_dir, exist_ok=True)

            # 尝试自动检测编码
            with open(CORPUS_PATHS[corpus_type], 'rb') as file:
                raw_data = file.read(1024)  # 只读取一部分数据进行编码检测
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                print(f"文件编码格式: {encoding}")

            # 定义可能的编码列表，包含检测到的编码和其他常见编码
            possible_encodings = [encoding, 'utf-8', 'gbk', 'latin-1']

            # 尝试使用各种编码读取文件内容
            content = None
            for enc in possible_encodings:
                try:
                    with open(CORPUS_PATHS[corpus_type], 'r',
                              encoding=enc) as file:
                        content = file.read()
                    print(f"成功使用编码 {enc} 读取文件。")
                    break  # 成功读取后退出循环
                except UnicodeDecodeError:
                    print(f"无法使用编码 {enc} 读取文件，尝试下一个编码。")

            # 如果所有编码都尝试失败，则抛出错误
            if content is None:
                raise UnicodeDecodeError("尝试所有编码后仍无法解码文件")

            # 截取前100000个字符
            truncated_content = content[:100000]

            # 合并姓名（如 王/nrf  小明/nrg -> 王小明/nr）
            truncated_content = re.sub(
                r'(\b\w+/nrf\s+)(\w+/nrg\b)',
                lambda m: ''.join([m.group(1).split()[0].split('/')[0],
                                   m.group(2).split('/')[0], '/nr']),
                truncated_content
            )

            # 合并中括号内的内容（如 [中央/n  人民/n  广播/vn  电台/n]nt -> 中央人民广播电台/nt）
            truncated_content = re.sub(
                r'\[(.*?)\](\w+)',
                lambda m: ''.join([word.split('/')[0] for word in
                                   m.group(1).split()]) + '/' + m.group(2),
                truncated_content
            )

            # 合并时间（如 1997年/t 3月/t -> 1997年3月/t）
            truncated_content = re.sub(
                r'(\d+年/t)\s+(\d+月/t)',
                lambda m: m.group(1).replace('/t', '') + m.group(2).replace(
                    '/t', '')[2:] + '/t',
                truncated_content
            )

            # 全角转半角
            truncated_content = truncated_content.translate(
                str.maketrans('１２３４５６７８９０', '1234567890'))

            # 根据空格重新组织内容为多行
            lines = truncated_content.split()

            # 去除每行的首尾空格，并删除以/m或/w结尾的行
            processed_lines = []
            for line in lines:
                stripped_line = line.strip()
                if stripped_line and not stripped_line.endswith(
                        ('/m', '/w')):
                    processed_lines.append(stripped_line)

            # 将处理后的内容写回文件，覆盖原文件
            with open(CORPUS_PATHS[corpus_type], 'w',
                      encoding='utf-8') as file:
                file.write('\n'.join(processed_lines))

            print(f"文件已成功处理。文件路径：{CORPUS_PATHS[corpus_type]}")

        except Exception as e:
            print(f"处理文件时发生错误：{e}")

    def toWordRestore(self, corpus_type):
        """
        此函数去除语料中的分词和词性标记，形成原始文本

        参数：
            corpus_type - 语料类型
        """
        try:
            # 构建还原文件的路径
            restored_path = CORPUS_PATHS[corpus_type].replace('_a.', '_o.')

            # 确保输出目录存在
            output_dir = os.path.dirname(restored_path)
            os.makedirs(output_dir, exist_ok=True)

            # 读取标注文件
            with open(CORPUS_PATHS[corpus_type], 'r', encoding='utf-8') as f:
                content = f.read()

            # 删除标注字符和换行符
            restored_content = (
                content.replace(" ", "")
                .replace("]", "").replace("[", "")
                .replace("/", "")
                .replace("v", "").replace("n", "")
                .replace("u", "").replace("t", "")
                .replace("m", "").replace("w", "")
                .replace("a", "").replace("r", "")
                .replace("q", "").replace("p", "")
                .replace("c", "").replace("d", "")
                .replace("f", "").replace("Ng", "")
                .replace("z", "").replace("i", "")
                .replace("s", "").replace("Tg", "")
                .replace("k", "").replace("j", "")
                .replace("b", "").replace("l", "")
                .replace("Vg", "").replace("y", "")
                .replace("Dg", "").replace("Ag", "")
                .replace("Bg", "").replace("e", "")
                .replace("Rg", "").replace("h", "")
                .replace("Mg", "").replace("o", "")
                .replace("x", "")
                .replace("\n", "")
            )

            # 写入还原后的内容
            with open(restored_path, 'w', encoding='utf-8') as f:
                f.write(restored_content)

            print(f"文件已成功还原。文件路径：:{restored_path}")

        except Exception as e:
            print(f"还原文件时发生错误：{e}")

    def clean_segmentation_result(self, seg_result):
        """清理分词结果，去除标注信息和特殊符号"""
        cleaned_result = []
        skip_special_chars = False  # 用于处理成对出现的特殊符号（如括号）
        for word in seg_result:
            # 去除标注信息（如 '/m'）
            clean_word = word.split('/')[0].strip()
            # 过滤空字符串和部分特殊符号
            if not clean_word:
                continue
            # 处理特殊符号
            if clean_word in ('-', '(', ')', '（', '）', '《', '》', '“', '”'):
                skip_special_chars = not skip_special_chars  # 切换特殊符号跳过状态
                continue
            if not skip_special_chars:
                cleaned_result.append(clean_word)
        return cleaned_result

    def parse_annotated_text(self, annotated_text):
        """解析带标注的新闻语料，提取纯净词串"""
        words = []
        for line in annotated_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            # 按照空格分割每个标注词
            annotated_words = line.split()
            for annotated_word in annotated_words:
                # 按 '/' 分割词汇和标注，仅保留词汇部分
                word_part = annotated_word.split('/')[0].strip()
                # 过滤空字符串和特殊符号
                if word_part and not word_part.startswith(
                        ('-', '(', ')', '《', '》', '‘', '’', '"')):
                    words.append(word_part)
        return words

    @timing
    def run_test(self, text, tool):
        """执行分词，支持标注数据对比"""
        if tool == "jieba":
            # 默认是精确模式
            result = list(self.jieba.lcut(text, HMM=True))
        elif tool == "pkuseg_news":
            result = self.pkuseg_news.cut(text)
        elif tool == "pkuseg_web":
            result = self.pkuseg_web.cut(text)
        elif tool == "thulac":
            result = self.thulac.cut(text, text=True).split()
        elif tool == "hanlp":
            result = self.hanlp(text)
            result = [word[0] for word in result]
        else:
            raise ValueError("未知工具")

        # 清理分词结果
        cleaned_result = self.clean_segmentation_result(result)

        return cleaned_result

    def test_pretreatment_and_restore(self, corpus_type):
        """测试预处理和还原功能"""
        self.pretreatment(corpus_type)
        self.toWordRestore(corpus_type)

    def test_all(self, corpus_type1, corpus_type2):
        """全流程测试"""
        # 读取原始文本（_o文件）
        try:
            with open(CORPUS_PATHS[corpus_type2], "r", encoding="utf-8") as f:
                text = f.read()[:90000]
        except UnicodeDecodeError:
            # 尝试自动检测编码
            with open(CORPUS_PATHS[corpus_type2], 'rb') as file:
                raw_data = file.read()[:90000]
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                print(f"文件编码格式: {encoding}")
            # 尝试按检测到的编码读取文件
            try:
                with open(CORPUS_PATHS[corpus_type2], "r",
                          encoding=encoding) as f:
                    text = f.read()[:90000]
            except UnicodeDecodeError:
                # 如果仍然失败，使用 latin-1 编码（通用解码方式）
                with open(CORPUS_PATHS[corpus_type2], "r",
                          encoding="latin-1") as f:
                    text = f.read()[:90000]

        # 打印调试信息
        print("\n=== 原始文本信息 ===")
        print(f"原始文本类型: {corpus_type2}")
        print(f"原始文本前20字符: {text[:20]}")
        text_length = len(text)
        print(
            f"实际处理文本长度: {text_length}字 ({text_length / 10000:.2f}万字)")

        # 对比工具列表
        tools = ["jieba", "pkuseg_news", "thulac", "hanlp"]

        # 执行测试
        results = {}
        for tool in tools:
            seg_result = self.run_test(text, tool)
            assert len(text) == text_length, "文本长度在分词过程中被修改！"
            # 使用带标注的文本（_a文件）计算性能指标
            with open(CORPUS_PATHS[corpus_type1], "r", encoding="utf-8") as f:
                annotated_text = f.read()
            true_words = self.parse_annotated_text(annotated_text)
            # 计算综合评估指标
            metrics = evaluate_segmentation(seg_result, true_words)

            # 打印调试信息
            print(f"\n工具: {tool}")
            print(f"分词结果前25个词: {seg_result[:25]}")
            print(f"标注数据前25个词: {true_words[:25]}")
            print(f"分词结果长度: {len(seg_result)}")
            print(f"标注数据长度: {len(true_words)}")

            results[tool] = {
                "time": self.run_test.last_time,
                "result": seg_result,
                "metrics": metrics
            }

        # 打印结果
        print("\n=== 分词性能对比 ===")
        print(
            f"{'工具':<11} | {'速度(万字/秒)':<9} | {'词精':<5} | {'词召':<5} | "
            f"{'词F':<5} | {'词准':<5} | {'字精':<5} | {'字召':<5} | {'字F':<5} | "
            f"{'字准':<5} | {'粒度差异':<5} | {'长度差异':<10}"
        )
        for tool, data in results.items():
            metrics = data["metrics"]
            print(
                f"{tool:<12} | {self.run_test.last_speed:<12.6f} |"
                f" {metrics['word_level']['precision']:.4f} "
                f"| {metrics['word_level']['recall']:.4f} |"
                f" {metrics['word_level']['f_score']:.4f} "
                f"| {metrics['word_level']['accuracy']:.4f} |"
                f" {metrics['char_level']['precision']:.4f} "
                f"| {metrics['char_level']['recall']:.4f} |"
                f" {metrics['char_level']['f_score']:.4f} "
                f"| {metrics['char_level']['accuracy']:.4f} | "
                f"{metrics['granularity_difference']:.4f} "
                f" | {metrics['length_difference']}"
            )


if __name__ == "__main__":
    tester = SegmenterTester()
    # 测试预处理和还原功能
    tester.test_pretreatment_and_restore("news_a")
    # 进行分词测试，使用news_o作为原始文本，news_a作为标注数据
    tester.test_all("news_a", "news_o")  # 修改为其他语料测试
    # tester.test_all("dianshang_reviews")
    # tester.test_all("bilbil_reviews")
