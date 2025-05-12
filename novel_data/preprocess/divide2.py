
import re
import json
import logging
from typing import List, Dict
import jieba.analyse

# 用来处理有章节标题的小说
# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 中文数字识别列表
CHINESE_NUMBERS = [
    "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
    "十一", "十二", "十三", "十四", "十五", "十六", "十七", "十八", "十九", "二十",
    "二十一", "二十二", "二十三", "二十四", "二十五", "二十六", "二十七", "二十八", "二十九", "三十"
]

# 尝试导入 OpenAI
try:
    from openai import OpenAI
except ImportError as e:
    logging.error(f"无法导入 openai 包: {e}")
    logging.error("请运行 'pip install --upgrade openai' 安装或更新 openai 包")
    raise

# OpenAI API 配置
try:
    client = OpenAI(api_key="YOUR_API_KEY", base_url="https://api.deepseek.com")
except Exception as e:
    logging.error(f"初始化 OpenAI 客户端失败: {e}")
    raise

def call_llm_api(text: str, max_length: int = 200) -> str:
    """
    调用 OpenAI API 生成章节简介，确保输出完整句子。
    :param text: 章节全文
    :param max_length: 简介最大长度（字）
    :return: 生成的完整简介
    """
    # 限制输入长度（避免 API 超限）
    max_input_len = 20000
    text = text[:max_input_len]

    # 构造提示，明确要求完整句子和字数范围
    prompt = (
        f"请为以下小说章节生成一个简洁的内容简介（30-70字），突出主要事件、人物或情节转折，确保内容以完整句子结束，避免泛泛而谈,直接写简介内容，不要有任何开头：\n\n{text}\n\n"
    )

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7
        )
        summary = response.choices[0].message.content.strip()
        print(summary)
        # 移除多余的“简介：”前缀（如果模型返回）
        if summary.startswith("简介："):
            summary = summary[3:].strip()

        # 截断到 max_length，确保完整句子
        if len(summary) <= max_length:
            return summary

        # 查找最后一个完整句子（以句号、叹号、问号结束）
        end_chars = ['。', '！', '？']
        truncated = summary[:max_length]
        last_end_pos = max([truncated.rfind(char) for char in end_chars] + [-1])

        if last_end_pos > 0 and last_end_pos >= max_length - 20:
            # 如果接近 max_length 且有完整句子，截断到该位置
            return truncated[:last_end_pos + 1]
        else:
            # 如果找不到合适句子或截断后太短，尝试缩短到前一句
            sentences = [s for s in re.split(r'[。！？]', summary) if s.strip()]
            result = ""
            total_len = 0
            for s in sentences:
                sentence = s + '。'  # 假设句号结束
                if total_len + len(sentence) <= max_length - 2:
                    result += sentence
                    total_len += len(sentence)
                else:
                    break
            if result and len(result) >= 50:
                return result
            # 若无法生成完整简介，使用备用
            raise Exception("无法生成完整简介，切换到备用")

    except Exception as e:
        logging.warning(f"OpenAI API 调用失败或无法生成完整简介: {e}")
        # 备用简介：基于关键词
        keywords = jieba.analyse.extract_tags(text, topK=5)
        backup = f"涉及{'、'.join(keywords)}的事件。"
        # 确保备用简介完整
        if len(backup) <= max_length:
            return backup
        # 截断备用简介到最后一个完整句子
        end_chars = ['。', '！', '？']
        truncated = backup[:max_length]
        last_end_pos = max([truncated.rfind(char) for char in end_chars] + [-1])
        return truncated[:last_end_pos + 1] if last_end_pos > 0 else truncated

def extract_keywords(text: str, top_k: int = 10) -> List[str]:
    """
    提取章节关键词。
    :param text: 章节全文
    :param top_k: 返回的关键词数量
    :return: 关键词列表
    """
    try:
        keywords = jieba.analyse.extract_tags(text, topK=top_k, withWeight=False)
        return keywords
    except Exception as e:
        logging.warning(f"关键词提取失败: {e}")
        return []

def merge_short_paragraphs(paragraphs: List[str], min_len: int = 100, target_len: int = 300) -> List[str]:
    """
    合并短段落，确保段落长度适中。
    :param paragraphs: 原始段落列表
    :param min_len: 最小段落长度（字数）
    :param target_len: 目标段落最大长度
    :return: 合并后的段落列表
    """
    merged_paragraphs = []
    current_paragraph = []
    current_length = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_length = len(para)
        current_paragraph.append(para)
        current_length += para_length
        if current_length >= min_len:
            merged_paragraphs.append("\n".join(current_paragraph))
            current_paragraph = []
            current_length = 0
        elif current_length > target_len:
            merged_paragraphs.append("\n".join(current_paragraph))
            current_paragraph = []
            current_length = 0

    if current_paragraph:
        merged_paragraphs.append("\n".join(current_paragraph))

    final_paragraphs = []
    buffer = []
    buffer_length = 0

    for para in merged_paragraphs:
        para_length = len(para)
        if para_length < min_len and final_paragraphs:
            buffer.append(para)
            buffer_length += para_length
        else:
            if buffer:
                final_paragraphs[-1] = "\n".join([final_paragraphs[-1]] + buffer)
                buffer = []
                buffer_length = 0
            final_paragraphs.append(para)

    if buffer:
        if final_paragraphs:
            final_paragraphs[-1] = "\n".join([final_paragraphs[-1]] + buffer)
        else:
            final_paragraphs.append("\n".join(buffer))

    return final_paragraphs

def split_by_chapter(raw_text: str) -> List[Dict]:
    """
    按章节分割小说文本，为每章生成简介和关键词。
    :param raw_text: 小说文本
    :return: 章节结构列表，包含标题、简介、关键词和段落
    """
    lines = raw_text.strip().splitlines()
    # 正则表达式匹配章节标题：中文数字 或 第X章，可选空格+标题
    chapter_pattern = re.compile(
        r'^(?:第([一二三四五六七八九十百千万〇零两\d]+)[章回节部卷篇]|([一二三四五六七八九十百千万〇零两\d]+))'  # 编号部分
        r'[\s:：．、，]+'  # 分隔符
        r'(.{1,40})$'  # 标题
    )



    chapters = []
    current_chapter = {"title": None, "summary": "", "keywords": [], "paragraphs": []}
    buffer = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:  # 跳过空行
            logging.debug(f"行 {i+1}: 空行，跳过")
            continue
        match = chapter_pattern.match(stripped)
        # 调试：打印正则匹配详情
        if match and any(num in stripped for num in CHINESE_NUMBERS):
            logging.info(f"行 {i+1}: 匹配章节标题: {repr(stripped)}, 匹配结果: {match.groups()}")
            if len(stripped) >= 25:
                logging.warning(f"行 {i+1}: 标题过长（{len(stripped)}字），可能不是标题: {stripped}")
            if current_chapter["title"]:
                paragraphs = [p.strip() for p in "\n".join(buffer).split("\n") if p.strip()]
                current_chapter["paragraphs"] = merge_short_paragraphs(paragraphs)
                chapter_text = "\n".join(current_chapter["paragraphs"])
                logging.info(f"正在为章节 {current_chapter['title']} 生成简介和关键词（字数：{len(chapter_text)}）")
                current_chapter["summary"] = call_llm_api(chapter_text)
                current_chapter["keywords"] = extract_keywords(chapter_text)
                chapters.append(current_chapter)
                buffer = []
            current_chapter = {"title": stripped, "summary": "", "keywords": [], "paragraphs": []}
        else:
            logging.debug(f"行 {i+1}: 非章节标题: {repr(stripped)}")
            buffer.append(stripped)

    if current_chapter["title"]:
        paragraphs = [p.strip() for p in "\n".join(buffer).split("\n") if p.strip()]
        current_chapter["paragraphs"] = merge_short_paragraphs(paragraphs)
        chapter_text = "\n".join(current_chapter["paragraphs"])
        logging.info(f"正在为章节 {current_chapter['title']} 生成简介和关键词（字数：{len(chapter_text)}）")
        current_chapter["summary"] = call_llm_api(chapter_text)
        current_chapter["keywords"] = extract_keywords(chapter_text)
        chapters.append(current_chapter)

    # 调试：打印所有章节标题
    logging.info("最终章节标题：")
    for chapter in chapters:
        logging.info(f"章节: {chapter['title']}")

    return chapters

# 主函数：处理新书
import os
import json
import logging


if __name__ == "__main__":
    input_dir = "D:/4schoolwork/vscode/IR-project/novel_data/source_txt"
    output_dir = "D:/4schoolwork/vscode/IR-project/novel_data/novel_index"
    exclude_file = "一-雪山飞狐.txt"

    try:
        for filename in os.listdir(input_dir):
            if filename.endswith(".txt") and filename != exclude_file:
                input_path = os.path.join(input_dir, filename)
                output_filename = os.path.splitext(filename)[0] + ".json"
                output_path = os.path.join(output_dir, output_filename)

                try:
                    with open(input_path, "r", encoding="utf-8") as f:
                        raw_text = f.read()

                    structured = split_by_chapter(raw_text)

                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(structured, f, ensure_ascii=False, indent=2)

                    logging.info(f"✅ 已保存章节 JSON：{output_path}")
                except Exception as e:
                    logging.error(f"处理文件失败：{input_path}，错误：{e}")

    except Exception as e:
        logging.error(f"批处理失败：{e}")
