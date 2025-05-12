import json
import logging
from typing import List, Dict
import jieba.analyse
import numpy as np
from sentence_transformers import SentenceTransformer

# 用来处理只有章节分割，没有章节标题的小说
# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 加载嵌入模型
try:
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
except Exception as e:
    logging.error(f"加载 sentence-transformers 模型失败: {e}")
    logging.error("请运行 'pip install sentence-transformers' 安装")
    raise

def load_json_data(file_path: str) -> List[Dict]:
    """
    加载 JSON 数据。
    :param file_path: JSON 文件路径
    :return: 章节数据列表
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"JSON 文件未找到: {file_path}")
        raise
    except Exception as e:
        logging.error(f"加载 JSON 失败: {e}")
        raise

def compute_embeddings(texts: List[str], model) -> np.ndarray:
    """
    计算文本嵌入向量。
    :param texts: 文本列表
    :param model: 嵌入模型
    :return: 嵌入向量数组
    """
    try:
        embeddings = model.encode(texts, convert_to_tensor=False, show_progress_bar=True)
        return embeddings.tolist()
    except Exception as e:
        logging.error(f"嵌入向量计算失败: {e}")
        return []

def build_index(chapters: List[Dict], model) -> Dict:
    """
    建立索引。
    :param chapters: 章节数据
    :param model: 嵌入模型
    :return: 索引字典
    """
    index = {
        "chapters": [],
        "paragraphs": [],
        "embeddings": {
            "summaries": [],
            "paragraphs": []
        }
    }

    paragraph_global_idx = 0
    for chapter_idx, chapter in enumerate(chapters):
        chapter_info = {
            "chapter_idx": chapter_idx,
            "title": chapter["title"],
            "summary": chapter["summary"],
            "keywords": chapter["keywords"],
            "paragraph_count": len(chapter["paragraphs"])
        }
        index["chapters"].append(chapter_info)

        for para_idx, para in enumerate(chapter["paragraphs"]):
            para_info = {
                "global_idx": paragraph_global_idx,
                "chapter_idx": chapter_idx,
                "chapter_title": chapter["title"],
                "paragraph_idx": para_idx,
                "content": para,
                "keywords": jieba.analyse.extract_tags(para, topK=5, withWeight=False)
            }
            index["paragraphs"].append(para_info)
            paragraph_global_idx += 1

        texts = [chapter["summary"]] + chapter["paragraphs"]
        embeddings = compute_embeddings(texts, model)
        if embeddings:
            index["embeddings"]["summaries"].append({
                "chapter_idx": chapter_idx,
                "embedding": embeddings[0]
            })
            index["embeddings"]["paragraphs"].extend([
                {
                    "global_idx": paragraph_global_idx - len(chapter["paragraphs"]) + i,
                    "embedding": emb
                }
                for i, emb in enumerate(embeddings[1:])
            ])

        logging.info(f"已处理章节 {chapter['title']}（字数：{sum(len(p) for p in chapter['paragraphs'])}）")

    return index

def save_index(index: Dict, output_file: str):
    """
    保存索引到文件。
    :param index: 索引字典
    :param output_file: 输出文件路径
    """
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        logging.info(f"✅ 索引已保存：{output_file}")
    except Exception as e:
        logging.error(f"保存索引失败: {e}")
        raise

if __name__ == "__main__":
    input_file = "D:/4schoolwork/vscode/IR-project/novel_data/雪山飞狐.json"
    output_file = "D:/4schoolwork/vscode/IR-project/novel_data/novel_index.json"

    try:
        chapters = load_json_data(input_file)
        index = build_index(chapters, model)
        save_index(index, output_file)
    except Exception as e:
        logging.error(f"索引建立失败: {e}")