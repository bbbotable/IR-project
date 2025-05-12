import json
import logging
from typing import List, Dict
import jieba.analyse
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 加载嵌入模型
try:
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
except Exception as e:
    logging.error(f"加载 sentence-transformers 模型失败: {e}")
    logging.error("请运行 'pip install sentence-transformers' 安装")
    raise

def load_index(file_path: str) -> Dict:
    """
    加载索引文件。
    :param file_path: 索引文件路径
    :return: 索引字典
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"索引文件未找到: {file_path}")
        raise
    except Exception as e:
        logging.error(f"加载索引失败: {e}")
        raise

def extract_query_keywords(query: str, top_k: int = 10) -> List[str]:
    """
    提取查询关键词。
    :param query: 自然语言查询
    :param top_k: 返回的关键词数量
    :return: 关键词列表
    """
    try:
        return jieba.analyse.extract_tags(query, topK=top_k, withWeight=False)
    except Exception as e:
        logging.warning(f"查询关键词提取失败: {e}")
        return []

def compute_keyword_score(query_keywords: List[str], chapter_keywords: List[str], paragraph_keywords: List[str]) -> float:
    """
    计算关键词匹配得分。
    :param query_keywords: 查询关键词
    :param chapter_keywords: 章节关键词
    :param paragraph_keywords: 段落关键词
    :return: 关键词得分
    """
    keyword_hits = sum(1 for kw in query_keywords if kw in chapter_keywords)
    paragraph_hits = sum(1 for kw in query_keywords if kw in paragraph_keywords)
    total_hits = keyword_hits * 2 + paragraph_hits
    return total_hits / (len(query_keywords) + 1)

def compute_semantic_score(query: str, embeddings: List[float], model) -> float:
    """
    计算查询与嵌入的语义相似度。
    :param query: 自然语言查询
    :param embeddings: 预计算的嵌入向量
    :param model: 嵌入模型
    :return: 相似度得分
    """
    try:
        query_embedding = model.encode([query], convert_to_tensor=False)[0]
        embedding = np.array(embeddings)
        return cosine_similarity([query_embedding], [embedding])[0][0]
    except Exception as e:
        logging.warning(f"语义相似度计算失败: {e}")
        return 0.0

def get_context(paragraphs: List[Dict], match_global_idx: int, context_size: int = 2) -> List[str]:
    """
    获取匹配段落的上下文。
    :param paragraphs: 段落索引
    :param match_global_idx: 匹配段落的全局索引
    :param context_size: 前后上下文段落数
    :return: 上下文段落列表
    """
    match_para = next((p for p in paragraphs if p["global_idx"] == match_global_idx), None)
    if not match_para:
        return []
    chapter_idx = match_para["chapter_idx"]
    para_idx = match_para["paragraph_idx"]
    
    chapter_paras = [p for p in paragraphs if p["chapter_idx"] == chapter_idx]
    start = max(0, para_idx - context_size)
    end = min(len(chapter_paras), para_idx + context_size + 1)
    return [p["content"] for p in chapter_paras[start:end]]

def search_index(query: str, index: Dict, model, top_k: int = 5, context_size: int = 2) -> List[Dict]:
    """
    自然语言检索。
    :param query: 自然语言查询
    :param index: 索引字典
    :param model: 嵌入模型
    :param top_k: 返回 Top-k 结果
    :param context_size: 上下文段落数
    :return: Top-k 检索结果
    """
    logging.info(f"处理查询: {query}")
    query_keywords = extract_query_keywords(query)
    logging.info(f"查询关键词: {query_keywords}")

    results = []
    for chapter in index["chapters"]:
        chapter_idx = chapter["chapter_idx"]
        chapter_keywords = chapter["keywords"]
        
        chapter_paras = [p for p in index["paragraphs"] if p["chapter_idx"] == chapter_idx]
        summary_emb = next((e["embedding"] for e in index["embeddings"]["summaries"] if e["chapter_idx"] == chapter_idx), None)
        
        best_score = 0.0
        best_para = None
        best_context = []
        
        for para in chapter_paras:
            global_idx = para["global_idx"]
            para_keywords = para["keywords"]
            para_emb = next((e["embedding"] for e in index["embeddings"]["paragraphs"] if e["global_idx"] == global_idx), None)
            
            keyword_score = compute_keyword_score(query_keywords, chapter_keywords, para_keywords)
            semantic_score = max(
                compute_semantic_score(query, summary_emb, model) if summary_emb else 0.0,
                compute_semantic_score(query, para_emb, model) if para_emb else 0.0
            )
            total_score = 0.5 * keyword_score + 0.5 * semantic_score
            
            if total_score > best_score:
                best_score = total_score
                best_para = para
                best_context = get_context(index["paragraphs"], global_idx, context_size)
        
        if best_para:
            results.append({
                "title": chapter["title"],
                "summary": chapter["summary"],
                "keywords": chapter["keywords"],
                "match_paragraph": best_para["content"],
                "context": best_context,
                "score": best_score
            })

    return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

def format_results(results: List[Dict]) -> str:
    """
    格式化检索结果。
    :param results: 检索结果
    :return: 格式化文本
    """
    output = []
    for i, result in enumerate(results, 1):
        output.append(f"Top-{i} (得分: {result['score']:.3f})")
        output.append(f"章节: {result['title']}")
        output.append(f"简介: {result['summary']}")
        output.append(f"关键词: {', '.join(result['keywords'])}")
        output.append("匹配段落:")
        output.append(result["match_paragraph"])
        output.append("上下文:")
        output.append("\n".join(f"  - {para}" for para in result["context"]))
        output.append("-" * 50)
    return "\n".join(output)

if __name__ == "__main__":
    index_file = "D:/4schoolwork/vscode/IR-project/novel_data/novel_index.json"
    query = "陶子安与铁盒的冲突"
    top_k = 5
    context_size = 2

    try:
        index = load_index(index_file)
        results = search_index(query, index, model, top_k, context_size)
        print(format_results(results))
        
        with open("D:/4schoolwork/vscode/IR-project/novel_data/search_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logging.info("✅ 检索结果已保存：search_results.json")
    except Exception as e:
        logging.error(f"检索失败: {e}")