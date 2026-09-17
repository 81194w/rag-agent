"""RAG 检索评测：3 个变体 × 3 个指标，检验「召回是精排上限」。

EVAL 是人工标注的评测集（query + 正确答案的 chunk 编号集合），
三个变体统一签名 (query) -> 编号列表，evaluate 逐题跑分后求平均。

用法：
    python eval_rag.py
"""
from rag_retrieve import build_index, search, rerank, embed, embed_2gram, N
from rag_chunk import split_text, SEPARATORS

K = N        # 每个变体最终取 top-K 参与打分
RECALL = 5   # rerank 变体的候选数：先粗召回 5 个，再精排到 K


def recall(top_ids, gt):
    return len(set(top_ids)&gt)/len(gt)
def hit(top_ids, gt):
    if set(top_ids)&gt:
        return 1
    return 0
def mrr(top_ids, gt):
    for i,idx in enumerate(top_ids):
        if idx in gt:
            return 1/(i+1)
    return 0


# 评测集：query + 人工标注的 ground truth（正确 chunk 编号集合）
EVAL = [
    ("什么是用户记忆", {0, 1, 9, 91}),
    ("Memobase 的设计理念", {22, 23}),
    ("什么是稠密嵌入", {30, 31, 33}),
    ("什么是稀疏嵌入", {36}),
    ("什么是 RAPTOR", {53, 57, 58}),
    ("什么是 GraphRAG", {54, 56, 57, 58}),
    ("上下文感知检索是什么", {80, 81}),
    ("语义搜索", {30, 31}),
    ]

# 三个变体共用同一套语料、分块与 query——控制变量
CHUNK_SIZE = 500
with open("corpus_chapter3.txt", encoding="utf-8") as f:
    corpus = f.read()
chunks = split_text(corpus, SEPARATORS, CHUNK_SIZE)
index_2gram = build_index(chunks, embed_2gram)
index_bge   = build_index(chunks, embed)


def v_2gram(query):
    return [i for i, _ in search(query, index_2gram, K, embed_2gram)]

def v_bge(query):
    return [i for i, _ in search(query, index_bge, K, embed)]

def v_rerank(query):
    return [i for i, _ in rerank(query, search(query, index_bge, RECALL, embed))]


def evaluate(v_fn):
    """对 EVAL 逐题跑分，返回 (recall 均值, hit 均值, mrr 均值)。"""
    rs,hs,ms=[],[],[]
    for (query,gt) in EVAL:
        ids=v_fn(query)
        rs.append(recall(ids, gt)); 
        hs.append(hit(ids, gt)); 
        ms.append(mrr(ids, gt))
    return [sum(rs) / len(rs), sum(hs) / len(hs), sum(ms) / len(ms)]

if __name__ == "__main__":
    variants = [("2-gram", v_2gram), ("BGE", v_bge), ("BGE+rerank", v_rerank)]
    r_col, h_col = f"recall@{K}", f"hit@{K}"
    print(f"{'variant':<12}{r_col:>12}{h_col:>12}{'MRR':>12}")
    for name, v in variants:
        r, h, m = evaluate(v)
        print(f"{name:<12}{r:>12.3f}{h:>12.3f}{m:>12.3f}")
