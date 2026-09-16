

from rag_retrieve import build_index, search, rerank, embed, embed_2gram, N
from rag_chunk import split_text, SEPARATORS

K = N
RECALL = 5



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

  # 0. 准备（只做一次）：读语料 → 分块 → 建两个索引
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

CHUNK_SIZE = 500
with open("corpus_chapter3.txt", encoding="utf-8") as f:
    corpus = f.read()
chunks = split_text(corpus, SEPARATORS, CHUNK_SIZE)
index_2gram = build_index(chunks, embed_2gram)
index_bge   = build_index(chunks, embed)

  # 1. 三个变体，签名统一：变体(query) -> 编号列表（top-3）
def v_2gram(query):
    return [i for i, _ in search(query, index_2gram, K, embed_2gram)]

def v_bge(query):
    return [i for i, _ in search(query, index_bge, K, embed)]

def v_rerank(query):
    return [i for i, _ in rerank(query, search(query, index_bge, RECALL, embed))]



  # 2. evaluate(变体)：
  #    三个空列表 rs, hs, ms
  #    对 EVAL 里每个 (query, gt)：
  #        ids = 变体(query)
  #        rs.append(recall(ids, gt)); hs.append(hit(ids, gt)); ms.append(mrr(ids, gt))
  #    返回 (rs平均, hs平均, ms平均)
def evaluate(v_fn):
    rs,hs,ms=[],[],[]
    for (query,gt) in EVAL:
        ids=v_fn(query)
        rs.append(recall(ids, gt)); 
        hs.append(hit(ids, gt)); 
        ms.append(mrr(ids, gt))
    return [sum(rs) / len(rs), sum(hs) / len(hs), sum(ms) / len(ms)]

  # 3. 主循环：对三个变体各调一次 evaluate，拼成 3×3 表打印
if __name__ == "__main__":
    variants = [("2-gram", v_2gram), ("BGE", v_bge), ("BGE+rerank", v_rerank)]
    for name, v in variants:
        r, h, m = evaluate(v)      # 解包元组：三个变量各接一个
        print(name, r, h, m)



