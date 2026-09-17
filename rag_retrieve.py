"""RAG 全链路：召回 → rerank → 溯源 → 生成。

块与 query 各自向量化，按余弦相似度召回 top-k，再由 LLM 精排打分，
最后拼进 prompt 让模型带 [编号] 引用作答。

用法：
    python rag_retrieve.py
"""
import os
import json
import sys
import math
import time
from openai import OpenAI
from fastembed import TextEmbedding
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


model=TextEmbedding("BAAI/bge-small-zh-v1.5")

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
    timeout=60,
)
MODEL = "deepseek-chat"


N = 3


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def embed(text):
    vec=list(model.embed([text]))[0]
    return vec.tolist()

def embed_2gram(text, D=256):
    vec=[0]*D
    for i in range(len(text) - 1):
        h=0
        h=h*31+ord(text[i])
        h=h*31+ord(text[i+1])
        h=h%D
        vec[h]+=1
    return vec


def cos(a, b):
    """余弦相似度，值域 [-1, 1]。"""
    return dot(a,b)/(math.sqrt(dot(a, a))*math.sqrt(dot(b, b)))


def build_index(chunks, embed_fn):
    # {块编号: (文本, 向量)}——文本跟着向量一起存，search 才能取回原文
    index={}
    for i,chunk in enumerate(chunks):
        index[i]=(chunk,embed_fn(chunk))
    return index


def search(query, index, k, embed_fn):
    """按余弦相似度召回 top-k，返回 [(块编号, 文本), ...]。"""
    score={}
    q=embed_fn(query)
    for i, (text, vec) in index.items():
        score[i]=cos(q,vec)
    top=sorted(score.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [(i, index[i][0]) for i,_ in top]


def rerank(query, candidates):
    """用 LLM 给候选块打 0~100 分并重排，返回 top-N。"""
    scored_list=[]
    for (idx,chunk) in candidates:
        score=None
        system_prompt=(
                "你是一个相关性打分器。\n"
                "判断下面这段文本能否回答用户问题\n"
                "相关度打 0~100 分，输出一行JSON，格式如下：\n"
                f"""{{"score": 85}}\n"""
                f"【问题】：{query}\n"
                f"【文本】：{chunk}\n"
                )
        messages = [{"role": "system", "content": system_prompt}]
        for step in range(3):
            print(f"\n===== 第 {step} 轮 =====")
            try:
                resp = client.chat.completions.create(model=MODEL, messages=messages)
            except Exception as e:
                print(f"  调用失败：{e}，5 秒后重试")
                time.sleep(5)
                continue
            content = resp.choices[0].message.content
            print(f"模型回复: {content}")
            try:
                score = json.loads(content)["score"]
                break
            except:
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": "你的输出不是合法 JSON，请重新只输出 JSON"})    
        if score is None:
            score=0
        scored_list.append((score,idx,chunk))
    scored_list.sort(key=lambda k: k[0], reverse=True)
    return [(idx, text) for (_, idx, text) in scored_list[:N]]


def generate(query, top):
    """让模型只依据候选材料作答并标注 [编号]，返回 (答案, {编号: 文本})。"""
    context = "\n".join(f"【{idx}】 {text}" for idx,text in top)
    system = "只根据下面材料回答，每句话用 [编号] 标注依据，材料里没有就说没提及\n" + context
    messages = [ {"role":"system", "content": system}, {"role":"user", "content": query} ]
    resp = client.chat.completions.create(model=MODEL, messages=messages)
    content = resp.choices[0].message.content
    print(f"模型回复: {content}")
    sources = {idx: text for idx,text in top}
    return (content,sources)

if __name__ == "__main__":
    from rag_chunk import split_text, SEPARATORS   # 复用分块逻辑，避免重复实现

    CHUNK_SIZE = 500
    with open("corpus_chapter3.txt", encoding="utf-8") as f:
        corpus = f.read()

    chunks = split_text(corpus, SEPARATORS, CHUNK_SIZE)
    index = build_index(chunks, embed)
    print(f"已建索引：{len(chunks)} 个块，向量维度 {768}\n")

    for query in ["检索增强生成", "知识过期", "用户记忆"]:
        print(f"=== 查询：{query} ===")
        candidates = search(query, index, k=5, embed_fn=embed)   # 先召回 5 个候选
        top = rerank(query, candidates)          # LLM 打分精排，取 top-N
        print(f"--- 精排后 top-{N} ---")
        for (idx,text) in top:
            preview = text[:100].replace("\n", " ")
            print(f"  - [{idx}] {preview}...")
        answer, sources = generate(query, top)
        print(f"--- 答案 ---")
        print(answer)
        print(f"--- 来源 ---")
        for idx, text in sources.items():
            print(f"[{idx}] {text[:80]}...")    
        print()