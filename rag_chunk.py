"""递归分块：按分隔符优先级把长文档切成 <= chunk_size 的小块。

优先在段落/句子边界下刀而非按固定字数硬切，避免把语义单元劈成两半。

用法：
    python rag_chunk.py
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
# 分隔符优先级：从粗到细，最后的 "" 表示逐字符兜底
SEPARATORS = ["\n\n", "\n", "。", " ", ""]


def split_text(text, separators, chunk_size):
    """递归切块：返回长度 <= chunk_size 的字符串列表。"""
    # 已经够小，直接成块
    if len(text) <= chunk_size:
        return [text]

    # 分隔符用尽，逐字符硬切
    if not separators:
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    # 用当前最粗的分隔符切，超长的再递归换更细的
    sep = separators[0]
    parts = text.split(sep)

    # 贪心合并：能装进一块就装，装不下开新块
    merged = []
    cur = parts[0]
    for p in parts[1:]:
        if len(cur) + len(sep) + len(p) <= chunk_size:
            cur = cur + sep + p
        else:
            merged.append(cur)
            cur = p
    merged.append(cur)

    # 够小的收下，超长的用更细的分隔符递归
    result = []
    for chunk in merged:
        if len(chunk) <= chunk_size:
            result.append(chunk)
        else:
            result += split_text(chunk, separators[1:], chunk_size)
    return result


if __name__ == "__main__":
    CHUNK_SIZE = 500  # 每块的字符数上限
    with open("corpus_chapter3.txt", encoding="utf-8") as f:
        corpus = f.read()

    chunks = split_text(corpus, SEPARATORS, CHUNK_SIZE)

    print(f"原文 {len(corpus)} 字 -> 切成 {len(chunks)} 块")
    print("块长度：最小", min(map(len, chunks)),
          "最大", max(map(len, chunks)),
          "平均", sum(map(len, chunks)) // len(chunks))
    print("\n--- 前 2 块预览 ---")
    for i, c in enumerate(chunks[:2], 1):
        print(f"\n[块{i}] ({len(c)} 字)\n{c[:200]}...")
