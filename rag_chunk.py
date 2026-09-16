"""RAG 单元1：递归分块（recursive chunking）

把长文档按「分隔符优先级」递归切成 <= chunk_size 的小块，
尽量在段落/句子边界下刀，避免劈开语义。

用法：
    python rag_chunk.py
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
# pyright: ignore[reportUndefinedVariable]
# 刀列表：从粗到细，最后一个 "" 是逐字符兜底
SEPARATORS = ["\n\n", "\n", "。", " ", ""]


def split_text(text, separators, chunk_size):
    """递归切块：返回长度 <= chunk_size 的字符串列表。"""
    # 基准情形1：已经够小，直接成一整块
    if len(text) <= chunk_size:
        return [text]

    # 基准情形2：刀用完了，逐字符硬切
    if not separators:
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    # 递归情形：用最粗的刀切
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

    # 够小的收下，超长的换更细的刀递归
    result = []
    for chunk in merged:
        if len(chunk) <= chunk_size:
            result.append(chunk)
        else:
            result += split_text(chunk, separators[1:], chunk_size)
    return result


if __name__ == "__main__":
    CHUNK_SIZE = 500  # 块大小（几百~一千多字）
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
