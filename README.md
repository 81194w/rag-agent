# RAG 全链路实践（带溯源 + 评测）

> 手写 RAG 五段管线（分块 → 检索 → rerank → 溯源 → 生成），
> 附「三变体 × 三指标」评测，实证两条核心结论：
> **召回是精排上限**、**必须用真 embedding（稠密检索）**。

## 是什么 / 解决什么问题

手写 RAG 全链路（分块 → 检索 → rerank → 溯源 → 生成），解决 LLM 的三大短板：
知识过时、幻觉、私有数据。附带「三变体 × 三指标」评测，用数据说明
「为什么必须真 embedding」和「召回是精排上限」。

## 架构

```
离线建库（一次）                在线问答（每次）
  语料 corpus_chapter3.txt        问题 query
     │                              │
     ▼ 语义分块 rag_chunk.py        ▼ 向量化（BGE 稠密 / 2-gram 稀疏）
  96 个 chunk                       │
     │                              ▼
     ▼ 向量化 + 建索引              召回 top-5（余弦相似度）
  {编号: (文本, 向量)}              │
                                    ▼
                                   精排 rerank（LLM 打分 → top-3）
                                    │
                                    ▼
                                   拼 context + 生成（要求 [编号] 溯源）
                                    │
                                    ▼
                                  答案 + 来源块
```

## 怎么跑

### 依赖

```bash
pip install openai fastembed
```

- 首次运行会自动下载 BGE 中文 embedding 模型 `BAAI/bge-small-zh-v1.5`。
- `rerank` / `generate` 两步调用 DeepSeek，需要环境变量 `DEEPSEEK_API_KEY`。

### 命令

```bash
# 单 query 全链路：分块 → 检索 → rerank → 溯源 → 生成
python rag_retrieve.py

# 评测：3 变体 × 3 指标，打印对比表
python eval_rag.py
```

## 评测结果

评测集：8 个 query + 人工标注 ground truth（正确 chunk 编号集合）。

| 变体 | recall@3 | hit@3 | MRR |
|---|---|---|---|
| 2-gram 稀疏 | 0.333 | 0.75 | 0.5625 |
| BGE 稠密 | 0.271 | 0.375 | 0.375 |
| BGE + rerank | 0.438 | 0.5 | 0.438 |

**结论**：rerank 能拉高 BGE（recall .271 → .438），但**追不平 2-gram 的 hit/MRR**——
本评测集以「字面题」为主，BGE 在召回一步就漏了，rerank 排不到漏掉的块。
这实锤了 **召回是精排上限：rerank 不能无中生有**。

**补充证据**（同义改写场景）：query「知识过期」时，2-gram 字面检索漏检、最高分仅 35；
换 BGE 稠密后最高分 78——改写/同义词场景下真 embedding 才追得到。

## 设计决策（面试三问）

1. **为什么语义分块，而不是按固定字数硬切？**
   固定字数硬切无视语义边界，会把一个完整段落/知识点劈成两半——检索时命中一半、另一半就丢了，块也不完整。
   语义分块在段落/句子边界下刀，让每块尽量是一个自足的语义单元，命中一块就能拿到完整答案。

2. **为什么 rerank 只是精排上限，救不了召回？**
   检索是两段式：召回先粗筛（快但糙），rerank 再对召回到的候选精排（准但只在候选内）。
   rerank 的上限就是召回的上限——召回漏掉的块根本不在 rerank 的输入里，再准也排不出来，只能「矮子里拔高个」。
   所以想提升效果，先补召回（recall），不是叠更多 rerank。

3. **为什么必须真 embedding（稠密），字符 2-gram 不够？**
   字符 2-gram 是字面匹配，只能抓「字符相邻」的巧合，有两个致命盲区：同义不同字（改写/同义词）漏检、同字不同义误判。
   稠密 embedding 把语义编进向量，语义近的句子向量就近，天然扛住改写和同义。
   但稠密也不是全面碾压——评测里 2-gram 在「字面题」上反超 BGE（hit .75 vs .375），两者各有主场，生产里常用混合检索。

## 已知改进点

- **rerank 换本地 cross-encoder**（如 `bge-reranker-base`）：当前 rerank 用远程 LLM 打分，
  脆（外部 API 抽风整条评估瘫）、慢（40 次串行）、贵（烧 token）。
  生产用本地 cross-encoder 批量打分：稳、快、可复现。计划投递后落地。
- **评测集扩充「改写题」比例**，避免 2-gram 的字面优势掩盖稠密检索的价值。

## 语料来源与授权

`corpus_chapter3.txt` 抽取自 **《深入理解 AI Agent：设计原理与工程实践》**（李博杰 著）第 3 章「用户记忆与知识库」，
原书以 **Apache License 2.0** 开源：https://github.com/bojieli/ai-agent-book

本仓库中的语料仅用于技术演示，版权归原作者所有；`extract_corpus.py` 提供从原始 HTML 抽取语料的脚本。
