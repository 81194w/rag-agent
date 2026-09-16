import re, html, sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

with open("chapter3.html", encoding="utf-8") as f:
    raw = f.read()

# 提取正文容器
m = re.search(r'<article class="md-content__inner md-typeset">(.*?)</article>', raw, re.S)
body = m.group(1) if m else raw

# 块级标签先转成换行，保留段落/标题/列表结构
body = re.sub(r'<(br|/p|/h[1-6]|/li|/tr|/div|/pre|/blockquote)[^>]*>', '\n', body, flags=re.I)
# 去掉剩余所有标签
text = re.sub(r'<[^>]+>', '', body)
# 解 HTML 实体（&amp; -> & 等）
text = html.unescape(text)
# 去掉 MkDocs 标题的 permalink 锚点符号
text = text.replace('¶', '')
# 压缩 3 个以上连续换行
text = re.sub(r'\n{3,}', '\n\n', text).strip()

with open("corpus_chapter3.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("总字数:", len(text))
print("--- 开头 400 字 ---")
print(text[:400])
