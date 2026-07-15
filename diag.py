import generate_qa as g, os

PDFS = g.PDFS
files = sorted(f for f in os.listdir(PDFS) if f.endswith(".pdf"))[:3]
lines = []
for fn in files:
    text = g.extract_text(os.path.join(PDFS, fn))
    total = len(text)
    secs = g._split_sections(text)
    n_sec = len(secs) if secs else 0
    n_match = len(list(g.SECTION_PAT.finditer(text)))
    pieces = g.chunk_text(text)
    kept = sum(len(p) for p in pieces)
    line = (f"\n{fn}\n"
            f"  原始字数={total}  章节锚点数={n_match}  章节段数={n_sec}\n"
            f"  当前保留块数={len(pieces)}  保留字数={kept}  保留率={kept/total:.1%}")
    if secs:
        ratios = [g.digit_ratio(s) for s in secs]
        drop = [i for i, r in enumerate(ratios) if r >= 0.35]
        line += (f"\n  各章节段 digit_ratio: 均值={sum(ratios)/len(ratios):.2f}  "
                 f">=0.35(被丢)段={len(drop)}/{n_sec}")
        # 列出被丢的章节段开头, 看丢了什么
        for i in drop:
            head = secs[i][:40].replace("\n", " ")
            line += f"\n    [丢] 段{i}: {head}"
    lines.append(line)
open("diag.txt", "w", encoding="utf-8").write("\n".join(lines))
print("done")
