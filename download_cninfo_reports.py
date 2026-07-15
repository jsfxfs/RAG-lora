#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量下载巨潮(CNINFO)上市公司年度报告 PDF —— 取最近 50 份
用法:
  python download_cninfo_reports.py --test      # 只查询并打印前几条, 不下载
  python download_cninfo_reports.py             # 下载最近 50 份年报到 ./01_年报PDF
"""
import os
import sys
import json
import time
import re
import datetime
import urllib.request
import urllib.parse
import ssl

BASE_QUERY = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
STATIC_HOSTS = ["https://static.cninfo.com.cn/", "http://static.cninfo.com.cn/"]
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(OUT_DIR, "01_年报PDF")
MANIFEST = os.path.join(OUT_DIR, "manifest.json")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "http://www.cninfo.com.cn/new/disclosure/overview/business",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "X-Requested-With": "XMLHttpRequest",
}

CATEGORY = "category_ndbg_szsh;"   # 年度报告(沪深)
SE_DATE = "2025-01-01~2026-07-09"  # 时间窗口(足够覆盖最近披露季)
TOP_N = 50
PAGE_SIZE = 30
MAX_PAGES = 20
TEST = "--test" in sys.argv


def http_post_json(url, data):
    body = urllib.parse.urlencode(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=HEADERS, method="POST")
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, context=ctx, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def http_get_bytes(url):
    req = urllib.request.Request(url, headers={
        "User-Agent": HEADERS["User-Agent"],
        "Referer": "http://www.cninfo.com.cn/",
    }, method="GET")
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, context=ctx, timeout=60) as r:
        return r.read()


def query(page, page_size=PAGE_SIZE):
    data = {
        "pageNum": page, "pageSize": page_size, "column": "",
        "tabName": "fulltext", "plate": "", "stock": "", "seDate": SE_DATE,
        "isHL": "", "sortName": "", "sortType": "", "limit": "",
        "category": CATEGORY,
    }
    return http_post_json(BASE_QUERY, data)


def ts_to_date(ms):
    try:
        return datetime.datetime.fromtimestamp(
            int(ms) / 1000, tz=datetime.timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        return "unknown"


def safe_name(s):
    return re.sub(r'[\\/:*?"<>|\r\n\t]+', "_", str(s)).strip("_")


def _is_full_text(title):
    """判断是否为中文年度报告全文(排除 摘要 / 英文版)。

    注: '全文'/'更正后'/'修订版'/'补充后'/'图文版' 均为年度报告全文, 保留;
        仅 '摘要' 与 '英文' 类排除(语料要中文全文)。
    """
    t = title or ""
    if any(k in t for k in ("摘要", "英文")):
        return False
    return "年度报告" in t


def fetch_recent(n=TOP_N):
    # 按公司(secCode)去重, 且只接受"中文年度报告全文"(排除 摘要/英文);
    # 某公司近期若只发了英文版/摘要, 则跳过, 顺延到下一家, 保证 50 份皆为中文全文
    seen = {}
    for p in range(1, MAX_PAGES + 1):
        try:
            j = query(p)
        except Exception as e:
            print(f"  [warn] 第{p}页查询失败: {e}")
            break
        a = j.get("announcements") or []
        if not a:
            break
        for it in a:
            code = it.get("secCode")
            if not code or code in seen:
                continue
            if not _is_full_text(it.get("announcementTitle")):
                continue
            seen[code] = it
        if len(seen) >= n:
            break
    items = sorted(seen.values(),
                    key=lambda x: int(x.get("announcementTime", 0) or 0),
                    reverse=True)
    return items[:n]


def main():
    print(f"[*] 查询巨潮年报, 窗口={SE_DATE}, 目标={TOP_N} 份 ...")
    top = fetch_recent(TOP_N)
    print(f"[*] 共取得 {len(top)} 条候选(已按披露时间倒序)")

    if TEST:
        out = [{
            "secCode": it.get("secCode"),
            "secName": it.get("secName"),
            "announcementTitle": it.get("announcementTitle"),
            "announcementTime": ts_to_date(it.get("announcementTime")),
            "adjunctUrl": it.get("adjunctUrl"),
            "announcementId": it.get("announcementId"),
        } for it in top]
        with open(os.path.join(OUT_DIR, "test_output.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[*] TEST 完成, 共 {len(out)} 条 -> test_output.json")
        return

    os.makedirs(PDF_DIR, exist_ok=True)
    manifest = []
    ok = 0
    for i, it in enumerate(top, 1):
        code = it.get("secCode", "NA")
        name = it.get("secName", "NA")
        title = it.get("announcementTitle", "")
        date = ts_to_date(it.get("announcementTime"))
        adj = (it.get("adjunctUrl") or "").strip()
        if not adj:
            print(f"[{i}/{TOP_N}] 跳过(无 adjunctUrl): {code} {name}")
            continue
        fname = f"{i:02d}_{code}_{safe_name(name)}_{date}.pdf"
        path = os.path.join(PDF_DIR, fname)
        last_err = None
        for host in STATIC_HOSTS:
            url = host + adj.lstrip("/")
            try:
                data = http_get_bytes(url)
                with open(path, "wb") as f:
                    f.write(data)
                ok += 1
                manifest.append({
                    "idx": i, "code": code, "name": name, "title": title,
                    "date": date, "url": url, "file": fname, "size": len(data),
                })
                print(f"[{i}/{TOP_N}] OK ({len(data)}B) {code} {name} {date}")
                last_err = None
                break
            except Exception as e:
                last_err = e
        if last_err:
            print(f"[{i}/{TOP_N}] FAIL {code} {name}: {last_err}")
        time.sleep(0.3)

    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump({"count": ok, "items": manifest}, f, ensure_ascii=False, indent=2)
    print(f"\n完成: 成功 {ok}/{TOP_N}, manifest -> {MANIFEST}")


if __name__ == "__main__":
    main()
