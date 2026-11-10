"""把 report.html 渲染成 PNG（逐页截图）。

用法：
    python render_report.py
输出：
    D:\\project\\math_modeling_learn\\report\\png\\ 01_xxx.png ...
"""
import os

from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.abspath(__file__))
HTML = os.path.join(ROOT, "report.html")
OUT = os.path.join(ROOT, "png")
SCALE = 2                      # 2 倍超采样，文字更清晰
WIDTH = 1080

TITLES = [
    "cover",
    "00_总纲_建模在干什么",
    "01_武器一_翻译成线性规划",
    "02_武器二_线性化",
    "03_武器三_整数性",
    "04_武器四_凸性",
    "05_武器五_最优性证明",
    "06_武器六_灵敏度分析",
    "07_武器七_验证",
    "08_总表与陷阱清单",
    "09_练习路线",
]


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        if f.endswith(".png"):
            os.remove(os.path.join(OUT, f))

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": WIDTH, "height": 1600},
                                device_scale_factor=SCALE)
        page.goto("file:///" + HTML.replace("\\", "/"))
        page.wait_for_load_state("networkidle")
        page.add_style_tag(content="body{background:#fff;} .page{margin-bottom:0 !important;}")

        pages = page.query_selector_all("section.page")
        print(f"共 {len(pages)} 节")
        results = []
        for i, sec in enumerate(pages):
            name = TITLES[i] if i < len(TITLES) else f"page{i:02d}"
            path = os.path.join(OUT, f"{name}.png")
            sec.screenshot(path=path)
            size = os.path.getsize(path)
            # 超过 2MB 会超出下游图片显示上限，量化到 256 色（对文字/图表几乎无损）
            if size > 1_900_000:
                try:
                    from PIL import Image
                    im = Image.open(path).convert("RGB")
                    q = im.quantize(colors=256, method=Image.MEDIANCUT,
                                    dither=Image.FLOYDSTEINBERG)
                    tmp = path + ".tmp.png"
                    q.save(tmp, optimize=True, compress_level=9)
                    if os.path.getsize(tmp) < size:
                        os.replace(tmp, path)
                        size = os.path.getsize(path)
                    else:
                        os.remove(tmp)
                except Exception as e:
                    print("   (压缩跳过:", e, ")")
            results.append((name, size))
            print(f"  {name:<34} {size/1024:8.1f} KB")
        browser.close()

    print(f"\n输出目录: {OUT}")
    print(f"合计 {sum(s for _, s in results)/1024/1024:.2f} MB")


if __name__ == "__main__":
    main()
