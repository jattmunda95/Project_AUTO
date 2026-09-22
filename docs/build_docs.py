"""Build the offline documentation portal with Python's standard library.

Supported Markdown subset: headings, paragraphs, flat lists, tables, fenced code,
inline code, and links. Numbered Markdown chapters are the source of truth.
"""

from html import escape
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
REVIEWED = "14 September 2026"


def inline(text: str, html_links: bool = True) -> str:
    parts = []
    offset = 0
    for match in re.finditer(r"`([^`]+)`|\[([^\]]+)\]\(([^)]+)\)", text):
        parts.append(escape(text[offset:match.start()]))
        if match.group(1) is not None:
            parts.append(f"<code>{escape(match.group(1))}</code>")
        else:
            href = match.group(3)
            if html_links and re.match(r"^\d\d-[^/]+\.md$", href):
                href = href[:-3] + ".html"
            parts.append(f'<a href="{escape(href, quote=True)}">{escape(match.group(2))}</a>')
        offset = match.end()
    parts.append(escape(text[offset:]))
    return "".join(parts)


def render(markdown: str) -> tuple[str, list[tuple[str, str]]]:
    lines = markdown.splitlines()
    output, headings = [], []
    used = set()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith("```"):
            code = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i])
                i += 1
            output.append("<pre><code>" + escape("\n".join(code)) + "</code></pre>")
            i += 1
            continue
        heading = re.match(r"^(#{1,3}) (.+)$", line)
        if heading:
            level, title = len(heading[1]), heading[2]
            anchor = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
            original, suffix = anchor, 2
            while anchor in used:
                anchor = f"{original}-{suffix}"
                suffix += 1
            used.add(anchor)
            output.append(f'<h{level} id="{anchor}">{inline(title)}</h{level}>')
            if level == 2:
                headings.append((anchor, title))
            i += 1
            continue
        if line.startswith("| "):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [cell.strip() for cell in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                    rows.append(cells)
                i += 1
            table = '<div class="table-wrap" tabindex="0" role="region" aria-label="Scrollable reference table"><table><thead><tr>'
            table += "".join(f'<th scope="col">{inline(cell)}</th>' for cell in rows[0])
            table += "</tr></thead><tbody>"
            for row in rows[1:]:
                table += "<tr>" + "".join(f"<td>{inline(cell)}</td>" for cell in row) + "</tr>"
            output.append(table + "</tbody></table></div>")
            continue
        item = re.match(r"^(?:- |\d+\. )(.+)$", line)
        if item:
            tag = "ul" if line.startswith("- ") else "ol"
            items = []
            while i < len(lines):
                match = re.match(r"^- (.+)$" if tag == "ul" else r"^\d+\. (.+)$", lines[i])
                if not match:
                    break
                items.append(f"<li>{inline(match[1])}</li>")
                i += 1
            output.append(f"<{tag}>" + "".join(items) + f"</{tag}>")
            continue
        paragraph = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#|```|\| |\d+\. |- )", lines[i]):
            paragraph.append(lines[i])
            i += 1
        output.append("<p>" + inline(" ".join(paragraph)) + "</p>")
    return "\n".join(output), headings


CSS = """
:root{--ink:#172c3a;--muted:#526875;--line:#d7e2e7;--paper:#fff;--bg:#f2f6f7;--accent:#006c68;--nav:#112c3c}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:24px}body{margin:0;color:var(--ink);background:var(--bg);font:16px/1.7 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}a{color:var(--accent);text-underline-offset:3px}a:hover{text-decoration-thickness:2px}a:focus-visible,input:focus-visible,button:focus-visible,[tabindex]:focus-visible{outline:3px solid #dc8f16;outline-offset:4px}.skip{position:fixed;top:-80px;left:16px;z-index:10;background:white;padding:12px}.skip:focus{top:12px}.layout{display:grid;grid-template-columns:265px minmax(0,1fr);min-height:100vh}.sidebar{background:var(--nav);color:#dce9ee;padding:34px 22px;position:sticky;top:0;height:100vh;overflow:auto}.brand{font-size:22px;font-weight:800;letter-spacing:.08em;color:white;text-decoration:none}.eyebrow{font-size:12px;text-transform:uppercase;letter-spacing:.16em;font-weight:700}.sidebar .eyebrow{color:#91b5c1;margin:4px 0 32px}.sidebar nav a{display:block;color:#d7e5eb;text-decoration:none;padding:9px 12px;border-radius:6px;font-size:14px;margin:3px 0}.sidebar nav a:hover{background:#244353}.sidebar nav a[aria-current=page]{background:#d9f3e9;color:#153e3d;font-weight:700}.nav-label{font-size:11px;text-transform:uppercase;letter-spacing:.13em;color:#9bb6c2;margin:24px 12px 8px}.sidebar .meta{border-top:1px solid #35505e;margin-top:28px;padding-top:20px;font-size:12px;color:#a6c0cb}.sidebar .meta a{color:#c9e3e9}main{padding:42px clamp(24px,4vw,72px);max-width:1450px;min-width:0}.topline{display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-bottom:28px}.badge{display:inline-block;border:1px solid #b6d8ce;background:#e3f4ed;color:#15554e;border-radius:20px;padding:3px 11px;font-size:12px;font-weight:650}h1{font-size:clamp(30px,3.4vw,48px);line-height:1.15;letter-spacing:-.04em;margin:12px 0 24px;max-width:950px}h2{font-size:24px;line-height:1.3;letter-spacing:-.02em;margin:44px 0 15px;padding-top:7px}h3{font-size:19px}p{max-width:88ch;margin:14px 0 20px}.lead{font-size:20px;color:var(--muted);max-width:780px}.hero{border-bottom:1px solid var(--line);padding-bottom:32px;margin-bottom:30px}.summary{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:28px 0}.summary div{border-left:3px solid #278b7c;padding-left:16px}.summary strong{display:block;font-size:20px}.summary span{font-size:13px;color:var(--muted)}.routes{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:32px}.route{background:#e4eeef;padding:18px;border-radius:8px}.route strong{display:block}.route p{font-size:14px;margin:6px 0}.search{margin:20px 0}.search label{display:block;font-weight:700;margin-bottom:8px}.search input{width:100%;padding:14px 16px;border:1px solid #97afb9;border-radius:6px;font:inherit;background:white;color:var(--ink)}.count{color:var(--muted);font-size:13px}.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.card{background:white;border:1px solid var(--line);border-radius:9px;padding:23px;box-shadow:0 3px 9px #112c3c04}.card[hidden]{display:none}.card .number{font-size:12px;color:var(--muted);letter-spacing:.1em;text-transform:uppercase}.card h2{font-size:21px;margin:10px 0;padding:0}.card h2 a{text-decoration:none}.card p{font-size:14px;color:var(--muted);margin:8px 0 15px}.card .source{font-size:12px}.notice{background:#fff7e6;border-left:3px solid #be821b;padding:17px 21px;margin:24px 0;font-size:14px}.on-page{background:#e6eef1;border-radius:8px;padding:16px 20px;margin:22px 0 32px}.on-page strong{font-size:12px;text-transform:uppercase;letter-spacing:.1em}.on-page ul{display:flex;flex-wrap:wrap;gap:5px 22px;list-style:none;margin:9px 0 0;padding:0;font-size:13px}.content{background:white;border:1px solid var(--line);border-radius:10px;padding:clamp(20px,3vw,42px);min-width:0}.content h1{font-size:36px}.content>h1:first-child{margin-top:0}li{margin:7px 0}code{font-family:Consolas,"Liberation Mono",monospace;font-size:.88em;overflow-wrap:anywhere;background:#edf2f4;padding:2px 5px;border-radius:3px}pre{background:#102b3a;color:#e1f3f3;padding:22px;border-radius:7px;overflow:auto;font-size:13px;line-height:1.7}pre code{background:none;color:inherit;padding:0;overflow-wrap:normal}.table-wrap{overflow-x:auto;margin:20px 0 26px}table{border-collapse:collapse;width:100%;font-size:14px}th{text-align:left;background:#edf3f4;color:#254552;font-size:12px}th,td{padding:12px 14px;border-bottom:1px solid var(--line);vertical-align:top}td:first-child{font-weight:550}tbody tr:nth-child(even){background:#fafcfc}.page-controls{display:flex;justify-content:space-between;gap:15px;margin:28px 0;font-size:14px}.toolbar{display:flex;gap:20px;align-items:center}button{font:inherit;border:1px solid var(--line);background:white;color:var(--accent);border-radius:5px;padding:4px 11px;cursor:pointer}footer{border-top:1px solid var(--line);padding-top:24px;margin-top:38px;font-size:12px;color:var(--muted)}
@media(min-width:1500px){main{padding-left:72px;padding-right:72px}}@media(max-width:900px){.layout{grid-template-columns:220px minmax(0,1fr)}.sidebar{padding:25px 12px}main{padding:28px 20px}.routes{grid-template-columns:1fr}.cards{grid-template-columns:1fr}.summary strong{font-size:17px}.content{padding:23px}}@media(max-width:620px){.layout{display:block}.sidebar{position:relative;height:auto;padding:18px}.sidebar .eyebrow{margin-bottom:12px}.sidebar nav{display:flex;overflow:auto;gap:5px;padding-bottom:6px}.sidebar nav a{white-space:nowrap}.sidebar .nav-label,.sidebar .meta{display:none}main{padding:25px 15px}.content{padding:18px}.content h1{font-size:30px}.summary{grid-template-columns:1fr;gap:16px}.topline{margin-bottom:18px}.on-page ul{display:block}.page-controls{flex-wrap:wrap}th,td{min-width:135px;padding:9px}.routes{gap:10px}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}@media print{body{background:white;font-size:10pt}.layout{display:block}.sidebar,.skip,.toolbar,.search,.on-page,.page-controls{display:none}main{max-width:none;padding:0}.content{border:0;padding:0}.hero{padding:0}.cards,.routes{display:block}.card{break-inside:avoid;margin:10px 0;box-shadow:none}.table-wrap{overflow:visible}table{font-size:9pt}th,td{min-width:0;padding:6px}pre{white-space:pre-wrap;color:black;background:#eee;font-size:8pt}h1,h2,h3{break-after:avoid}tr{break-inside:avoid}a{color:inherit}.summary{display:flex;gap:24px}footer{margin-top:20px}}
"""

JS = """'use strict';
const field = document.querySelector('#search');
if (field) {
  const cards = [...document.querySelectorAll('.card')];
  const count = document.querySelector('#result-count');
  const empty = document.querySelector('#empty');
  field.addEventListener('input', () => {
    const terms = field.value.toLocaleLowerCase().trim().split(/\\s+/).filter(Boolean);
    let visible = 0;
    for (const card of cards) {
      card.hidden = !terms.every(term => card.dataset.search.includes(term));
      if (!card.hidden) visible++;
    }
    count.textContent = `${visible} of ${cards.length} chapters`;
    empty.hidden = visible !== 0;
  });
}
document.querySelectorAll('[data-print]').forEach(button => {
  button.hidden = false;
  button.addEventListener('click', () => window.print());
});
"""


def build() -> None:
    chapters = []
    for path in sorted(ROOT.glob("[0-9][0-9]-*.md")):
        text = path.read_text(encoding="utf-8")
        title = text.splitlines()[0].removeprefix("# ")
        body, headings = render(text)
        chapters.append(dict(path=path, title=title, text=text, body=body, headings=headings))
    labels = ["Overview", "Architecture", "Modules", "Stack & dependencies", "Data & contracts", "Configuration", "Performance", "Scope & assumptions", "Setup & operations", "Verification", "Regions & spatial memory"]
    summaries = [
        "Purpose, current capabilities, maturity, and reading routes for each audience.",
        "Runtime flow, main-thread and worker responsibilities, identity decisions, and boundaries.",
        "A source-linked catalog of implemented modules, interfaces, scripts, and placeholders.",
        "Technology layers, declared version ranges, model assets, and environment requirements.",
        "Permanent identities, event history, reference vectors, message contracts, and schema evolution.",
        "All six YAML files: camera settings, thresholds, lifecycle timing, and worker scheduling.",
        "Configured limits, derived estimates, implementation caveats, and a proposed benchmark protocol.",
        "Operating assumptions, known gaps, current exclusions, and follow-up priorities.",
        "Installation, model preparation, database preflight, running, stopping, and troubleshooting.",
        "Test coverage boundaries, hardware validation, source precedence, and documentation upkeep.",
        "Polygon calibration, event-based location state, region queries, and spatial contracts.",
    ]

    def shell(title: str, current: str, body: str) -> str:
        nav = '<a href="index.html"' + (' aria-current="page"' if current == "index" else '') + '>Documentation home</a>'
        for i, chapter in enumerate(chapters):
            if i in (0, 1, 8):
                nav += '<div class="nav-label">' + {0: "Understand", 1: "Technical reference", 8: "Operate & maintain"}[i] + '</div>'
            href = chapter["path"].with_suffix(".html").name
            active = ' aria-current="page"' if current == href else ''
            nav += f'<a href="{href}"{active}>{i + 1:02d} &nbsp; {escape(labels[i])}</a>'
        return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="description" content="Project AUTO technical documentation: local computer vision, permanent object identity, and event memory."><title>{escape(title)} | Project AUTO</title><link rel="stylesheet" href="assets/docs.css"><script src="assets/docs.js" defer></script></head>
<body><a class="skip" href="#main">Skip to content</a><div class="layout"><aside class="sidebar"><a class="brand" href="index.html">PROJECT AUTO</a><div class="eyebrow">Engineering documentation</div><nav aria-label="Documentation">{nav}</nav><div class="meta">Version 0.1.0<br>Source reviewed {REVIEWED}<br>Local prototype · CPU inference<br><br><a href="../README.md">Repository README</a></div></aside><main id="main"><div class="topline"><span>PROJECT AUTO / DOCUMENTATION</span><span>Reviewed {REVIEWED}</span></div>{body}<footer>Project AUTO · Source-grounded documentation · Offline HTML edition<br>Configured values are not measured guarantees. Source chapters are maintained in Markdown.</footer></main></div></body></html>'''

    cards = []
    for i, chapter in enumerate(chapters):
        html_name = chapter["path"].with_suffix(".html").name
        search_text = escape((chapter["title"] + " " + chapter["text"]).lower(), quote=True)
        cards.append(f'<article class="card" data-search="{search_text}"><span class="number">Chapter {i + 1:02d}</span><h2><a href="{html_name}">{escape(chapter["title"])}</a></h2><p>{summaries[i]}</p><a class="source" href="{chapter["path"].name}">Markdown source</a></article>')
    home = '''<section class="hero"><span class="badge">Local computer vision · v0.1.0</span><h1>From seeing objects<br>to remembering them.</h1><p class="lead">A structured guide to Project AUTO: what it does, how its components work together, and what it takes to operate and validate it.</p><div class="summary"><div><strong>11 focused chapters</strong><span>Overview through implementation</span></div><div><strong>One local pipeline</strong><span>Camera, perception, identity, memory</span></div><div><strong>Evidence-led status</strong><span>Implemented, configured, and pending</span></div></div></section>
<section aria-labelledby="routes"><h2 id="routes">Choose your reading route</h2><div class="routes"><div class="route"><strong>Understand the project</strong><p>Purpose, scope, and delivery maturity.</p><a href="01-overview.html">Start with the overview →</a></div><div class="route"><strong>Explore the engineering</strong><p>Architecture, modules, data, and stack.</p><a href="02-architecture.html">Follow the pipeline →</a></div><div class="route"><strong>Run and validate</strong><p>Setup, configuration, and verification.</p><a href="09-operations.html">Open the operations guide →</a></div></div></section>
<div class="notice"><strong>Current maturity:</strong> live identity, event persistence, polygon regions, and console spatial queries are integrated. Live region validation, recognition calibration, measured performance, and deliberate schema recovery remain open. See the relevant chapter before treating a capability as production-ready.</div>
<section aria-labelledby="library"><h2 id="library">Documentation library</h2><div class="search"><label for="search">Search all chapter content</label><input id="search" type="search" placeholder="Try: database, queue, dependencies, camera…" autocomplete="off" aria-describedby="result-count"><p class="count" id="result-count" role="status" aria-live="polite">11 of 11 chapters</p><noscript><p>Search needs JavaScript; all chapters remain available below.</p></noscript></div><div class="cards">''' + "".join(cards) + '''</div><p id="empty" hidden>No chapters match. Try a shorter term such as “model” or “event”.</p></section>'''
    (ROOT / "index.html").write_text(shell("Documentation home", "index", home), encoding="utf-8")
    for i, chapter in enumerate(chapters):
        html_name = chapter["path"].with_suffix(".html").name
        toc = '<div class="on-page"><strong>On this page</strong><ul>' + ''.join(f'<li><a href="#{anchor}">{escape(title)}</a></li>' for anchor, title in chapter["headings"]) + '</ul></div>'
        toolbar = f'<div class="toolbar"><a href="{chapter["path"].name}">Markdown source</a><button data-print hidden type="button">Print chapter</button></div>'
        before = '<a href="index.html">← Documentation home</a>' if i == 0 else f'<a href="{chapters[i-1]["path"].with_suffix(".html").name}">← {escape(labels[i-1])}</a>'
        after = '<a href="index.html">Back to library →</a>' if i == len(chapters)-1 else f'<a href="{chapters[i+1]["path"].with_suffix(".html").name}">{escape(labels[i+1])} →</a>'
        content = toolbar + toc + '<article class="content">' + chapter["body"] + '</article><nav class="page-controls" aria-label="Adjacent chapters">' + before + after + '</nav>'
        (ROOT / html_name).write_text(shell(chapter["title"], html_name, content), encoding="utf-8")
    assets = ROOT / "assets"
    assets.mkdir(exist_ok=True)
    (assets / "docs.css").write_text(CSS.strip() + "\n", encoding="utf-8")
    (assets / "docs.js").write_text(JS, encoding="utf-8")
    print(f"Built {len(chapters)} chapter pages and index.html")


if __name__ == "__main__":
    build()
