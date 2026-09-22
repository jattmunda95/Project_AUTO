"""Validate offline documentation links and structure without project imports."""

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent


class Page(HTMLParser):
    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        self.ids = set()
        self.links = []
        self.duplicates = []
        self.h1 = 0
        self.title = 0
        self.feed(path.read_text(encoding="utf-8"))

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            if attrs["id"] in self.ids:
                self.duplicates.append(attrs["id"])
            self.ids.add(attrs["id"])
        if tag == "h1":
            self.h1 += 1
        if tag == "title":
            self.title += 1
        for attr in ("href", "src"):
            if attr in attrs:
                self.links.append(attrs[attr])


def check() -> None:
    pages = {path.resolve(): Page(path) for path in ROOT.glob("*.html")}
    errors = []
    links = 0
    for path, page in pages.items():
        if page.h1 != 1 or page.title != 1 or "main" not in page.ids:
            errors.append(f"{path.name}: requires one h1/title and main landmark target")
        if page.duplicates:
            errors.append(f"{path.name}: duplicate IDs {page.duplicates}")
        for href in page.links:
            links += 1
            url = urlsplit(href)
            if url.scheme or url.netloc:
                errors.append(f"{path.name}: unexpected external dependency/link {href}")
                continue
            target = (path.parent / unquote(url.path)).resolve() if url.path else path
            if not target.is_file():
                errors.append(f"{path.name}: missing target {href}")
            elif url.fragment and target in pages and unquote(url.fragment) not in pages[target].ids:
                errors.append(f"{path.name}: missing fragment {href}")
    for source in ROOT.glob("[0-9][0-9]-*.md"):
        if source.with_suffix(".html").resolve() not in pages:
            errors.append(f"{source.name}: missing HTML chapter")
    if not pages or not (ROOT / "index.html").is_file():
        errors.append("Missing portal pages")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"PASS: {len(pages)} HTML pages, {links} local links/assets, chapter coverage and unique anchors")


if __name__ == "__main__":
    check()
