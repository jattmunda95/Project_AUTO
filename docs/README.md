# Project AUTO documentation

Open [the HTML documentation portal](index.html) for an offline, searchable guide to the project.

## Documentation levels

- Orientation: [project overview](01-overview.md) and [scope and assumptions](08-scope.md).
- Visual guide: [complete project map](12-project-map.md) and [Python and file index](13-source-index.md).
- System design: [architecture](02-architecture.md), [modules](03-modules.md), [technology stack](04-stack.md), and [data model](05-data.md), and [regions and spatial memory](11-regions.md).
- Operational reference: [configuration](06-configuration.md), [performance](07-performance.md), [operations](09-operations.md), and [verification](10-verification.md).

Numbered Markdown chapters are the maintained source; HTML is generated for browser reading. The visual map and source index were integrated on 26 September 2026. Runtime code takes precedence over historical project notes. No benchmark, production accuracy guarantee, or live-camera verification is implied by this documentation.

From the project root, rebuild and check using Node.js 22 or newer and standard-library Python:

```powershell
node docs/build_docs.mjs
python docs/check_docs.py
```

The builder supports headings, paragraphs, flat lists, tables, fenced code, inline code, bold text, links, local diagram images and generated source catalogs. diagrams.json is the maintained diagram description; build_docs.mjs creates the offline SVG assets. Shared styles and browser behavior are maintained in assets/docs.css and assets/docs.js. The original build_docs.py is retained unchanged as a legacy builder; use the JavaScript command above for the expanded portal. No server, package installation, external fonts, or network access is required to view the HTML pages. Keep the docs directory together; source-code links additionally require the surrounding repository.
