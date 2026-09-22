# Project AUTO documentation

Open [the HTML documentation portal](index.html) for an offline, searchable guide to the project.

## Documentation levels

- Orientation: [project overview](01-overview.md) and [scope and assumptions](08-scope.md).
- System design: [architecture](02-architecture.md), [modules](03-modules.md), [technology stack](04-stack.md), and [data model](05-data.md), and [regions and spatial memory](11-regions.md).
- Operational reference: [configuration](06-configuration.md), [performance](07-performance.md), [operations](09-operations.md), and [verification](10-verification.md).

Numbered Markdown chapters are the maintained source; HTML is generated for browser reading. The snapshot was reviewed on 14 September 2026. Runtime code takes precedence over historical project notes. No benchmark, production accuracy guarantee, or live-camera verification is implied by this documentation.

From the project root, rebuild and check using standard-library Python:

```powershell
python docs/build_docs.py
python docs/check_docs.py
```

The builder supports headings, paragraphs, flat lists, tables, fenced code, inline code, and links. No server, package installation, external fonts, or network access is required to view the HTML pages. Keep the docs directory together; source-code links additionally require the surrounding repository.
