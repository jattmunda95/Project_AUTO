# Python and file index

This index is generated during the documentation build from the current repository. Use the [project map](12-project-map.md) to understand responsibilities, or the [module reference](03-modules.md) for concise contracts. Expand a file below to see its classes, functions, methods, source line numbers and project imports.

## Every Python file and definition

Source links open the local file. Line numbers are listed beside each definition; they are build-time locations, not permanent identifiers. Descriptions come from docstrings or test names and can lag the implementation. Imports include local and type-only imports and do not imply runtime call order.

```catalog
python
```

## Supporting files by folder

This inventory includes model and research asset filenames, configuration, documentation and generated package metadata. Git internals, the installed virtual environment, bytecode and pytest caches are excluded. Binary model and image contents are not interpreted by the builder.

```catalog
files
```

## Keeping this index current

Rebuild the documentation after adding or renaming source files. No application modules are imported, models loaded, cameras opened or databases changed by index generation.

```powershell
node docs/build_docs.mjs
python docs/check_docs.py
```
