// Offline portal builder. Python is used only to parse source syntax, never import it.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {execFileSync} from 'node:child_process';
const root = path.dirname(fileURLToPath(import.meta.url));
const repo = path.dirname(root);
const esc = value => String(value).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const read = name => fs.readFileSync(path.join(root,name),'utf8');
const write = (name,text) => fs.writeFileSync(path.join(root,name),text+'\n');
const excluded = new Set(['.git','.venv','venv','__pycache__','.pytest_cache','.mypy_cache','.ruff_cache','node_modules']);
function walk(dir) {
  return fs.readdirSync(dir,{withFileTypes:true}).sort((a,b)=>a.name.localeCompare(b.name,'en')).flatMap(e => {
    if (excluded.has(e.name) || e.isSymbolicLink()) return [];
    const p = path.join(dir,e.name);
    return e.isDirectory() ? walk(p) : [path.relative(repo,p).replaceAll('\\','/')];
  });
}
const files = walk(repo);
const pythonFiles = files.filter(p=>p.endsWith('.py'));
const syntax = String.raw`import ast,json,pathlib,sys
root=pathlib.Path(sys.argv[1])
result=[]
for name in json.load(sys.stdin):
 text=(root/name).read_text(encoding='utf-8-sig')
 tree=ast.parse(text,filename=name)
 definitions=[]
 def visit(node,prefix=''):
  for child in ast.iter_child_nodes(node):
   if isinstance(child,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
    qualified=prefix+child.name
    description=(ast.get_docstring(child) or '').split('\n')[0]
    if not description and child.name.startswith('test_'): description=child.name[5:].replace('_',' ')
    definitions.append([qualified,child.lineno,description or 'See source and module reference.'])
    visit(child,qualified+'.')
   else: visit(child,prefix)
 visit(tree)
 imports=set()
 for node in ast.walk(tree):
  if isinstance(node,ast.Import): imports.update(a.name for a in node.names if a.name.startswith('project_auto'))
  if isinstance(node,ast.ImportFrom) and (node.level or (node.module or '').startswith('project_auto')): imports.add('.'*node.level+(node.module or ''))
 result.append(dict(name=name,definitions=definitions,imports=sorted(imports),empty=not text.strip()))
print(json.dumps(result))`;
const sources = JSON.parse(execFileSync(process.env.PYTHON || 'python',['-B','-c',syntax,repo],{input:JSON.stringify(pythonFiles),encoding:'utf8',maxBuffer:16*1024*1024}));
function table(rows) { return '<div class="table-wrap" tabindex="0" role="region" aria-label="Reference table"><table><thead><tr>'+rows[0].map(c=>'<th scope="col">'+c+'</th>').join('')+'</tr></thead><tbody>'+rows.slice(1).map(r=>'<tr>'+r.map(c=>'<td>'+c+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>'; }
function catalog(type) {
 if(type==='python') return `<p class="catalog-count">${sources.length} Python files. Expand a file for definitions and project imports.</p>`+sources.map(s=>`<details class="catalog-group"><summary>${esc(s.name)} · ${s.definitions.length} definitions</summary><p><a href="../${esc(s.name)}">Open source</a></p>`+(s.imports.length?'<p>Project imports (including local/type-only imports): '+s.imports.map(i=>'<code>'+esc(i)+'</code>').join(', ')+'</p>':'')+(s.definitions.length?table([['Definition','Line','Source description'],...s.definitions.map(([n,l,d])=>['<code>'+esc(n)+'</code>',l,esc(d)])]):'<p>'+(s.empty?'Empty package marker or placeholder.':'Top-level script; no class or function definitions.')+'</p>')+'</details>').join('\n');
 if(type!=='files') throw Error('Unknown catalog '+type);
 const groups = Object.groupBy(files,p=>path.posix.dirname(p));
 return `<p class="catalog-count">${files.length} files, excluding environments and caches.</p>`+Object.entries(groups).map(([folder,entries])=>`<details class="catalog-group"><summary>${esc(folder)} · ${entries.length} files</summary><ul>`+entries.map(p=>`<li><a href="../${esc(p)}">${esc(path.posix.basename(p))}</a></li>`).join('')+'</ul></details>').join('\n');
}
function inline(text) {
 return text.split(/(`[^`]+`|\*\*[^*]+\*\*|\[[^\]]+\]\([^)]+\))/g).map(t=>{
  if(t.startsWith('`'))return '<code>'+esc(t.slice(1,-1))+'</code>';
  if(t.startsWith('**'))return '<strong>'+esc(t.slice(2,-2))+'</strong>';
  const m=t.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
  if(m)return `<a href="${esc(m[2].replace(/^(\d\d-[^/]+)\.md(?=#|$)/,'$1.html'))}">${esc(m[1])}</a>`;
  return esc(t);
 }).join('');
}
function figure(alt,url) {return `<figure class="diagram"><div class="diagram-scroll"><img src="${esc(url)}" alt="${esc(alt)}" loading="lazy"></div><figcaption>${esc(alt)} · <a href="${esc(url)}">Open full-size diagram</a></figcaption></figure>`;}
function render(text) {
 const lines=text.split(/\r?\n/), out=[], headings=[], used=new Set();
 let i=0;
 while(i<lines.length){
  let line=lines[i++],m;
  if(!line.trim())continue;
  if(line.startsWith('```')){const language=line.slice(3).trim(),code=[];while(i<lines.length&&!lines[i].startsWith('```'))code.push(lines[i++]);i++;out.push(language==='catalog'?catalog(code.join('').trim()):'<pre><code>'+esc(code.join('\n'))+'</code></pre>');continue;}
  if((m=line.match(/^(#{1,6}) (.+)$/))){const level=m[1].length,title=m[2],base=title.toLowerCase().replace(/[^a-z0-9]+/g,'-').replace(/^-|-$/g,'');let id=base,n=2;while(used.has(id))id=base+'-'+n++;used.add(id);out.push(`<h${level} id="${id}">${inline(title)}</h${level}>`);if(level===2)headings.push([id,title]);continue;}
  if((m=line.match(/^!\[([^\]]*)\]\(([^)]+)\)$/))){out.push(figure(m[1],m[2]));continue;}
  if(line.startsWith('|')){const rows=[];do{const cells=line.trim().replace(/^\||\|$/g,'').split('|').map(c=>c.trim());if(!cells.every(c=>/^:?-+:?$/.test(c)))rows.push(cells.map(inline));line=lines[i++];}while(line?.startsWith('|'));i--;out.push(table(rows));continue;}
  if(/^(?:- |\d+\. )/.test(line)){const tag=line.startsWith('- ')?'ul':'ol',pattern=tag==='ul'?/^- (.+)$/:/^\d+\. (.+)$/,items=[];do{items.push('<li>'+inline(line.replace(pattern,'$1'))+'</li>');line=lines[i++];}while(line&&pattern.test(line));i--;out.push(`<${tag}>${items.join('')}</${tag}>`);continue;}
  const p=[line];while(i<lines.length&&lines[i].trim()&&!/^(#|```|\||- |\d+\. |!\[)/.test(lines[i]))p.push(lines[i++]);out.push('<p>'+inline(p.join(' '))+'</p>');
 }
 return {body:out.join('\n'),headings};
}
// Diagram geometry is maintained as data; all output remains usable offline.
for(const [name,d] of Object.entries(JSON.parse(read('diagrams.json')))){
 const nodes=Object.fromEntries(d.nodes.map(n=>[n.id,n]));
 const port=(id,side)=>{const [x,y,w,h]=nodes[id].box;return {top:[x+w/2,y],bottom:[x+w/2,y+h],left:[x,y+h/2],right:[x+w,y+h/2]}[side];};
 let svg=`<svg xmlns="http://www.w3.org/2000/svg" width="${d.size[0]}" height="${d.size[1]}" viewBox="0 0 ${d.size.join(' ')}" role="img" aria-labelledby="title desc"><title id="title">${esc(d.title)}</title><desc id="desc">${esc(d.description)}</desc><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="#526875"/></marker></defs><rect width="100%" height="100%" fill="white"/><style>text{font-family:system-ui,Segoe UI,sans-serif;fill:#172c3a;font-size:15px}.title{font-weight:600}.label{font-size:13px;paint-order:stroke;stroke:white;stroke-width:5px;stroke-linejoin:round}</style>`;
 for(const e of d.edges){const points=[port(e.from,e.start||'bottom'),...(e.via||[]),port(e.to,e.end||'top')];svg+=`<path d="M${points.map(p=>p.join(',')).join(' L')}" fill="none" stroke="#526875" stroke-width="1.5" ${e.dashed?'stroke-dasharray="6 4"':''} marker-end="url(#arrow)"/>`;if(e.label)svg+=`<text class="label" x="${e.label_at[0]}" y="${e.label_at[1]}" text-anchor="middle">${esc(e.label)}</text>`;}
 for(const n of d.nodes){const [x,y,w,h]=n.box;svg+=`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="7" fill="${n.accent?'#e3f4ed':'#edf3f4'}" stroke="#b9cdd4"/>`;n.lines.forEach((line,j)=>{svg+=`<text ${j===0?'class="title"':''} x="${x+w/2}" y="${y+h/2-(n.lines.length-1)*10+5+j*20}" text-anchor="middle">${esc(line)}</text>`;});}
 for(const [x,y,label] of d.labels||[])svg+=`<text class="label" x="${x}" y="${y}">${esc(label)}</text>`;
 fs.mkdirSync(path.join(root,'assets/diagrams'),{recursive:true});write(`assets/diagrams/${name}.svg`,svg+'</svg>');
}
const chapters=fs.readdirSync(root).filter(p=>/^\d\d-.*\.md$/.test(p)).sort().map(name=>{const text=read(name);return {name,html:name.replace(/\.md$/,'.html'),title:text.split(/\r?\n/)[0].slice(2),text,...render(text)};});
const reviewed='26 September 2026';
function shell(title,current,body){const nav=`<a href="index.html" ${current==='index.html'?'aria-current="page"':''}>Documentation home</a>`+chapters.map((c,i)=>`<a href="${c.html}" ${current===c.html?'aria-current="page"':''}>${String(i+1).padStart(2,'0')} &nbsp; ${esc(c.title)}</a>`).join('');return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>${esc(title)} | Project AUTO</title><link rel="stylesheet" href="assets/docs.css"><script src="assets/docs.js" defer></script></head><body><a class="skip" href="#main">Skip to content</a><div class="layout"><aside class="sidebar"><a class="brand" href="index.html">PROJECT AUTO</a><div class="eyebrow">Engineering documentation</div><nav aria-label="Documentation">${nav}</nav><div class="meta">Version 0.1.0<br>Source reviewed ${reviewed}<br>Local prototype · CPU inference<br><a href="../README.md">Repository README</a></div></aside><main id="main"><div class="topline"><span>PROJECT AUTO / DOCUMENTATION</span><span>Reviewed ${reviewed}</span></div>${body}<footer>Project AUTO · Source-grounded documentation · Offline HTML edition<br>Configured values are not measured guarantees. Source chapters are maintained in Markdown.</footer></main></div></body></html>`;}
for(const [i,c] of chapters.entries()){
 const toc='<div class="on-page"><strong>On this page</strong><ul>'+c.headings.map(([id,t])=>`<li><a href="#${id}">${esc(t)}</a></li>`).join('')+'</ul></div>';
 const previous=chapters[i-1],next=chapters[i+1];
 write(c.html,shell(c.title,c.html,`<div class="toolbar"><a href="${c.name}">Markdown source</a><button data-print hidden type="button">Print chapter</button></div>${toc}<article class="content">${c.body}</article><nav class="page-controls" aria-label="Adjacent chapters"><a href="${previous?.html||'index.html'}">← ${esc(previous?.title||'Documentation home')}</a><a href="${next?.html||'index.html'}">${esc(next?.title||'Back to library')} →</a></nav>`));
}
const cards=chapters.map((c,i)=>`<article class="card" data-search="${esc((c.title+' '+c.text).toLowerCase())}"><span class="number">Chapter ${String(i+1).padStart(2,'0')}</span><h2><a href="${c.html}">${esc(c.title)}</a></h2><p>${inline(c.text.split(/\r?\n/).find((l,j)=>j>0&&l.trim()&&!/^(#|\||!\[|`)/.test(l))||'')}</p><a class="source" href="${c.name}">Markdown source</a></article>`).join('');
write('index.html',shell('Documentation home','index.html',`<section class="hero"><span class="badge">Local computer vision · v0.1.0</span><h1>From seeing objects<br>to remembering them.</h1><p class="lead">The complete guide to Project AUTO: follow the live pipeline, explore each Python file, and understand how identity, events and spatial memory fit together.</p><div class="summary"><div><strong>${chapters.length} chapters</strong><span>Purpose through implementation</span></div><div><strong>${sources.length} Python files</strong><span>Expandable source and function index</span></div><div><strong>7 visual maps</strong><span>Offline diagrams with source explanations</span></div></div></section><section><h2>The project at a glance</h2>${figure('Project AUTO: live pipeline','assets/diagrams/overview.svg')}</section><section aria-labelledby="routes"><h2 id="routes">Choose your reading route</h2><div class="routes"><div class="route"><strong>Understand the project</strong><p>Purpose, scope and capabilities.</p><a href="01-overview.html">Start with the overview →</a></div><div class="route"><strong>Explore all files and functions</strong><p>Seven diagrams explain responsibilities; the source index lists every Python definition.</p><a href="12-project-map.html">Open the complete map →</a><br><a href="13-source-index.html">Browse the file index →</a></div><div class="route"><strong>Run and validate</strong><p>Setup, configuration and verification.</p><a href="09-operations.html">Open operations →</a></div></div></section><div class="notice">This is a source review, not a camera or recognition-accuracy test. Known implementation limits are explained in the project map and verification chapters.</div><section aria-labelledby="library"><h2 id="library">Documentation library</h2><div class="search"><label for="search">Search chapter content</label><input id="search" type="search" placeholder="Try: database, queue, camera…" aria-describedby="result-count"><p id="result-count" class="count" role="status" aria-live="polite">${chapters.length} of ${chapters.length} chapters</p><noscript>Search needs JavaScript; all chapters remain available below.</noscript></div><div class="cards">${cards}</div><p id="empty" hidden>No chapters match. Try a shorter term.</p></section>`));
console.log(`Built ${chapters.length} chapters, 7 diagrams and an index of ${sources.length} Python files.`);
