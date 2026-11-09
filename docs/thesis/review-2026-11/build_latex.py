from __future__ import annotations
import re, shutil, sys
from pathlib import Path

HERE = Path('/home/jerrycao/projects/demo-new/docs/thesis/review-2026-11')
ROOT = HERE.parents[2]
OUT = HERE / 'latex'
(OUT / 'assets').mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import build_review

SOURCE = HERE / 'review.md'
BIB = ROOT / 'docs/thesis/references.bib'
PILOT = ROOT / 'outputs/adaptive-dev-seed42-wsl-20261102/summary.json'
summary = __import__('json').loads(PILOT.read_text())
specs, _ = build_review.table_specs(summary)

PREAMBLE = r'''% Generated from ../review.md by ../build_latex.py.
% Compile with XeLaTeX and Biber; see README.md.
\documentclass[11pt,a4paper]{article}
\usepackage[a4paper,margin=24mm,headheight=14pt]{geometry}
\usepackage{fontspec}
\setmainfont{Latin Modern Roman}
\setsansfont{Latin Modern Sans}
\usepackage{microtype}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs,tabularx,array,colortbl}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,calc}
\usepackage[most]{tcolorbox}
\usepackage{enumitem}
\usepackage{caption}
\usepackage{fancyhdr}
\usepackage{xcolor}
\definecolor{Ink}{HTML}{182D3B}
\definecolor{Teal}{HTML}{006C73}
\definecolor{LightTeal}{HTML}{EAF3F3}
\definecolor{TableAlt}{HTML}{F2F6F7}
\usepackage[backend=biber,style=numeric-comp,sorting=none]{biblatex}
\addbibresource{references.bib}
\usepackage[colorlinks=true,allcolors=Teal]{hyperref}
\setlength{\parskip}{0.48em}
\setlength{\parindent}{0pt}
\renewcommand{\arraystretch}{1.15}
\captionsetup{font=small,labelfont=bf,skip=6pt}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small Compression for Federated Learning}
\fancyhead[R]{\small Literature review}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.3pt}
\setlist[itemize]{leftmargin=1.5em,itemsep=0.2em,topsep=0.3em}
\newcolumntype{R}[1]{>{\hsize=#1\hsize\raggedright\arraybackslash}X}
\newtcolorbox{reviewcallout}{colback=LightTeal,colframe=Teal!35,boxrule=0.5pt,arc=2mm,left=2mm,right=2mm,top=1.5mm,bottom=1.5mm,before skip=7pt,after skip=12pt}
\tikzset{reviewbox/.style={draw=Teal!45,fill=LightTeal,rounded corners=2pt,align=center,text=Ink,font=\small,minimum height=9mm,inner sep=4pt},reviewarrow/.style={-{Stealth[length=2mm]},draw=Teal,thick}}
\title{Compression for Federated Learning\\[3pt]\large Quantization, sparsification, learned coding and adaptive Lempel--Ziv methods}
\author{Cao\\\normalsize Thesis research preparation}
\date{Updated 2 November 2026}
\begin{document}
\maketitle
'''

def tex_escape(s: str) -> str:
    # Escape plain prose while leaving TeX markup generated below untouched.
    replacements = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(replacements.get(c, c) for c in s)

INLINE = re.compile(r'(\[@[^\]]+\]|\*\*.+?\*\*|`[^`]+`)')
def inline(s: str) -> str:
    parts = INLINE.split(s)
    out = []
    for part in parts:
        if not part:
            continue
        if part.startswith('[@'):
            keys = [x.strip().lstrip('@') for x in part[2:-1].split(';')]
            out.append(r'\parencite{' + ','.join(keys) + '}')
        elif part.startswith('**') and part.endswith('**'):
            out.append(r'\textbf{' + inline(part[2:-2]) + '}')
        elif part.startswith('`') and part.endswith('`'):
            out.append(r'\texttt{' + tex_escape(part[1:-1]) + '}')
        else:
            out.append(tex_escape(part))
    return ''.join(out)

EQUATIONS = {
    'w_(t+1) = w_t + sum_i a_i C(u_i,t)': r'w_{t+1} = w_t + \sum_i a_i C(u_{i,t})',
    'E[Q_s(x) | x] = x': r'\mathbb{E}[Q_s(x)\mid x] = x',
    'E[||Q_s(x)-x||_2^2 | x] <= min(d/s^2, sqrt(d)/s) ||x||_2^2': r'\mathbb{E}[\|Q_s(x)-x\|_2^2\mid x] \leq \min(d/s^2,\sqrt{d}/s)\|x\|_2^2',
    '||x-Top_k(x)||_2^2 <= (1-k/d) ||x||_2^2': r'\|x-\operatorname{Top}_k(x)\|_2^2 \leq (1-k/d)\|x\|_2^2',
    'E[||(d/k) Rand_k(x)-x||_2^2] = (d/k-1) ||x||_2^2': r'\mathbb{E}[\|(d/k)\operatorname{Rand}_k(x)-x\|_2^2] = (d/k-1)\|x\|_2^2',
    'v_i,t = u_i,t + e_i,t;   c_i,t = decode(encode(v_i,t))': r'v_{i,t}=u_{i,t}+e_{i,t};\qquad c_{i,t}=\operatorname{decode}(\operatorname{encode}(v_{i,t}))',
    'e_i,t+1 = v_i,t - c_i,t': r'e_{i,t+1}=v_{i,t}-c_{i,t}',
    'E_extra(p) = sum_tensors ||z_p-z||_2^2': r'E_{\mathrm{extra}}(p)=\sum_{\mathrm{tensors}}\|z_p-z\|_2^2',
    'E_extra(p) <= tau * sum_tensors ||u||_2^2': r'E_{\mathrm{extra}}(p)\leq \tau\sum_{\mathrm{tensors}}\|u\|_2^2',
    '||z_p-u||_2 <= ||z-u||_2 + sqrt(tau) ||u||_2': r'\|z_p-u\|_2\leq\|z-u\|_2+\sqrt{\tau}\|u\|_2',
}

def equation(expr):
    if expr not in EQUATIONS:
        raise ValueError(f'Unmapped equation: {expr}')
    return r'\begin{equation*}' + EQUATIONS[expr] + r'\end{equation*}'

def table_tex(key):
    if key == 'contents':
        return r'\tableofcontents\clearpage'
    headers, rows, fractions, caption = specs[key]
    n = len(headers)
    cols = ''.join('R{' + f'{f*n:.5f}' + '}' for f in fractions)
    out = [r'\begin{table}[htbp]', r'\centering', r'\small', r'\begin{tabularx}{\textwidth}{' + cols + '}', r'\toprule']
    out.append(' & '.join(r'\textbf{' + inline(str(x)) + '}' for x in headers) + r' \\\midrule')
    for i, row in enumerate(rows):
        if i % 2 == 1:
            out.append(r'\rowcolor{TableAlt}')
        out.append(' & '.join(inline(str(x)) for x in row) + r' \\')
    out.extend([r'\bottomrule', r'\end{tabularx}', r'\caption{' + inline(re.sub(r'^Table\s+\d+\.\s*', '', caption)) + '}', r'\label{tab:' + key + '}', r'\end{table}'])
    return '\n'.join(out)

DIAGRAM_CAPTIONS = {
    'fl': 'Local training and compressed model-delta aggregation. Original schematic based on FedAvg; mathematical setting is defined in the text.',
    'ef': 'Error feedback in model-delta units. Original schematic; residuals refer to the complete decoded compressor.',
    'learned': 'Learned update coding and the decoder dependency. Original schematic inspired by the AE literature; no learned-codec FL run is claimed.',
    'adaptive': 'Current adaptive selector. All p candidates share the same QSGD draw; only the selected complete reply is sent.',
}
def diagram_tex(name):
    common = [r'\begin{figure}[htbp]\centering', r'\begin{tikzpicture}[node distance=8mm and 7mm]']
    if name == 'fl':
        body = [r'\node[reviewbox] (s1) {Server\\broadcast $w_t$};', r'\node[reviewbox,right=of s1] (c) {Client $i$\\$H$ local steps};', r'\node[reviewbox,right=of c] (enc) {Encode/decode\\delta $u_{i,t}$};', r'\node[reviewbox,right=of enc] (s2) {Server\\weighted update};', r'\draw[reviewarrow] (s1)--(c);\draw[reviewarrow] (c)--(enc);\draw[reviewarrow] (enc)--(s2);', r'\node[below=4mm of c,text=Ink,font=\footnotesize] {Repeated for sampled clients; preserve the local-delta sign and units.};']
    elif name == 'ef':
        body = [r'\node[reviewbox] (a) {Correct input\\$v=u+e$};', r'\node[reviewbox,right=of a] (b) {Codec + decode\\$c=C(v)$};', r'\node[reviewbox,right=of b] (c) {Store residual\\$e_{\mathrm{next}}=v-c$};', r'\draw[reviewarrow] (a)--(b);\draw[reviewarrow] (b)--(c);', r'\draw[reviewarrow] (c.south) -- ++(0,-7mm) -| (a.south);', r'\node[below=12mm of b,text=Ink,font=\footnotesize] {Residual retained locally for the next client update};']
    elif name == 'learned':
        body = [r'\node[reviewbox] (a) {Client encoder\\local calibration};', r'\node[reviewbox,right=of a] (b) {Latent payload\\declared precision};', r'\node[reviewbox,right=of b] (c) {Server decoder\\synchronized version};', r'\draw[reviewarrow] (a)--(b);\draw[reviewarrow] (b)--(c);', r'\node[below=4mm of b,text=Ink,font=\footnotesize] {Decoder distribution and later refresh are additional transfers.};']
    elif name == 'adaptive':
        common = [r'\begin{figure}[htbp]\centering', r'\begin{tikzpicture}[node distance=8mm and 8mm]']
        body = [r'\node[reviewbox] (u) {Delta $u$};', r'\node[reviewbox,right=of u] (q) {One QSGD draw $z$};', r'\node[reviewbox,right=of q,yshift=13mm] (p0) {$p=0$};', r'\node[reviewbox,right=of q] (p1) {$p=1$};', r'\node[reviewbox,right=of q,yshift=-13mm] (p2) {$p=2$};', r'\node[reviewbox,right=17mm of p1] (measure) {Local decode\\error + bytes};', r'\node[reviewbox,right=of measure] (choose) {Choose feasible\\minimum bytes};', r'\draw[reviewarrow] (u)--(q);', r'\draw[reviewarrow] (q.east) -- ++(4mm,0) |- (p0.west);', r'\draw[reviewarrow] (q)--(p1);', r'\draw[reviewarrow] (q.east) -- ++(4mm,0) |- (p2.west);', r'\draw[reviewarrow] (p0.east) -- ++(4mm,0) |- (measure.west);', r'\draw[reviewarrow] (p1)--(measure);', r'\draw[reviewarrow] (p2.east) -- ++(4mm,0) |- (measure.west);', r'\draw[reviewarrow] (measure)--(choose);', r'\node[above=4mm of q,text=Teal,font=\small\bfseries] {Fixed first stage; adaptive second-stage tolerance};']
    else:
        raise KeyError(name)
    return '\n'.join(common + body + [r'\end{tikzpicture}', r'\caption{' + DIAGRAM_CAPTIONS[name] + '}', r'\label{fig:' + name + '}', r'\end{figure}'])

FIGURE_CAPTION = ('Completed WSL DigiFace seed-42 development batch. Left: selected validation accuracy versus training-upload MB (log x-axis). Right: serialized Flower-object totals split into training uploads and all other exchanges. No error bars are inferred from one seed; test-set evaluation was disabled.')

def render_block(block):
    block = block.strip()
    if not block:
        return ''
    if block.startswith('{{table:'):
        return table_tex(block[len('{{table:'):-2])
    if block.startswith('{{diagram:'):
        return diagram_tex(block[len('{{diagram:'):-2])
    if block == '{{figure:pilot}}':
        return '\n'.join([r'\begin{figure}[htbp]\centering', r'\includegraphics[width=0.97\textwidth]{assets/pilot-communication.png}', r'\caption{' + FIGURE_CAPTION + '}', r'\label{fig:pilot}', r'\end{figure}'])
    if block.startswith('{{equation:'):
        return equation(block[len('{{equation:'):-2])
    if block.startswith('{{callout:'):
        title, body = block[len('{{callout:'):-2].split(' | ', 1)
        return r'\begin{reviewcallout}\textbf{' + inline(title) + r'}\\' + inline(body) + r'\end{reviewcallout}'
    if block.startswith('# '):
        heading = block[2:]
        match = re.match(r'^\d+\s*\|\s*(.*)$', heading)
        if match:
            return r'\section{' + inline(match.group(1)) + '}'
        if heading == 'Compression for Federated Learning':
            return ''
        raise ValueError('Unexpected top-level heading in body: ' + heading)
    if block == '## Abstract':
        return r'\begin{abstract}'
    if block.startswith('## '):
        heading = block[3:]
        # The subtitle is already included in the title block.
        if heading.startswith('Quantization, sparsification, learned coding'):
            return ''
        return r'\subsection*{' + inline(heading) + '}'
    if block.startswith('- '):
        items = [line[2:].strip() for line in block.splitlines() if line.strip()]
        if not all(line.startswith('- ') for line in block.splitlines() if line.strip()):
            raise ValueError('Mixed list block: ' + block[:60])
        return r'\begin{itemize}' + ''.join(r'\item ' + inline(item) for item in items) + r'\end{itemize}'
    # The title metadata is represented by \author and \date in the preamble.
    if block.startswith('Literature review and research positioning | Updated '):
        return ''
    return inline(' '.join(line.strip() for line in block.splitlines()))

manuscript = SOURCE.read_text(encoding='utf-8')
blocks = re.split(r'\n\s*\n', manuscript.replace('}}\n{{', '}}\n\n{{').strip())
rendered = []
for block in blocks:
    part = render_block(block)
    if part:
        rendered.append(part)
# Close the abstract before the generated contents page.
body = '\n\n'.join(rendered)
body = body.replace(r'\begin{abstract}', r'\begin{abstract}', 1)
# Abstract is a single prose block; the TOC token immediately follows it.
body = re.sub(r'(\\begin\{abstract\})(.*?)(\\tableofcontents)', r'\1\2\\end{abstract}\n\n\3', body, count=1, flags=re.S)
if r'\end{abstract}' not in body:
    raise RuntimeError('Could not close abstract before contents')
# Replace the original first two title blocks with the formal title page.
# The title block is generated from metadata; remove the source title/subtitle blocks.
body = re.sub(r'^Compression for Federated Learning\s*\n\n', '', body)
# Ensure all special placeholders were consumed and citations resolve.
if '{{' in body:
    raise RuntimeError('Unexpanded template marker remains: ' + repr(re.findall(r'.{0,30}\}\}.{0,60}', body)))
keys = set(re.findall(r'\\parencite\{([^}]+)\}', body))
bibtext = BIB.read_text(encoding='utf-8')
bibkeys = set(re.findall(r'@\w+\s*\{\s*([^,\s]+)', bibtext))
used = {key for group in keys for key in group.split(',')}
missing = used - bibkeys
if missing:
    raise RuntimeError(f'Missing bibliography keys: {sorted(missing)}')

# Check source template markers were all mapped, and reproduce the associated figure from the same frozen pilot JSON.
expected = set(re.findall(r'\{\{(?:table|diagram|equation|callout|figure):?([^}]*)\}\}', manuscript))
for key in ('contents','accounting','quantization','costs','ae','llz-example','adaptive','recent','pilot','evidence'):
    if key not in specs and key != 'contents':
        raise RuntimeError(f'Missing table specification: {key}')
chart = OUT / 'assets/pilot-communication.png'
build_review.pilot_figure(summary, chart)
shutil.copy2(BIB, OUT / 'references.bib')
(OUT / 'review.tex').write_text(PREAMBLE + '\n' + body + '\n\n\\printbibliography\n\\end{document}\n', encoding='utf-8')
(OUT / 'README.md').write_text('''# LaTeX source for the literature review\n\nThis folder contains a LaTeX conversion of the November 2, 2026 review. It is generated from `../review.md`, the project BibTeX bibliography, and the saved seed-42 development summary. The manuscript wording and evidence cutoff are preserved.\n\n## Files\n\n- `review.tex`: complete article source, tables, equations and TikZ diagrams.\n- `references.bib`: bibliography used by the review.\n- `assets/pilot-communication.png`: chart generated from the saved development summary.\n- `../build_latex.py`: rebuilds this folder from project sources.\n\n## Compile\n\nUse XeLaTeX and Biber (Overleaf: select XeLaTeX, then compile):\n\n```sh\nxelatex review.tex\nbiber review\nxelatex review.tex\nxelatex review.tex\n```\n\nThe project WSL image does not currently include a TeX engine, so this environment generated and statically checked the source but did not compile a second PDF. The existing PDF remains the visually reviewed version.\n''', encoding='utf-8')
print(f'Created {OUT / "review.tex"}')
print(f'Used bibliography entries: {len(used)}; defined entries: {len(bibkeys)}')
print(f'Template markers expanded; source chars={len(body)}; chart={chart.stat().st_size} bytes')
