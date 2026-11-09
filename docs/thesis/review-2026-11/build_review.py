"""Build the November literature review from Markdown, BibTeX and frozen results.

No network, training, dataset access or application-code changes are required.
Usage: .venv/bin/python docs/thesis/review-2026-11/build_review.py
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, KeepTogether, Image,
)
from reportlab.graphics.shapes import Drawing, Rect, Line, Polygon, String
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).with_name('review.md')
BIB = ROOT / 'docs/thesis/references.bib'
PILOT = ROOT / 'outputs/adaptive-dev-seed42-wsl-20261102'
AE = ROOT / 'outputs/ae-feasibility-20261006T091559343464Z'
PAGE_W, PAGE_H = A4
WIDTH = PAGE_W - 96
INK = colors.HexColor('#182D3B')
TEAL = colors.HexColor('#006C73')
LIGHT = colors.HexColor('#EAF3F3')
GRAY = colors.HexColor('#526370')
BORDER = colors.HexColor('#D2DFE4')


def read_bib(path):
    text = path.read_text(encoding='utf-8')
    entries = {}
    for m in re.finditer(r'@(\w+)\s*\{\s*([^,\s]+)\s*,', text):
        at = m.end(); start = at; depth = 1
        while depth and at < len(text):
            if text[at] == '\\': at += 2; continue
            depth += (text[at] == '{') - (text[at] == '}'); at += 1
        if depth: raise ValueError('Unbalanced bibliography entry')
        body = text[start:at-1]; fields = {}; at = 0
        while at < len(body):
            at += re.match(r'[\s,]*', body[at:]).end()
            if at == len(body): break
            fm = re.match(r'(\w+)\s*=\s*\{', body[at:])
            if not fm: raise ValueError('Unsupported bibliography syntax')
            name = fm.group(1).lower(); at += fm.end(); begin = at; depth = 1
            while depth and at < len(body):
                if body[at] == '\\': at += 2; continue
                depth += (body[at] == '{') - (body[at] == '}'); at += 1
            fields[name] = body[begin:at-1]
        if m.group(2) in entries: raise ValueError('Duplicate bibliography key')
        entries[m.group(2)] = fields
    return entries


def plain_tex(text):
    # The bibliography uses a small, checked subset of TeX name accents.
    for a, b in [(r'\"u', 'ü'), (r'\"o', 'ö'), (r'\"a', 'ä'),
                 (r'\v{s}', 'š'), (r'\v{S}', 'Š')]:
        text = text.replace(a, b)
    return text.replace('{', '').replace('}', '')


def setup_fonts():
    base = Path('/usr/share/fonts/truetype/dejavu')
    fonts = {'Body': 'DejaVuSerif.ttf', 'BodyBold': 'DejaVuSerif-Bold.ttf',
             'Sans': 'DejaVuSans.ttf', 'SansBold': 'DejaVuSans-Bold.ttf',
             'Mono': 'DejaVuSansMono.ttf'}
    for name, filename in fonts.items():
        pdfmetrics.registerFont(TTFont(name, str(base / filename)))
    pdfmetrics.registerFontFamily('Body', normal='Body', bold='BodyBold', italic='Body', boldItalic='BodyBold')
    pdfmetrics.registerFontFamily('Sans', normal='Sans', bold='SansBold', italic='Sans', boldItalic='SansBold')


def make_styles():
    return {
        'body': ParagraphStyle('body', fontName='Body', fontSize=9.6, leading=13.9,
                               textColor=INK, spaceAfter=8.5, allowWidows=0, allowOrphans=0),
        'title': ParagraphStyle('title', fontName='SansBold', fontSize=27, leading=32,
                                textColor=INK, spaceAfter=14),
        'h1': ParagraphStyle('h1', fontName='SansBold', fontSize=18.5, leading=23,
                             textColor=INK, spaceAfter=13),
        'h2': ParagraphStyle('h2', fontName='SansBold', fontSize=11.2, leading=15,
                             textColor=TEAL, spaceBefore=5, spaceAfter=7, keepWithNext=True),
        'table': ParagraphStyle('table', fontName='Sans', fontSize=8.0, leading=10.7, textColor=INK),
        'tablehead': ParagraphStyle('tablehead', fontName='SansBold', fontSize=8.0, leading=10.5, textColor=colors.white),
        'caption': ParagraphStyle('caption', fontName='Sans', fontSize=8.1, leading=11,
                                  textColor=GRAY, spaceBefore=5, spaceAfter=10),
        'equation': ParagraphStyle('equation', fontName='Mono', fontSize=8.8, leading=13,
                                   textColor=TEAL, backColor=LIGHT, borderPadding=7,
                                   spaceBefore=5, spaceAfter=13),
        'callout': ParagraphStyle('callout', fontName='Sans', fontSize=9.0, leading=13.2,
                                  textColor=INK, backColor=LIGHT, borderPadding=10,
                                  spaceBefore=7, spaceAfter=17),
        'reference': ParagraphStyle('reference', fontName='Sans', fontSize=8.0, leading=11.2,
                                    textColor=INK, spaceAfter=10),
    }


def citation_keys(text):
    result = []
    for match in re.finditer(r'\[@([^\]]+)\]', text):
        for part in match.group(1).split(';'):
            key = part.strip().lstrip('@')
            if key not in result: result.append(key)
    return result


def inline(text, numbers):
    # Escape prose before adding the few supported markup constructs.
    result = []; start = 0
    for match in re.finditer(r'\[@([^\]]+)\]', text):
        result.append(html.escape(text[start:match.start()]))
        keys = [x.strip().lstrip('@') for x in match.group(1).split(';')]
        result.append(', '.join(f'<link href="#ref{numbers[k]}" color="#006C73">[{numbers[k]}]</link>' for k in keys))
        start = match.end()
    result.append(html.escape(text[start:]))
    return ''.join(result)


def table_flow(headers, rows, fractions, caption, styles, numbers):
    data = [[Paragraph(inline(x, numbers), styles['tablehead']) for x in headers]]
    data += [[Paragraph(inline(str(x), numbers), styles['table']) for x in row] for row in rows]
    table = Table(data, colWidths=[WIDTH*x for x in fractions], repeatRows=1, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),TEAL), ('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
        ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F2F6F7')]),
        ('LINEBELOW',(0,-1),(-1,-1),0.6,BORDER),
    ]))
    return KeepTogether([table, Paragraph(inline(caption, numbers), styles['caption'])])


def label_box(d, x, y, w, h, lines, fill=LIGHT):
    d.add(Rect(x,y,w,h,rx=5,ry=5,fillColor=fill,strokeColor=BORDER,strokeWidth=.7))
    for i, line in enumerate(lines):
        d.add(String(x+w/2,y+h/2+(len(lines)-1)*5.5-i*11-3,line,
                     fontName='SansBold' if i==0 else 'Sans',fontSize=8.1,
                     fillColor=INK,textAnchor='middle'))


def arrow(d,x1,y1,x2,y2,color=TEAL):
    import math
    d.add(Line(x1,y1,x2,y2,strokeColor=color,strokeWidth=1))
    angle=math.atan2(y2-y1,x2-x1); length=5; side=2.6
    x=x2-length*math.cos(angle); y=y2-length*math.sin(angle)
    d.add(Polygon([x2,y2,x+side*math.sin(angle),y-side*math.cos(angle),
                   x-side*math.sin(angle),y+side*math.cos(angle)],fillColor=color,strokeColor=color))


def diagram(name, styles):
    d=Drawing(WIDTH,100)
    if name=='fl':
        boxes=[['Server','broadcast w_t'],['Client i','H local steps'],['Encode / decode','delta u_i,t'],['Server','weighted update']]
        for i, lines in enumerate(boxes):
            x=3+i*128
            label_box(d,x,36,108,43,lines)
            if i<3:arrow(d,x+111,57,x+124,57)
        d.add(String(WIDTH/2,15,'Repeated for sampled clients; preserve the local-delta sign and units.',fontName='Sans',fontSize=8,fillColor=GRAY,textAnchor='middle'))
        cap='Figure 1. Local training and compressed model-delta aggregation. Original schematic based on FedAvg; mathematical setting is defined in the text.'
    elif name=='ef':
        for x,lines in [(4,['Correct input','v = u + e']), (185,['Codec + decode','c = C(v)']), (366,['Store residual','e_next = v - c'])]:
            label_box(d,x,35,127,43,lines)
        arrow(d,134,57,180,57);arrow(d,315,57,361,57)
        d.add(Line(428,34,428,12,strokeColor=TEAL));d.add(Line(428,12,66,12,strokeColor=TEAL));arrow(d,66,12,66,33)
        d.add(String(247,1,'Residual retained locally for the next client update',fontName='Sans',fontSize=8,fillColor=GRAY,textAnchor='middle'))
        cap='Figure 2. Error feedback in model-delta units. Original schematic; residuals refer to the complete decoded compressor.'
    elif name=='learned':
        for x,lines in [(4,['Client encoder','local calibration']), (185,['Latent payload','declared precision']), (366,['Server decoder','synchronized version'])]:label_box(d,x,35,127,43,lines)
        arrow(d,134,57,180,57);arrow(d,315,57,361,57)
        d.add(String(WIDTH/2,13,'Decoder distribution and later refresh are additional transfers.',fontName='Sans',fontSize=8,fillColor=GRAY,textAnchor='middle'))
        cap='Figure 3. Learned update coding and the decoder dependency. Original schematic inspired by the AE literature; no learned-codec FL run is claimed.'
    elif name=='adaptive':
        d=Drawing(WIDTH,143)
        label_box(d,2,47,65,42,['Delta','u'])
        label_box(d,84,47,80,42,['One QSGD','draw z'])
        arrow(d,68,68,82,68)
        for y,p in [(99,0),(54,1),(9,2)]:
            label_box(d,190,y,57,29,[f'p = {p}'])
            arrow(d,165,68,188,y+14)
            arrow(d,249,y+14,278,68)
        label_box(d,281,40,93,57,['Local decode','error + bytes'])
        label_box(d,401,40,94,57,['Choose feasible','minimum bytes'])
        arrow(d,376,68,399,68)
        d.add(String(WIDTH/2,132,'Fixed first stage; adaptive second-stage tolerance',fontName='SansBold',fontSize=8.3,fillColor=TEAL,textAnchor='middle'))
        cap='Figure 4. Current adaptive selector. All p candidates share the same QSGD draw; only the selected complete reply is sent.'
    else:raise KeyError(name)
    return KeepTogether([d,Paragraph(cap,styles['caption'])])


def pilot_figure(summary, output):
    order=['qsgd','qsgd-llz-p0','qsgd-llz-p1','adaptive-tight']
    labels=['QSGD','LLZ p=0','LLZ p=1','Adaptive LLZ, tau=5.9']
    palette=['#24445B','#007C83','#92712F','#486EA7','#AA5353']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.3,'axes.labelsize':8.3,
                         'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,(ax,bx)=plt.subplots(1,2,figsize=(7.15,2.85),gridspec_kw={'width_ratios':[1.05,1.0]})
    offsets=[(-6,10),(0,11),(13,8),(4,-17),(13,-15)]
    for i,key in enumerate(order):
        v=summary['methods'][key];x=v['upload_bytes']/1e6;y=100*v['best_validation_accuracy']
        ax.scatter(x,y,color=palette[i],s=31,zorder=3)
        ax.annotate(labels[i],(x,y),xytext=offsets[i],textcoords='offset points',
                    ha='right' if i==0 else 'left',fontsize=7.0,color=palette[i])
    ax.set_xscale('log');ax.set_xlim(8,800);ax.set_ylim(77.0,85.0)
    ax.set_xticks([10,30,100,300]);ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel('Training-upload MB (log scale)');ax.set_ylabel('Selected validation accuracy (%)')
    ax.set_title('Learning versus uploads',loc='left',fontweight='bold',fontsize=9.1)
    ax.grid(alpha=.18)
    y=np.arange(len(order));upload=np.array([summary['methods'][k]['upload_bytes']/1e6 for k in order])
    total=np.array([summary['methods'][k]['total_bytes']/1e6 for k in order])
    bx.barh(y,upload,color=palette,height=.56,label='Training uploads')
    bx.barh(y,total-upload,left=upload,color='#D6DFE3',height=.56,label='Other exchanges')
    bx.set_yticks(y,labels,fontsize=7.1);bx.invert_yaxis();bx.set_xlim(0,4950)
    for j,v in enumerate(total):bx.text(v-70,j,f'{v:,.1f}',ha='right',va='center',fontsize=7.2,color='#182D3B')
    bx.set_xlabel('Total serialized MB');bx.set_title('Wider communication boundary',loc='left',fontweight='bold',fontsize=9.1)
    bx.legend(loc='upper center',bbox_to_anchor=(.50,-.30),ncol=2,frameon=False,fontsize=6.8,borderaxespad=0)
    fig.subplots_adjust(left=.09,right=.99,top=.89,bottom=.24,wspace=.57)
    fig.savefig(output,dpi=280,facecolor='white');plt.close(fig)


def table_specs(summary):
    methods=summary['methods']
    # Small worked source-code example; no training or experiment is started.
    from compression import llz_p
    source=np.array([2,0,2,0,3,1,2,0],dtype=np.int32)
    example=[];trace=[]
    for p in (0,1):
        triples=llz_p.parse(source,alphabet_bound=3,p=p,window_size=8)
        packet=llz_p.encode(source,alphabet_bound=3,p=p,window_size=8)
        decoded=llz_p.decode(packet)
        if np.max(np.abs(decoded-source))>p:raise ValueError('Worked LLZ example inconsistent')
        if p==0 and not np.array_equal(source,decoded):raise ValueError('Worked p=0 example inconsistent')
        example.append([f'p={p}', '; '.join(str(t) for t in triples), ' '.join(map(str,decoded))])
        trace.append(dict(p=p,source=source.tolist(),triplets=triples,decoded=decoded.tolist(),packet_bytes=len(packet)))
    specs={
      'contents':(['Research line','Representative methods'],[
        ['Quantization / local updates','QSGD, signSGD, FedPAQ'],
        ['Coordinate sparsity / feedback','Top-k, Rand-k, memory/EF, DGC'],
        ['Low-rank approximation','PowerSGD'],
        ['Learned representation','Autoencoder update coding'],
        ['Source / side-information coding','LLZ-p, LLZ-SI, KLMS, NNCodec'],
        ['Adaptive policies','AdaQuantFL, DAdaQuant, ASCEND']], [.43,.57],
        'Research map. Families can compose; EF and adaptation are cross-cutting mechanisms. Main review: pages 2-10; separate project evidence: 11-12; references: 13-14.'),
      'accounting':(['Boundary','What is counted'],[
        ['Codec packet','Compressed content plus codec header, shape and padding.'],
        ['Serialized Flower object','Codec plus records, metadata, routing and diagnostics.'],
        ['Whole experiment ledger','All declared uploads, downloads and evaluation exchanges.'],
        ['Physical network / energy','Requires transport/device measurement; absent from this pilot.']], [.30,.70],
        'Table 1. Communication boundaries used throughout this review; each saving needs its stated boundary.'),
      'quantization':(['Method','Numerical object','Key limitation'],[
        ['QSGD','Stochastic levels and norm','Expected error; representation must be specified. [@alistarh2017qsgd]'],
        ['signSGD','Signs / majority update','Changes the optimization rule. [@bernstein2018signsgd]'],
        ['FedPAQ','Quantized local differences','IID/unbiased-compressor theory is conditional. [@reisizadeh2020fedpaq]']], [.19,.32,.49],
        'Table 2. Quantization-related methods and why their guarantees are not interchangeable.'),
      'costs':(['Representation','Declared content / packet cost','Important omission'],[
        ['Dense FP32','32d bits','Protocol framing'],['k coordinates','k(32 + ceil(log2 d)) bits','Padding / synchronization'],
        ['Project Top-k','12 + 8k bytes','Flower layout/state records'],
        ['Rank-r n by m factors','32r(n+m) bits','Other tensors / collectives'],
        ['Project example: d=4096, k=64','Top-k 524 bytes; dense 16,384 bytes','Not a learning benchmark']], [.25,.43,.32],
        'Table 3. Illustrative accounting; formats and numerical units are explicit. Sparse and low-rank context: [@stich2018memory; @vogels2019powersgd].'),
      'ae':(['Offline quantity','Block AE','Top-k, no EF'],[
        ['Constructed bytes / update','924,310','992,594'],['Pooled relative squared error','0.665316','0.264914'],
        ['Mean full-update cosine','0.585770','0.861918'],['Projected 60 uploads + decoder setup','55,732,156 bytes','59,555,640 bytes']], [.48,.26,.26],
        'Table 4. Existing 6 October offline assessment, not a new run. Four AE decoder artifacts total 273,556 bytes; projection excludes Flower/downlink/control traffic.'),
      'llz-example':(['Tolerance','Triplets: (position, length, literal)','Decoded symbols'],example,[.14,.56,.30],
        'Table 5. Original worked example: source [2, 0, 2, 0, 3, 1, 2, 0], alphabet bound 3 and window 8. Exact tuples come from the current project parser; packet sizes are deliberately not used to claim compression on this tiny vector.'),
      'adaptive':(['Method','Control / objective','Shared or local state'],[
        ['AdaQuantFL','Precision across training; bound/bit budget','Loss / initial precision'],
        ['DAdaQuant','Time/client levels; variance proxy','Loss history / weights'],
        ['KLMS','Side-information sample coding','Distributions / seeds / blocks'],
        ['ASCEND','Candidate method; learning/time reward','Bandit utility / residuals'],
        ['Our LLZ selector','p after QSGD; bytes/error constraint','Locally decoded candidates']], [.23,.47,.30],
        'Table 6. Different meanings of adaptation. The first three methods are discussed below; ASCEND is examined on page 9. Sources: [@jhunjhunwala2021adaquantfl; @honig2022dadaquant; @isik2024sideinformation; @liu2026ascend].'),
      'recent':(['2025-26 candidate','Read depth','Relevance / status'],[
        ['IEEE LLZ, 2026','Existing full text reused','Direct coding anchor; Early Access.'],
        ['ComParFreFL, 2026','Targeted full text','Optimizer/feedback theory; ICML.'],
        ['Discrepancy-aware calibration','ICLR 2026 full text; follow-up 2 Nov','Closest task-sensitive alternative; calibration-set output discrepancy. [@wang2026discrepancy]'],
        ['EA-AQF, 2026','Metadata / abstract','Quantization and freezing; journal.'],
        ['ASCEND, 2026','Targeted full text','Candidate selection; preprint.'],
        ['NNCodec FL, 2025','Two-page full text','Practical update coding; poster report.'],
        ['PAC-MCoFL, 2025','Metadata / abstract','Joint resource/assignment control; preprint.']], [.39,.24,.37],
        'Table 7. Seven screened sources; the closest discrepancy-aware candidate received a dated full-text follow-up on 2 November. Publication status and reading depth are distinguished.'),
      'pilot':(['Method','Val. accuracy','Train upload MB','Total MB'],[
        [label, f"{100*methods[key]['best_validation_accuracy']:.2f}%", f"{methods[key]['upload_bytes']/1e6:.3f}", f"{methods[key]['total_bytes']/1e6:.3f}"]
        for key,label in [('qsgd','QSGD'),('qsgd-llz-p0','QSGD + LLZ p=0'),('qsgd-llz-p1','QSGD + LLZ p=1'),('adaptive-tight','Adaptive LLZ, tau=5.9')]], [.40,.18,.22,.20],
        'Table 8. Completed WSL seed-42 development batch; checkpoint selected by validation loss. MB = 1,000,000 bytes. Uploads and total serialized Flower objects are distinct accounting boundaries; no physical network traffic is claimed.'),
      'evidence':(['Established in this review','Still open / not established'],[
        ['p=0 replay and saved pilot accounting','Learning-safe budget or statistical equivalence'],
        ['Finite feasible selector and complete-reply costs','Optimal aggregate traffic across changing trajectories'],
        ['Published compression/EF precedents','A theorem for our actual no-EF local epochs'],
        ['AE offline feasibility evidence','AE identification accuracy in FL'],
        ['Seven-candidate recent search','An exhaustive novelty or priority conclusion']], [.49,.51],
        'Table 9. Evidence boundaries for the thesis. Proposed extensions are not relabeled as completed results.'),
    }
    return specs,trace


class ReviewDoc(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(str(filename),pagesize=A4,leftMargin=48,rightMargin=48,
                         topMargin=54,bottomMargin=54,title='Compression for Federated Learning',
                         author='Cao - thesis research preparation',
                         subject='Literature review and adaptive LLZ research positioning')
        self.section_pages=[]
        frame=Frame(48,54,WIDTH,PAGE_H-110,id='review',leftPadding=0,rightPadding=0,
                    topPadding=0,bottomPadding=0)
        self.addPageTemplates(PageTemplate(id='review',frames=frame,onPage=self.furniture))

    def furniture(self, canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(BORDER);canvas.setLineWidth(.6)
        canvas.line(48,PAGE_H-37,PAGE_W-48,PAGE_H-37)
        canvas.setFont('SansBold',7.2);canvas.setFillColor(TEAL)
        canvas.drawString(48,PAGE_H-28,'THESIS LITERATURE REVIEW')
        canvas.setFont('Sans',7.2);canvas.setFillColor(GRAY)
        canvas.drawRightString(PAGE_W-48,PAGE_H-28,'Search cutoff: 29 Oct; follow-up: 2 Nov 2026')
        canvas.line(48,41,PAGE_W-48,41)
        canvas.drawString(48,28,'Compression for FL | literature, evidence and adaptive LLZ')
        canvas.setFont('SansBold',8);canvas.drawRightString(PAGE_W-48,28,f'{doc.page:02d}')
        canvas.restoreState()

    def afterFlowable(self, flowable):
        section=getattr(flowable,'section_id',None)
        if section is not None:
            self.canv.bookmarkPage('section'+str(section))
            self.canv.addOutlineEntry(flowable.getPlainText(),'section'+str(section),level=0,closed=False)
            self.section_pages.append(dict(section=section,page=self.page,title=flowable.getPlainText()))


def add_references(story, keys, entries, styles, numbers):
    for chunk_start in range(0,len(keys),12):
        story.append(PageBreak())
        heading=Paragraph('References' if chunk_start==0 else 'References | continued',styles['h1'])
        heading.section_id='refs'+str(chunk_start);story.append(heading)
        if chunk_start==0:
            story.append(Paragraph('Numbering follows first citation. Primary sources and their checked publication/version status are listed below. Abstract-only screening is identified explicitly; the 2 November full-text follow-up is logged in the source notes.',styles['caption']))
        for key in keys[chunk_start:chunk_start+12]:
            fields=entries[key];n=numbers[key]
            authors=plain_tex(fields['author']).replace(' and others',' et al.').replace(' and ','; ')
            title=plain_tex(fields['title']);venue=plain_tex(fields.get('journal',fields.get('booktitle','Author preprint')))
            vol=fields.get('volume');pages=fields.get('pages')
            details=venue+(', vol. '+vol if vol else '')+(', pp. '+pages.replace('--','-') if pages else '')+'. '+fields['year']+'.'
            note=''
            if key=='cao2026llz':note='IEEE Early Access, 21 October; accepted author version, not fully edited.'
            elif key=='cao2026llzmanuscript':note='Historical user-supplied revision, 2 June 2026; retained as implementation provenance.'
            elif key=='liu2026ascend':note='arXiv v1, 16 August 2026; preprint. Proof-report reference unresolved in the inspected version.'
            elif key=='wang2026discrepancy':note='Published as an ICLR 2026 conference paper; final OpenReview PDF checked 2 November 2026.'
            elif key=='becking2025tinyfl':note='arXiv v1; authors report EuCNC & 6G Summit poster acceptance; two-page paper.'
            elif key=='ikram2026eaaqf':note='Article 694; updated publisher metadata/abstract only were screened.'
            elif key=='tan2025pacmcofl':note='arXiv v2, 28 August 2025; metadata/abstract only; archival venue unverified.'
            elif key=='kairouz2021advances':note='Cited author preprint v3, 9 March 2021; first submitted 2019.'
            if fields.get('url'):
                url=fields['url']
                label=fields.get('doi') or ('arXiv:'+fields['eprint'] if fields.get('eprint') else 'Primary source')
                source=f'<link href="{html.escape(url,quote=True)}" color="#006C73">{html.escape(label)}</link>'
            else:source='Local historical source; provenance in bibliography.'
            paragraph=(f'<a name="ref{n}"/><b>[{n}]</b> '+html.escape(authors)+'. '
                       +'<b>'+html.escape(title)+'</b>. '+html.escape(details)+' '
                       +html.escape(note)+' '+source)
            story.append(Paragraph(paragraph,styles['reference']))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(output_dir):
    setup_fonts();styles=make_styles()
    manuscript=SOURCE.read_text(encoding='utf-8')
    entries=read_bib(BIB);keys=citation_keys(manuscript)
    numbers={key:i+1 for i,key in enumerate(keys)}
    unknown=set(keys)-set(entries)
    if unknown:raise ValueError('Unknown citations: '+str(unknown))
    output_dir=Path(output_dir);final_dir=output_dir/'output/pdf';assets=output_dir/'assets'
    final_dir.mkdir(parents=True,exist_ok=True);assets.mkdir(parents=True,exist_ok=True)
    summary=json.loads((PILOT/'summary.json').read_text())
    protocol=json.loads((PILOT/'protocol.json').read_text())
    study=json.loads((PILOT/'study.json').read_text())
    if summary['status']!='complete' or study['status']!='completed':raise ValueError('Development study is not complete')
    expected={'qsgd','qsgd-llz-p0','qsgd-llz-p1','adaptive-tight'}
    if set(summary['methods'])!=expected:raise ValueError('Unexpected method set in saved summary')
    assert study['runs'][0]['config']['seed']==42 and study['runs'][0]['config']['num-server-rounds']==15
    assert study['runs'][0]['config']['evaluate-final-test'] is False
    chart=assets/'pilot-communication.png';pilot_figure(summary,chart)
    specs,trace=table_specs(summary)
    pdf=final_dir/'compression-for-federated-learning-review.pdf'
    story=[];page_index=0
    blocks=re.split(r'\n\s*\n',manuscript.replace('}}\n{{', '}}\n\n{{').strip())
    for block in blocks:
        block=block.strip()
        if block.startswith('# '):
            if page_index:story.append(PageBreak())
            heading=Paragraph(inline(block[2:],numbers),styles['title' if page_index==0 else 'h1'])
            heading.section_id=page_index+1;story.append(heading);page_index+=1
        elif block.startswith('## '):
            story.append(Paragraph(inline(block[3:],numbers),styles['h2']))
        elif block.startswith('{{table:'):
            key=block[len('{{table:'):-2]
            story.append(table_flow(*specs[key],styles,numbers))
        elif block.startswith('{{diagram:'):
            story.append(diagram(block[len('{{diagram:'):-2],styles))
        elif block=='{{figure:pilot}}':
            story.append(KeepTogether([Image(str(chart),width=WIDTH,height=WIDTH*2.85/7.15),
                Paragraph('Figure 5. Completed WSL DigiFace seed-42 development batch. Left: selected validation accuracy versus training-upload MB (log x-axis). Right: serialized Flower-object totals split into training uploads and all other exchanges. No error bars are inferred from one seed; test-set evaluation was disabled.',styles['caption'])]))
        elif block.startswith('{{equation:'):
            equation=block[len('{{equation:'):-2]
            if pdfmetrics.stringWidth(equation,'Mono',8.8)>WIDTH-14:
                raise ValueError('Equation exceeds line width: '+equation)
            story.append(Paragraph(html.escape(equation),styles['equation']))
        elif block.startswith('{{callout:'):
            parts=block[len('{{callout:'):-2].split(' | ',1)
            story.append(Paragraph('<b>'+html.escape(parts[0])+'</b><br/>'+html.escape(parts[1]),styles['callout']))
        else:
            story.append(Paragraph(inline(block.replace('\n',' '),numbers),styles['body']))
    # Table-only citations are checked and numbered too; no silent dangling keys.
    table_keys=[]
    for spec in specs.values():
        for key in citation_keys(json.dumps(spec)):
            if key not in numbers:table_keys.append(key)
    if table_keys:raise ValueError('Table citations must also occur in prose: '+str(table_keys))
    add_references(story,keys,entries,styles,numbers)
    doc=ReviewDoc(pdf);doc.build(story)
    reader=PdfReader(pdf)
    expected_main=list(range(1,page_index+1))
    actual_main=[x['page'] for x in doc.section_pages if isinstance(x['section'],int)]
    if expected_main!=actual_main:
        raise ValueError('Unplanned main-page overflow; inspect layout: '+str(doc.section_pages))
    if len(reader.pages)!=14:raise ValueError('Expected 12 main pages + 2 reference pages; got '+str(len(reader.pages)))
    extracted='\n'.join(page.extract_text() or '' for page in reader.pages)
    if '[@' in extracted or '{{' in extracted or '\ufffd' in extracted:
        raise ValueError('Unresolved citation/template or replacement glyph')
    for number in numbers.values():
        if f'[{number}]' not in extracted:raise ValueError('Missing reference '+str(number))
    sources=[SOURCE,Path(__file__),BIB,PILOT/'summary.json',PILOT/'protocol.json',PILOT/'study.json',AE/'decision.md',
             ROOT/'outputs/literature-blocks-20261029/source-manifest.json']
    manifest=dict(created_date='2026-11-02',review_cutoff='2026-11-02 targeted follow-up',pages=len(reader.pages),
                  sections=doc.section_pages,citation_numbers=numbers,
                  inputs=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sources],
                  pdf=dict(path=str(pdf.relative_to(ROOT)),sha256=sha(pdf)),
                  llz_worked_example=trace,
                  evidence='Completed WSL seed-42 development batch; summary, protocol and study records hashed; no test-set evaluation',
                  limitations=['One seed only; no statistical-equivalence claim','Manifest digest changed after WSL relocation; exact input-file byte identity is not asserted',
                               'Tau remains a historical training-only heuristic, not a learning-safe bound','ASCEND proof-report reference unresolved',
                               'No convergence theorem for the actual local-epoch selector'])
    (output_dir/'build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(pdf=str(pdf),pages=len(reader.pages),citations=len(keys),
                          sections=doc.section_pages,llz_example=trace),indent=2))
    return pdf


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/literature-review-20261102')
    args=parser.parse_args()
    build(args.output_dir)
