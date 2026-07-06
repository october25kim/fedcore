#!/usr/bin/env python3
"""Post-process the pandoc-generated docx into Information Sciences (Elsevier)
manuscript style: Times New Roman 12pt, double line spacing, continuous line
numbering, centered page numbers, black headings, centered title block.

Run after pandoc (see build_docx.sh). Idempotent.
"""
import sys
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH as AL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

P = sys.argv[1] if len(sys.argv) > 1 else "Fed-CORE_draft.docx"
d = Document(P)

# 1) document default font = Times New Roman 12pt
styles_el = d.styles.element
dd = styles_el.find(qn('w:docDefaults'))
if dd is None:
    dd = OxmlElement('w:docDefaults'); styles_el.insert(0, dd)
rp = dd.find(qn('w:rPrDefault'))
if rp is None:
    rp = OxmlElement('w:rPrDefault'); dd.append(rp)
rPrDef = rp.find(qn('w:rPr'))
if rPrDef is None:
    rPrDef = OxmlElement('w:rPr'); rp.append(rPrDef)
rf = rPrDef.find(qn('w:rFonts'))
if rf is None:
    rf = OxmlElement('w:rFonts'); rPrDef.insert(0, rf)
for a in ('w:ascii', 'w:hAnsi', 'w:cs'):
    rf.set(qn(a), 'Times New Roman')
sz = rPrDef.find(qn('w:sz'))
if sz is None:
    sz = OxmlElement('w:sz'); rPrDef.append(sz)
sz.set(qn('w:val'), '24')  # 12pt


def set_style(nm, size=None, bold=None):
    try:
        st = d.styles[nm]
    except KeyError:
        return
    st.font.name = 'Times New Roman'
    if size:
        st.font.size = Pt(size)
    if bold is not None:
        st.font.bold = bold


for nm in ('Normal', 'Body Text', 'First Paragraph', 'Compact', 'Footer', 'Header'):
    set_style(nm, 12)
for nm, szp in (('Heading 1', 14), ('Heading 2', 13), ('Heading 3', 12), ('Title', 16)):
    set_style(nm, szp, True)

# 2) double line spacing on body
for nm in ('Normal', 'Body Text'):
    try:
        pf = d.styles[nm].paragraph_format
        pf.line_spacing = 2.0
        pf.space_after = Pt(0)
    except KeyError:
        pass

# 3) black headings (strip theme color at style and run level)
for nm in ('Heading 1', 'Heading 2', 'Heading 3', 'Title'):
    try:
        rpr = d.styles[nm].element.get_or_add_rPr()
        for c in rpr.findall(qn('w:color')):
            rpr.remove(c)
        col = OxmlElement('w:color'); col.set(qn('w:val'), '000000'); rpr.append(col)
    except Exception:
        pass


def force_black(run):
    rpr = run._r.get_or_add_rPr()
    for c in rpr.findall(qn('w:color')):
        rpr.remove(c)
    col = OxmlElement('w:color'); col.set(qn('w:val'), '000000'); rpr.append(col)


# 4) center the title block + black heading runs
prefixes = ('Fed-CORE', 'Sanghoon', '[Department', 'Corresponding', 'E-mail')
for i, p in enumerate(d.paragraphs):
    sn = (p.style.name or '')
    if sn.startswith('Heading') or sn == 'Title':
        for r in p.runs:
            force_black(r)
    if i < 8 and (p.text or '').strip().startswith(prefixes):
        p.alignment = AL.CENTER

# 5) qFormat must precede pPr/rPr inside each w:style
pre = (qn('w:pPr'), qn('w:rPr'))
for st in styles_el.findall(qn('w:style')):
    q = st.find(qn('w:qFormat'))
    if q is None:
        continue
    first_pr = next((c for c in list(st) if c.tag in pre), None)
    if first_pr is not None and list(st).index(q) > list(st).index(first_pr):
        st.remove(q); first_pr.addprevious(q)

# 6) continuous line numbering + centered page-number footer
for sec in d.sections:
    sectPr = sec._sectPr
    if sectPr.find(qn('w:lnNumType')) is None:
        ln = OxmlElement('w:lnNumType')
        ln.set(qn('w:countBy'), '1'); ln.set(qn('w:restart'), 'continuous'); ln.set(qn('w:distance'), '454')
        cols = sectPr.find(qn('w:cols'))
        (cols.addprevious(ln) if cols is not None else sectPr.append(ln))
    sec.different_first_page_header_footer = False
    f = sec.footer; f.is_linked_to_previous = False
    fp = f.paragraphs[0] if f.paragraphs else f.add_paragraph()
    fp.text = ''; fp.alignment = AL.CENTER
    fld = OxmlElement('w:fldSimple'); fld.set(qn('w:instr'), ' PAGE ')
    r = OxmlElement('w:r'); tt = OxmlElement('w:t'); tt.text = '1'; r.append(tt); fld.append(r)
    fp._p.append(fld)

# 7) single-space the reference entries (paragraphs starting with "[n] ")
import re as _re
for p in d.paragraphs:
    if _re.match(r'^\[\d+\]\s', (p.text or '')):
        pf = p.paragraph_format
        pf.line_spacing = 1.0
        pf.space_after = Pt(6)

d.save(P)
print(f"[ins_format] applied Information Sciences style to {P}")
