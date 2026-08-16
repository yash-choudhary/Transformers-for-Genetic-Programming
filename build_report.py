"""Insert Chapter 8 and the figures into the capstone report.

Writes a NEW file; the source document is never modified. Content is placed at
named anchors rather than appended, because a results chapter that arrives after
the conclusion, and figures that arrive after the chapter discussing them, are
both things a reader notices.
"""
import copy
import os
import sys

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

SRC = r"C:\Users\yc199\Downloads\MIS41170-Capstone.docx"
DST = sys.argv[1] if len(sys.argv) > 1 else \
    r"D:\MSBA\Capstone\TSGP\MIS41170-Capstone-with-Ch8.docx"
FIG = "figures"

GREY = RGBColor(0x5F, 0x63, 0x68)
doc = Document(SRC)


# ------------------------------------------------------------------ anchors
def find_heading(text, level=None):
    """First paragraph whose text starts with `text` and is a heading."""
    for p in doc.paragraphs:
        if not p.style.name.startswith("Heading"):
            continue
        if level and not p.style.name.endswith(str(level)):
            continue
        if p.text.strip().lower().startswith(text.lower()):
            return p
    raise LookupError(f"anchor not found: {text!r}")


def para_before(anchor, text="", *, bold=False, italic=False, size=10.5,
                color=None, style=None, align=None):
    p = anchor.insert_paragraph_before("", style=style)
    if align is not None:
        p.alignment = align
    if text:
        r = p.add_run(text)
        r.bold, r.italic = bold, italic
        r.font.size = Pt(size)
        if color:
            r.font.color.rgb = color
    return p


def rich_before(anchor, parts, size=10.5):
    p = anchor.insert_paragraph_before("")
    for t, b in parts:
        r = p.add_run(t)
        r.bold = b
        r.font.size = Pt(size)
    return p


def heading_before(anchor, text, level):
    """Insert a heading using the document's own heading styles, so it picks up
    the report's existing numbering, fonts and table-of-contents behaviour."""
    p = anchor.insert_paragraph_before("", style=f"Heading {level}")
    p.add_run(text)
    return p


def figure_before(anchor, filename, caption_text, width=6.2):
    path = os.path.join(FIG, filename)
    if not os.path.exists(path):
        print(f"  !! missing figure {path}")
        return
    p = anchor.insert_paragraph_before("")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(path, width=Inches(width))
    c = anchor.insert_paragraph_before("", style=CAP_STYLE)
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = c.add_run(caption_text)
    if not CAP_STYLE:
        r.italic = True
        r.font.size = Pt(9)
        r.font.color.rgb = GREY
    print(f"  + {filename}")


def shade(cell, fill):
    el = OxmlElement("w:shd")
    el.set(qn("w:val"), "clear")
    el.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(el)


STYLES = {s.name for s in doc.styles}
# Use the document's own styles where it has them, so inserted tables and
# captions are indistinguishable from the ones already in the report. This
# document defines only 'Normal Table', with paragraph styles tablehead1,
# tabletext1 and Tablecaption, so borders are applied by hand.
HEAD_STYLE = "tablehead1" if "tablehead1" in STYLES else None
BODY_STYLE = "tabletext1" if "tabletext1" in STYLES else None
CAP_STYLE = "Tablecaption" if "Tablecaption" in STYLES else None


def set_borders(table, colour="BFBFBF", sz=4):
    tblPr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), str(sz))
        e.set(qn("w:color"), colour)
        borders.append(e)
    tblPr.append(borders)


def table_before(anchor, headers, rows, widths, *, bold_cols=(), left_cols=()):
    """Tables cannot be inserted directly, so build at the end of the body and
    move the element into place."""
    def al(i):
        return (WD_ALIGN_PARAGRAPH.LEFT if (i == 0 or i in left_cols)
                else WD_ALIGN_PARAGRAPH.CENTER)
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_borders(t)
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]
        c.text = ""
        p = c.paragraphs[0]
        if HEAD_STYLE:
            p.style = doc.styles[HEAD_STYLE]
        p.alignment = al(i)
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(9)
        shade(c, "ECEFF1")
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            if BODY_STYLE:
                p.style = doc.styles[BODY_STYLE]
            p.alignment = al(i)
            p.paragraph_format.space_after = Pt(2)
            r = p.add_run(str(v))
            r.font.size = Pt(9)
            if i in bold_cols:
                r.bold = True
    for row in t.rows:
        for i, c in enumerate(row.cells):
            c.width = Inches(widths[i])
    anchor._p.addprevious(t._tbl)
    return t


def caption_before(anchor, text):
    c = anchor.insert_paragraph_before("", style=CAP_STYLE)
    r = c.add_run(text)
    if not CAP_STYLE:
        r.italic = True
        r.font.size = Pt(9)
        r.font.color.rgb = GREY
    return c


# ============================================================ figures into
# ============================================================ existing chapters
print("inserting figures into existing chapters ...")

a = find_heading("Phase 1")
figure_before(a, "fig_semantic_distance.png",
              "Figure 3.2  Semantics and semantic distance. The semantics of an "
              "expression is the vector of values it produces on a fixed set of "
              "probe inputs; the semantic distance between two expressions is "
              "the Euclidean distance between those vectors. Two expressions "
              "are semantically similar when the vectors are close, however "
              "differently the expressions are built.", width=6.4)

a = find_heading("Discussion", level=1)
figure_before(a, "fig_size_generations.png",
              "Figure 6.3  Size of the best solution across a run on ESL, thin "
              "lines individual runs and thick lines the median. Standard GP "
              "continues to build structure for as long as it keeps finding "
              "improvements; TSGP stops early and stays small.", width=5.6)

a = find_heading("Standard GP anneals")
figure_before(a, "fig_step_floor.png",
              "Figure 7.1  Achieved parent-offspring distance against the "
              "distance requested, both on log scales. Below a request of "
              "roughly 0.1 the achieved distance does not fall further. The "
              "dashed line shows where the points would lie if the request "
              "were honoured; the method's operating point lies beneath the "
              "floor.", width=5.6)

a = find_heading("What this means for the paper")
figure_before(a, "fig_locality_control.png",
              "Figure 7.2  The control condition. Distance from an offspring to "
              "its own parent, against distance from the same offspring to an "
              "unrelated parent, at each requested distance. The ratio above "
              "each pair is the first quantity divided by the second; a value "
              "near 1 would mean the parent was ignored.", width=5.6)

# ============================================================ Chapter 8
print("inserting Chapter 8 ...")
A = find_heading("Threats to Validity", level=1)

heading_before(A, "Extending TSGP to Binary Classification", 1)

para_before(A, "The preceding chapters test a published claim and find that it "
               "does not reproduce as specified. This chapter asks a different "
               "question. The mechanism at the centre of TSGP, a transformer "
               "that proposes structurally free but semantically close "
               "variations, is not specific to regression. If it has value, "
               "that value should show somewhere. This chapter extends the "
               "method to binary classification, a task the original paper does "
               "not address, and reports what it finds.")
rich_before(A, [
    ("The extension is not a second replication. ", True),
    ("It is an original application of the method, and its results should be "
     "read as such: they say nothing about whether the published regression "
     "result is correct, and they are not evidence for or against the "
     "diagnosis of the preceding chapter.", False)])

heading_before(A, "Keeping semantics continuous: decision values instead of "
                  "labels", 2)
para_before(A, "The obvious way to apply a symbolic method to classification is "
               "to have the evolved expression emit a class label. That would "
               "break TSGP specifically, and it is worth being precise about "
               "why.")
para_before(A, "The method's two load-bearing definitions are that the "
               "semantics of an expression is the vector of values it produces "
               "on a set of probe inputs, and that the semantic distance "
               "between two expressions is the Euclidean distance between those "
               "vectors. An expression that emits only zeros and ones has a "
               "semantics that is a binary vector, and the distance between two "
               "such vectors is simply a count of the samples they disagree on. "
               "That measure is coarse and largely flat: very many structurally "
               "unrelated expressions sit at exactly the same distance from one "
               "another. The k-nearest-neighbour search that manufactures the "
               "transformer's training pairs would then be matching "
               "near-arbitrary expressions, and the operator would have nothing "
               "to learn.")
para_before(A, "The expression is therefore kept real-valued and its output is "
               "reinterpreted, as shown in Figure 8.1. The predicted class is "
               "the sign of the output, with labels encoded as -1 and +1; the "
               "surface on which the expression evaluates to zero is the "
               "decision boundary, and the magnitude of the output is a measure "
               "of confidence. This is the same device that underlies logistic "
               "regression and support vector machines. Its value here is "
               "specific: semantics remain continuous vectors, Euclidean "
               "distance retains its meaning, and the entire transformer "
               "apparatus transfers without modification.")
figure_before(A, "fig_decision_value.png",
              "Figure 8.1  A real-valued expression read as a classifier. The "
              "sign of the output gives the class and its magnitude gives "
              "confidence, so the semantics of the expression remain a "
              "continuous vector and semantic distance keeps its meaning.",
              width=6.0)

heading_before(A, "Fitness: a smooth surrogate rather than accuracy", 2)
para_before(A, "Selection cannot act on accuracy directly. Accuracy is "
               "piecewise constant in the parameters of an expression: a small "
               "change to a coefficient leaves it completely unchanged until "
               "some sample crosses the boundary, at which point it jumps. "
               "Selection would be climbing a staircase, with no gradient "
               "between steps.")
para_before(A, "Fitness is therefore the mean logistic loss of the decision "
               "value, log(1 + exp(-y f(x))), where the quantity y f(x) is the "
               "classification margin: positive when the prediction is correct "
               "and larger when it is more confidently correct. This is smooth, "
               "so every small improvement registers. Accuracy and the area "
               "under the ROC curve are computed for reporting only and never "
               "used for selection.")

heading_before(A, "Constructing the benchmarks", 2)
para_before(A, "The benchmark set had to be constructed, and the reason should "
               "be stated plainly. The trained operator's terminal set is fixed "
               "at four variables, so a problem must be expressible in four "
               "features. The Penn Machine Learning Benchmarks collection "
               "contains exactly two four-feature binary classification "
               "datasets, and neither is usable for a benchmark: one has 264 "
               "samples at 73% class imbalance, the other 50 samples.")
para_before(A, "The five regression benchmarks are therefore converted into "
               "binary problems by thresholding the target. The features are "
               "unchanged real-world features, every problem is natively "
               "four-dimensional so nothing is discarded, and the "
               "classification and regression results are measured on identical "
               "data, which allows the operator's behaviour to be compared "
               "across the two tasks rather than across different benchmarks. "
               "Two constructions are used, and the second exists for a reason "
               "established in the next section.")
table_before(A, ["Construction", "Definition", "Rationale"],
             [["Median split",
               "y above a cut chosen from the observed values to divide the "
               "training half as close to evenly as possible",
               "The natural construction. A plain median is insufficient "
               "because these targets are discrete ordinal ratings with heavy "
               "ties: on LEV it produces a 21/79 split whose majority baseline "
               "would swamp any signal."],
              ["Middle band",
               "y inside the central third of the training distribution",
               "Not a linear function of the features even when y is, so a "
               "linear model cannot represent it. Introduced after the analysis "
               "in the next section."]],
             [1.05, 2.15, 3.0], left_cols=(1, 2))
caption_before(A, "Table 8.1  The two label constructions. Every result file "
                  "records which construction produced it.")

heading_before(A, "Establishing that the benchmarks discriminate", 2)
para_before(A, "Comparing a new method only against the baseline it is designed "
               "to improve on is a common way to obtain a flattering result. "
               "Standard classifiers were therefore run on the same splits, and "
               "the outcome changed the design of the study.")
rich_before(A, [
    ("On the median-split task, logistic regression with five coefficients "
     "matches or beats TSGP on all five datasets", True),
    (", and a random forest performs worse than logistic regression on four of "
     "them. That pattern is the signature of a decision boundary that is close "
     "to linear, and it follows directly from the construction: thresholding a "
     "monotonic target produces a boundary a linear model captures almost "
     "exactly. The median-split task therefore cannot test whether structural "
     "flexibility is worth anything, because there is no non-linearity for "
     "flexibility to exploit.", False)])
para_before(A, "The middle-band construction was introduced to address this, "
               "and was selected on the evidence of the standard classifiers "
               "alone, before any TSGP run was performed on it. Figure 8.2 "
               "shows the effect. On that task logistic regression falls to the "
               "majority baseline on ERA and on pollen, while a random forest "
               "reaches 0.8525 on ESL against logistic regression's 0.6434. "
               "There is genuine non-linear structure present that a linear "
               "model cannot express.")
figure_before(A, "fig_task_difficulty.png",
              "Figure 8.2  Accuracy above the majority-class baseline for three "
              "standard classifiers, on each construction. On the median split "
              "a linear model is as good as a forest; on the middle band it "
              "collapses while the tree-based models hold up. This is why the "
              "second construction exists.", width=6.4)

heading_before(A, "Results at equal budget", 2)
para_before(A, "All results below use the operator exactly as the paper "
               "specifies it, with one offspring sampled per parent, so that "
               "TSGP and standard GP consume the same number of model "
               "evaluations. Thirty independent runs per method per dataset, as "
               "elsewhere in this report. The step-control variant is discussed "
               "at the end of the section.")
table_before(A, ["Dataset", "TSGP acc", "stdGP acc", "p", "TSGP size",
                 "stdGP size", "ratio", "p (size)"],
             [["ERA", "0.6870", "0.7020", "0.0015", "9", "31", "3.4x", "7.4e-05"],
              ["ESL", "0.9324", "0.9344", "0.918", "29", "78", "2.7x", "6.7e-10"],
              ["Galaxy", "0.9444", "0.9444", "0.842", "27", "137", "5.1x", "7.4e-11"],
              ["LEV", "0.8060", "0.8330", "<1e-4", "15", "27", "1.8x", "1.2e-04"],
              ["pollen", "0.8381", "0.8363", "0.647", "23", "65", "2.8x", "4.7e-08"]],
             [0.72, 0.82, 0.82, 0.66, 0.82, 0.82, 0.6, 0.84], bold_cols=(4, 6))
caption_before(A, "Table 8.2  Median-split task, equal budget, 30 runs. "
                  "Accuracy compared by Wilcoxon rank-sum.")
table_before(A, ["Dataset", "TSGP acc", "stdGP acc", "p", "TSGP size",
                 "stdGP size", "ratio", "p (size)"],
             [["ERA", "0.5850", "0.5890", "0.762", "9", "79", "8.8x", "3.6e-10"],
              ["ESL", "0.7705", "0.8299", "0.0002", "12", "70", "5.8x", "6.3e-08"],
              ["Galaxy", "0.8704", "0.8735", "0.156", "28", "98", "3.5x", "1.7e-07"],
              ["LEV", "0.6880", "0.6930", "0.264", "10", "21", "2.1x", "1.8e-02"],
              ["pollen", "0.6783", "0.6596", "0.0095", "11", "42", "3.8x", "9.1e-03"]],
             [0.72, 0.82, 0.82, 0.66, 0.82, 0.82, 0.6, 0.84], bold_cols=(4, 6))
caption_before(A, "Table 8.3  Middle-band task, equal budget, 30 runs.")

para_before(A, "Two things stand out, and they should be reported together "
               "because the result is a trade rather than a dominance. Figure "
               "8.3 shows both at once.")
rich_before(A, [
    ("Solution size. ", True),
    ("TSGP produces significantly smaller solutions on all five datasets in "
     "both tasks, by factors of 1.8 to 8.8, with p-values as low as 7.4e-11. On "
     "Galaxy under the median split it matches standard GP's accuracy exactly "
     "using 27 nodes against 137. This is the paper's central claim about "
     "compactness, holding on a task the paper does not address, and at equal "
     "computational budget.", False)])
rich_before(A, [
    ("Accuracy. ", True),
    ("TSGP is statistically indistinguishable from standard GP on three of five "
     "datasets in each task, and significantly worse on two under the median "
     "split and one under the middle band. It does not match standard GP's "
     "accuracy across the board, and reporting it as though it did would "
     "overstate the result.", False)])
figure_before(A, "fig_size_vs_accuracy.png",
              "Figure 8.3  Test accuracy against solution size on the "
              "median-split task, one point per method. TSGP sits far to the "
              "left of standard GP at comparable height: the same accuracy for "
              "a fraction of the structure. Logistic regression, at five "
              "coefficients, is the strongest model on this task.", width=6.6)
rich_before(A, [
    ("The exception is instructive. ", True),
    ("On pollen under the middle-band construction TSGP is significantly better "
     "than standard GP (0.6783 against 0.6596, p = 0.0095), and the detail that "
     "gives this meaning is that standard GP there sits below the "
     "majority-class baseline (-0.011) while TSGP sits above it (+0.007). The "
     "area under the ROC curve tells the same story more starkly, 0.7421 "
     "against 0.5946 (p < 1e-4). On the hardest non-linear problem in the set, "
     "standard GP finds essentially nothing and the learned operator finds real "
     "structure.", False)])
para_before(A, "The step-control variant was also run on both tasks. It behaves "
               "comparably and wins on pollen under the middle band by a "
               "smaller margin (p = 0.042). Those runs use eight model "
               "evaluations per offspring and were performed at ten runs rather "
               "than thirty; since the equal-budget arms demonstrate that ten "
               "runs can conceal genuine differences, they are reported here as "
               "preliminary and are not relied upon.")

heading_before(A, "Whether retraining on classification data was necessary", 2)
para_before(A, "The operator used above was trained on a pool generated "
               "specifically for this task. The reason was measured rather than "
               "assumed: under logistic loss nothing constrains the magnitude of "
               "the expression's output, because a more confident prediction "
               "always scores better, so a classification search drifts towards "
               "ever larger semantic norms. The population's median norm was "
               "observed to climb from 19.5 to 27.3 to 57.7 across generations, "
               "against a regression control that stays near 10, which is where "
               "the regression training pool sits. A pool built from regression "
               "problems therefore leaves a classification search outside the "
               "region the operator was trained on for most of its run.")
para_before(A, "Whether that mattered is a separate question, and it is "
               "answered by running the regression-trained operator on the same "
               "benchmarks, at the same budget and the same number of runs.")
table_before(A, ["Task", "Classification-trained better on", "Significant"],
             [["Median split", "4 of 5",
               "2 of 5: pollen +0.054 (p < 1e-4), ESL +0.010 (p = 0.0065). "
               "Worse on ERA (p = 0.030)."],
              ["Middle band", "3 of 5",
               "1 of 5: pollen +0.026 (p = 0.0004)."]],
             [1.25, 1.85, 3.1], left_cols=(1, 2))
caption_before(A, "Table 8.4  Effect of retraining the operator on "
                  "classification-regime data, both arms at equal budget and "
                  "30 runs.")
rich_before(A, [
    ("The effect is real but modest, and concentrated on a single dataset. ",
     True),
    ("Set against its cost, a complete regeneration of the training pool and "
     "several hours of retraining, this is a negative result worth recording: "
     "an operator trained on regression semantics transfers to classification "
     "better than the semantic-regime measurement predicted it would.", False)])
para_before(A, "One further observation carries across from the preceding "
               "chapter. The step-size floor described there is present here "
               "unchanged: sweeping the requested semantic distance moves the "
               "achieved distance by less than a factor of two. Retraining the "
               "operator on a different task, with a different pool and a "
               "different semantic scale, did not affect it. That strengthens "
               "the reading of the floor as a structural property of the "
               "operator rather than an artefact of any particular training set.")

heading_before(A, "What the extension establishes, and what it does not", 2)
para_before(A, "It establishes that the method transfers. A learned semantic "
               "operator, applied to a task its training never anticipated, "
               "produces solutions two to nine times more compact than standard "
               "genetic programming at equal budget, with comparable accuracy on "
               "the majority of datasets, and on one non-linear problem finds "
               "structure that standard genetic programming does not find at "
               "all. For an application where a model must be read and defended "
               "as well as scored, that trade is the one worth having.")
para_before(A, "It does not establish that TSGP is a competitive classifier in "
               "general. Logistic regression, at five coefficients, remains the "
               "better model on the median-split task. The benchmarks are "
               "constructed rather than native, and inherit the four-feature "
               "restriction of the trained operator. The comparison is against "
               "standard genetic programming alone, not against the wider "
               "field. And the accuracy losses on two datasets are real and are "
               "not explained here.")

doc.save(DST)
print(f"\nwritten: {DST}")
