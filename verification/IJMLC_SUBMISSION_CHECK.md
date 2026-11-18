# IJMLC submission file check

Checked on 18 November 2026 against the International Journal of Machine Learning and Cybernetics guidance. Status: **text updates complete; artwork corrections remain**. This is a file and presentation check, not a new scientific validation or a journal submission.

## Version lineage

| Item | Source | Revised file |
| --- | --- | --- |
| Manuscript | FedCORE_IJMLC_Submission_Final_v5.docx | FedCORE_IJMLC_Submission_Final_v6.docx |
| Cover letter | FedCORE_IJMLC_Cover_Letter_v4.docx | FedCORE_IJMLC_Cover_Letter_v5.docx |

The manuscript source is the v5 file recorded at repository commit `202c3949029fd66fc3a32795a30a954c6c38217d`. No alternative manuscript version was used as the editing base. The cover letter is supplied separately, outside the public repository.

| File | SHA-256 |
| --- | --- |
| Source manuscript v5 | `0a7093f55b10d50e1bd12f95869fa024b4ee5ad69a35392628a8eb7d2fffe5c6` |
| Revised manuscript v6 | `d2860e939d654ae3bd50bcb0785aba35c74d8a0533eb0168d3009a527148a591` |
| Revised cover letter v5 | `fa7f598a47581a4b37b8214ed19a9195137a2c187adf919b8cef5e0e48cd12a4` |

## Completed edits

- Replaced the historical tag/future-release wording with the public repository URL: https://github.com/october25kim/fedcore. Updated the manuscript's code and data availability statements and the cover letter consistently.
- Stated the public release's count-level replay scope and exclusions precisely, without claiming independent model retraining.
- Corrected obsolete appendix pointers from A.19 to C.3 and from A.15 to B.9.
- Adjusted early figure references and added explicit callouts for Figures 3, 6 and 8. Made the Figure 3 caption label bold.
- Added a statement describing the use of existing benchmark datasets and the absence of new participant recruitment; no institutional exemption was asserted.

All native equation elements, table elements, drawing elements and embedded image bytes were preserved exactly. Scientific results, numerical values and experiment protocols were not revised. No new training or NICO++ evaluation was performed.

## File findings

| Check | Observation |
| --- | --- |
| Title and author information | Title, author, affiliation and corresponding email are present. |
| Abstract and keywords | 179 words; six keywords. |
| Editable structure | DOCX with native equations and 13 numbered tables; 15 Word tables including two layout tables. |
| Declarations | Author contribution, competing interests, funding, data/code availability and AI-assisted tools disclosure are present; ethics scope added. Existing funding statements retained. |
| Citations and references | Author-year citations, alphabetically arranged references and DOI links are present. Individual bibliographic records and DOI destinations were not comprehensively reverified. |
| Review artifacts | No tracked changes or comments detected; automatic page numbering retained. |
| Rendered layout | Manuscript: 45 pages. Cover letter: one page. Every page was visually inspected; edited pages were rechecked after the final changes. No layout clipping was observed. Figure 7's small lettering remains an artwork concern. |
| Public repository | Manuscript points to the current repository and explains its replay limits. |

## Artwork findings and remaining corrections

Effective resolution was measured from the embedded pixel dimensions and the actual insertion width. All nine embedded figures are PNG color diagrams, plots or mixed illustrations. At their current sizes, all fall below 600 dpi; these are not purely photographic halftones.

| Figure | Embedded pixels | Width (inches) | Effective dpi |
| --- | --- | --- | --- |
| 1 | 1872 × 875 | 6.24 | 300.0 |
| 2 | 2175 × 1695 | 6.20 | 350.8 |
| 3 | 1606 × 704 | 6.10 | 263.3 |
| 4 | 1872 × 960 | 6.24 | 300.0 |
| 5 | 1872 × 1425 | 5.80 | 322.8 |
| 6 | 1872 × 1605 | 6.24 | 300.0 |
| 7 | 3000 × 1850 | 5.30 | 566.0 |
| 8 | 3600 × 1440 | 6.24 | 576.9 |
| 9 | 1872 × 1185 | 5.80 | 322.8 |

1. Re-export artwork from the existing plotting sources or vector originals at the final display size. Do not increase PNG pixel dimensions by interpolation merely to change the reported resolution. Matching vector companions were located for Figures 1, 2, 3, 4 and 6, but have not been substituted into this version.
2. Change Figure 3 panel labels A/B to a/b and synchronize its caption.
3. Move the overall titles embedded in Figures 1, 4, 5, 6, 7, 8 and 9 into their captions. Retain necessary panel descriptions and axis labels.
4. Increase Figure 7's smallest lettering during export and inspect it at the final physical size.

These corrections concern presentation and require no additional experiments. The current v6 retains the original figures and should not be described as fully compliant with all artwork requirements.

## Journal guidance used

The [submission guidelines](https://link.springer.com/journal/13042/submission-guidelines) specify a 150–250-word abstract, four to six keywords, editable Word text, native equations/tables, automatic page numbers and author-year references. Artwork guidance specifies 600 dpi for combination art, 1200 dpi for bitmap line art and 300 dpi for halftones. Vector EPS is preferred, with fonts embedded. Panels use lowercase letters, overall titles belong outside illustrations, and lettering is usually 8–12 pt at final size. Author contributions and competing interests also require submission-system entries.

The [ethics and disclosures page](https://link.springer.com/journal/13042/ethics-and-disclosures) identifies single-anonymous review; retaining author information and the public repository link is consistent with that model.

At submission, enter the required portal metadata and confirm the existing originality/exclusive-submission declaration. Current consideration status at other journals was not independently verified. No manuscript has been submitted by this file-check task.
