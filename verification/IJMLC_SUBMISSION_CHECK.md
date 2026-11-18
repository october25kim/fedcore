# IJMLC submission file check

Checked on 18 November 2026 against the International Journal of Machine Learning and Cybernetics guidance. Status: **text and requested artwork corrections complete**. This is a file and presentation check, not a new scientific validation or a journal submission.

## Version lineage

| Item | Source | Revised file |
| --- | --- | --- |
| Manuscript, editorial revision | FedCORE_IJMLC_Submission_Final_v5.docx | FedCORE_IJMLC_Submission_Final_v6.docx |
| Manuscript, artwork revision | FedCORE_IJMLC_Submission_Final_v6.docx | FedCORE_IJMLC_Submission_Final_v7.docx |
| Cover letter | FedCORE_IJMLC_Cover_Letter_v4.docx | FedCORE_IJMLC_Cover_Letter_v5.docx |

The manuscript source is the v5 file recorded at repository commit `202c3949029fd66fc3a32795a30a954c6c38217d`. The artwork revision uses v6 from commit `50f34f2c93e3938de4f37a36bdabe0062b7884f7`, maintaining the v5 → v6 → v7 lineage. The cover letter is supplied separately, outside the public repository.

| File | SHA-256 |
| --- | --- |
| Source manuscript v5 | `0a7093f55b10d50e1bd12f95869fa024b4ee5ad69a35392628a8eb7d2fffe5c6` |
| Revised manuscript v6 | `d2860e939d654ae3bd50bcb0785aba35c74d8a0533eb0168d3009a527148a591` |
| Revised manuscript v7 | `4d03f3c2638afdd59032aa4192e717dc03963562407feeddc2fff0c328a04673` |
| Revised cover letter v5 | `fa7f598a47581a4b37b8214ed19a9195137a2c187adf919b8cef5e0e48cd12a4` |

## Completed edits

- Replaced the historical tag/future-release wording with the public repository URL: https://github.com/october25kim/fedcore. Updated the manuscript's code and data availability statements and the cover letter consistently.
- Stated the public release's count-level replay scope and exclusions precisely, without claiming independent model retraining.
- Corrected obsolete appendix pointers from A.19 to C.3 and from A.15 to B.9.
- Adjusted early figure references and added explicit callouts for Figures 3, 6 and 8. Made the Figure 3 caption label bold.
- Added a statement describing the use of existing benchmark datasets and the absence of new participant recruitment; no institutional exemption was asserted.

In v6, native equation elements, table elements, drawing elements and embedded image bytes were preserved exactly. In v7, all 312 native equation elements and 15 Word table elements remain byte-preserved; the nine images and their display dimensions were intentionally updated. Only the captions of Figures 3, 5 and 9 changed; all other paragraph text is unchanged. Scientific results, numerical values and experiment protocols were not revised. No new training or NICO++ evaluation was performed.

## File findings

| Check | Observation |
| --- | --- |
| Title and author information | Title, author, affiliation and corresponding email are present. |
| Abstract and keywords | 179 words; six keywords. |
| Editable structure | DOCX with native equations and 13 numbered tables; 15 Word tables including two layout tables. |
| Declarations | Author contribution, competing interests, funding, data/code availability and AI-assisted tools disclosure are present; ethics scope added. Existing funding statements retained. |
| Citations and references | Author-year citations, alphabetically arranged references and DOI links are present. Individual bibliographic records and DOI destinations were not comprehensively reverified. |
| Review artifacts | No tracked changes or comments detected; automatic page numbering retained. |
| Rendered layout | Manuscript: 45 pages. Cover letter: one page. The v6 review covered every page. In v7, 36 pages are pixel-identical to that review; all nine changed pages (5, 14, 19, 29, 30, 31, 33, 34 and 35) were visually inspected. No clipping or legend overlap was observed in the final renders. |
| Public repository | Manuscript points to the current repository and explains its replay limits. |

## Completed artwork corrections

All nine figures were regenerated from original vector geometry or unchanged archived plot data, rather than enlarging old PNGs. The manuscript embeds 600 dpi PNG exports; lossless TIFF and EPS/PDF/SVG companions are supplied in [figures](../figures/). Fonts are embedded in all nine PDF companions. Vector companions are supplied for line art. The native 28 × 28 PathMNIST patches retain their original information.

| Figure | New raster pixels | Width (inches) | Effective horizontal dpi |
| --- | --- | --- | --- |
| 1 | 3744 × 1750 | 6.24 | 600.0 |
| 2 | 3720 × 2900 | 6.20 | 600.0 |
| 3 | 3660 × 1605 | 6.10 | 600.0 |
| 4 | 3744 × 1920 | 6.24 | 600.0 |
| 5 | 3480 × 2649 | 5.80 | 600.0 |
| 6 | 3744 × 3210 | 6.24 | 600.0 |
| 7 | 3744 × 1950 | 6.24 | 600.0 |
| 8 | 3744 × 1500 | 6.24 | 600.0 |
| 9 | 3744 × 2370 | 6.24 | 600.0 |

- Figure 3: A/B changed to (a)/(b), with matching caption.
- Figure 5: all three visual sections labeled (a)–(c), with matching caption.
- Figures 1, 4, 5, 6, 7, 8 and 9: overall titles removed from the image; descriptions remain in captions and text.
- Figure 1: mathematical membership symbol outlined to prevent font substitution.
- Figure 7: final width increased to 6.24 inches; lettering at least 8 pt; dataset, certification and quartile legends separated. The original footer's numerical statements remain in Appendix B.5.
- Figure 8: original saved confidence intervals preserved; vertical limits expanded to include the full bands and the n = 514 reference line retained.
- Figure 9: label/colorbar margins corrected; caption identifies the native 28 × 28-pixel patches.
- Word image compression disabled; all figure placement and caption layouts visually checked.

The source hashes, geometric preservation, image-format checks and manuscript invariants are recorded in [IJMLC_ARTWORK_CHECK.json](IJMLC_ARTWORK_CHECK.json). This is an artwork and document check, not an independent scientific replication. No new experiments were needed.

## Journal guidance used

The [submission guidelines](https://link.springer.com/journal/13042/submission-guidelines) specify a 150–250-word abstract, four to six keywords, editable Word text, native equations/tables, automatic page numbers and author-year references. Artwork guidance specifies 600 dpi for combination art, 1200 dpi for bitmap line art and 300 dpi for halftones. Vector EPS is preferred, with fonts embedded. Panels use lowercase letters, overall titles belong outside illustrations, and lettering is usually 8–12 pt at final size. Author contributions and competing interests also require submission-system entries.

The [ethics and disclosures page](https://link.springer.com/journal/13042/ethics-and-disclosures) identifies single-anonymous review; retaining author information and the public repository link is consistent with that model.

At submission, enter the required portal metadata and confirm the existing originality/exclusive-submission declaration. Current consideration status at other journals was not independently verified. No manuscript has been submitted by this file-check task.
