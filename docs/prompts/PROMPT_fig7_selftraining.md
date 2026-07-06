# Claude Code prompt — regenerate Figure 7 (certified pseudo-label admission)

Purpose. The reviewer asked that Figure 7's panel (b) stop showing downstream accuracy
(which invites the wrong "accuracy booster" reading) and instead show the **admission /
halt behavior** — so the figure reads as a certified admission gate, not a failed accuracy
experiment. The draft caption is already rewritten to match; this prompt regenerates the PNG
to match the caption. Everything must come from the real logged self-training run — **no
synthetic or hand-drawn curves**.

Data already has the needed columns. `runs/selftrain_cifar10_resnet18_d5_none0.0_seed0.csv`
columns: `mode, round, cert_risk_ucb, cert_coverage_lcb, realized_contam, n_pseudo, test_acc,
test_coverage, test_risk, admitted, infeasible_round`. So panel (b) can be built from
`admitted`, `n_pseudo`, and `infeasible_round` — no new training needed.

Paste the fenced block into Claude Code in the FedCORE repo.

```text
READ CLAUDE.md AND AGENTS.md FIRST. This is a plotting-only task (CPU, no GPU, no retraining).
Do NOT fabricate any value; plot only what is in the CSV. Output must overwrite
experiments/fedcore/figs/F8_selftraining.png (and a .pdf) so the manuscript picks it up.

GOAL. Regenerate Figure 7 ("Certified pseudo-label admission prevents unsafe self-training")
with three panels, all from runs/selftrain_cifar10_resnet18_d5_none0.0_seed0.csv (alpha=delta=0.1,
T=5 rounds, ResNet-GN d=5). Keep panel (a); replace the accuracy panel with admission/halt;
add a small validity inset.

PANEL (a) — pseudo-label contamination per round (KEEP).
  x = round, y = realized_contam, one line per mode in {naive, certified, none} (or whatever
  the 'mode' column contains). Add a horizontal dashed line at y=alpha=0.10. Naive should rise
  (~0.19 -> 0.67); certified stays <= alpha. Label the y-axis "pseudo-label contamination".

PANEL (b) — admission / halt behavior (NEW; replaces accuracy).
  x = round. Plot the **admitted pseudo-label fraction** = admitted / n_pseudo per round (or, if
  'admitted' is already a fraction, use it directly — inspect the column first). Draw naive and
  certified as separate series. For the certified series, mark each round where
  infeasible_round is true with a distinct "HALT" marker (e.g. a red X at y=0) so the reader
  sees that certified admits a batch only when feasible and halts otherwise. Label the y-axis
  "admitted pseudo-label fraction"; annotate the first halted round with the word "halt".
  This panel must show that certified admits fewer (or zero) pseudo-labels on infeasible rounds,
  NOT accuracy.

PANEL (c) / inset — Proposition-4 validity contract (small).
  A 2-bar inset: simultaneous unsafe rate for "delta/T split" (0.086) vs "no split / delta per
  round" (0.386), with a dashed line at delta=0.10. These two numbers are the Table-6 values;
  read them from the self-training validity output if present (T7-style), else annotate them as
  the reported contract values. Title the inset "round-wise validity".

STYLE. Match the existing figure family (same fonts/size as the other figs/*.png). Use
matplotlib, Agg backend, dpi=200, savefig both .png and .pdf to experiments/fedcore/figs/
F8_selftraining.{png,pdf}. Keep it readable at half-column width.

HONESTY. Only plot values present in the CSV. If 'admitted' or 'infeasible_round' is missing for
some mode, plot what exists and state in the run log which series were available — do not invent.
If the 'mode' labels differ from {naive, certified, none}, map them faithfully and report the
mapping. Do not change panel (a)'s numbers.

REPORT (fixed format): 진단 요약 / 확인한 명령 / 핵심 결과(어떤 컬럼으로 각 panel을 그렸는지,
admitted-fraction 범위, halt가 표시된 round) / 판정 / 다음 행동. Confirm the regenerated PNG
matches the draft caption: (a) contamination, (b) admission/halt, (c) delta/T validity inset.
```

---

### Notes for Sanghoon
- 새 코드/데이터는 필요 없습니다 — 기존 `selftrain_*.csv`의 `admitted` / `n_pseudo` /
  `infeasible_round` 컬럼만으로 panel (b)를 만들 수 있습니다(순수 plotting).
- 본문 캡션·문장은 이미 "admission gate" 메시지로 바꿔 두었으니, PNG만 교체되면 Figure 7이
  완성됩니다.
- accuracy는 본문에서 "보장 대상 아님"으로만 언급 — panel에서 빠지면 reviewer 우려가 해소됩니다.
- 산출되면 Mac에서 `bash build_docx.sh`만 다시 돌리면 docx에 반영됩니다(별도 draft 수정 불필요).
