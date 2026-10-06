# Prompt for the deep-research runs (paste verbatim into Claude and ChatGPT)

Attach `protocol.md` to the run. The text below is the instruction.

---

You are doing a small, documented literature review for a master's thesis on
automated curriculum learning for constrained reinforcement learning. The
protocol is attached; follow it exactly. Do not improve it, do not skip steps;
if a step is impossible for you, say so and say what you did instead.

## Context you need

The thesis will build **situations**: distributions over a family of
constrained environments, each isolating one property that makes constrained
training hard, on which curriculum methods are compared (the structure of the
TeachMyAgent benchmark, applied to constrained RL). A situation is justified
only if the literature shows that practitioners actually run into that
property. Your job is to find that evidence — or its absence.

Terms, so your output uses the same words as the thesis:

- *Constrained RL / safe RL*: an agent maximises reward subject to a cost
  constraint (CMDP; PPO-Lagrangian, CPO, PID-Lagrangian, Sauté, etc.).
- *Environment family*: one task with varying parameters — layout, hazard
  count, physics, goal size. One parameter setting is a *context*.
- *Property*: a feature of the family that makes constrained training fail or
  struggle (e.g. "most contexts are infeasible under the budget").
- *Evidence*: a place where someone reports, designs for, reviews, or argues
  that property. Reports and designs count more than arguments.

## Question

Which properties of a family of constrained environments make constrained RL
training fail or struggle, **as actually reported by people who tried**?

Candidate properties P1–P6 are in the protocol. Test each. Add new ones
(label `NEW:<short-name>`) whenever the literature reports a property not on
the list. Do not force evidence onto P1–P6; "no evidence" is a valid and
useful answer.

## Procedure

1. Run the eight keyword queries (protocol table) on arXiv. For Q2, screen the
   first 60 by relevance.
2. For each seed paper (arXiv ids are given in the protocol): list its
   references and the papers citing it (Semantic Scholar). Screen those at
   stage 1.
3. Stage 1 (title + abstract), then stage 2 (full text), with the protocol's
   inclusion and exclusion rules. Every paper you see gets a row in the
   screening table, including excluded ones, with the criterion that decided.
4. For every included paper, fill extraction rows (one per evidence item). A
   paper may yield several rows; a row has exactly one property.
5. Snowball one round from newly included papers. Stop when a round adds
   nothing.
6. Write the synthesis.

## Hard rules

- **Every paper needs an arXiv id or a DOI**, in the `paper_id` column, in
  this exact form: `2103.09815` (arXiv, no `arXiv:` prefix, no version
  suffix) or `10.24963/ijcai.2024/913` (DOI). No id, no inclusion. Each id
  will be checked by a script; an id that does not resolve disqualifies the
  row and will be reported as a fabrication.
- **`title` is the paper's full title**, not an acronym or short name. The
  script compares it against the title the id resolves to; an id that exists
  but names a different paper is caught this way. (Example: `1910.01708` is a
  real paper, but it is not Safety Gym.)
- `evidence_type` is exactly one of `REPORTED`, `DESIGNED`, `REVIEWED`,
  `ARGUED` (definitions in the protocol). Prefer the first two.
- `quote` is verbatim, ≤ 40 words, with its section or figure in `location`.
  If you could not access the full text, write `full text not accessed` in
  `location` and leave `quote` empty. Never reconstruct a quote.
- Keep the actual search strings and dates you used, as executed. If your
  tooling does not expose them, say so explicitly.
- If a paper fits two properties, make two rows. If you cannot decide which
  property, pick the closest and add `?` after it (e.g. `P2?`) — I will
  resolve these.

## Output

Return **four fenced code blocks, in this order, each starting with a comment
line naming the file**, then a short free-text section. Nothing else between
the blocks. Column names and order exactly as given; one header row; no blank
lines; commas inside fields quoted with double quotes.

### Block 1 — `screening.csv`

```
paper_id,title,year,source,stage_reached,decision,criterion
```

- `source`: `Q1`..`Q6`, `QA1`, `QA2`, `seed:<arXiv id>`, or
  `snowball:<arXiv id of the paper it was found from>`
- `stage_reached`: `1` or `2`
- `decision`: `include` or `exclude`
- `criterion`: the protocol rule that decided, in ≤ 12 words

### Block 2 — `evidence.csv`

```
paper_id,title,year,venue,property,statement,environment,evidence_type,location,quote
```

Field definitions are the protocol's extraction form. `property` is
`P1`..`P6`, optionally followed by `?`, or `NEW:<short-name>`.

### Block 3 — `by_product_A.csv`

```
paper_id,title,year,interaction
```

One row per paper that combines a curriculum, task distribution, environment
design, or domain randomisation with a constrained or safe learner.
`interaction`: one sentence on how the curriculum touches the constraint.

### Block 4 — `synthesis.md`

For each of P1..P6 and each NEW property, in this fixed shape:

```
## <property id> — <name>
counts: REPORTED=<n> DESIGNED=<n> REVIEWED=<n> ARGUED=<n>
verdict: supported | argued only | no evidence
strongest:
- <paper_id>: <one sentence>
- <paper_id>: <one sentence>
notes: <anything that would change the verdict; ≤ 3 sentences>
```

Then a section `## Ambiguities` listing the `?` rows and why, and a section
`## Search log` with the queries and dates as executed (or the statement that
your tooling hides them).

### Free text (after the blocks)

At most 300 words: what surprised you, what the protocol missed, what you
would search next with another hour.

Length is not a goal. A correct 40-row screening table beats a 200-row one
with invented entries. Dropping a paper you are unsure of is better than
including it with a guessed id.
