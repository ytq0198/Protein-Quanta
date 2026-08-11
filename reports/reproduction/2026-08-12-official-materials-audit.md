# GOAI official/public supplementary-material audit

## Scope and source priority

Audit date: 2026-08-12 (Asia/Shanghai).

This note compares the local competition handbook/template with the public
GOAI AI for Research page and the Direction 2 tutorial linked from that page.
It is an internal compliance record, not submission prose.

When sources conflict, use this priority:

1. current GOAI track page and authenticated submission portal;
2. latest organizer notice/email/community message;
3. competition handbook/template;
4. tutorial slides, livestreams and learning-platform incentives.

The supplementary PDF itself repeatedly says that the website/handbook is the
source of truth.

## Confirmed current public facts

- Public track page: `https://goaihz.com/en/tracks?track=ai4s`
- Preliminary deadline date: 2026-08-16. No exact clock time was published on
  the public track page at audit time.
- Preliminary advancement: Top 40 total, split into 20 algorithmic and 20 open
  exploration teams.
- Final advancement after the semi-final: Top 20 total, split into 10 and 10.
- Direction 2 tutorial entry:
  `https://ailc.datawhale.cn/hall/group/100001094/task/100001290`
- The tutorial states that each algorithmic team may submit only one direction,
  may upload at most three competition entries, and the last successful entry
  before the deadline is the reviewed version.
- The tutorial instructs competitors to complete the Word template, manually
  add Section 6 (team introduction), compress the Word file into a ZIP, and
  upload that ZIP through the competition website.
- Datawhale check-in and screenshot steps are tied to its Token Plan incentive;
  they are not treated here as the GOAI competition submission itself.

## Resolved conflict

The July Direction 2 interpretation PDF says Top 50/Top 15 on pages 13 and 17.
The current GOAI track page says Top 40/Top 20 and further gives the 20+20 and
10+10 composition. Because the PDF is older and explicitly defers to the
website, the repository uses Top 40/Top 20.

Source PDF retained outside Git:
`文献参考/GOAI_方向二_赛题解读分享材料.pdf` (19 pages, 1,561,081 bytes).

## Technical-content caution

The interpretation PDF describes several items as technical suggestions, not
scoring rules: explicit solvent/ions, forces/free-energy labels, selected
distance/hydrogen-bond/PCA features, generative trajectories, ML-MD and active
sampling. These statements do not override the actual inspected MISATO-100
schema or the competition handbook. The current project therefore does not
claim that forces, free energies, a binding event from free ligand to bound
state, or official Phys labels were available in the experimental subset.

## Still unresolved

- Exact submission cutoff time in the authenticated portal.
- ZIP size limit and filename restrictions.
- Whether the portal requires extra text fields in addition to the ZIP.
- Whether a repository URL is accepted as an optional initial-round artifact.
- Project-level license choice and upstream NeuralMD license clarification.

These items require a team member to inspect the authenticated portal or a
newer organizer notice. No value is inferred from learning-platform reward
timestamps because they are not the competition submission task.

