# Clarynt Quality Benchmark

Use this before spending generation credits or putting new examples on the public site.

## Goal

Prove one product ad can stand next to mainstream AI ad demos before scaling outreach.

The benchmark is not a volume test. It is a quality-bar test.

## Default benchmark product

Use an existing product/reference pack when available. The current local benchmark candidate is `drinkweird_motion_v8` because it already has grouped shot boards and continuity refs.

## Pre-generation approval packet

Before any paid generation run, send Logan:

1. Full script / shot list
2. The exact reference image or approved board that will drive generation
3. Provider and endpoint
4. Number of generations / estimated paid calls
5. Pass/fail rubric

Do not run the paid job until approval is explicit.

## Required provider path

The generation path must honor `VIDEO_GEN_PROVIDER`.

For Seedance benchmark runs, verify:

- `.env` has `VIDEO_GEN_PROVIDER=seedance`
- the entrypoint does not instantiate `VeoProvider()` directly
- the provider supports reference images
- generation uses image/reference-to-video, not text-only, when a product board is available

## Minimum generation protocol

For one product:

1. Select one full-resolution approved product board.
2. Generate 5 takes for the same 6-8 second motion prompt.
3. Keep only the best 1-2 takes.
4. Edit the winner with hook text, product benefit overlay, CTA, and sound.
5. Compare against public Higgsfield / Arcads / Creatify examples before publishing.

## Hard reject criteria

Reject the take if any of these happen:

- Product is not instantly identifiable in the first 2 seconds
- Label/text turns into obvious mush when readability matters
- Product geometry warps, duplicates, melts, or changes form
- Hands/faces become uncanny or physically wrong
- Camera motion makes the product unreadable
- Lighting/color makes the product feel cheap
- The clip looks like AI concept art rather than a usable ad
- The edit lacks a clear hook, benefit, and CTA

## Scorecard

Score each candidate 1-5:

- Product readability
- Product fidelity
- Motion realism
- Lighting / premium feel
- Ad usefulness
- Social-native pacing
- Artifact control
- Competitive parity with mainstream demos

A public portfolio sample needs:

- No category below 4
- Average score of 4.3+
- Explicit human approval

Anything below that can stay as internal R&D only.

## Site rule

Do not use mediocre samples to prove a premium offer. If a clip is not at or near mainstream-demo quality, remove it from the public portfolio rather than explaining it away with positioning.
