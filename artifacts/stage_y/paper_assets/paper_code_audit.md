# Stage Y paper-code acquisition audit

## Decision

```text
PUBLIC_ASSET_SEARCH_STATUS = COMPLETED
OFFICIAL_PAPER_CODE_FOUND = false
```

No executable repository, training script, model config, preprocessing code,
checkpoint, 3D DISCO implementation, AthenaK run input, or coupling patch was
found in an official paper asset or an author-provided asset explicitly bound
to this paper.

## Official assets checked

The verified arXiv v1 source archive contains exactly `main.tex`, two
bibliographies, style files, `00README.json`, and eight figure assets.  A
recursive search finds no GitHub/GitLab/Zenodo/Figshare/Hugging Face URL, code
availability statement, dataset link, supplementary archive, or executable
source.  The PDF and source hashes are recorded in
`public_asset_manifest.csv`.

The NeurIPS 2025 workshop page links an OpenReview project page and no code or
data.  The author homepage lists the paper with only its arXiv link.

Public searches performed on 2026-08-17 included the exact title, arXiv ID,
authors plus LocalNO/DISCO/GRMHD, GitHub repository search, and searches of
Zenodo, Figshare, and Hugging Face.  GitHub's unauthenticated repository search
for the exact arXiv ID returned zero repositories.  Search-engine hits did not
produce an author/paper-bound code repository.

## Author public repositories

`cwwangcal/neuraloperator` was cloned read-only to
`/tmp/stage_y_cwwang_neuraloperator`.  It is an MIT-licensed fork of
`neuraloperator/neuraloperator`, has one public branch (`main`), no tags, and
HEAD `8719ad20733d4b58414a65706c62dda634a642bc` dated 2025-08-12.  Recursive
search finds no paper title/arXiv ID, GRMHD, Athena, volumetric DISCO,
`DiscreteContinuousConv3d`, paper data loader, or coupling implementation.  Its
local-integral code remains 2D.

The same author's public-repository listing contained five repositories.  A
recursive tree audit of `light_fno` (tree
`6c22dc8b9b37f7e23334aca8a01485b56efd336b`) and `wcwhmpg` (tree
`509b820a3f672a75e38160a91684a8513e4104a7`) found no paper/GRMHD/LocalNO/DISCO
path; the latter is a website repository.  These are not classified as paper
assets.

## What the paper text does recover

The paper gives substantial training-level prose: 1200 epochs, batch 4,
accumulation 4, Adam, learning rate and weight decay, warmup/cosine schedule,
gradient clipping, losses, normalization, shells, and split.  It also names a
"3D Local Neural Operator with equidistant DISCO specialized to volumetric
inputs" (`main.tex:535-555`).  It does not give the volumetric DISCO source or
the exact layer count, hidden width, Fourier modes, 3D kernel basis/support,
boundary semantics, per-layer branch flags, or executable config.  Therefore
paper prose is not a substitute for official training code.

The project-pinned `external/neuraloperator` at
`86a8bc7812a31b42c4f7895693cf4ac11521c066` is an upstream dependency, not the
paper repository.
