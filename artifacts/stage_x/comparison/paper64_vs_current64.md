# Paper 64^3 versus current 64^3

Equal array shape does not imply equal scientific content.

| property | paper | current |
|---|---|---|
| grid | 64_x × 64_y × 64_z Cartesian KS cube | 64_phi × 64_theta × 64_r spherical KS tensor |
| radial sampling | Cartesian distance emerges from x/y/z cells | explicitly logarithmic r centres from 1.1 to 200 |
| angular/polar behavior | Cartesian cells, no spherical coordinate pole | uniform theta/phi coordinates; physical azimuthal scale shrinks as sin(theta) |
| raw source | exact AMR/remap path not specified | Athena++ spherical-KS AMR, nearest leaf centre |
| vector fields | Bx/y/z and vx/y/z | Bcc1/2/3 and vel1/2/3 retained without basis conversion |
| thermal field | eint in C.2 | press |
| default position signal | Cartesian-index radial shells | physical spherical-r shells |
| conservation/div B | magnetic live coupling uses CT | regrid is non-conservative and not divergence preserving |

Current 64^3 is therefore a workflow/method adaptation, not a scientifically equivalent realization of the paper's 64^3 target.
