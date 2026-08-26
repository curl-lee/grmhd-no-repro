# Vector basis audit

The paper states Cartesian Kerr–Schild simulation coordinates and labels components `Bx,By,Bz` and `vx,vy,vz` (Appendices A.2/C.2). The current raw ATHDF states only `Coordinates=kerr-schild` and variables `Bcc1,Bcc2,Bcc3,vel1,vel2,vel3`, with x1/x2/x3 numerically identified as r/theta/phi. The project deliberately retains those as spherical-KS coordinate-component names (`src/build_regrid_from_athdf.py:710-712`) and performs no vector conversion.

Position mapping and vector transformation are distinct. Knowing `(r,theta,phi)` locations is sufficient to design positional features; it is not sufficient to transform vector components without the simulation's mapping and component conventions.

`VECTOR_BASIS_GAP = FUNDAMENTAL`.
