# Spherical Embedding Audit

Cell centres are embedded only for local relative distances as
`(X,Y,Z)=(r sin(theta) cos(phi), r sin(theta) sin(phi), r cos(theta))`.
This is a Euclidean spherical-coordinate embedding proxy. It is not a
Kerr-Schild Cartesian transformation for vector components and is not used to
transform Bcc or velocity components. No black-hole spin or unknown metric
parameter is used, and no `1/sin(theta)` operation appears.
