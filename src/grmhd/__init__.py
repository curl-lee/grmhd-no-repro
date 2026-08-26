"""GRMHD neural-operator surrogate utilities.

The channel order used throughout this package is intentionally expressed in
the native spherical Kerr-Schild coordinate basis.  The components must not be
interpreted as Cartesian vectors.
"""

CHANNELS = ("Bcc1", "Bcc2", "Bcc3", "rho", "press", "vel1", "vel2", "vel3")

__all__ = ["CHANNELS"]
