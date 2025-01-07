from ase.atoms import Atoms
from ase.build import fcc111


def create_surface(element: str,
                   size: tuple[int, int, int],
                   a: float,
                   vacuum: float) -> Atoms:
    """Create a metal fcc(111) surface used during the structure search.

    Parameters
    ----------
    element : str
        Element to create the surface from.
    size : tuple[int, int, int]
        Number of atoms (a, b, c) in the surface.
    a : float
        Lattice parameter for the fcc(111) surface (in Å).
    vacuum : float
        Total vacuum size between top layer of surface atoms and bottom layer
        in next periodic copy (in Å).

    Returns
    -------
    Atoms
        Created surface.
    """

    surface = fcc111(symbol=element,
                     size=size,
                     a=a,
                     vacuum=vacuum / 2,
                     orthogonal=False)

    # headroom above the surface for placing nanocluster atoms
    surface.translate([0, 0, -vacuum / 4])

    return surface


def transfer_surface(atoms: Atoms, old_surface: Atoms, new_surface: Atoms) -> Atoms:
    """Transfer a supported structure to a different surface.

    Parameters
    ----------
    atoms : Atoms
        Structure supported on a surface.
    old_surface : Atoms
        The clean surface contained in `atoms`.
    new_surface : Atoms
        The clean surface to place the supported structure on.

    Returns
    -------
    Atoms
        Structure supported on the new surface.
    """

    old_top_layer = old_surface[old_surface.get_tags() == 1]
    new_top_layer = new_surface[new_surface.get_tags() == 1]

    old_z = old_top_layer.positions[:, 2].mean()
    new_z = new_top_layer.positions[:, 2].mean()

    cluster = atoms[len(old_surface):]
    cluster.positions[:, 2] += new_z - old_z

    return new_surface + cluster
