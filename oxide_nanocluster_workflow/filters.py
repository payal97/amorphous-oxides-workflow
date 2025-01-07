from collections import defaultdict

import numpy as np
from agox.candidates import StandardCandidate
from agox.models.descriptors import VoronoiSite
from ase.atoms import Atoms


def energy_filter(structures: list[Atoms],
                  threshold: float = 1.0) -> list[Atoms]:
    """Filter structures by energy, removing all structures that have an energy
    higher than `threshold` above the lowest-energy structure.

    Parameters
    ----------
    structures : list[Atoms]
        List of structures to filter.
    threshold : float, optional
        Energy threshold (eV), by default 1.0.

    Returns
    -------
    list[Atoms]
        Filtered structures.
    """

    energies = [s.get_potential_energy() for s in structures]
    e_min = min(energies)
    return [s for s in structures
            if s.get_potential_energy() < e_min + threshold]


def graph_filter(structures: list[Atoms],
                 template: Atoms) -> list[Atoms]:
    """Filter structures by their graph-based structure fingerprint.

    Parameters
    ----------
    structures : list[Atoms]
        List of structures to filter.
    template : Atoms
        Surface template.

    Returns
    -------
    list[Atoms]
        Most stable structure from each fingerprint group, sorted by energy.
    """

    voronoi_site = _get_voronoi_site(n_atoms=len(structures[0]),
                                     template=template)

    # identify structure groups
    structures_by_feature: dict[str, list[Atoms]] = defaultdict(list)
    for structure in structures:
        candidate = StandardCandidate.from_atoms(template, structure)
        feature = voronoi_site.create_features(candidate)
        structures_by_feature[feature].append(structure)

    # get most stable structure from each group
    most_stable_structures = [min(feature_structures, key=lambda s: s.get_potential_energy())
                              for feature_structures in structures_by_feature.values()]

    return sorted(most_stable_structures, key=lambda s: s.get_potential_energy())


def joined_filter(structures: list[Atoms],
                  template: Atoms) -> list[Atoms]:
    """Filter structures based on whether the adsorbed atoms form a single
    joined nanocluster.

    Parameters
    ----------
    structures : list[Atoms]
        List of structures to filter.
    template : Atoms
        Surface template.

    Returns
    -------
    list[Atoms]
        Structures forming a single joined nanocluster.
    """

    voronoi_site = _get_voronoi_site(n_atoms=len(structures[0]),
                                     template=template)

    return [s for s in structures if _is_joined(voronoi_site, s, template)]


def _get_voronoi_site(n_atoms: int, template: Atoms) -> VoronoiSite:
    """Create a VoronoiSite descriptor object configured to include the top
    layer of the template and the nanocluster atoms.

    Parameters
    ----------
    n_atoms : int
        Total number of atoms in the full structure.
    template : Atoms
        Surface template.

    Returns
    -------
    VoronoiSite
        Descriptor object.
    """

    top_layer_indices = [atom.index for atom in template if atom.tag == 1]
    graph_indices = top_layer_indices + list(range(len(template), n_atoms))

    return VoronoiSite(site_mapping='fcc111',
                       template=template,
                       indices=graph_indices,
                       n_points=0,
                       environment=None)


def _is_joined(voronoi_site: VoronoiSite, structure: Atoms, template: Atoms) -> bool:
    """Return whether the adsorbed atoms form a single joined nanocluster.

    Parameters
    ----------
    voronoi_site : VoronoiSite
        Graph descriptor to use.
    structure : Atoms
        Structure to check.
    template : Atoms
        Surface template.

    Returns
    -------
    bool
        Whether the adsorbed atoms form a single joined nanocluster.
    """

    candidate = StandardCandidate.from_atoms(template, structure)
    M = voronoi_site.get_bond_matrix(candidate)

    num_top_layer_atoms = np.sum(template.get_tags() == 1)
    num_cluster_atoms = len(candidate) - len(template)

    matrix_cluster_indices = range(num_top_layer_atoms, num_top_layer_atoms + num_cluster_atoms)

    # set up a new graph based on the Voronoi graph: edges between all cluster
    # atoms that have a maximal shortest-path distance of 2 (i.e., via
    # maximally one surface atom)
    neighbors = {}

    for cluster_index in matrix_cluster_indices:
        # BFS for each cluster atom to find the shortest-path distances to
        # other cluster atoms
        explored = {cluster_index}
        queue = [cluster_index]
        distances = {cluster_index: 0}
        while len(queue) > 0 and not set(distances.keys()) >= set(matrix_cluster_indices):
            index = queue.pop(0)
            voronoi_neighbors = np.flatnonzero(M[index, :] == 1)
            for neighbor in voronoi_neighbors:
                if neighbor not in explored:
                    explored.add(neighbor)
                    queue.append(neighbor)
                    distances[neighbor] = distances[index] + 1

        # find neighbors of this atom
        neighbors[cluster_index] = set(other for other in matrix_cluster_indices
                                       if distances[other] <= 2)

    # determine whether the cluster is non-disjoint: a search through the new
    # neighbor graph finds all vertices
    explored = set()
    queue = [matrix_cluster_indices[0]]
    while len(queue) > 0:
        index = queue.pop(0)
        if index not in explored:
            explored.add(index)
            queue += list(neighbors[index])

    return explored == set(matrix_cluster_indices)
