"""FEMTIC output: the built deal.II mesh as a FEMTIC non-conforming deformed
hexahedral mesh (``DHEXA``).

FEMTIC (Usui) reads ``mesh.dat`` + ``resistivity_block_iter0.dat``.  Its DHEXA
mesh is the same kind of mesh pyGoFEM builds: hexahedra refined as an octree
with hanging faces, vertices moved vertically to fit the topography.  So the
FEMTIC mesh *is* the GoFEM mesh, written differently; this module only converts.

Format (FEMTIC ``MeshDataNonConformingHexaElement::inputMeshData``, and the
writer of Usui's own mesher, makeDHexaMesh ``outputMeshData``):

  ``DHEXA``
  ``n_nodes`` then ``id x y z`` per node
  ``n_elements`` then per element ``id n0 .. n7`` and, per face 0-5,
  ``count id ..`` (``-1`` = outer boundary, ``1 id``, or ``4 id id id id``)
  six boundary planes: ``n`` then ``element face`` lines (plane p = face p)
  the earth surface: ``n`` then ``element 4`` lines

Frames and numbering, all checked against FEMTIC's example mesh:
  * the same axes as GoFEM: x = north, y = east, z down, metres;
  * the same face numbers as deal.II: 0 x-, 1 x+, 2 y-, 3 y+, 4 top, 5 bottom;
  * corners: FEMTIC goes round each horizontal face (0, 1, 2, 3 at the top,
    4-7 below), deal.II is lexicographic, so FEMTIC node k = deal.II vertex
    ``_DEAL_OF_FEMTIC[k]``;
  * a face with 4 neighbours lists them as ``i1 + 2 * i2``, i1 / i2 along the
    face's first / second direction (y, z on faces 0-1; x, z on 2-3; x, y on 4-5).
    FEMTIC asserts that order through shared edges.

FEMTIC ties each fine edge on a hanging face straight to the coarse face's edge
and does not follow chains, so a coarse edge must never itself hang on an even
coarser face.  deal.II never makes that: it keeps cells that share an edge within
one level (its level jumps are at corners only, which FEMTIC doesn't mind), so the
GoFEM mesh goes out as it is.  :func:`check_mesh` still tests for it.

Resistivity blocks: block 0 is air (FEMTIC requires it, fixed); every other fixed
material (sea, mantle) is one fixed block; each free (earth) cell is its own block.
"""
from __future__ import annotations

import os

import numpy as np

MESH_DAT = "mesh.dat"
BLOCKS_DAT = "resistivity_block_iter0.dat"

#: FEMTIC node k is deal.II vertex _DEAL_OF_FEMTIC[k]
_DEAL_OF_FEMTIC = np.array([0, 1, 3, 2, 4, 5, 7, 6])

#: per face, the deal.II vertices at in-plane positions (i1, i2) = (0,0), (1,0), (0,1), (1,1)
_FACE_VERTS = np.array([
    [0, 2, 4, 6],   # 0: x-, (y, z)
    [1, 3, 5, 7],   # 1: x+
    [0, 1, 4, 5],   # 2: y-, (x, z)
    [2, 3, 6, 7],   # 3: y+
    [0, 1, 2, 3],   # 4: top (z-), (x, y)
    [4, 5, 6, 7],   # 5: bottom
])

#: FEMTIC's face -> local nodes and face -> edge -> nodes tables (for check_mesh)
_F_FACE_NODES = [[0, 3, 7, 4], [1, 2, 6, 5], [0, 1, 5, 4], [3, 2, 6, 7], [0, 1, 2, 3], [4, 5, 6, 7]]
_F_FACE_EDGES = [[4, 5, 8, 10], [6, 7, 9, 11], [0, 2, 8, 9], [1, 3, 10, 11], [0, 1, 4, 6], [2, 3, 5, 7]]
_F_EDGE_NODES = [[0, 1], [3, 2], [4, 5], [7, 6], [0, 3], [4, 7], [1, 2], [5, 6],
                 [0, 4], [1, 5], [3, 7], [2, 6]]

# resistivity block types (FEMTIC ResistivityBlockIsotropic)
FREE_AND_CONSTRAINED, FIXED_AND_ISOLATED = 0, 1


class FemticMesh:
    """A DHEXA mesh in arrays: ``nodes`` (n, 3) x/y/z; ``elements`` (m, 8) node ids in
    FEMTIC order; ``neighbours`` [m][6] lists (empty = outer boundary);
    ``planes`` six (k, 2) arrays of (element, face); ``surface`` (k, 2) likewise."""

    def __init__(self, nodes, elements, neighbours, planes, surface):
        self.nodes = np.asarray(nodes, dtype="float64")
        self.elements = np.asarray(elements, dtype="int64")
        self.neighbours = neighbours
        self.planes = [np.asarray(p, dtype="int64").reshape(-1, 2) for p in planes]
        self.surface = np.asarray(surface, dtype="int64").reshape(-1, 2)

    @property
    def n_elements(self) -> int:
        return len(self.elements)


# --------------------------------------------------------------------------- #
# deal.II -> arrays
# --------------------------------------------------------------------------- #
def tria_arrays(tria):
    """``(xyz, vid, level, material)`` of the active cells, in ``active_cells()`` order:
    vertex coordinates by deal.II vertex index, (n, 8) vertex ids, levels, material ids."""
    cells = tria.active_cells()
    n = len(cells)
    vid = np.empty((n, 8), dtype="int64")
    lv = np.empty(n, dtype="int64")
    mat = np.empty(n, dtype="int64")
    coords: dict[int, tuple] = {}
    for k, c in enumerate(cells):
        lv[k] = c.level()
        mat[k] = c.material_id
        for i in range(8):
            j = c.vertex_index(i)
            vid[k, i] = j
            if j not in coords:
                p = c.get_vertex(i)
                coords[j] = (p.x, p.y, p.z)
    xyz = np.full((int(vid.max()) + 1, 3), np.nan)
    ids = np.fromiter(coords.keys(), dtype="int64", count=len(coords))
    xyz[ids] = np.array(list(coords.values()), dtype="float64")
    return xyz, vid, lv, mat


def build_mesh(xyz, vid, level, material, *, surface_below_ids, earth_top_ids) -> FemticMesh:
    """Convert active-cell arrays (see :func:`tria_arrays`) to a :class:`FemticMesh`.

    ``earth_top_ids``: materials whose top face counts as the earth surface where it
    meets a cell of ``surface_below_ids`` (air, sea) -- FEMTIC's land-surface list,
    which makeDHexaMesh fills with every land cell under air *or* sea.
    """
    vid = np.asarray(vid, dtype="int64")
    level = np.asarray(level, dtype="int64")
    material = np.asarray(material, dtype="int64")
    n = len(vid)

    # nodes: only the vertices the active cells use, renumbered 0..
    used = np.unique(vid)
    remap = np.full(int(used.max()) + 1, -1, dtype="int64")
    remap[used] = np.arange(len(used))
    nodes = np.asarray(xyz)[used]
    v = remap[vid]                                   # (n, 8) node ids, deal.II order
    elements = v[:, _DEAL_OF_FEMTIC]

    # every face's corners at in-plane positions q = i1 + 2 i2
    fv = v[:, _FACE_VERTS]                           # (n, 6, 4)
    neighbours: list[list[list[int]]] = [[[] for _f in range(6)] for _e in range(n)]

    # 1. conforming faces: the same four corners
    flat = np.sort(fv.reshape(-1, 4), axis=1)
    _u, inv, counts = np.unique(flat, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    if counts.max() > 2:
        raise ValueError("FEMTIC export: a face is shared by more than two cells")
    order = np.argsort(inv, kind="stable")
    sinv = inv[order]
    pair_start = np.flatnonzero((counts[sinv] == 2) & (np.r_[True, sinv[1:] != sinv[:-1]]))
    a, b = order[pair_start], order[pair_start + 1]  # flat face ids (cell * 6 + face)
    for fa, fb in zip(a.tolist(), b.tolist()):
        neighbours[fa // 6][fa % 6].append(fb // 6)
        neighbours[fb // 6][fb % 6].append(fa // 6)
    unmatched = np.flatnonzero(counts[inv] == 1)

    # 2. hanging faces: a coarse face C (face f) is covered by four faces (face f ^ 1) of
    #    cells one level finer; the child at position q has C's corner q at its own q.
    by_corner: dict[tuple, int] = {}
    for g in unmatched.tolist():
        e, f = divmod(g, 6)
        for q in range(4):
            by_corner[(f, int(fv[e, f, q]), q)] = g
    hung = set()
    for g in unmatched.tolist():
        e, f = divmod(g, 6)
        kids = [by_corner.get((f ^ 1, int(fv[e, f, q]), q)) for q in range(4)]
        if any(k is None for k in kids):
            continue
        kid_cells = [k // 6 for k in kids]
        centre = {int(fv[kc, f ^ 1, 3 - q]) for q, kc in enumerate(kid_cells)}
        if len(centre) != 1 or any(level[kc] != level[e] + 1 for kc in kid_cells):
            continue
        neighbours[e][f] = kid_cells
        for kc in kid_cells:
            neighbours[kc][f ^ 1] = [e]
        hung.add(g)
        hung.update(kids)

    # 3. the rest must be the outer boundary
    planes = [[] for _p in range(6)]
    lo, hi = nodes.min(axis=0), nodes.max(axis=0)
    tol = 1e-6 * max(float(np.ptp(nodes, axis=0).max()), 1.0)
    for g in unmatched.tolist():
        if g in hung:
            continue
        e, f = divmod(g, 6)
        axis, side = f // 2, f % 2
        c = nodes[fv[e, f], axis]
        edge = hi[axis] if side else lo[axis]
        if axis < 2 and not np.all(np.abs(c - edge) < tol):
            raise ValueError("FEMTIC export: element %d face %d has no neighbour but is "
                             "not on the model boundary" % (e, f))
        if axis == 2 and np.ptp(c) > tol:       # the top and bottom must be flat
            raise ValueError("FEMTIC export: boundary face %d of element %d is not flat" % (f, e))
        planes[f].append((e, f))

    # 4. the earth surface: earth cells whose top face touches air / sea
    below = np.isin(material, list(surface_below_ids))
    earth = np.isin(material, list(earth_top_ids))
    surface = [(e, 4) for e in np.flatnonzero(earth).tolist()
               if any(below[nb] for nb in neighbours[e][4])]
    return FemticMesh(nodes, elements, neighbours, planes, surface)


# --------------------------------------------------------------------------- #
# checks FEMTIC would make (and the ones it would just get wrong silently)
# --------------------------------------------------------------------------- #
def check_mesh(m: FemticMesh, *, eps: float = 1e-6) -> list[str]:
    """Problems that would stop FEMTIC or break its solution; empty when fine."""
    problems: list[str] = []
    P, E = m.nodes, m.elements

    # node order: 0 -> 1 along +x, 0 -> 3 along +y
    flipped = (P[E[:, 1], 0] <= P[E[:, 0], 0]) | (P[E[:, 3], 1] <= P[E[:, 0], 1])
    if flipped.any():
        problems.append("%d elements have their corners in the wrong order (e.g. element %d)"
                        % (flipped.sum(), np.flatnonzero(flipped)[0]))
    # side faces must be vertical planes (FEMTIC checkWhetherSideFaceIsParallelTo...)
    for f in range(4):
        axis = 0 if f < 2 else 1
        c = P[E[:, _F_FACE_NODES[f]], axis]
        bad = np.flatnonzero(np.ptp(c, axis=1) > eps)
        if len(bad):
            problems.append("%d elements: face %d is not a vertical plane (e.g. element %d)"
                            % (len(bad), f, bad[0]))
            break

    edge_id: dict[tuple, int] = {}

    def eid(e, local_edge):
        a, b = E[e, _F_EDGE_NODES[local_edge]]
        key = (a, b) if a < b else (b, a)
        return edge_id.setdefault((int(key[0]), int(key[1])), len(edge_id))

    slave_edges, master_edges = set(), set()
    n_wrong = n_asym = n_order = 0
    for e in range(m.n_elements):
        for f in range(6):
            nb = m.neighbours[e][f]
            if len(nb) not in (0, 1, 4):        # FEMTIC can read 2 too, but deal.II never makes it
                n_wrong += 1
                continue
            for k in nb:
                if e not in m.neighbours[k][f ^ 1]:
                    n_asym += 1
            if len(nb) != 4:
                continue
            fn = f ^ 1
            s = [[eid(k, le) for le in _F_FACE_EDGES[fn]] for k in nb]
            # the asserts in Forward3DNonConformingHexaElement0thOrder (neighbour order)
            if not (s[0][1] == s[2][0] and s[1][1] == s[3][0]
                    and s[0][3] == s[1][2] and s[2][3] == s[3][2]):
                n_order += 1
            for row in s:
                slave_edges.update(row)
            for le in _F_FACE_EDGES[f]:
                master_edges.add(eid(e, le))
    if n_wrong:
        problems.append("%d faces have a neighbour count other than 0, 1 or 4" % n_wrong)
    if n_asym:
        problems.append("%d neighbour links are one-way" % n_asym)
    if n_order:
        problems.append("%d hanging faces list their four neighbours in the wrong order" % n_order)
    chained = master_edges & slave_edges
    if chained:
        problems.append("%d coarse edges hang on an even coarser face (FEMTIC doesn't follow "
                        "chained constraints)" % len(chained))
    for p in range(6):
        if not len(m.planes[p]):
            problems.append("boundary plane %d is empty" % p)
    if not len(m.surface):
        problems.append("no earth-surface faces")
    return problems


def folded_elements(m: FemticMesh) -> np.ndarray:
    """Elements with a vertical edge that points up (bottom corner at or above its top).

    The topography fit can make these, for GoFEM as much as for FEMTIC: deal.II puts a
    hanging vertex at the middle of the coarse cell's edge, so across a steep step in
    the DEM it can end up above the top of the thin cell under it.  A warning, not an
    error: the GoFEM mesh is the same."""
    P, E = m.nodes, m.elements
    return np.flatnonzero((P[E[:, 4:], 2] <= P[E[:, :4], 2]).any(axis=1))


# --------------------------------------------------------------------------- #
# files
# --------------------------------------------------------------------------- #
def write_mesh_dat(m: FemticMesh, path: str) -> str:
    with open(path, "w") as fh:
        fh.write("DHEXA\n%10d\n" % len(m.nodes))
        for i, (x, y, z) in enumerate(m.nodes):
            fh.write("%10d %20.6f %20.6f %20.6f\n" % (i, x, y, z))
        fh.write("%10d\n" % m.n_elements)
        for e in range(m.n_elements):
            fh.write("%10d " % e + " ".join("%10d" % k for k in m.elements[e]) + "\n")
            for nb in m.neighbours[e]:
                fh.write(("%10d " % len(nb) + " ".join("%10d" % k for k in nb)) if nb
                         else "%10d" % -1)
                fh.write("\n")
        for p in range(6):
            fh.write("%10d\n" % len(m.planes[p]))
            for e, f in m.planes[p]:
                fh.write("%10d %10d\n" % (e, f))
        fh.write("%10d\n" % len(m.surface))
        for e, f in m.surface:
            fh.write("%10d %10d\n" % (e, f))
    return path


def resistivity_blocks(material, materials: dict, free_ids, air_id: int):
    """``(element -> block, [(rho, fixed), ...])``: block 0 = air, one block per other
    fixed material, then one per free element (FEMTIC's parameter cells)."""
    material = np.asarray(material, dtype="int64")
    unknown = set(np.unique(material).tolist()) - set(materials)
    if unknown:
        raise ValueError("FEMTIC export: cells with material ids %s have no resistivity"
                         % sorted(unknown))
    free_ids = set(int(i) for i in free_ids)
    blocks = [(1.0 / materials[air_id][1], True)]
    block = np.full(len(material), -1, dtype="int64")
    block[material == air_id] = 0
    for mid in sorted(set(np.unique(material).tolist()) - free_ids - {air_id}):
        block[material == mid] = len(blocks)
        blocks.append((1.0 / materials[mid][1], True))
    free = np.flatnonzero(np.isin(material, list(free_ids)))
    block[free] = len(blocks) + np.arange(len(free))
    blocks.extend((1.0 / materials[int(material[e])][1], False) for e in free.tolist())
    return block, blocks


def write_blocks_dat(block, blocks, path: str) -> str:
    """``resistivity_block_iter0.dat``: element -> block, then per block
    ``id rho min max weight type`` (makeDHexaMesh's bounds and weight)."""
    with open(path, "w") as fh:
        fh.write("%10d%10d\n" % (len(block), len(blocks)))
        for e, b in enumerate(block.tolist()):
            fh.write("%10d%10d\n" % (e, b))
        for b, (rho, fixed) in enumerate(blocks):
            fh.write("%10d %15e %15e %15e %15e %5d\n"
                     % (b, rho, 1e-20, 1e20, 1.0,
                        FIXED_AND_ISOLATED if fixed else FREE_AND_CONSTRAINED))
    return path


def write_mesh(tria, materials: dict, out_dir: str, *, air_id: int, free_ids,
               sea_ids=(), log=print) -> dict:
    """Write ``mesh.dat`` and ``resistivity_block_iter0.dat`` for the built mesh.
    Raises if the mesh breaks a FEMTIC rule (see :func:`check_mesh`)."""
    xyz, vid, lv, mat = tria_arrays(tria)
    earth_ids = set(np.unique(mat).tolist()) - {air_id} - set(sea_ids)
    m = build_mesh(xyz, vid, lv, mat, surface_below_ids={air_id, *sea_ids},
                   earth_top_ids=earth_ids)
    problems = check_mesh(m)
    if problems:
        raise ValueError("the mesh can't be written for FEMTIC: " + "; ".join(problems))
    folded = folded_elements(m)
    if len(folded):
        c = m.nodes[m.elements[folded]].mean(axis=1)
        log("WARNING: %d cells are folded over by the topography fit (a bottom corner at or "
            "above the top; the GoFEM mesh has them too), around x %.0f..%.0f km, y %.0f..%.0f km "
            "(north, east): a steep DEM step next to thin cells. Smoothing the DEM or refining "
            "there more evenly removes them."
            % (len(folded), c[:, 0].min() / 1e3, c[:, 0].max() / 1e3,
               c[:, 1].min() / 1e3, c[:, 1].max() / 1e3))
    block, blocks = resistivity_blocks(mat, materials, free_ids, air_id)
    os.makedirs(out_dir, exist_ok=True)
    paths = {"femtic_mesh": write_mesh_dat(m, os.path.join(out_dir, MESH_DAT)),
             "femtic_blocks": write_blocks_dat(block, blocks, os.path.join(out_dir, BLOCKS_DAT))}
    n_free = sum(1 for _rho, fixed in blocks if not fixed)
    n_hang = sum(1 for e in range(m.n_elements) for nb in m.neighbours[e] if len(nb) == 4)
    log("FEMTIC: wrote %s (%d nodes, %d elements, %d hanging faces, %d earth-surface faces) "
        "and %s (%d blocks, %d free)"
        % (MESH_DAT, len(m.nodes), m.n_elements, n_hang, len(m.surface),
           BLOCKS_DAT, len(blocks), n_free))
    paths["femtic_folded"] = int(len(folded))
    return paths


# --------------------------------------------------------------------------- #
# reading (tests now, the plotting app later)
# --------------------------------------------------------------------------- #
def read_mesh_dat(path: str) -> FemticMesh:
    """Parse a DHEXA ``mesh.dat`` (ours or makeDHexaMesh's)."""
    tok = open(path).read().split()
    if not tok or not tok[0].startswith("DHEXA"):
        raise ValueError(f"{path}: not a DHEXA mesh")
    i = 1
    nn = int(tok[i]); i += 1
    nodes = np.array(tok[i:i + 4 * nn], dtype="float64").reshape(nn, 4)[:, 1:]; i += 4 * nn
    ne = int(tok[i]); i += 1
    elements = np.empty((ne, 8), dtype="int64")
    neighbours = []
    for e in range(ne):
        i += 1
        elements[e] = np.array(tok[i:i + 8], dtype="int64"); i += 8
        row = []
        for _f in range(6):
            k = max(int(tok[i]), 0); i += 1
            row.append([int(t) for t in tok[i:i + k]]); i += k
        neighbours.append(row)
    planes = []
    for _p in range(6):
        k = int(tok[i]); i += 1
        planes.append(np.array(tok[i:i + 2 * k], dtype="int64").reshape(k, 2)); i += 2 * k
    k = int(tok[i]); i += 1
    surface = np.array(tok[i:i + 2 * k], dtype="int64").reshape(k, 2)
    return FemticMesh(nodes, elements, neighbours, planes, surface)


def read_blocks_dat(path: str):
    """``(element -> block, rho per block, fixed per block)``."""
    tok = open(path).read().split()
    ne, nb = int(tok[0]), int(tok[1])
    eb = np.array(tok[2:2 + 2 * ne], dtype="int64").reshape(ne, 2)[:, 1]
    rows = np.array(tok[2 + 2 * ne:2 + 2 * ne + 6 * nb], dtype="float64").reshape(nb, 6)
    return eb, rows[:, 1], np.isin(rows[:, 5].astype(int), (1, 2))
