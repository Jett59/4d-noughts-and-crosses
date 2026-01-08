import numpy as np

all_points = [np.array([x, y, z, w]) for x in range(-1, 2) for y in range(-1, 2) for z in range(-1, 2) for w in range(-1, 2)]

def is_solution_subseq_fast(pt_idxs):
    '''
    Check if the given points represent a subsequence of a solution plane
    A valid solution sequence has 9 coplanar points in lexicographically sorted order (equivalently, sorted index order)
    In other words, this function takes a list of point indices and checks if the corresponding points are coplanar, assuming they are sorted and consecutive in the solution sequence
    It is sufficient and necessary to check that the difference between consecutive points takes at most 2 values
    Proof:
    ## For sufficiency:
        If the set is empty or contains a single point then it is trivially coplanar.
        If there is only 1 difference vector then the points are colinear and hence also coplanar.
        Otherwise take the first point as the origin and the take two difference vectors as basis vectors of the plane.
        Any other point can be reached by adding some combination of these two vectors by definition - the two vectors allow one to 'hop' between each of the points.
    ## For necessity:
        Consider the sequence of 9 points in the plane, rather than the subsequence given as input.
        Note This is valid since the set of difference vectors in the subsequence is a subset of the difference vectors in the full sequence.
        Notice that, since the 9 points cannot all be colinear, there must be at least 2 difference vectors. Call the smallest ones \vec{d}_1 and \vec{d}_2 (where \vec{d}_1 < \vec{d}_2).
        We will first prove that they are linearly independent and hence form a basis for the plane (fixing some origin, of course).
        Notice that, since every component of the points -1, 0 or 1, the components of a difference vector must be in the set {-2, -1, 0, 1, 2}.
        So if they were linearly dependent then one must be the negation of or twice the other (or twice the negation).
        But note that since the vectors are sorted, the first non-zero component of the difference vectors must be positive, hence one is not the negation of the other, nor twice the negation, leaving one case remaining.
        Say WLOG that consecutive points \vec{a} and \vec{b} differed by \vec{d}_2 = 2\vec{d}_1: \vec{b}-\vec{a} = \vec{d}_2 = 2\vec{d}_1
        Note that \vec{d}_1 cannot have any components of magnitude 2, since then \vec{d}_2 would have components of magnitude 4, which is impossible.
        The point halfway between \vec{a} and \vec{b} = \vec{a}+\vec{d}_1 is also in the plane, and must have all components in the set {-1, 0, 1} since \vec{a} and \vec{d}_1 only have these components, and \vec{b} is in the sequence.
        Therefore there must exist an element of the sequence between \vec{a} and \vec{b}, contradicting the assumption that they are consecutive.

        Suppose there is a third difference vector d_3. This can be expressed as a linear combination of d_1 and d_2 since they form a basis.
        Notice that, for all components to be integers in our set, the coefficients of this representation must be integers as well (if one component had a non-integer coefficient, the resulting component would be non-integer as well since d_1 and d_2 only have integer components and are linearly independent).
        In fact, the coefficients must be in the set {-2, -1, 0, 1, 2} since otherwise \vec{d}_3 would have components outside the set {-2, -1, 0, 1, 2}.
        Also note that \vec{d}_3 must be strictly greater than both \vec{d}_1 and \vec{d}_2 in lexicographic order.
        Suppose that \vec{a} and \vec{b} are two consecutive points differing by \vec{d}_3 = c_1\vec{d}_1+c_2\vec{d}_2.
        A similar argument to above proves that neither c_1 nor c_2 is zero (otherwise there would be a point between \vec{a} and \vec{b}, or the sequence of points would not be in ascending order, or \vec{d}_3 would not be distinct).
        Similarly, c_1 and c_2 cannot both be negative as then the first non-zero component of \vec{d}_3 would be negative, contradicting the sortedness of the sequence.
        To be continued
    '''
    pts = [all_points[i] for i in pt_idxs]
    deltas = set(tuple(pts[i] - pts[i-1]) for i in range(1, len(pts)))
    return len(deltas) <= 2

def is_solution_subseq(pt_idxs):
    if len(pt_idxs) == 0:
        return True
    # We take the first point as the origin, the next two usable points as the basis vectors
    # We orthogonalise the basis
    # Then we attempt to orthogonalise all other points against the basis
    # All should reduce to zero iff they are coplanar
    pts = [all_points[i] for i in pt_idxs]
    origin = pts[0]
    pts = pts[1:]
    for i in range(len(pts)):
        pts[i] = pts[i] - origin
    if len(pts) < 2:
        return True # Trivially coplanar
    basis1 = pts[0]
    # This is not the origin, since that was removed from the list
    # Find a second basis vector
    # After this choice there must be at least 2 dimensions that are not both zero in the two basis vectors, otherwise they would be linearly dependent
    basis2 = None
    for p in pts[1:]:
        if not np.array_equal(p, -basis1) and not np.array_equal(p, -2*basis1) and not np.array_equal(p, 2*basis1): # These are the allowed values (differences are in {-2, -1, 0, 1, 2})
            basis2 = p
            break
    if basis2 is None:
        return True # All points are colinear with basis1, hence coplanar
    # Orthogonalise basis2
    basis2 = basis2-np.dot(basis1, basis2)/np.dot(basis1, basis1)*basis1
    for p in pts:
        orthogonalised = p - np.dot(basis1, p)/np.dot(basis1, basis1)*basis1
        orthogonalised = orthogonalised - np.dot(basis2, orthogonalised)/np.dot(basis2, basis2)*basis2
        if not np.allclose(orthogonalised, np.array([0,0,0,0])):
            return False
    return True

# Generate the list of planes
# We consider every nonet of points (excluding permutations) and check if they are coplanar
# There may be a faster algorithm but I can't be bothered to find it right now
def find_planes(start_idx=0, base=[], remaining = 9):
    if remaining != 0:
        # If the base is already not coplanar then adding more points won't help
        if not is_solution_subseq_fast(base):
            return
        for i in range(start_idx, len(all_points) - remaining + 1):
            yield from find_planes(i + 1, base + [i], remaining - 1)
    else:
        if is_solution_subseq(base):
            print('.', end='', flush = True)
            yield [all_points[i] for i in base]

planes = list(find_planes())
print(len(planes))

for plane in planes:
    for point in plane:
        print(' '.join(str(int(x)) for x in point))
    print('-'*15)

# Count the planes that pass through the origin
origin_planes = [plane for plane in planes if any(np.array_equal(point, np.array([0,0,0,0])) for point in plane)]
print(f'Planes through origin: {len(origin_planes)}')
