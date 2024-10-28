import sympy as sp
import numpy as np

# Define the coordinates of the points
points = {
    'A': (-10.0, 0, 0),
    'B': (-5.0, -8.660254, 0),
    'C': (5, -8.660254, 0),
    'D': (10, 0, 0),
    'E': (-5.0, 8.660254, 0),
    'F': (5, 8.660254, 0),
    'G': (-8.660254, -5.0, 12.5),
    'H': (0, -10.0, 12.5),
    'I': (8.660254, -5.0, 12.5),
    'J': (-8.660254, 5, 12.5),
    'K': (0, 10, 12.5),
    'L': (8.660254, 5, 12.5)
}

# Define the facets
facets = [
    ('A', 'H', 'B'),
    ('B', 'H', 'I'),
    ('B', 'I', 'C'),
    ('C', 'I', 'L'),
    ('C', 'L', 'D'),
    ('D', 'L', 'K'),
    ('D', 'K', 'F'),
    ('F', 'K', 'J'),
    ('F', 'J', 'E'),
    ('E', 'J', 'G'),
    ('E', 'G', 'A'),
    ('A', 'G', 'H')
]

# Define the symbolic variables for the coordinates
coords = {p: (sp.symbols(f'x{p}'), sp.symbols(f'y{p}'), sp.symbols(f'z{p}')) for p in points}

# Define the constraints for the facets
constraints = []
for facet in facets:
    p1, p2, p3 = facet
    x1, y1, z1 = coords[p1]
    x2, y2, z2 = coords[p2]
    x3, y3, z3 = coords[p3]
    constraints.append((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2 - sp.Rational((points[p1][0] - points[p2][0])**2 + (points[p1][1] - points[p2][1])**2 + (points[p1][2] - points[p2][2])**2))
    constraints.append((x2 - x3)**2 + (y2 - y3)**2 + (z2 - z3)**2 - sp.Rational((points[p2][0] - points[p3][0])**2 + (points[p2][1] - points[p3][1])**2 + (points[p2][2] - points[p3][2])**2))
    constraints.append((x3 - x1)**2 + (y3 - y1)**2 + (z3 - z1)**2 - sp.Rational((points[p3][0] - points[p1][0])**2 + (points[p3][1] - points[p1][1])**2 + (points[p3][2] - points[p1][2])**2))

# Define the boundary constraints
fixed_points = ['A', 'B', 'C', 'D', 'E', 'F']
for p in fixed_points:
    x, y, z = coords[p]
    constraints.append(x - points[p][0])
    constraints.append(y - points[p][1])
    constraints.append(z - points[p][2])

# Calculate the Jacobian matrix
variables = [var for p in coords for var in coords[p]]
jacobian_matrix = sp.Matrix(constraints).jacobian(variables)

# Substitute the initial coordinates into the Jacobian matrix
subs = {coords[p][i]: points[p][i] for p in points for i in range(3)}
jacobian_matrix_subs = jacobian_matrix.subs(subs)

# Convert the Jacobian matrix to a numpy array and calculate its rank
jacobian_matrix_np = np.array(jacobian_matrix_subs).astype(np.float64)
rank = np.linalg.matrix_rank(jacobian_matrix_np)

# Calculate the DOF
num_variables = len(variables)
dof = num_variables - rank

print(f"Degrees of Freedom (DOF): {dof}")