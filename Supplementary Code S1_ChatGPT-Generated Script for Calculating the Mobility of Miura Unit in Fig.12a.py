import sympy as sp
import numpy as np

# Define the symbolic variables for the coordinates of each point
xA, yA, zA = sp.symbols('xA yA zA')
xB, yB, zB = sp.symbols('xB yB zB')
xC, yC, zC = sp.symbols('xC yC zC')
xD, yD, zD = sp.symbols('xD yD zD')
xE, yE, zE = sp.symbols('xE yE zE')
xF, yF, zF = sp.symbols('xF yF zF')
xG, yG, zG = sp.symbols('xG yG zG')
xH, yH, zH = sp.symbols('xH yH zH')
xI, yI, zI = sp.symbols('xI yI zI')

# Initial coordinates
initial_coords = {
    xA: -2.928932, yA: 6.532815, zA: 2.705981,
    xB: 7.071068, yB: 6.532815, zB: 2.705981,
    xC: 14.516276, yC: 6.532815, zC: 9.381973,
    xD: -10, yD: 0, zD: 0,
    xE: 0, yE: 0, zE: 0,
    xF: 7.445208, yF: 0, zF: 6.675992,
    xG: -2.928932, yG: -6.532815, zG: 2.705981,
    xH: 7.071068, yH: -6.532815, zH: 2.705981,
    xI: 14.516276, yI: -6.532815, zI: 9.381973
}

# Define the constraints for each facet
def edge_constraint(x1, y1, z1, x2, y2, z2):
    return (x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2

def diagonal_constraint(x1, y1, z1, x2, y2, z2):
    return (x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2

def coplanar_constraint(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4):
    v12 = sp.Matrix([x2 - x1, y2 - y1, z2 - z1])
    v13 = sp.Matrix([x3 - x1, y3 - y1, z3 - z1])
    v14 = sp.Matrix([x4 - x1, y4 - y1, z4 - z1])
    return v12.dot(v13.cross(v14))

# Facet 1: A-B-E-D
constraints = [
    edge_constraint(xA, yA, zA, xB, yB, zB),
    edge_constraint(xB, yB, zB, xE, yE, zE),
    edge_constraint(xE, yE, zE, xD, yD, zD),
    edge_constraint(xD, yD, zD, xA, yA, zA),
    diagonal_constraint(xA, yA, zA, xE, yE, zE),
    diagonal_constraint(xB, yB, zB, xD, yD, zD),
    coplanar_constraint(xA, yA, zA, xB, yB, zB, xE, yE, zE, xD, yD, zD)
]

# Facet 2: B-C-F-E
constraints += [
    edge_constraint(xB, yB, zB, xC, yC, zC),
    edge_constraint(xC, yC, zC, xF, yF, zF),
    edge_constraint(xF, yF, zF, xE, yE, zE),
    edge_constraint(xE, yE, zE, xB, yB, zB),
    diagonal_constraint(xB, yB, zB, xF, yF, zF),
    diagonal_constraint(xC, yC, zC, xE, yE, zE),
    coplanar_constraint(xB, yB, zB, xC, yC, zC, xF, yF, zF, xE, yE, zE)
]

# Facet 3: D-E-H-G
constraints += [
    edge_constraint(xD, yD, zD, xE, yE, zE),
    edge_constraint(xE, yE, zE, xH, yH, zH),
    edge_constraint(xH, yH, zH, xG, yG, zG),
    edge_constraint(xG, yG, zG, xD, yD, zD),
    diagonal_constraint(xD, yD, zD, xH, yH, zH),
    diagonal_constraint(xE, yE, zE, xG, yG, zG),
    coplanar_constraint(xD, yD, zD, xE, yE, zE, xH, yH, zH, xG, yG, zG)
]

# Facet 4: E-F-I-H
constraints += [
    edge_constraint(xE, yE, zE, xF, yF, zF),
    edge_constraint(xF, yF, zF, xI, yI, zI),
    edge_constraint(xI, yI, zI, xH, yH, zH),
    edge_constraint(xH, yH, zH, xE, yE, zE),
    diagonal_constraint(xE, yE, zE, xI, yI, zI),
    diagonal_constraint(xF, yF, zF, xH, yH, zH),
    coplanar_constraint(xE, yE, zE, xF, yF, zF, xI, yI, zI, xH, yH, zH)
]

# Boundary constraints (fixed points A, B, D, E)
boundary_constraints = [
    xA - initial_coords[xA],
    yA - initial_coords[yA],
    zA - initial_coords[zA],
    xB - initial_coords[xB],
    yB - initial_coords[yB],
    zB - initial_coords[zB],
    xD - initial_coords[xD],
    yD - initial_coords[yD],
    zD - initial_coords[zD],
    xE - initial_coords[xE],
    yE - initial_coords[yE],
    zE - initial_coords[zE]
]

# Combine all constraints
all_constraints = constraints + boundary_constraints

# Calculate the Jacobian matrix
variables = [xA, yA, zA, xB, yB, zB, xC, yC, zC, xD, yD, zD, xE, yE, zE, xF, yF, zF, xG, yG, zG, xH, yH, zH, xI, yI, zI]
jacobian_matrix = sp.Matrix(all_constraints).jacobian(variables)

# Substitute initial coordinates into the Jacobian matrix
jacobian_subs = jacobian_matrix.subs(initial_coords)

# Convert the Jacobian matrix to a numpy array and calculate its rank
jacobian_np = np.array(jacobian_subs).astype(np.float64)
rank = np.linalg.matrix_rank(jacobian_np)

# Calculate the DOF
num_variables = len(variables)
num_constraints = len(all_constraints)
dof = num_variables - rank

print(f"Degrees of Freedom (DOF): {dof}")