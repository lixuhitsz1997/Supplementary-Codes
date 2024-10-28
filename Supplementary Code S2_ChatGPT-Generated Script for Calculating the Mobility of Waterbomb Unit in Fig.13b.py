import sympy as sp
import numpy as np

# Define symbolic variables for the coordinates of each point
xA, yA, zA = sp.symbols('xA yA zA')
xB, yB, zB = sp.symbols('xB yB zB')
xC, yC, zC = sp.symbols('xC yC zC')
xD, yD, zD = sp.symbols('xD yD zD')
xE, yE, zE = sp.symbols('xE yE zE')
xF, yF, zF = sp.symbols('xF yF zF')
xG, yG, zG = sp.symbols('xG yG zG')

# Initial coordinates
initial_coords = {
    xA: 0, yA: 0, zA: 0,
    xB: -9.914449, yB: 10.0, zB: 1.305262,
    xC: 0, yC: 9.664966, zC: 2.566793,
    xD: 9.914449, yD: 10.0, zD: 1.305262,
    xE: -9.914449, yE: -10, zE: 1.305262,
    xF: 0, yF: -9.664966, zF: 2.566793,
    xG: 9.914449, yG: -10, zG: 1.305262
}

# Define the constraints for each facet
def edge_constraint(x1, y1, z1, x2, y2, z2):
    return (x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2

constraints = [
    edge_constraint(xB, yB, zB, xC, yC, zC),
    edge_constraint(xC, yC, zC, xA, yA, zA),
    edge_constraint(xA, yA, zA, xB, yB, zB),
    
    edge_constraint(xC, yC, zC, xD, yD, zD),
    edge_constraint(xD, yD, zD, xA, yA, zA),
    edge_constraint(xA, yA, zA, xC, yC, zC),
    
    edge_constraint(xA, yA, zA, xD, yD, zD),
    edge_constraint(xD, yD, zD, xG, yG, zG),
    edge_constraint(xG, yG, zG, xA, yA, zA),
    
    edge_constraint(xA, yA, zA, xG, yG, zG),
    edge_constraint(xG, yG, zG, xF, yF, zF),
    edge_constraint(xF, yF, zF, xA, yA, zA),
    
    edge_constraint(xA, yA, zA, xF, yF, zF),
    edge_constraint(xF, yF, zF, xE, yE, zE),
    edge_constraint(xE, yE, zE, xA, yA, zA),
    
    edge_constraint(xA, yA, zA, xE, yE, zE),
    edge_constraint(xE, yE, zE, xB, yB, zB),
    edge_constraint(xB, yB, zB, xA, yA, zA)
]

# Boundary conditions (fixing points A, B, and C)
boundary_conditions = [
    xA - initial_coords[xA],
    yA - initial_coords[yA],
    zA - initial_coords[zA],
    
    xB - initial_coords[xB],
    yB - initial_coords[yB],
    zB - initial_coords[zB],
    
    xC - initial_coords[xC],
    yC - initial_coords[yC],
    zC - initial_coords[zC]
]

# Combine all constraints
all_constraints = constraints + boundary_conditions

# Compute the Jacobian matrix
variables = [xA, yA, zA, xB, yB, zB, xC, yC, zC, xD, yD, zD, xE, yE, zE, xF, yF, zF, xG, yG, zG]
jacobian_matrix = sp.Matrix(all_constraints).jacobian(variables)

# Substitute initial coordinates into the Jacobian matrix
jacobian_subs = jacobian_matrix.subs(initial_coords)

# Convert the symbolic Jacobian matrix to a numerical one
jacobian_numeric = np.array(jacobian_subs).astype(np.float64)

# Calculate the rank of the Jacobian matrix
rank = np.linalg.matrix_rank(jacobian_numeric)

# Calculate the DOF
num_variables = len(variables)
dof = num_variables - rank

print(f"Degrees of Freedom (DOF): {dof}")