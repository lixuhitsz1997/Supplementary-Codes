import sympy as sp

# Define the symbolic variables
a1, a2, a3 = sp.symbols('a1 a2 a3')
b1, b2, b3 = sp.symbols('b1 b2 b3')
c1, c2, c3, c4 = sp.symbols('c1 c2 c3 c4')

# Define the unit direction vectors for the rotational joints
L1 = sp.Matrix([0, 1, 0])
L2 = sp.Matrix([0, 1, 0])
L3 = sp.Matrix([0, 1, 0])
L4 = sp.Matrix([1, 0, 0])
L5 = sp.Matrix([1, 0, 0])
L6 = sp.Matrix([1, 0, 0])

# Define the position vectors for the rotational joints
r1 = sp.Matrix([a1, 0, 0])
r2 = sp.Matrix([a2, 0, c1])
r3 = sp.Matrix([a3, 0, c2])
r4 = sp.Matrix([0, b1, 0])
r5 = sp.Matrix([0, b2, c3])
r6 = sp.Matrix([0, b3, c4])

# Compute the moment vectors for the rotational joints
M1 = r1.cross(L1)
M2 = r2.cross(L2)
M3 = r3.cross(L3)
M4 = r4.cross(L4)
M5 = r5.cross(L5)
M6 = r6.cross(L6)

# Define the motion screws for each joint
S1 = sp.Matrix.vstack(L1, M1)
S2 = sp.Matrix.vstack(L2, M2)
S3 = sp.Matrix.vstack(L3, M3)
S4 = sp.Matrix.vstack(L4, M4)
S5 = sp.Matrix.vstack(L5, M5)
S6 = sp.Matrix.vstack(L6, M6)

# Concatenate the motion screws for each branch
S_branch1 = sp.Matrix.hstack(S1, S2, S3)
S_branch2 = sp.Matrix.hstack(S4, S5, S6)

# Compute the reciprocal screw system for each branch
C_branch1 = S_branch1.T.nullspace()
C_branch2 = S_branch2.T.nullspace()

# Convert the list of basis vectors to a matrix
C_branch1 = sp.Matrix.hstack(*C_branch1)
C_branch2 = sp.Matrix.hstack(*C_branch2)

# Concatenate the constraint screws from both branches
C_platform = sp.Matrix.hstack(C_branch1, C_branch2)

# Compute the motion screw system of the moving platform
M_platform = C_platform.T.nullspace()

# Convert the list of basis vectors to a matrix
M_platform = sp.Matrix.hstack(*M_platform)

# Print the motion screw system of the moving platform
sp.pprint(M_platform)