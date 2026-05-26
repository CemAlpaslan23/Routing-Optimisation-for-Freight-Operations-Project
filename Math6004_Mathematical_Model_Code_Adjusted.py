import pandas as pd
import numpy as np
import gurobipy
from gurobipy import GRB, Model, quicksum
import matplotlib.pyplot as plt
import math


# Haversine distance calculation function, Dij parameter calculation method
def haversine_distance(p1, p2):

    lat1, lon1 = p1
    lat2, lon2 = p2

    # Earth radius in kilometers
    R = 6371.0

    # Convert degrees to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # Calculate the differences
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Calculating a in the formula
    a = math.sin(dlat / 2)**2 + \
        math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2

    # Calculating c in the formula
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Calculating D in the formula
    return R * c

# Reading the dataset and defining the sets and parameters
node_df = pd.read_excel("data_math6004.xlsx", sheet_name="Nodes", index_col=0)
shipment_df = pd.read_excel("data_math6004.xlsx", sheet_name="Shipments", index_col=0)

# Reading nodes sheet in the data
xcoord = list(node_df['Longitude']) # Reading X-coordinate (Longitude) from the dataset
ycoord = list(node_df['Latitude']) # Reading Y-coordinate (Latitude) from the dataset
e = list(node_df['Earliest']) # Reading earliest service start time from the dataset
l = list(node_df['Latest'])  # Reading latest service start time from the dataset
s_time = list(node_df['ServiceTime']) # Reading service duration time from the dataset
node_type = list(node_df['Type']) # Reading node type from the dataset: 'depot', 'pickup', 'delivery'

# Reading shipment sheet in the data
p_node = list(shipment_df['PickupNode']) # p(s): Pickup node index for each shipment
d_node = list(shipment_df['DeliveryNode']) # d(s): Delivery node index for each shipment
q_s = list(shipment_df['Quantity']) # qs: Shipment quantity
alpha = list(shipment_df['AlreadyPickedUp']) # αs: Pickup status parameter, 1 = already at depot, 0 = pickup must be done within the horizon

# Defining the sets
location_set = len(node_df) # Total number of nodes

N = list(range(location_set)) # N = {0} ∪ P ∪ D: Full node set
P_set = [i for i in N if node_type[i] == 'pickup'] # P: Pickup node set
D_set = [i for i in N if node_type[i] == 'delivery'] # D: Delivery node set
S_set = list(range(len(shipment_df))) # S: Shipment set

n_days = 2 # Number of planning days, set T
n_vehicles = 3 # Number of vehicles, 1 = Small, 2 = Medium, 3 = Large, set K
# R_max = len(P_set) + len(D_set) # R_max = |P| + |D|: Maximum number of trips per vehicle per day, set R
R_max = 4 # R_max can be 3 or 4, I calculated according to the maximum daily distance limitation

T_set = list(range(1, n_days + 1)) # T = {1, 2} +1 because python does not include the last index
K_set = list(range(1, n_vehicles + 1)) # K = {1 = Small, 2 = Medium, 3 = Large} +1 because python does not include the last index
R_set = list(range(1, R_max + 1)) # R = {1, ..., Rmax} +1 because python does not include the last index

# Defining dict for model parameters, because it is a heteregenous fleet
cap_by_vehicle = {1: 22, 2: 30, 3: 38} # Qk: Cargo capacity parameter
fixed_cost_by_vehicle = {1: 1750, 2: 2000, 3: 2500} # Fk: Daily rental cost, fixed cost parameter
dist_cost_by_vehicle  = {1: 15, 2: 18, 3: 22} # Ak: Distance cost per km parameter

Q = {k: cap_by_vehicle[k] for k in K_set}  # Cargo capacity of vehicle k parameter
F = {k: fixed_cost_by_vehicle[k] for k in K_set}  # Daily rental cost of vehicle k parameter
A_cost = {k: dist_cost_by_vehicle[k] for k in K_set}  # Distance cost per km for vehicle k parameter

H  = {(k, t): 7 for k in K_set for t in T_set}  # Hkt: Maximum daily travel distance limit (km) parameter
E0 = {1: 360, 2: 1560}  # E0t: Depot earliest opening time (minutes), cumulative: Day 1=360, Day 2=1560 (=1200+360)
L0 = {1: 1200, 2: 2400} # L0t: Depot latest closing time (minutes), cumulative: Day 1=1200, Day 2=2400 (=1200+1200)

tau = 30 # τ: Reload/handling time at depot between two trips (minutes) parameter
C_cd = 2500 # C^cd: Cross-docking transfer (load exchange) cost parameter
C_dep = 200 # C^dep: Storage capacity of the depot (units) parameter

vehicle_speed = 60  # Average vehicle speed 60 km/h, I thought as a intra-city delivery and gave 60

# Dij parameter: Haversine distance matrix
D_dist = [[haversine_distance((ycoord[i], xcoord[i]), (ycoord[j], xcoord[j])) for j in N] for i in N]
D_dist = np.array(D_dist) # Converting into a numpy array

# Tij parameter: Travel time matrix, distance / vehicle speed in minutes * 60 because unit must be per km hour
T_time = [[(D_dist[i][j] / vehicle_speed) * 60 for j in N] for i in N]

# Δi parameter: Net load change at node i (positive at pickup nodes, negative at delivery nodes)
Delta = [0.0] * location_set
for s in S_set:
    if alpha[s] == 0:
        Delta[p_node[s]] += q_s[s]   # Pickup node: load increases, when alpa equals to 0
    Delta[d_node[s]] -= q_s[s]   # Delivery node: load decreases, all time

# I0 parameter: Initial depot inventory, shipments already at depot before planning horizon
I0 = sum(alpha[s] * q_s[s] for s in S_set)

# Defining the model and giving a name
VRP = gurobipy.Model("VRP")

# Defining the decision variables
x = VRP.addVars(N, N, K_set, T_set, R_set, lb=0, ub=1, vtype=GRB.BINARY, name='x') # 1 if vehicle k travels arc (i,j) on trip r on day t; 0 otherwise
w = VRP.addVars(K_set, T_set, R_set, lb=0, ub=1, vtype=GRB.BINARY, name='w') # 1 if trip r of vehicle k is active on day t; 0 otherwise
v = VRP.addVars(K_set, T_set, lb=0, ub=1, vtype=GRB.BINARY, name='v') # 1 if vehicle k is rented on day t; 0 otherwise
b = VRP.addVars(N, K_set, T_set, R_set, lb=0, vtype=GRB.CONTINUOUS, name='b') # Service start time at node i for vehicle k on trip r on day t
u = VRP.addVars(N, K_set, T_set, R_set, lb=0, vtype=GRB.CONTINUOUS, name='u') # Load of vehicle k after leaving node i on trip r on day t
l_out = VRP.addVars(K_set, T_set, R_set, lb=0, vtype=GRB.CONTINUOUS, name='l_out') # Load taken from depot at start of trip r by vehicle k on day t
l_in = VRP.addVars(K_set, T_set, R_set, lb=0, vtype=GRB.CONTINUOUS, name='l_in') # Load dropped off at depot at end of trip r by vehicle k on day t
delta_s = VRP.addVars(S_set, lb=0, ub=1, vtype=GRB.BINARY, name='delta') # 1 if shipment s changes vehicles at depot (cross-docking); 0 otherwise
I_inv = VRP.addVars(T_set, lb=0, vtype=GRB.CONTINUOUS, name='I') # Depot inventory at the end of day t


# Arc-based and constraint-specific tight Big-M values for gap reduction, I am trying to find the minimum values of the maximum for each big-M`s

# Big-M for Constraints 14 and 16
M_load_lower = {(j,k): Q[k] + max(0, Delta[j])  for j in N if j!=0 for k in K_set}

# Big-M for Constraints 15 and 17
M_load_upper = {(j,k): Q[k] + max(0,-Delta[j])  for j in N if j!=0 for k in K_set}

# Big-M for Constraint 18
M_lin = max(Q.values())

# Big-M for Constraint 21
Mij = np.zeros((location_set, location_set))
for i in range(location_set):
    for j in range(location_set):
        if i == j:
            continue
        Mij[i, j] = max(0, l[i] + s_time[i] + T_time[i][j] - e[j])

# Big-M for Constraint 22
M_tw_lo = {i: max(0, l[i] - e[i], e[i]) for i in N if i != 0}

# Big-M for Constraint 22
M_tw_hi = {i: max(0, max(L0.values()) - e[i]) for i in N if i != 0}

# Big-M for Constraint 24
M_seq = max(L0.values()) + sum(max(s_time[i] for i in N if i != 0) +
            max(T_time[i][j] for j in N) for i in N if i != 0) + tau

# Big-M for Constraint 27
M_prec = max(l[i] for i in N if i != 0) - min(e[i] for i in N if i != 0)

# Big-M for Constraint 28
M_arr = max(L0.values()) + max(s_time) + max(T_time[i][0] for i in N if i != 0)

# Big-M for Constraints 39 and 40
M_cd = {s: q_s[s] for s in S_set}


# Arc Pre-elimination: set x[i,j,k,t,r]=0 for arcs that are infeasible
for i in N:
    for j in N:
        if i == j: continue
        # Standard time window check the loop appplies to all days
        if j != 0 and e[i] + s_time[i] + T_time[i][j] > l[j] + 1e-3:
            for k in K_set:
                for t in T_set:
                    for r in R_set:
                        x[i, j, k, t, r].ub = 0
            continue  # No further check needed for this (i,j) pair
        # Day-specific time window feasibility check
        for t in T_set:
            if j == 0:
                # Return-to-depot: can node i finish and return by L0[t]?
                elimination = e[i] + s_time[i] + T_time[i][0] > L0[t] + 1e-3
            else:
                # Customer node j: can j finish and return to depot by L0[t]?
                elimination = e[j] + s_time[j] + T_time[j][0] > L0[t] + 1e-3
            if elimination:
                for k in K_set:
                    for r in R_set:
                        x[i, j, k, t, r].ub = 0


# Defining the objective function which is minimizing the total cost (fixed rental cost + distance per km cost + cross-docking(load exchange) cost)
VRP.setObjective(quicksum(F[k] * v[k, t] for k in K_set for t in T_set)
    +
    quicksum(A_cost[k] * D_dist[i, j] * x[i, j, k, t, r] for k in K_set for t in T_set for r in R_set for i in N for j in N if i != j)
    +
    C_cd * quicksum(delta_s[s] for s in S_set),
    GRB.MINIMIZE)

# Adding the constraints

# Constraint 2
VRP.addConstrs((w[k, t, r] <= v[k, t] for k in K_set for t in T_set for r in R_set), name="Constraint_2")

# Constraint 3
VRP.addConstrs((w[k, t, r] <= w[k, t, r_prime] for k in K_set for t in T_set for r in R_set for r_prime in R_set if r_prime < r), name="Constraint_3")

# Constraint 4
VRP.addConstrs((quicksum(x[0, j, k, t, r] for j in N if j != 0) == w[k, t, r] for k in K_set for t in T_set for r in R_set), name="Constraint_4")

# Constraint 5
VRP.addConstrs((quicksum(x[i, 0, k, t, r] for i in N if i != 0) == w[k, t, r] for k in K_set for t in T_set for r in R_set), name="Constraint_5")
 
# Constraint 6
VRP.addConstrs(
    (quicksum(x[i, j, k, t, r] for j in N if j != i) - quicksum(x[j, i, k, t, r] for j in N if j != i) == 0
    for i in N if i != 0 for k in K_set for t in T_set for r in R_set),
    name="Constraint_6"
)

# Constraint 7
VRP.addConstrs(
    (quicksum(x[p_node[s], j, k, t, r] for j in N if j != p_node[s] for k in K_set for t in T_set for r in R_set) == 1
    for s in S_set if alpha[s] == 0),
    name="Constraint_7"
)

# Constraint 8
VRP.addConstrs(
    (quicksum(x[d_node[s], j, k, t, r] for j in N if j != d_node[s] for k in K_set for t in T_set for r in R_set) == 1
    for s in S_set),
    name="Constraint_8"
)
 
# Constraints 9 and 10
VRP.addConstrs((l_out[k, t, r] <= Q[k] * w[k, t, r] for k in K_set for t in T_set for r in R_set), name="Constraint_9")

VRP.addConstrs((l_in[k, t, r]  <= Q[k] * w[k, t, r] for k in K_set for t in T_set for r in R_set), name="Constraint_10")

# Constraint 11
VRP.addConstrs((u[0, k, t, 1] == l_out[k, t, 1] for k in K_set for t in T_set), name="Constraint_11")

# Constraint 12
VRP.addConstrs((u[0, k, t, r] == l_out[k, t, r] for k in K_set for t in T_set for r in R_set if r > 1), name="Constraint_12")
 
# Constraint 13
for t in T_set:
    if t == 1:
        continue  # No previous day for t=1, handled by Constraint 11
    VRP.addConstr(
        quicksum(u[0, k, t, r] for k in K_set for r in R_set)
        ==
        quicksum(u[0, k, t - 1, r] for k in K_set for r in R_set)
        + quicksum(Delta[j] * x[i, j, k, t - 1, r]
                   for k in K_set for r in R_set for i in N for j in N if i != j)
        - quicksum(l_in[k, t - 1, r] for k in K_set for r in R_set)
        + quicksum(l_out[k, t, r] for k in K_set for r in R_set),
        name= "Constraint_13"
    )
 
# Constraint 14
VRP.addConstrs(
    (u[j, k, t, r] >= u[0, k, t, r] + Delta[j] - M_load_lower[j, k] * (1 - x[0, j, k, t, r])
    for j in N if j != 0 for k in K_set for t in T_set for r in R_set),
    name="Constraint_14"
)

# Constraint 15
VRP.addConstrs(
    (u[j, k, t, r] <= u[0, k, t, r] + Delta[j] + M_load_upper[j, k] * (1 - x[0, j, k, t, r])
    for j in N if j != 0 for k in K_set for t in T_set for r in R_set),
    name="Constraint_15"
)
 
# Constraint 16
VRP.addConstrs(
    (u[j, k, t, r] >= u[i, k, t, r] + Delta[j] - M_load_lower[j, k] * (1 - x[i, j, k, t, r])
    for i in N for j in N if i != j and j != 0
    for k in K_set for t in T_set for r in R_set),
    name="Constraint_16"
)

# Constraint 17
VRP.addConstrs(
    (u[j, k, t, r] <= u[i, k, t, r] + Delta[j] + M_load_upper[j, k] * (1 - x[i, j, k, t, r])
    for i in N for j in N if i != j and j != 0
    for k in K_set for t in T_set for r in R_set),
    name="Constraint_17"
)
 
# Constraint 18
VRP.addConstrs(
    (l_in[k, t, r] <= u[i, k, t, r] + M_lin * (1 - x[i, 0, k, t, r])
    for i in N if i != 0 for k in K_set for t in T_set for r in R_set),
    name="Constraint_18_ub"
)
VRP.addConstrs(
    (l_in[k, t, r] >= u[i, k, t, r] - M_lin * (1 - x[i, 0, k, t, r])
    for i in N if i != 0 for k in K_set for t in T_set for r in R_set),
    name="Constraint_18_lb"
)
 
# Constraint 19
VRP.addConstrs((u[i, k, t, r] <= Q[k] * w[k, t, r] for i in N for k in K_set for t in T_set for r in R_set), name="Constraint_19")

# Constraint 20
VRP.addConstrs(
    (u[j, k, t, r] <= Q[k] * quicksum(x[i, j, k, t, r] for i in N if i != j)
    for j in N if j != 0 for k in K_set for t in T_set for r in R_set),
    name="Constraint_20"
)
 
# Constraint 21
VRP.addConstrs(
    (b[j, k, t, r] >= b[i, k, t, r] + s_time[i] + T_time[i][j] - Mij[i, j] * (1 - x[i, j, k, t, r])
    for i in N for j in N if i != j and j != 0 for k in K_set for t in T_set for r in R_set),
    name="Constraint_21"
)

# Constraint 22
VRP.addConstrs(
    (b[i, k, t, r] >= e[i] - M_tw_lo[i] * (1 - quicksum(x[i, j, k, t, r] for j in N if j != i))
    for i in N if i != 0 for k in K_set for t in T_set for r in R_set),
    name="Constraint_22_lb"
)
VRP.addConstrs(
    (b[i, k, t, r] <= l[i] + M_tw_hi[i] * (1 - quicksum(x[i, j, k, t, r] for j in N if j != i))
    for i in N if i != 0 for k in K_set for t in T_set for r in R_set),
    name="Constraint_22_ub"
)

# Constraint 23
VRP.addConstrs((b[0, k, t, r] >= E0[t] * w[k, t, r] for k in K_set for t in T_set for r in R_set), name="Constraint_23_lb")

VRP.addConstrs((b[0, k, t, r] <= L0[t] * w[k, t, r] for k in K_set for t in T_set for r in R_set), name="Constraint_23_ub")
 
# Constraint 24
VRP.addConstrs(
    (b[0, k, t, r_prime] >= b[i, k, t, r] + s_time[i] + T_time[i][0] + tau
    - M_seq * (2 - x[i, 0, k, t, r] - w[k, t, r_prime])
    for i in N if i != 0
    for k in K_set for t in T_set for r in R_set for r_prime in R_set if r_prime > r),
    name="Constraint_24"
)
 
# # Old Constraint 25, I commented it because upper bound of Constraint 22 already forces b[i,k,t,r] to be 0 when x[i,j,k,t,r] is 0 for all j, so it is redundant
# VRP.addConstrs(
#     b[i, k, t, r] <= l[i] * quicksum(x[i, j, k, t, r] for j in N if j != i)
#     for i in N if i != 0 for k in K_set for t in T_set for r in R_set, name = "Constraint_25"
# )
 
# Constraint 25
VRP.addConstrs(
    (b[0, k, t, r]
    + quicksum(s_time[i] * quicksum(x[i, j, k, t, r] for j in N if j != i) for i in N if i != 0)
    + quicksum(T_time[i][j] * x[i, j, k, t, r] for i in N for j in N if i != j)
    <= L0[t]
    for k in K_set for t in T_set for r in R_set),
    name="Constraint_25"
)
 
# Constraint 26
VRP.addConstrs(
    (quicksum(t * quicksum(x[p_node[s], j, k, t, r]
                               for j in N if j != p_node[s]
                               for k in K_set for r in R_set)
              for t in T_set)
     <=
     quicksum(t * quicksum(x[d_node[s], j, k, t, r]
                               for j in N if j != d_node[s]
                               for k in K_set for r in R_set)
              for t in T_set)
     for s in S_set if alpha[s] == 0),
    name="Constraint_26"
)

# Constraint 27
VRP.addConstrs(
    (b[p_node[s], k, t, r] <= b[d_node[s], k, t, r]
    + M_prec * (2
        - quicksum(x[p_node[s], j, k, t, r] for j in N if j != p_node[s])
        - quicksum(x[d_node[s], j, k, t, r] for j in N if j != d_node[s]))
    for s in S_set if alpha[s] == 0
    for k in K_set for t in T_set for r in R_set),
    name="Constraint_27"
)


# Constraint 28
VRP.addConstrs(
    (b[0, k_d, t, r_d]
    >= b[i, k_p, t, r_p] + s_time[i] + T_time[i][0]
    - M_arr * (1 - x[i, 0, k_p, t, r_p])
    - M_arr * (2
        - quicksum(x[p_node[s], j, k_p, t, r_p] for j in N if j != p_node[s])
        - quicksum(x[d_node[s], j, k_d, t, r_d] for j in N if j != d_node[s]))
    for s in S_set if alpha[s] == 0
    for i in N if i != 0
    for t in T_set
    for k_p in K_set for r_p in R_set
    for k_d in K_set for r_d in R_set),
    name="Constraint_28"
)
 
# Constraint 29
VRP.addConstrs(
    (quicksum(D_dist[i, j] * x[i, j, k, t, r] for r in R_set for i in N for j in N if i != j) <= H[k, t]
    for k in K_set for t in T_set),
    name="Constraint_29"
)
 
# # Constraint 30
# VRP.addConstrs(I0 = quicksum(alpha[s] * q_s[s] for s in S_set), name = "Constraint_30") # It is already computed as a parameter above thats why I did not include it
 
# Constraints 31 and 32
for t in T_set:
    prev_inv = I0 if t == 1 else I_inv[t - 1]
 
    # Constraint 31
    VRP.addConstr(
        I_inv[t] == prev_inv
        + quicksum(q_s[s] * quicksum(x[p_node[s], j, k, t, r] for j in N if j != p_node[s] for k in K_set for r in R_set)
                   for s in S_set if alpha[s] == 0)
        - quicksum(q_s[s] * quicksum(x[d_node[s], j, k, t, r] for j in N if j != d_node[s] for k in K_set for r in R_set)
                   for s in S_set),
        name="Constraint_31"
    )

    # Constraint 32
    VRP.addConstr(
        quicksum(l_out[k, t, r] for k in K_set for r in R_set)
        <= prev_inv
        + quicksum(q_s[s] * quicksum(x[p_node[s], j, k, t, r] for j in N if j != p_node[s] for k in K_set for r in R_set)
                   for s in S_set if alpha[s] == 0),
        name="Constraint_32"
    )
 
# Constraint 33
VRP.addConstrs((I_inv[t] <= C_dep for t in T_set), name="Constraint_33")
 
# Constraints 34 and 35
VRP.addConstrs(
    (delta_s[s] >= quicksum(x[p_node[s], j, k, t, r] for j in N if j != p_node[s] for r in R_set)
                - quicksum(x[d_node[s], j, k, t, r] for j in N if j != d_node[s] for r in R_set)
    for s in S_set if alpha[s] == 0
    for k in K_set for t in T_set),
    name="Constraint_34"
)

VRP.addConstrs(
    (delta_s[s] >= quicksum(x[d_node[s], j, k, t, r] for j in N if j != d_node[s] for r in R_set)
                - quicksum(x[p_node[s], j, k, t, r] for j in N if j != p_node[s] for r in R_set)
    for s in S_set if alpha[s] == 0
    for k in K_set for t in T_set),
    name="Constraint_35"
)
 
# Constraints 36 and 37
VRP.addConstrs(
    (delta_s[s] >= quicksum(x[p_node[s], j, k, t, r] for j in N if j != p_node[s] for r in R_set)
                - quicksum(x[d_node[s], j, k, t_prime, r_prime] for j in N if j != d_node[s] for r_prime in R_set)
    for s in S_set if alpha[s] == 0
    for k in K_set for t in T_set for t_prime in T_set if t_prime > t),
    name="Constraint_36"
)

VRP.addConstrs(
    (delta_s[s] >= quicksum(x[d_node[s], j, k, t_prime, r_prime] for j in N if j != d_node[s] for r_prime in R_set)
                - quicksum(x[p_node[s], j, k, t, r] for j in N if j != p_node[s] for r in R_set)
    for s in S_set if alpha[s] == 0
    for k in K_set for t in T_set for t_prime in T_set if t_prime > t),
    name="Constraint_37"
)
 
# Constraint 38
VRP.addConstrs(
    (delta_s[s] <= 2 - quicksum(x[p_node[s], j, k, t, r] for j in N if j != p_node[s] for r in R_set)
                     - quicksum(x[d_node[s], j, k, t, r] for j in N if j != d_node[s] for r in R_set)
    for s in S_set if alpha[s] == 0
    for k in K_set for t in T_set),
    name="Constraint_38"
)

# Constraint 39
VRP.addConstrs(
    (l_in[k, t, r] >= q_s[s] * quicksum(x[p_node[s], j, k, t, r] for j in N if j != p_node[s]) - M_cd[s] * (1 - delta_s[s])
    for s in S_set for k in K_set for t in T_set for r in R_set),
    name="Constraint_39"
)

# Constraint 40
VRP.addConstrs(
    (l_out[k, t, r] >= q_s[s] * quicksum(x[d_node[s], j, k, t, r] for j in N if j != d_node[s]) - M_cd[s] * (1 - delta_s[s])
    for s in S_set for k in K_set for t in T_set for r in R_set),
    name="Constraint_40"
)

# Constraint 41
VRP.addConstrs(
    (l_out[k, t, r] >= quicksum(
        q_s[s] * quicksum(x[d_node[s], j, k, t, r] for j in N if j != d_node[s])
        for s in S_set if alpha[s] == 1)
    for k in K_set for t in T_set for r in R_set),
    name="Constraint_41"
)

# Constraint 42
VRP.addConstrs((delta_s[s] == 0 for s in S_set if alpha[s] == 1), name="Constraint_42")
 

# Adding the optimisation parameters, settings
VRP.setParam("TimeLimit", 7200)      # Time limit, decide the time
VRP.setParam("MIPFocus", 2)          # 0 balance, 1 focuses on finding a feasible solution (incumbent), 2 focuses on gap improvement, 3 focuses on incumbent improvement, solution improvement
VRP.setParam("Heuristics", 0.3)      # Effort given to heuristics; 0 closed, 1 max
VRP.setParam("NoRelHeurTime", 180)   # First NoReL Heuristic then B&B, decide the time, now it is 180 seconds for NoReL
VRP.setParam("RINS", 10)             # Relaxation Induced Neighborhood Search, it tries 10 times, it improves the current incumbent 
VRP.setParam("Presolve", 2)          # 0 closed, 1 conservative, 2 aggressive; it tightens bounds
VRP.setParam("Cuts", 3)              # Level of cuts: 0 closed, 1 low, 2 medium, 3 aggressive
VRP.setParam("ImproveStartTime", 6300) # It makes MIP focus 3 after 6300 seconds
VRP.setParam("Symmetry", 2)          # Detects and removes symmetry
VRP.setParam("Threads", 6)           # Number of CPU threads used concurrently, my CPU has 12 threads

  

# Finding the solution
VRP.update()
VRP.optimize()

status = VRP.status
object_Value = VRP.objVal

print()
print("Model status is: ", status)
print()
print("Objective Function value is: ", object_Value)


# Printing the decision variables which are not zeros
if status != 3 and status != 4:
    for var in VRP.getVars():
        if VRP.objVal < 1e99 and var.x != 0:
            if var.VarName.startswith('b['):
                parts = var.VarName[2:-1].split(',')
                i_idx, k_idx, t_idx, r_idx = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                if i_idx != 0:
                    visited = any(x[i_idx, j, k_idx, t_idx, r_idx].X > 0.5 for j in N if j != i_idx)
                    if not visited:
                        continue
            print('%s %f' % (var.VarName, var.x))

# Print section
if VRP.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] and VRP.SolCount > 0:

    routes = {} # Defining an empty dict to track the established routes

    # Extract routes per vehicle, day, and trip
    for k in K_set:
        for t in T_set:
            for r in R_set:
                if w[k, t, r].X > 0.5:
                    route   = [0]
                    current = 0

                    while True:
                        next_nodes = [j for j in N if j != current and x[current, j, k, t, r].X > 0.5]
                        if not next_nodes:
                            break
                        next_node = next_nodes[0]
                        route.append(next_node)
                        current = next_node
                        if current == 0:
                            break

                    routes[(k, t, r)] = route

    def route_distance(route): # Calculating the distance function
        return sum(D_dist[route[i], route[i + 1]] for i in range(len(route) - 1))

    # Printing the routes
    print("\nEstablished Routes:")
    for (k, t, r), route in routes.items():
        print(f"Vehicle {k} | Day {t} | Trip {r} | Route: {route}")

    # Printing the route summary
    print("\nRoute Summary:")
    for (k, t, r), route in routes.items():
        dist = route_distance(route)
        print(f"Vehicle {k} | Day {t} | Trip {r} | Distance: {dist:.4f} km | Load Out: {l_out[k, t, r].X:.2f} | Cap: {Q[k]}")

    # Printing shipment outcomes with vehicle/day/trip detail and checks
    print("\nShipment Outcomes:")
    for s in S_set:
        # Find pickup vehicle/day/trip
        pickup_k = pickup_t = pickup_r = None
        if alpha[s] == 0:
            for k in K_set:
                for t in T_set:
                    for r in R_set:
                        if any(x[p_node[s], j, k, t, r].X > 0.5 for j in N if j != p_node[s]):
                            pickup_k, pickup_t, pickup_r = k, t, r

        # Find delivery vehicle/day/trip
        delivery_k = delivery_t = delivery_r = None
        for k in K_set:
            for t in T_set:
                for r in R_set:
                    if any(x[d_node[s], j, k, t, r].X > 0.5 for j in N if j != d_node[s]):
                        delivery_k, delivery_t, delivery_r = k, t, r

        if delta_s[s].X > 0.5:
            # Classify cross-docking type
            if pickup_k != delivery_k and pickup_t != delivery_t:
                cd_type = "different vehicle, different day"
            elif pickup_k != delivery_k:
                cd_type = "different vehicle, same day"
            elif pickup_t != delivery_t:
                cd_type = "same vehicle, different day"
            else:
                cd_type = "same vehicle same day — check the constraints"

            pickup_b   = b[p_node[s], pickup_k,   pickup_t,   pickup_r].X
            delivery_b = b[d_node[s], delivery_k, delivery_t, delivery_r].X
            p_tw = "OK" if e[p_node[s]] <= pickup_b   <= l[p_node[s]]   else "TW VIOLATED"
            d_tw = "OK" if e[d_node[s]] <= delivery_b <= l[d_node[s]]   else "TW VIOLATED"
            prec = "OK" if pickup_t <= delivery_t else "PREDECESSOR VIOLATED"

            print(f"Shipment {s+1} [p={p_node[s]}, d={d_node[s]}, q={q_s[s]}]: CROSS-DOCKING ({cd_type})")
            print(f"    Pickup  : V{pickup_k}  | Day {pickup_t} | Trip {pickup_r} | b={pickup_b:.1f}   TW=[{e[p_node[s]]}, {l[p_node[s]]}] {p_tw}")
            print(f"    Delivery: V{delivery_k} | Day {delivery_t} | Trip {delivery_r} | b={delivery_b:.1f} TW=[{e[d_node[s]]}, {l[d_node[s]]}] {d_tw}")
            print(f"    Predecessor: pickup Day {pickup_t} <= delivery Day {delivery_t} — {prec}")

        elif alpha[s] == 1:
            delivery_b = b[d_node[s], delivery_k, delivery_t, delivery_r].X
            d_tw = "OK" if e[d_node[s]] <= delivery_b <= l[d_node[s]] else "TW VIOLATED"
            print(f"Shipment {s+1} [p=depot, d={d_node[s]}, q={q_s[s]}]: Direct (pre-positioned at depot)")
            print(f"    Delivery: V{delivery_k} | Day {delivery_t} | Trip {delivery_r} | b={delivery_b:.1f} TW=[{e[d_node[s]]}, {l[d_node[s]]}] {d_tw}")

        else:
            pickup_b   = b[p_node[s], pickup_k,   pickup_t,   pickup_r].X
            delivery_b = b[d_node[s], delivery_k, delivery_t, delivery_r].X
            p_tw = "OK" if e[p_node[s]] <= pickup_b   <= l[p_node[s]]   else "TW VIOLATED"
            d_tw = "OK" if e[d_node[s]] <= delivery_b <= l[d_node[s]]   else "TW VIOLATED"
            if pickup_k == delivery_k and pickup_t == delivery_t and pickup_r == delivery_r:
                desc = "same vehicle, same day, same trip"
            elif pickup_k == delivery_k and pickup_t == delivery_t:
                desc = f"same vehicle, same day, Trip {pickup_r}→{delivery_r}"
            else:
                desc = f"same vehicle, Day {pickup_t}→{delivery_t}"
            print(f"Shipment {s+1} [p={p_node[s]}, d={d_node[s]}, q={q_s[s]}]: DIRECT DELIVERY ({desc})")
            print(f"    Pickup  : V{pickup_k}  | Day {pickup_t} | Trip {pickup_r} | b={pickup_b:.1f}   TW=[{e[p_node[s]]}, {l[p_node[s]]}] {p_tw}")
            print(f"    Delivery: V{delivery_k} | Day {delivery_t} | Trip {delivery_r} | b={delivery_b:.1f} TW=[{e[d_node[s]]}, {l[d_node[s]]}] {d_tw}")

    # Printing depot inventory levels
    print("\nDepot Inventory Levels:")
    print(f"  Day 0 (initial I0): {I0:.2f}")
    for t in T_set:
        print(f"  End of Day {t}: {I_inv[t].X:.2f}")
        print()

# Visualization section
if VRP.SolCount > 0:
    from matplotlib.lines import Line2D

    all_kr      = sorted(set((k, r) for (k, t, r) in routes.keys()))
    cmap        = plt.cm.get_cmap('tab20' if len(all_kr) <= 20 else 'hsv', len(all_kr))
    trip_colors = {kr: cmap(i) for i, kr in enumerate(all_kr)}

    pickup_nodes   = set(P_set)
    delivery_nodes = set(D_set)

    rental    = sum(F[k] * v[k, t].X for k in K_set for t in T_set)
    dist_cost = sum(A_cost[k] * D_dist[i, j] * x[i, j, k, t, r].X
                    for k in K_set for t in T_set for r in R_set
                    for i in N for j in N if i != j)
    cd_cost   = C_cd * sum(delta_s[s].X for s in S_set)

    fig, axes = plt.subplots(1, n_days, figsize=(18, 8))
    if n_days == 1:
        axes = [axes]

    fig.suptitle(
        f"Established Routes with Mathematical Formulation — Objective Function Value: {object_Value:,.2f} AUD\n",
        fontsize=13, fontweight='bold'
    )

    for day_idx, t in enumerate(T_set):
        ax = axes[day_idx]
        ax.set_title(f"Day {t}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        # Plot all nodes
        for i in N:
            xi, yi = xcoord[i], ycoord[i]
            if i == 0:
                ax.plot(xi, yi, marker='*', color='black', markersize=18, zorder=5)
                ax.annotate('DEPOT', (xi, yi), textcoords="offset points",
                            xytext=(6, 6), fontsize=8, fontweight='bold')
            else:
                if i in pickup_nodes and i in delivery_nodes:
                    color, marker = '#ff7f0e', 's'
                elif i in pickup_nodes:
                    color, marker = '#9467bd', '^'
                else:
                    color, marker = '#17becf', 'o'
                ax.plot(xi, yi, marker=marker, color=color, markersize=10,
                        markeredgecolor='black', markeredgewidth=0.8, zorder=4)
                ax.annotate(str(i), (xi, yi), textcoords="offset points",
                            xytext=(5, 5), fontsize=8)

        # Draw routes with arrows
        active_trips = [(k, day, r) for (k, day, r) in sorted(routes.keys()) if day == t]
        for (k, day, r) in active_trips:
            route = routes[(k, day, r)]
            color = trip_colors.get((k, r), '#333333')
            for idx in range(len(route) - 1):
                ii, jj = route[idx], route[idx + 1]
                ax.annotate(
                    '', xy=(xcoord[jj], ycoord[jj]), xytext=(xcoord[ii], ycoord[ii]),
                    arrowprops=dict(
                        arrowstyle='->', color=color, lw=2.0,
                        linestyle='-',
                        connectionstyle='arc3,rad=0.05'
                    ), zorder=3
                )

        ax.grid(True, alpha=0.2)

        # Legend: vehicles and trips
        legend_handles = []
        for (k, day, r) in active_trips:
            legend_handles.append(Line2D(
                [0], [0], color=trip_colors.get((k, r), '#333333'),
                linewidth=2, linestyle='-', label=f'V{k} Trip{r}'
            ))

        legend_handles += [
            Line2D([0], [0], marker='*', color='black',   markersize=12, linestyle='None', label='Depot'),
            Line2D([0], [0], marker='^', color='#9467bd', markersize=8,  linestyle='None', label='Pickup node'),
            Line2D([0], [0], marker='o', color='#17becf', markersize=8,  linestyle='None', label='Delivery node'),
            Line2D([0], [0], marker='s', color='#ff7f0e', markersize=8,  linestyle='None', label='Pickup+Delivery node'),
        ]

        ax.legend(handles=legend_handles, loc='best', fontsize=7, framealpha=0.9)

    plt.tight_layout()
    plt.show()
    