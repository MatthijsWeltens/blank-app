import streamlit as st
st.markdown(
    """
    <style>
    .custom-slider-label {
        font-size: 24px;
        font-weight: bold;
        color: black;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFCC00;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    '<h1 style="color:#D40511; text-align: center;">KPIs Linehaul Planning</h1>',
    unsafe_allow_html=True
)


#Code
import numpy as np
import pandas as pd
from itertools import combinations, permutations
import random
import math

# Parameters
truck_capacity_diesel = 26
truck_capacity_electric = 18

cost_per_km_diesel = 1
cost_per_km_electric = 1.1
cost_per_hour = 40

# Terminals
terminal_ids = ["ALK", "AME", "AMS", "ANT", "ARN", "AWS", "BEE", "DRA", "EIN", "FOE", "GRA",
                "GUL", "HEN", "HER", "OUD", "ROO", "ROT", "SVC", "TER", "UTR", "ZLB", "ZWO"]

n = len(terminal_ids)

loading_and_unloading_cost = 0.67 # Cost per pallet, because one pallets takes 1 minute
cross_dock_cost_per_pallet = 0.50 # Extra cross docking cost per pallet
fixed_cross_dock_cost = 20 # Cross takes 30 minutes which costs 20 euro in salary of the driver

# Below is the maximum time an optimized route from and to the cross docks can take
max_route_time = 195 # 195 + 30 + 195 minutes = 7 hours

# Hyper parameters
st.markdown('<div class="custom-slider-label">Select the maximum kilometer range of electric trucks</div>', unsafe_allow_html=True)

km_range = st.slider(
    label=" ",  # Lege string om label weg te laten
    min_value=0,
    max_value=250,
    step=5,
    value=75,
    key="km_slider"
)
minimum_load_diesel = 22 # Accept direct deliveries from this amount of pallets
cross_dock_possible_loc = ['EIN', 'UTR']
cross_dock_possible_id = [terminal_ids.index(t) for t in cross_dock_possible_loc]
times_worse = 2.0 # We don't use cross docking when the transportation costs is this times higher

# Loading data
data = pd.read_excel("DHL Data for Python.xlsx", sheet_name=None, index_col=0)
df_km = data['Distance'].loc[terminal_ids, terminal_ids].to_numpy()
df_time = data['Time'].loc[terminal_ids, terminal_ids].to_numpy()
df_demand = data['Expected average #pallets'].loc[terminal_ids, terminal_ids].to_numpy()

# Determine the direct routes (electric vs diesel)
route_electric = df_km <= km_range
route_not_999 = df_km != 999

# Scale data per day
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
scale = [0.95, 1.2, 0.9, 1.1, 0.85]

# Delayed FOE-related demand
ein_foe_from_yesterday = [83, 67, 99, 94, 94]

ein_i_from_yesterday = np.array([
    [0., 0., 0., 1., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 5., 0., 0., 7.],
    [0., 0., 0., 2., 0., 3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 6., 0., 0., 9.],
    [0., 0., 0., 1., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 5., 0., 0., 7.],
    [0., 0., 0., 2., 0., 3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 6., 0., 0., 8.],
    [0., 0., 0., 1., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 5., 0., 0., 6.],
    ], dtype=int)

too_late_foe = sum(ein_foe_from_yesterday) + np.sum(ein_i_from_yesterday)

#%% Helper Functions

########################### Annealing With 2 Opt ################################

def route_distance(route, df_km):
    return sum(df_km[a, b] for a, b in zip(route, route[1:]))    

def route_time(route, df_time, pallets_for_dock, depot_idx):
    """
    Sum of travel time + loading time (1 min per pallet) at every stop except the depot.
    
    route:            list of terminal‐indices, includes depot_idx somewhere
    df_time:          matrix of travel times (minutes)
    pallets_for_dock: 1-D array of pallet counts for this cross‐dock
    depot_idx:        the index of the depot in your terminal list
    """
    # 1) travel time over each leg
    travel = sum(df_time[a, b] for a, b in zip(route, route[1:]))
    # 2) loading: 1 minute per pallet at each non‐depot stop
    loading = 2 * sum(pallets_for_dock[i] for i in route if i != depot_idx)
    return travel + loading

def route_cost(route, df_km, df_time, pallets_for_dock, depot_idx):
    """
    Compute total cost and number of pallets for a route that includes depot_idx.
    
    route:            list of terminal‐indices, includes depot_idx somewhere
    df_km, df_time:   your distance and time matrices
    pallets_for_dock: 1-D array of pallet counts
    depot_idx:        the index of the depot
    Returns: (total_cost, num_pallets)
    """
    # 1) distance & time (with loading baked in)
    dist = route_distance(route, df_km)
    tt   = route_time(route, df_time, pallets_for_dock, depot_idx)
    
    # 2) count all pallets except the depot
    num_pallets = sum(pallets_for_dock[i] for i in route if i != depot_idx)

    # 3) compute cost components
    load_cost = 2 * loading_and_unloading_cost * num_pallets
    time_cost = (tt / 60) * cost_per_hour
    diesel    = dist * cost_per_km_diesel

    return load_cost + time_cost + diesel, num_pallets

def two_opt_path(route, df_km, df_time):
    """2-opt on a route > optimizes the route and fixes the depot at the end of the route"""
    best = route[:]
    improved = True
    while improved:
        improved = False
        n = len(best)
        # i goes from 0 all the way up to n-3, so we can reverse segments
        for i in range(0, n - 2):
            # j goes from i+1 up to n-2, so best[n-1] (the depot) stays in place
            for j in range(i + 1, n - 1):
                candidate = best[:i] + best[i:j+1][::-1] + best[j+1:]
                if route_distance(candidate, df_km) < route_distance(best, df_km):
                    best = candidate
                    improved = True
        route = best
    return best
 
def initial_groups(valid_ids, pallets_for_dock, cross_dock_index, df_time, max_route_time):
    """Greedy fill: endpoints = cross_dock_index; check capacity and max driving time"""
    remaining = valid_ids.copy()
    groups = []
    while remaining:
        grp = [cross_dock_index]
        load = pallets_for_dock[cross_dock_index]
        # Voor de initiële terminal proberen we er één in te gooien:
        first = remaining.pop(0)
        # Tijd voor [first → depot]
        tentative_route = [first, cross_dock_index]
        if pallets_for_dock[first] <= truck_capacity_diesel and route_time(tentative_route, df_time, pallets_for_dock, cross_dock_index) <= max_route_time:
            grp.append(first)
            load += pallets_for_dock[first]
        else:
            continue

        i = 0
        while i < len(remaining):
            t = remaining[i]
            # Capacity OK?
            if load + pallets_for_dock[t] > truck_capacity_diesel:
                i += 1
                continue
            # Tijd OK?
            tentative_route = sorted((x for x in grp if x != cross_dock_index),
                                     key=lambda x: df_km[x, cross_dock_index],
                                     reverse=True) + [t, cross_dock_index]
            if route_time(tentative_route, df_time, pallets_for_dock, cross_dock_index) >= max_route_time:
                i += 1
                continue

            # Beide OK: voeg toe
            grp.append(t)
            load += pallets_for_dock[t]
            remaining.pop(i)

        groups.append(grp)
    return groups

def simulated_annealing(groups, df_km, df_time, pallets_for_dock, cross_dock_index,
                        max_route_time, alpha=1, beta=1,
                        initial_temp=5000, cooling_rate=0.995, max_iter=5000):
    current = [g[:] for g in groups]
    # Unpack only the cost (index 0 of the tuple)
    cur_cost = sum(route_cost(g, df_km, df_time, pallets_for_dock, cross_dock_index)[0] for g in current)
    cur_time = sum(route_time(g, df_time, pallets_for_dock, cross_dock_index) for g in current)
    cur_combined = alpha * cur_cost + beta * cur_time

    best, best_combined = [g[:] for g in current], cur_combined
    T = initial_temp

    for _ in range(max_iter):
        # 1) swap
        i, j = random.sample(range(len(current)), 2)
        non_i = [t for t in current[i] if t != cross_dock_index]
        non_j = [t for t in current[j] if t != cross_dock_index]
        if not non_i or not non_j:
            T *= cooling_rate; continue

        t_i, t_j = random.choice(non_i), random.choice(non_j)
        cand = [g[:] for g in current]
        cand[i].remove(t_i); cand[j].remove(t_j)
        cand[i].append(t_j);    cand[j].append(t_i)

        # capacity check
        if sum(pallets_for_dock[x] for x in cand[i]) > truck_capacity_diesel or \
           sum(pallets_for_dock[x] for x in cand[j]) > truck_capacity_diesel:
            T *= cooling_rate; continue

        # 2) 2-opt en tijdscheck per gewijzigde route
        valid = True
        for gidx in (i, j):
            route = sorted((x for x in cand[gidx] if x != cross_dock_index),
                           key=lambda x: df_km[x, cross_dock_index], reverse=True) + [cross_dock_index]
            opt_route = two_opt_path(route, df_km, df_time)
            # tijd check
            if route_time(opt_route, df_time, pallets_for_dock, cross_dock_index) > max_route_time:
                valid = False
                break
            cand[gidx] = opt_route[:-1] + [cross_dock_index]

        if not valid:
            T *= cooling_rate; continue

        # 3) objective: again take only the cost part
        new_cost = sum(route_cost(g, df_km, df_time, pallets_for_dock, cross_dock_index)[0] for g in cand)
        new_time = sum(route_time(g, df_time, pallets_for_dock, cross_dock_index) for g in cand)
        new_combined = alpha * new_cost + beta * new_time

        Δ = new_combined - cur_combined
        if Δ < 0 or random.random() < math.exp(-Δ / T):
            current, cur_combined = cand, new_combined
            if new_combined < best_combined:
                best, best_combined = [g[:] for g in cand], new_combined

        T *= cooling_rate

    return best  

def two_opt_path_out(route, df_km):
    """
    2-opt that never moves route[0] (the dock).
    route = [dock, a, b, c]
    """
    best = route[:]
    improved = True
    n = len(best)
    while improved:
        improved = False
        # start at 1 so index 0 (the dock) is never moved
        for i in range(1, n - 2):
            # j can go up to n-1 (the last stop)
            for j in range(i + 1, n):
                cand = best[:i] + best[i:j+1][::-1] + best[j+1:]
                if route_distance(cand, df_km) < route_distance(best, df_km):
                    best = cand
                    improved = True
        route = best
    return best

def initial_groups_out(valid_ids, pallets_for_dock, dock_idx, df_time, max_route_time):
    """
    Exactly like your inbound initial_groups, but we *never* append
    the dock at the end—only at the front.
    """
    remaining = valid_ids.copy()
    groups = []
    while remaining:
        grp = [dock_idx]
        load = pallets_for_dock[dock_idx]  # this will be zero
        first = remaining.pop(0)
        # test depot→first
        tentative = [dock_idx, first]
        if pallets_for_dock[first] <= truck_capacity_diesel and \
           route_time(tentative, df_time, pallets_for_dock, dock_idx) <= max_route_time:
            grp.append(first)
            load += pallets_for_dock[first]
        else:
            continue

        i = 0
        while i < len(remaining):
            t = remaining[i]
            if load + pallets_for_dock[t] > truck_capacity_diesel:
                i += 1; continue
            tentative = grp + [t]
            if route_time(tentative, df_time, pallets_for_dock, dock_idx) > max_route_time:
                i += 1; continue
            grp.append(t)
            load += pallets_for_dock[t]
            remaining.pop(i)
        groups.append(grp)
    return groups

def simulated_annealing_out(groups, df_km, df_time, pallets_for_dock,
                            dock_idx, max_route_time,
                            alpha=1, beta=1,
                            initial_temp=5000, cooling_rate=0.995, max_iter=5000):
    # Same structure as inbound SA, but using two_opt_path_out
    current = [g[:] for g in groups]
    cur_cost = sum(route_cost(g, df_km, df_time, pallets_for_dock, dock_idx)[0] for g in current)
    cur_time = sum(route_time(g, df_time, pallets_for_dock, dock_idx) for g in current)
    cur_obj  = alpha*cur_cost + beta*cur_time

    best, best_obj = [g[:] for g in current], cur_obj
    T = initial_temp

    for _ in range(max_iter):
        i,j = random.sample(range(len(current)),2)
        non_i = [t for t in current[i] if t!=dock_idx]
        non_j = [t for t in current[j] if t!=dock_idx]
        if not non_i or not non_j:
            T *= cooling_rate; continue

        t_i, t_j = random.choice(non_i), random.choice(non_j)
        cand = [g[:] for g in current]
        # swap
        cand[i].remove(t_i); cand[j].remove(t_j)
        cand[i].append(t_j);    cand[j].append(t_i)

        # capacity check
        if sum(pallets_for_dock[x] for x in cand[i]) > truck_capacity_diesel or \
           sum(pallets_for_dock[x] for x in cand[j]) > truck_capacity_diesel:
            T *= cooling_rate; continue

        # 2-opt + time check
        valid = True
        for grp_idx in (i,j):
            route0 = cand[grp_idx]
            # build initial route: dock first, then origins in current order
            # (you could re-sort by farthest-first, but we keep current order)
            opt_route = two_opt_path_out(route0, df_km)
            if route_time(opt_route, df_time, pallets_for_dock, dock_idx) > max_route_time:
                valid = False; break
            cand[grp_idx] = opt_route

        if not valid:
            T *= cooling_rate; continue

        # new objective
        new_cost = sum(route_cost(g, df_km, df_time, pallets_for_dock, dock_idx)[0] for g in cand)
        new_time = sum(route_time(g, df_time, pallets_for_dock, dock_idx) for g in cand)
        new_obj  = alpha*new_cost + beta*new_time

        Δ = new_obj - cur_obj
        if Δ < 0 or random.random() < math.exp(-Δ/T):
            current, cur_obj = cand, new_obj
            if new_obj < best_obj:
                best, best_obj = [g[:] for g in cand], new_obj
        T *= cooling_rate

    return best

########################### Final Routes ################################

# Function to calculate the total demand for a destination
def total_demand_for_destination(remaining_demand, destination_index):
    return np.sum(remaining_demand[:, destination_index])

# Function to calculate the travel distance for a given route (no need to return to the start)
def calculate_route_cost(route, destination_index, remaining_demand):
    total_cost = 0
    total_pallets_loaded = 0  # Track pallets loaded for cost calculations
    
    # Calculate the cost of visiting each origin in the route
    last_origin = route[0] # This is the starting point of the route
    total_pallets_loaded += remaining_demand[last_origin, destination_index]
    for origin in route[1:]: # Iterates over all origins in the route
        transportation_cost = df_km[last_origin, origin] * cost_per_km_diesel + (df_time[last_origin, origin] / 60) * cost_per_hour
        total_cost += transportation_cost
        total_pallets_loaded += remaining_demand[origin, destination_index]  # Adding loaded pallets starting at first cross dock origin
        last_origin = origin

    # Add transportation cost from last cross dock location to the destination
    total_cost += df_km[last_origin, destination_index] * cost_per_km_diesel
    total_cost += (df_time[last_origin, destination_index] / 60) * cost_per_hour
    
    # Add Loading and unloading cost 
    total_cost += total_pallets_loaded * loading_and_unloading_cost * 2

    return total_cost

# Function to generate all valid group combinations where each group's demand <= truck capacity
def generate_valid_groups(origins_with_demand, remaining_demand, j):
    valid_groups = []
    for r in range(1, len(origins_with_demand) + 1):
        for group in combinations(origins_with_demand, r):
            group_demand = sum(remaining_demand[i, j] for i in group)
            if group_demand <= truck_capacity_diesel:
                valid_groups.append(group)
    return valid_groups

def solve_tsp_exact(group, destination_index, remaining_demand):
    """
    Vind voor de gegeven groep (tuple van origin-indices) de permutatie
    met de laagste totale kosten (afstand+tijd+laden/lossen).
    Keert (beste_route, best_cost) terug, waarbij beste_route een list is.
    """
    best_route = None
    best_cost = float('inf')
    for perm in permutations(group):
        cost = calculate_route_cost(list(perm), destination_index, remaining_demand)
        if cost < best_cost:
            best_cost = cost
            best_route = list(perm)
    return best_route, best_cost

# Function to find the best set of groups from all the valid sets of groups
def find_best_grouping(remaining_demand):
    n = remaining_demand.shape[0]
    all_routes = []

    # Loop through each destination (column in remaining_demand)
    for j in range(n):
        # print(f"Thinking about routes for destination {j}...")
        total_demand = total_demand_for_destination(remaining_demand, j)
        
        # Determine how many trucks (routes) are needed based on total demand
        trucks_needed = total_demand // truck_capacity_diesel
        if total_demand % truck_capacity_diesel != 0:
            trucks_needed += 1
        
        # Gather origins with demand for this destination
        origins_with_demand = [i for i in range(n) if remaining_demand[i, j] > 0]
        
        # If there are no origins with demand, skip this destination
        if not origins_with_demand:
            continue

        # Generate groups of origins, where the total remaining demand in every group is at max 26
        # In other words: below codes creates groups of origins such that total demand is below the capacity
        valid_groups = generate_valid_groups(origins_with_demand, remaining_demand, j)
                
        # Step 1: In this step we find sets or combinations of groups such that every origin is covered exactly ones
        all_combinations = []
        for r in range(1, trucks_needed + 1):
            for groups in combinations(valid_groups, r):
                # Flatten the groups and ensure that every origin is covered exactly once
                all_origins = [origin for group in groups for origin in group]
                if sorted(set(all_origins)) == sorted(origins_with_demand):
                    all_combinations.append(groups)
        
        # If no valid combinations are found, try again with one extra truck. This happend before, so this part is needed
        if not all_combinations:
            
            for r in range(1, trucks_needed + 2):  # trucks_needed + 1 trucks
                for groups in combinations(valid_groups, r):
                    # Flatten the groups and ensure that every origin is covered exactly once
                    all_origins = [origin for group in groups for origin in group]
                    if sorted(set(all_origins)) == sorted(origins_with_demand):
                        all_combinations.append(groups)
                    
        # Stap 2+3: voor elke combinatie los je per groep het TSP exact op
        best_cost = float('inf')
        best_combination = None
        
        for combination in all_combinations:
            total_cost = 0
            optimized_groups = []
            # los per groep de exacte TSP op
            for group in combination:
                best_route, cost = solve_tsp_exact(group, j, remaining_demand)
                optimized_groups.append(best_route)
                total_cost += cost
            if total_cost < best_cost:
                best_cost = total_cost
                best_combination = optimized_groups

        # If no valid combination was found, set best_combination to None
        if best_combination is None:
            print(f"No valid combination found for destination {terminal_ids[j]}")
            continue

        # Store the best combination for this destination
        all_routes.append((terminal_ids[j], best_combination, total_demand, best_cost))

    return all_routes

#%% Loop Loop through the days

result = {}

# Load the final route pallets from friday such that we add the to the monday demand
# For the rest of the days the final route pallets are carried over within the loop
# Load the array back from the file
final_route_pallets_friday = np.load('final_route_pallets_friday.npy')
remaining_demand_day = final_route_pallets_friday

# Loop through the days
for d in range(len(days)):
    
    result_day = {}
    demand_day = np.ceil(df_demand * scale[d]).astype(int)
    
    total_demand_day = np.sum(demand_day)
    result_day['Total demand'] = total_demand_day
    
    # The remaining final route pallets of day before are added here such that they are handled by the direct routes
    # This are the pallets that are 'too late'
    demand_day += remaining_demand_day
    result_day['Too late'] = np.sum(remaining_demand_day)
    
    ########################### 1. FOE ################################

    foe, ein = terminal_ids.index("FOE"), terminal_ids.index("EIN")

    # — Step 1: Direct diesel trucks ≥ minimum_load to and from FOE
    minimum_load_foe = 13

    # Step 1: Direct diesel trucks ≥ minimum_load_foe (15) in FOE→j
    direct_fromfoe_trucks = direct_fromfoe_cost = 0
    for j in range(n):
        if j==foe or df_km[foe,j]==999:
            continue
    
        qty = demand_day[foe,j]
        if qty < minimum_load_foe:
            continue
    
        # 1) full trucks
        full_trucks = qty // truck_capacity_diesel
        shipped = full_trucks * truck_capacity_diesel
    
        # 2) maybe one more partial truck if remainder ≥ minimum_load
        remainder = qty - shipped
        partial_trucks = 1 if remainder >= minimum_load_foe else 0
        shipped += partial_trucks * remainder
    
        total_trucks = full_trucks + partial_trucks
    
        # unit cost per trip
        travel = df_km[foe,j]*cost_per_km_diesel
        wait   = (df_time[foe,j]/60)*cost_per_hour
    
        direct_fromfoe_trucks += total_trucks
        # loading/unloading applies to exactly 'shipped' pallets
        direct_fromfoe_cost   += total_trucks*(travel+wait) + shipped*loading_and_unloading_cost*2
    
        # remove only what you shipped
        demand_day[foe,j] -= shipped
    
    # Step 1b: Direct diesel trucks ≥ minimum_load_foe in i→FOE
    direct_tofoe_trucks = direct_tofoe_cost = 0
    for i in range(n):
        if i==foe or df_km[i,foe]==999:
            continue
    
        qty = demand_day[i,foe]
        if qty < minimum_load_foe:
            continue
    
        full_trucks = qty // truck_capacity_diesel
        shipped = full_trucks * truck_capacity_diesel
    
        remainder = qty - shipped
        partial_trucks = 1 if remainder >= minimum_load_foe else 0
        shipped += partial_trucks * remainder
    
        total_trucks = full_trucks + partial_trucks
    
        travel = df_km[i,foe]*cost_per_km_diesel
        wait   = (df_time[i,foe]/60)*cost_per_hour
    
        direct_tofoe_trucks += total_trucks
        direct_tofoe_cost   += total_trucks*(travel+wait) + shipped*loading_and_unloading_cost*2
    
        demand_day[i,foe] -= shipped

    result_day['Direct trucks FOE→X'] = direct_fromfoe_trucks
    result_day['Direct cost FOE→X']   = direct_fromfoe_cost
    result_day['Direct trucks X→FOE'] = direct_tofoe_trucks
    result_day['Direct cost X→FOE']   = direct_tofoe_cost

    # — Step B: Remaining demand from FOE can be delivered to eindhoven in one truck in one day
    rem_vec = demand_day[foe, :].copy()
    rem_total = rem_vec.sum()
    if rem_total > 0:
        unit = df_km[foe,ein]*cost_per_km_diesel + (df_time[foe,ein]/60)*cost_per_hour
        foe_ein_cost = unit + rem_total*loading_and_unloading_cost*2
        result_day['FOE→EIN trucks'] = 1
        result_day['FOE→EIN cost']   = foe_ein_cost

        # clear FOE row
        demand_day[foe, :] = 0

    # — Step C: The demand that is delivered in EIN yesterday should be delivered to FOE today
    ret_qty = ein_foe_from_yesterday[d]
    if ret_qty > 0:
        t = math.ceil(ret_qty / truck_capacity_diesel)
        unit = df_km[ein,foe]*cost_per_km_diesel + (df_time[ein,foe]/60)*cost_per_hour
        ein_foe_cost = t*unit + ret_qty*loading_and_unloading_cost*2
        result_day['EIN→FOE trucks'] = t
        result_day['EIN→FOE cost']   = ein_foe_cost

    # — Step D: Handle demand from and to EIN—
    # 1) Bring in yesterday’s FOE→X demand into EIN→X today
    demand_day[ein, :] += ein_i_from_yesterday[d]

    # 2) Today we bring demand FOE→X to EIN
    for i in range(n):
        qty = demand_day[i,foe]
        demand_day[i,ein] += qty
        demand_day[i,foe]  = 0
        
    total_FOE_cost = direct_fromfoe_cost + direct_fromfoe_cost + foe_ein_cost + ein_foe_cost
    result_day['Total cost FOE']  = total_FOE_cost
    
    ########################### 1. Direct delivery ################################
    
    # Full direct trucks
    trucks_direct_full_electric = demand_day // truck_capacity_electric * route_electric * route_not_999
    trucks_direct_full_diesel = demand_day // truck_capacity_diesel * (1 - route_electric) * route_not_999
    remaining_demand_day = demand_day - trucks_direct_full_electric * truck_capacity_electric - trucks_direct_full_diesel * truck_capacity_diesel
    
    # Send direct diesel trucks that exceed lower bound
    trucks_direct_partial_diesel = remaining_demand_day // minimum_load_diesel * (1 - route_electric) * route_not_999
    remaining_demand_day = remaining_demand_day - trucks_direct_partial_diesel * remaining_demand_day
    
    # Costs
    direct_pallets = demand_day - remaining_demand_day
    loading_cost = np.sum(direct_pallets) * loading_and_unloading_cost * 2
    transport_cost_electric = np.sum(trucks_direct_full_electric * (df_km * cost_per_km_electric + df_time * cost_per_hour / 60))
    transport_cost_diesel = np.sum((trucks_direct_full_diesel + trucks_direct_partial_diesel) * (df_km * cost_per_km_diesel + df_time * cost_per_hour / 60))
    total_cost_direct = loading_cost + transport_cost_electric + transport_cost_diesel
    
    # Save results
    result_day['Direct trucks electric full'] = trucks_direct_full_electric
    result_day['Direct trucks diesel full'] = trucks_direct_full_diesel
    result_day['Direct trucks diesel partial'] = trucks_direct_partial_diesel
    result_day['Direct pallets'] = direct_pallets
    result_day['Direct total cost'] = total_cost_direct
    result_day['Fill rate direct'] = np.sum(direct_pallets) / (np.sum(trucks_direct_full_electric) * truck_capacity_electric + np.sum(trucks_direct_full_diesel) * truck_capacity_diesel + np.sum(trucks_direct_partial_diesel) * truck_capacity_diesel)
    result_day['Average cost direct'] = total_cost_direct / np.sum(direct_pallets)
    
    ############################ 2. Cross dock ####################################
    
    # Initialize
    cross_dock_loc = np.full((n, n), np.nan)
    cross_dock_pallets = np.zeros((n,n), int)
    cross_dock_pallets_1 = np.zeros((n, len(cross_dock_possible_loc)), dtype=int) # TO the cross dock locations
    cross_dock_pallets_2 = np.zeros((len(cross_dock_possible_loc), n), dtype=int) # FROM the cross dock locations
    
    # Function to calculate cross-docking costs
    def calculate_cross_docking_cost(i, j, cross_dock_possible_id):    
        
        # Calculate cross docking options
        cross_docking_opt = []
        for k in cross_dock_possible_id:
            cross_dock_distance = df_km[i, k] + df_km[k, j]
            cross_dock_time = df_time[i, k] + df_time[k, j]
            cross_docking_opt.append(cross_dock_distance * cost_per_km_diesel + cross_dock_time * cost_per_hour / 60)
        
        # Select option with minimum costs
        cross_dock_cost_ij = np.min(cross_docking_opt) + demand_ij * cross_dock_cost_per_pallet
        cross_dock_loc_ij = cross_dock_possible_id[np.argmin(cross_docking_opt)]
        cross_dock_loc[i, j] = cross_dock_loc_ij
        return cross_dock_cost_ij, cross_dock_loc_ij
    
    # Compute cross docked pallets for each combination (i,j)
    for i in range(n):
        for j in range(n):
            
            # Extract demand
            demand_ij = remaining_demand_day[i, j]
            if (terminal_ids[i] in cross_dock_possible_loc) or (terminal_ids[j] in cross_dock_possible_loc) or (demand_ij == 0): 
                continue
        
            # Calculate direct delivery cost and minimum cross dock cost
            direct_route_cost = df_km[i, j] * cost_per_km_diesel + df_time[i, j] * cost_per_hour / 60
            cross_dock_cost_ij, cross_dock_loc_ij = calculate_cross_docking_cost(i, j, cross_dock_possible_id)
    
            # Compare the costs and cross dock if beneficial
            if cross_dock_cost_ij <= times_worse * direct_route_cost:
                cross_dock_pallets[i, j] += demand_ij
                remaining_demand_day[i, j] -= demand_ij
    
    # We also ship the small remaining demands to the cross dock locations
    cross_dock_pallets[:, cross_dock_possible_id] = remaining_demand_day[:, cross_dock_possible_id]
    cross_dock_pallets[cross_dock_possible_id, :] = remaining_demand_day[cross_dock_possible_id, :]
    remaining_demand_day[:, cross_dock_possible_id] = 0
    remaining_demand_day[cross_dock_possible_id, :] = 0

    # Derive the number of pallets for route part 1 and part 2
    for k in range(len(cross_dock_possible_id)):
        cross_dock_pallets_1[:, k] = ((cross_dock_loc == cross_dock_possible_id[k]) * cross_dock_pallets).sum(axis = 1) + cross_dock_pallets[:,cross_dock_possible_id[k]]
        cross_dock_pallets_2[k, :] = ((cross_dock_loc == cross_dock_possible_id[k]) * cross_dock_pallets).sum(axis = 0) + cross_dock_pallets[cross_dock_possible_id[k],:]
    
    ############################ 3. Optimal Routes Part 1 = TO cross dock locations ####################################

    # First compute the number of (nearly) full trucks and the corresponding costs
    cross_dock_trucks_1_full = cross_dock_pallets_1 // truck_capacity_diesel  
    cross_dock_trucks_1_partial = cross_dock_pallets_1 % truck_capacity_diesel // minimum_load_diesel
    cross_dock_trucks_1_full = cross_dock_trucks_1_full + cross_dock_trucks_1_partial
    
    # Compute the remaining demand for which we find optimal routes
    cross_dock_pallets_1_remaining = cross_dock_pallets_1 % truck_capacity_diesel * (1 - cross_dock_trucks_1_partial)
    cross_dock_pallets_1_full = cross_dock_pallets_1 - cross_dock_pallets_1_remaining
    
    # Keep track of total costs
    total_costs_cross_dock_1_remaining = 0
    cross_dock_trucks_1_remaining = 0    
    
    # Loop through the cross-dock locations using the cross_dock_possible_loc and cross_dock_possible_id arrays
    for i, cross_dock in enumerate(cross_dock_possible_loc):
        # print(f"Optimizing routes for {cross_dock}...")
    
        # Get the remaining pallets for this cross-dock location
        pallets1 = cross_dock_pallets_1_remaining[:, i]
    
        # Mask terminals with zero pallets for this cross-dock
        valid_terminal_ids = [idx for idx, pallet_count in enumerate(pallets1) if pallet_count > 0]
        
        # Optimize routes
        cross_dock_index = cross_dock_possible_id[i]  # Use the corresponding cross-dock index
        groups0 = initial_groups(valid_terminal_ids, pallets1, cross_dock_index, df_time, max_route_time)
        opt_groups = simulated_annealing(groups0, df_km, df_time, pallets1,
                                         cross_dock_index, max_route_time,
                                         alpha=1, beta=1,
                                         initial_temp=5000, cooling_rate=0.995, max_iter=5000)

        # Results
        optimal_routes_to_cross_dock = {}
        total_time = 0
        total_distance = 0
        total_cost = 0
        route_count = 0
        for idx, grp in enumerate(opt_groups, start=1):
            
            route = [terminal_ids[i] for i in grp]
            dist = route_distance(grp, df_km)
            time = route_time(grp, df_time, pallets1, cross_dock_index)
            cost, num_pallets = route_cost(grp, df_km, df_time, pallets1, cross_dock_index)
    
            total_time += time
            total_distance += dist
            total_cost += cost
            route_count += 1
        
            key = f"Route {idx} {cross_dock}"
            optimal_routes_to_cross_dock[key] = {
                "stops":    route,
                "time":     time,
                "pallets":  num_pallets,
                "distance": dist,
                "cost":     cost
            }
            
        result_day[f'Optimal Routes to {cross_dock}'] = optimal_routes_to_cross_dock
                    
        # Keep track of total costs across all cross dock locations
        cross_dock_trucks_1_remaining += route_count
        total_costs_cross_dock_1_remaining += total_cost
        
    # Compute costs for sending full and nearly full trucks for all cross dock locations
    loading_cost_cross_dock_1_full = np.sum(cross_dock_pallets_1_full) * loading_and_unloading_cost * 2
    transport_cost_cross_dock_1_full = np.sum(cross_dock_trucks_1_full * (df_km[np.arange(n)[:, None], cross_dock_possible_id] + df_time[np.arange(n)[:, None], cross_dock_possible_id] * cost_per_hour / 60))
    total_cost_cross_dock_1_full = loading_cost_cross_dock_1_full + transport_cost_cross_dock_1_full

    # Save results
    result_day['Cross dock pallets'] = cross_dock_pallets
    result_day['Cross dock pallets 1'] = cross_dock_pallets_1
    result_day['Cross dock pallets 1 full'] = cross_dock_pallets_1_full
    result_day['Cross dock pallets 1 remaining'] = cross_dock_pallets_1_remaining
    result_day['Cross dock pallets 1 full costs'] = total_cost_cross_dock_1_full
    result_day['Cross dock pallets 1 remaining costs'] = total_costs_cross_dock_1_remaining
    result_day['Cross dock pallets 1 total costs'] = total_cost_cross_dock_1_full + total_costs_cross_dock_1_remaining
    result_day['Cross dock trucks 1 full'] = cross_dock_trucks_1_full
    result_day['Cross dock trucks 1 remaining'] = cross_dock_trucks_1_remaining
    result_day['Fill rate cross dock 1'] = np.sum(cross_dock_pallets_1) / ((np.sum(cross_dock_trucks_1_full) + cross_dock_trucks_1_remaining) * truck_capacity_diesel)

    ############################ 4. Optimal Routes Part 2 = FROM cross dock locations ####################################

    # First compute the number of (nearly) full trucks and the corresponding costs
    cross_dock_trucks_2_full = cross_dock_pallets_2 // truck_capacity_diesel  
    cross_dock_trucks_2_partial = cross_dock_pallets_2 % truck_capacity_diesel // minimum_load_diesel
    cross_dock_trucks_2_full = cross_dock_trucks_2_full + cross_dock_trucks_2_partial
    
    # Compute the remaining demand for which we find optimal routes
    cross_dock_pallets_2_remaining = cross_dock_pallets_2 % truck_capacity_diesel * (1 - cross_dock_trucks_2_partial)
    cross_dock_pallets_2_full = cross_dock_pallets_2 - cross_dock_pallets_2_remaining
    
    # Keep track of results
    total_costs_cross_dock_2_remaining = 0
    cross_dock_trucks_2_remaining    = 0    
    
    for k, cross_dock in enumerate(cross_dock_possible_loc):
        dock_idx = cross_dock_possible_id[k]
        pallets2 = cross_dock_pallets_2_remaining[k, :]   # row k = from-dock pallets
    
        valid_ids = [i for i,p in enumerate(pallets2) if p > 0]
        if not valid_ids:
            result_day[f'Optimal Routes from {cross_dock}'] = {}
            continue
    
        # 1) initial greedy
        g0 = initial_groups_out(valid_ids, pallets2, dock_idx, df_time, max_route_time)
        # 2) anneal
        og = simulated_annealing_out(
            g0, df_km, df_time, pallets2, dock_idx, max_route_time,
            alpha=1, beta=1, initial_temp=5000, cooling_rate=0.995, max_iter=5000
        )
    
        # 3) collect and count
        routes_from = {}
        route_count_2 = 0
        cost_remain_2 = 0
    
        for idx, grp in enumerate(og, start=1):
            stops = [terminal_ids[i] for i in grp]
            dist, tt           = route_distance(grp, df_km), route_time(grp, df_time, pallets2, cross_dock_index)
            cost, pallets_loaded = route_cost(grp, df_km, df_time, pallets2, cross_dock_index)
    
            key = f"Route {idx} from {cross_dock}"
            routes_from[key] = {
                "stops":    stops,
                "pallets":  pallets_loaded,
                "distance": dist,
                "time":     tt,
                "cost":     cost
            }
    
            # increment this dock’s counters
            route_count_2 += 1
            cost_remain_2 += cost
    
        # store per-dock results
        result_day[f'Optimal Routes from {cross_dock}'] = routes_from
    
        # fix #4: use the *new* counters, not leftover inbound ones
        cross_dock_trucks_2_remaining += route_count_2
        total_costs_cross_dock_2_remaining += cost_remain_2
    
    # Compute costs
    loading_cost_cross_dock_2_full = np.sum(cross_dock_pallets_2_full) * loading_and_unloading_cost * 2
    transport_cost_cross_dock_2_full = np.sum(cross_dock_trucks_2_full * (df_km[np.ix_(cross_dock_possible_id, np.arange(n))] + df_time[np.ix_(cross_dock_possible_id, np.arange(n))] * cost_per_hour / 60))
    total_cost_cross_dock_2_full = loading_cost_cross_dock_2_full + transport_cost_cross_dock_2_full
    
    # Compute total costs
    total_cost_cross_dock = total_cost_cross_dock_1_full + total_costs_cross_dock_1_remaining + total_cost_cross_dock_2_full + total_costs_cross_dock_2_remaining
    
    # Save results
    result_day['Cross dock pallets 2'] = cross_dock_pallets_2
    result_day['Cross dock pallets 2 full'] = cross_dock_pallets_2_full
    result_day['Cross dock pallets 2 remaining'] = cross_dock_pallets_2_remaining
    result_day['Cross dock pallets 2 full costs'] = total_cost_cross_dock_2_full
    result_day['Cross dock pallets 2 remaining costs'] = total_costs_cross_dock_2_remaining
    result_day['Cross dock pallets 2 total costs'] = total_cost_cross_dock_2_full + total_costs_cross_dock_2_remaining
    result_day['Cross dock trucks 2 full'] = cross_dock_trucks_2_full
    result_day['Cross dock trucks 2 remaining'] = cross_dock_trucks_2_remaining
    result_day['Fill rate cross dock 2'] = np.sum(cross_dock_pallets_2) / ((np.sum(cross_dock_trucks_2_full) + cross_dock_trucks_2_remaining) * truck_capacity_diesel)
    
    result_day['Cross dock total cost'] = total_cost_cross_dock
    result_day['Average cost cross dock'] = total_cost_cross_dock / np.sum(cross_dock_pallets)

    # Compute total costs costs
    total_cost = total_FOE_cost + total_cost_direct + total_cost_cross_dock
    
    result_day['Total cost'] = total_cost
    result_day['Average cost'] = total_cost / total_demand_day
    result[days[d]] = result_day

# Aggregate days
total_pallets_week = 0
total_cost_week = 0
total_pallets_electric = 0
too_late_final_routes = 0
for d in days:
    total_pallets_week += np.sum(result[d]['Total demand'])
    total_cost_week += result[d]['Total cost']
    total_pallets_electric += np.sum(result[d]['Direct trucks electric full']) * truck_capacity_electric
    too_late_final_routes += result[d]['Too late']

# Calculate KPIs
avg_pallet_cost = np.round(total_cost_week / total_pallets_week, 2)
service_level = np.round((total_pallets_week - too_late_foe - too_late_final_routes) / total_pallets_week * 100, 2)
pct_pallets_electric = np.round(total_pallets_electric / total_pallets_week * 100, 2)

# --- Background color (DHL yellow)
background_color = '#FFD700'  # DHL yellow hex
dhl_red = '#D40511'
light_green = '#90EE90'

# --- Big average cost in DHL red, label below in black
st.markdown(
    f"""
    <div style="background-color:{background_color}; padding:30px; text-align:center;">
        <span style="font-size:60px; color:{dhl_red}; font-weight:bold;">€{avg_pallet_cost}</span><br>
        <span style="font-size:28px; color:black;">Average Pallet Cost</span>
    </div>
    """,
    unsafe_allow_html=True
)

import plotly.graph_objects as go

# --- Create pie charts
fig_service = go.Figure(go.Pie(
    values=[service_level, 100 - service_level],
    labels=['Within Limit', 'Out of Limit'],
    hole=0.6,
    marker=dict(colors=[dhl_red, '#E0E0E0']),
    textinfo='percent',
    textfont_size=28
))
fig_service.update_layout(
    showlegend=False,
    paper_bgcolor=background_color,
    plot_bgcolor=background_color,
    margin=dict(t=0, b=0, l=0, r=0),
)

fig_electric = go.Figure(go.Pie(
    values=[pct_pallets_electric, 100 - pct_pallets_electric],
    labels=['Electric', 'Diesel'],
    hole=0.6,
    sort=False,  # ➤ respecteer volgorde
    direction='clockwise',  # ➤ start rechtsom
    rotation=0,  # ➤ start bij 12 uur (bovenaan)
    marker=dict(colors=[light_green, '#E0E0E0']),
    textinfo='percent',
    textfont_size=28
))

fig_electric.update_layout(
    showlegend=False,
    paper_bgcolor=background_color,
    plot_bgcolor=background_color,
    margin=dict(t=0, b=0, l=0, r=0),
)

# --- Display charts side by side
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig_service, use_container_width=True)
    st.markdown('<div style="text-align:center; color:black; font-size:28px;">Service Level</div>', unsafe_allow_html=True)

with col2:
    st.plotly_chart(fig_electric, use_container_width=True)
    st.markdown('<div style="text-align:center; color:black; font-size:28px;">Percentage Pallets Electric</div>', unsafe_allow_html=True)

