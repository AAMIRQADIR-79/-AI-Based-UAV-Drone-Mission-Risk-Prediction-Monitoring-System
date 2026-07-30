"""
swarm.py - Multi-UAV Swarm Coordination

Extends the single-drone system to handle 2-5 simultaneous UAV missions,
detecting route conflicts (two drones getting too close to each other
at a similar point in time) and providing a swarm-level risk summary.

This reuses your EXISTING single-drone analysis (geocoding, route
interpolation, risk prediction) - it doesn't replace it. Each drone
in the swarm still gets analyzed with your normal pipeline; this
module only adds the cross-drone conflict-checking layer on top.
"""

from utils.geocoding import haversine_distance


# ─────────────────────────────────────────────
# CHECK FOR ROUTE CONFLICTS BETWEEN TWO DRONES
# ─────────────────────────────────────────────
def check_pairwise_conflict(
    waypoints_a: list, waypoints_b: list,
    drone_id_a: str, drone_id_b: str,
    min_safe_separation_km: float = 0.5,
    eta_per_waypoint_min: float = 5.0,
) -> list:
    """
    Compares every waypoint of drone A against every waypoint of
    drone B. Flags a conflict if:
      1. Their positions are within min_safe_separation_km of each other, AND
      2. Their estimated time-of-arrival at those points is close
         (within one waypoint-interval of each other).

    This models the real collision-risk logic used in air traffic
    deconfliction: proximity in BOTH space and time matters, not
    space alone (two drones can cross the same point safely if they
    pass through it at very different times).
    """
    conflicts = []

    for i, wp_a in enumerate(waypoints_a):
        eta_a = i * eta_per_waypoint_min
        for j, wp_b in enumerate(waypoints_b):
            eta_b = j * eta_per_waypoint_min

            dist = haversine_distance(wp_a[0], wp_a[1], wp_b[0], wp_b[1])
            time_gap = abs(eta_a - eta_b)

            if dist <= min_safe_separation_km and time_gap <= eta_per_waypoint_min:
                conflicts.append({
                    "drone_a": drone_id_a, "wp_a_index": i,
                    "drone_b": drone_id_b, "wp_b_index": j,
                    "distance_km": round(dist, 3),
                    "eta_a_min": round(eta_a, 1),
                    "eta_b_min": round(eta_b, 1),
                    "time_gap_min": round(time_gap, 1),
                    "severity": (
                        "critical" if dist <= min_safe_separation_km * 0.4
                        else "warning"
                    ),
                })

    return conflicts


# ─────────────────────────────────────────────
# CHECK ALL PAIRS IN A SWARM
# ─────────────────────────────────────────────
def check_swarm_conflicts(
    drones: dict,
    min_safe_separation_km: float = 0.5,
    eta_per_waypoint_min: float = 5.0,
) -> dict:
    """
    drones: dict like {
        "Drone A": {"waypoints": [...], "risk_score": 12.4},
        "Drone B": {"waypoints": [...], "risk_score": 42.1},
        ...
    }

    Checks every unique pair of drones for conflicts and returns a
    combined swarm status report.
    """
    drone_ids = list(drones.keys())
    all_conflicts = []

    for i in range(len(drone_ids)):
        for j in range(i + 1, len(drone_ids)):
            id_a, id_b = drone_ids[i], drone_ids[j]
            wps_a = drones[id_a]["waypoints"]
            wps_b = drones[id_b]["waypoints"]

            pair_conflicts = check_pairwise_conflict(
                wps_a, wps_b, id_a, id_b,
                min_safe_separation_km, eta_per_waypoint_min
            )
            all_conflicts.extend(pair_conflicts)

    # Sort worst-first
    all_conflicts.sort(key=lambda c: c["distance_km"])

    individual_risks = [d.get("risk_score", 0) for d in drones.values()]
    avg_risk = round(sum(individual_risks) / len(individual_risks), 1) if individual_risks else 0
    max_risk = round(max(individual_risks), 1) if individual_risks else 0

    critical_conflicts = sum(1 for c in all_conflicts if c["severity"] == "critical")

    if critical_conflicts > 0:
        swarm_verdict = "CRITICAL — Collision risk between UAVs, immediate deconfliction required"
    elif all_conflicts:
        swarm_verdict = "CAUTION — Route proximity detected, monitor separation"
    elif max_risk >= 60:
        swarm_verdict = "One or more UAVs at high individual risk"
    else:
        swarm_verdict = "Swarm operating within safe parameters"

    return {
        "n_drones": len(drone_ids),
        "drone_ids": drone_ids,
        "conflicts": all_conflicts,
        "n_conflicts": len(all_conflicts),
        "n_critical_conflicts": critical_conflicts,
        "avg_individual_risk": avg_risk,
        "max_individual_risk": max_risk,
        "swarm_verdict": swarm_verdict,
    }


# ─────────────────────────────────────────────
# FORMAT CONFLICT AS ALERT
# ─────────────────────────────────────────────
def conflict_to_alert(conflict: dict) -> dict:
    level = conflict["severity"]
    return {
        "level": level,
        "message": (
            f"{conflict['drone_a']} (WP{conflict['wp_a_index']+1}) and "
            f"{conflict['drone_b']} (WP{conflict['wp_b_index']+1}) — "
            f"{conflict['distance_km']}km apart, "
            f"ETA gap {conflict['time_gap_min']}min"
        ),
        "action": (
            "Immediate altitude/route separation required"
            if level == "critical" else
            "Monitor both UAVs closely, consider minor reroute"
        ),
    }