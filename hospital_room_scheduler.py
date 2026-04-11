"""
MSE 435 – Examination Room Scheduling via Column Generation
===========================================================
Blocked time windows are SOFT constraints with a per-minute penalty.
Appointments overlapping admin/lunch blocks are still schedulable,
but incur a penalty (ADMIN_PENALTY_PER_MIN) added to the pattern cost.

New KPI: Room Utilization
  - room_utilization_pct: percentage of available room-minutes actually used
  - rooms_used_count: number of distinct rooms used across all chosen patterns
  - avg_appts_per_room_used: average appointments per room that was used
  - room_utilization_by_room: per-room breakdown (printed separately)

Run:
    python hospital_room_scheduler.py --csv AppointmentDataWeek1.csv [--day 11-10-2025]
"""

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# ──────────────────────────────────────────────────────────────────────────────
# 0.  PROBLEM CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

_RAW_DISTANCES: Dict[Tuple[int, int], float] = {
    (1, 2): 0.5, (1, 3): 1.5, (1, 4): 2.5, (1, 5): 4.0, (1, 6): 4.7,
    (1, 7): 2.0, (1, 8): 3.0, (1, 9): 4.7, (1, 10): 4.1, (1, 11): 3.3,
    (1, 12): 1.5, (1, 13): 3.0, (1, 14): 6.1, (1, 15): 11.5, (1, 16): 12.0,
    (2, 3): 1.0, (2, 4): 2.0, (2, 5): 3.5, (2, 6): 4.2,
    (2, 7): 1.5, (2, 8): 2.5, (2, 9): 4.2, (2, 10): 3.6, (2, 11): 2.8,
    (2, 12): 2.0, (2, 13): 3.5, (2, 14): 5.6, (2, 15): 11.0, (2, 16): 11.5,
    (3, 4): 1.0, (3, 5): 2.5, (3, 6): 3.2,
    (3, 7): 0.5, (3, 8): 1.5, (3, 9): 3.2, (3, 10): 2.6, (3, 11): 1.8,
    (3, 12): 3.0, (3, 13): 4.5, (3, 14): 4.6, (3, 15): 10.0, (3, 16): 10.5,
    (4, 5): 1.5, (4, 6): 2.2,
    (4, 7): 1.5, (4, 8): 1.5, (4, 9): 2.2, (4, 10): 1.6, (4, 11): 2.8,
    (4, 12): 4.0, (4, 13): 5.5, (4, 14): 3.6, (4, 15): 9.0, (4, 16): 9.5,
    (5, 6): 0.7,
    (5, 7): 3.0, (5, 8): 3.0, (5, 9): 2.5, (5, 10): 3.1, (5, 11): 4.3,
    (5, 12): 5.5, (5, 13): 6.0, (5, 14): 3.9, (5, 15): 7.5, (5, 16): 8.0,
    (6, 7): 3.7, (6, 8): 3.7, (6, 9): 3.2, (6, 10): 3.8, (6, 11): 5.0,
    (6, 12): 6.2, (6, 13): 6.7, (6, 14): 4.6, (6, 15): 6.8, (6, 16): 7.3,
    (7, 8): 1.0, (7, 9): 2.7, (7, 10): 2.1, (7, 11): 1.3,
    (7, 12): 2.5, (7, 13): 4.0, (7, 14): 4.1, (7, 15): 9.5, (7, 16): 10.0,
    (8, 9): 1.7, (8, 10): 1.1, (8, 11): 1.3,
    (8, 12): 2.5, (8, 13): 4.0, (8, 14): 3.1, (8, 15): 8.5, (8, 16): 9.0,
    (9, 10): 0.6, (9, 11): 1.8,
    (9, 12): 3.2, (9, 13): 5.7, (9, 14): 1.4, (9, 15): 6.8, (9, 16): 7.3,
    (10, 11): 1.2,
    (10, 12): 2.6, (10, 13): 5.1, (10, 14): 2.0, (10, 15): 7.4, (10, 16): 7.9,
    (11, 12): 1.8, (11, 13): 4.3, (11, 14): 2.8, (11, 15): 8.2, (11, 16): 8.7,
    (12, 13): 2.5, (12, 14): 4.6, (12, 15): 10.0, (12, 16): 10.5,
    (13, 14): 7.1, (13, 15): 12.5, (13, 16): 13.0,
    (14, 15): 5.4, (14, 16): 5.9,
    (15, 16): 0.5,
}

NUM_ROOMS = 16
ROOMS: List[int] = list(range(1, NUM_ROOMS + 1))

# ── Soft-constraint penalty ───────────────────────────────────────────────────
# Cost added per minute an appointment overlaps a blocked admin/lunch window.
# Set higher to more strongly discourage scheduling into blocked slots.
ADMIN_PENALTY_PER_MIN: float = 2.0

# ── Operating day window for room-utilization denominator ─────────────────────
# A room is "available" from DAY_START_MIN to DAY_END_MIN each day it appears
# in the schedule.  Blocked admin/lunch windows are subtracted so the
# denominator reflects only clinically schedulable time.
DAY_START_MIN: int = 8 * 60   # 08:00
DAY_END_MIN:   int = 17 * 60  # 17:00


def room_distance(r1: int, r2: int) -> float:
    if r1 == r2:
        return 0.0
    key = (min(r1, r2), max(r1, r2))
    return _RAW_DISTANCES.get(key, 0.0)


BLOCKED_MON_THU = [
    (9 * 60,       9 * 60 + 30),
    (11 * 60 + 30, 12 * 60),
    (12 * 60,      13 * 60),
    (16 * 60 + 30, 17 * 60),
]
BLOCKED_FRI = [
    (8 * 60,       8 * 60 + 30),
    (11 * 60 + 30, 12 * 60),
    (12 * 60,      13 * 60),
    (15 * 60,      15 * 60 + 30),
]


def is_friday(date_str: str) -> bool:
    from datetime import datetime
    try:
        dt = datetime.strptime(date_str, "%m-%d-%Y")
        return dt.weekday() == 4
    except ValueError:
        return False


def blocked_windows(date_str: str) -> List[Tuple[int, int]]:
    return BLOCKED_FRI if is_friday(date_str) else BLOCKED_MON_THU


def time_str_to_minutes(t: str) -> int:
    parts = t.strip().split(":")
    return int(parts[0]) * 60 + int(parts[1])


def is_available(start_min: int, end_min: int,
                 blocks: List[Tuple[int, int]]) -> bool:
    """Used for display/reporting only — no longer gates scheduling."""
    for (bs, be) in blocks:
        if start_min < be and end_min > bs:
            return False
    return True


def admin_overlap_minutes(start_min: int, end_min: int,
                          blocks: List[Tuple[int, int]]) -> int:
    """Minutes of [start_min, end_min) overlapping any blocked window."""
    total = 0
    for (bs, be) in blocks:
        total += max(0, min(end_min, be) - max(start_min, bs))
    return total


def schedulable_minutes_per_day(date_str: str) -> int:
    """
    Total schedulable room-minutes in one day = (DAY_END - DAY_START)
    minus the sum of all blocked-window durations.
    This is the denominator used in room-utilization calculations.
    """
    gross = DAY_END_MIN - DAY_START_MIN
    blocks = blocked_windows(date_str)
    blocked = sum(be - bs for (bs, be) in blocks)
    return max(0, gross - blocked)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  DATA STRUCTURES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Appointment:
    appt_id: int
    patient_id: str
    provider: str
    date: str
    start_min: int
    duration: int
    appt_type: str
    no_show: bool
    cancelled: bool
    deleted: bool

    @property
    def end_min(self) -> int:
        return self.start_min + self.duration

    @property
    def is_active(self) -> bool:
        return not self.cancelled and not self.deleted

    def __repr__(self):
        h, m = divmod(self.start_min, 60)
        return (f"Appt#{self.appt_id}[{self.patient_id} "
                f"{h:02d}:{m:02d}+{self.duration}min]")


@dataclass
class Pattern:
    pattern_id: int
    provider: str
    date: str
    assignment: Dict[int, int]   # appt_id -> room
    cost: float                  # travel distance + admin penalty

    def covers(self, appt_id: int) -> bool:
        return appt_id in self.assignment


# ──────────────────────────────────────────────────────────────────────────────
# 2.  DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_appointments(csv_path: str) -> List[Appointment]:
    appointments: List[Appointment] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            row = {k.strip(): (v.strip() if isinstance(v, str) else v)
                   for k, v in row.items()}
            duration_raw = row.get("Appt Duration", "0") or "0"
            duration = int(float(duration_raw)) if duration_raw else 0
            if duration <= 0:
                continue

            time_raw = row.get("Appt Time", "09:00:00") or "09:00:00"
            try:
                start = time_str_to_minutes(time_raw)
            except (ValueError, IndexError):
                start = 9 * 60

            appointments.append(Appointment(
                appt_id=idx,
                patient_id=row.get("Patient Id", ""),
                provider=row.get("Primary Provider", ""),
                date=row.get("Appt Date", ""),
                start_min=start,
                duration=duration,
                appt_type=row.get("Appt Type", "") or "",
                no_show=(row.get("No Show Appts", "N").strip().upper() == "Y"),
                cancelled=(row.get("Cancelled Appts", "N").strip().upper() == "Y"),
                deleted=(row.get("Deleted Appts", "N").strip().upper() == "Y"),
            ))
    return appointments


# ──────────────────────────────────────────────────────────────────────────────
# 3.  PRICING SUB-PROBLEM
# ──────────────────────────────────────────────────────────────────────────────

def compute_Pi_dp(appts: List[Appointment]) -> List[Tuple[int, int]]:
    sorted_appts = sorted(appts, key=lambda a: a.start_min)
    return [(sorted_appts[k].appt_id, sorted_appts[k + 1].appt_id)
            for k in range(len(sorted_appts) - 1)]


def compute_pattern_cost(
    assignment: Dict[int, int],
    appts: List[Appointment],
    date: Optional[str] = None,
    penalty_per_min: float = ADMIN_PENALTY_PER_MIN,
) -> float:
    """
    Pattern cost = travel distance  +  admin-overlap penalty.

    Travel distance: Σ_{(i,i') ∈ Π_{d,p}} c_{r^h_i, r^h_{i'}}
    Admin penalty:   Σ_{i ∈ A_{d,p}} penalty_per_min × τ_i
      where τ_i = minutes of appointment i overlapping a blocked window.

    When date=None the penalty term is skipped (backwards-compatible).
    """
    pi_dp = compute_Pi_dp(appts)
    travel = 0.0
    for (i, i_prime) in pi_dp:
        r_i      = assignment.get(i)
        r_iprime = assignment.get(i_prime)
        if r_i is not None and r_iprime is not None:
            travel += room_distance(r_i, r_iprime)

    penalty = 0.0
    if date is not None and penalty_per_min > 0:
        blocks = blocked_windows(date)
        appt_map = {a.appt_id: a for a in appts}
        for aid in assignment:
            a = appt_map.get(aid)
            if a is not None:
                overlap = admin_overlap_minutes(a.start_min, a.end_min, blocks)
                penalty += penalty_per_min * overlap

    return travel + penalty


def appointments_overlap(a1: Appointment, a2: Appointment) -> bool:
    return a1.start_min < a2.end_min and a2.start_min < a1.end_min


def solve_pricing_subproblem(
    appts: List[Appointment],
    date: str,
    rooms: List[int],
    duals_coverage: Dict[int, float],
    dual_assignment: float,
    single_room: bool = False,
    penalty_per_min: float = ADMIN_PENALTY_PER_MIN,
) -> Optional[Tuple[Dict[int, int], float]]:
    """
    Pricing sub-problem — blocked windows are soft constraints.
    All active appointments are eligible; scheduling into a blocked window
    incurs penalty_per_min × overlap_minutes added to the reduced cost.
    """
    blocks = blocked_windows(date)

    # ALL active appointments are eligible — no hard filter
    valid_appts = sorted(appts, key=lambda a: a.start_min)
    if not valid_appts:
        return None

    n = len(valid_appts)

    appt_penalty = {
        a.appt_id: penalty_per_min * admin_overlap_minutes(
            a.start_min, a.end_min, blocks)
        for a in valid_appts
    }

    total_dual = (
        sum(duals_coverage.get(a.appt_id, 0.0) for a in valid_appts)
        + dual_assignment
    )

    # ── Policy A fast path (single_room=True) ────────────────────────────────
    if single_room:
        occupied_until = 0
        feasible_single = True
        for a in valid_appts:
            if a.start_min < occupied_until:
                feasible_single = False
                break
            occupied_until = a.end_min
        if not feasible_single:
            return None

        single_cost = sum(appt_penalty[a.appt_id] for a in valid_appts)
        rc = single_cost - total_dual
        assignment = {a.appt_id: rooms[0] for a in valid_appts}
        return assignment, rc

    # ── Beam search (replaces exponential backtracking) ───────────────────────
    # State: (partial_cost, last_room, last_end, room_end_times, assignment)
    # We keep the BEAM_WIDTH cheapest partial states at each appointment step.
    BEAM_WIDTH = 8

    # Initial state
    beam = [(0.0, None, 0, {}, {})]  # (cost, last_room, last_end, room_end_times, assignment)

    for a in valid_appts:
        pen = appt_penalty[a.appt_id]
        candidates = []

        for (partial_cost, last_room, last_end, room_end_times, assignment) in beam:
            for r in rooms:
                if room_end_times.get(r, 0) > a.start_min:
                    continue  # room occupied

                travel_inc = 0.0
                if last_room is not None and a.start_min >= last_end:
                    travel_inc = room_distance(last_room, r)

                new_cost = partial_cost + travel_inc + pen

                new_room_end_times = dict(room_end_times)
                new_room_end_times[r] = a.end_min
                new_assignment = dict(assignment)
                new_assignment[a.appt_id] = r

                candidates.append((new_cost, r, a.end_min,
                                   new_room_end_times, new_assignment))

        if not candidates:
            return None  # no feasible extension

        # Keep best BEAM_WIDTH by partial cost
        candidates.sort(key=lambda x: x[0])
        beam = candidates[:BEAM_WIDTH]

    if not beam:
        return None

    best_cost, _, _, _, best_assignment = beam[0]
    best_rc = best_cost - total_dual
    return best_assignment, best_rc


def _greedy_pattern(
    valid_appts: List[Appointment],
    rooms: List[int],
    start_room: int,
    preferred_room: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    date: Optional[str] = None,
    penalty_per_min: float = ADMIN_PENALTY_PER_MIN,
) -> Optional[Dict[int, int]]:
    blocks = blocked_windows(date) if date else []
    assignment: Dict[int, int] = {}
    room_end_times: Dict[int, int] = {}
    last_r = start_room

    for a in valid_appts:
        free_rooms = [r for r in rooms if room_end_times.get(r, 0) <= a.start_min]
        if not free_rooms:
            return None

        pen = penalty_per_min * admin_overlap_minutes(a.start_min, a.end_min, blocks)

        def key(r: int) -> Tuple:
            jitter = float(rng.random()) * 0.01 if rng is not None else 0.0
            return (r != preferred_room, room_distance(last_r, r) + pen + jitter)

        chosen_r = min(free_rooms, key=key)
        assignment[a.appt_id] = chosen_r
        room_end_times[chosen_r] = a.end_min
        last_r = chosen_r

    return assignment


def generate_initial_patterns(
    appts: List[Appointment],
    provider: str,
    date: str,
    rooms: List[int] = ROOMS,
    n_initial: int = 50,
    seed: int = 0,
    single_room: bool = False,
    penalty_per_min: float = ADMIN_PENALTY_PER_MIN,
) -> List[Pattern]:
    """
    Generate initial patterns.  Blocked-window appointments are NOT filtered
    out — they are eligible columns that carry a higher cost (penalty).
    """
    valid_appts = sorted(appts, key=lambda a: a.start_min)
    if not valid_appts:
        return []

    rng = np.random.default_rng(seed)
    patterns: List[Pattern] = []
    pid = 0
    seen: set = set()

    def add(assignment: Optional[Dict[int, int]]) -> bool:
        nonlocal pid
        if assignment is None:
            return False
        if single_room and len(set(assignment.values())) > 1:
            return False
        sig = frozenset(assignment.items())
        if sig in seen:
            return False
        seen.add(sig)
        cost = compute_pattern_cost(assignment, valid_appts,
                                    date=date, penalty_per_min=penalty_per_min)
        patterns.append(Pattern(pid, provider, date, dict(assignment), cost))
        pid += 1
        return True

    # 1. Single-room patterns (zero travel)
    for r in rooms:
        occupied_until = 0
        ok = True
        for a in valid_appts:
            if a.start_min < occupied_until:
                ok = False
                break
            occupied_until = a.end_min
        if ok:
            add({a.appt_id: r for a in valid_appts})

    if not single_room:
        # 2. Greedy nearest-room
        for start_room in rooms:
            add(_greedy_pattern(valid_appts, rooms, start_room,
                                date=date, penalty_per_min=penalty_per_min))

        # 3. Greedy room-pinned
        for preferred in rooms:
            add(_greedy_pattern(valid_appts, rooms,
                                start_room=preferred, preferred_room=preferred,
                                date=date, penalty_per_min=penalty_per_min))

        # 4. Randomised greedy
        attempts = 0
        while len(patterns) < n_initial and attempts < n_initial * 10:
            attempts += 1
            start_room = int(rng.choice(rooms))
            preferred  = int(rng.choice(rooms))
            add(_greedy_pattern(valid_appts, rooms, start_room,
                                preferred_room=preferred, rng=rng,
                                date=date, penalty_per_min=penalty_per_min))

    return patterns


def generate_patterns(
    appts: List[Appointment],
    provider: str,
    date: str,
    rooms: List[int] = ROOMS,
    single_room: bool = False,
) -> List[Pattern]:
    return generate_initial_patterns(appts, provider, date, rooms,
                                     single_room=single_room)


def feasible_room_assignment(
    appts: List[Appointment],
    date: str,
    rooms: List[int],
    n_patterns: int = 50,
) -> List[Dict[int, int]]:
    """Return feasible raw assignment dicts.  All active appointments eligible."""
    valid_appts = sorted(appts, key=lambda a: a.start_min)
    if not valid_appts:
        return []

    seen: set = set()
    results: List[Dict[int, int]] = []

    for start_room in rooms:
        if len(results) >= n_patterns:
            break
        for secondary in rooms:
            if len(results) >= n_patterns:
                break
            assignment: Dict[int, int] = {}
            room_end_times: Dict[int, int] = {}
            last_r = start_room
            feasible = True
            for a in valid_appts:
                ordered = sorted(
                    [r for r in rooms if room_end_times.get(r, 0) <= a.start_min],
                    key=lambda r: (r != secondary, room_distance(last_r, r)),
                )
                if not ordered:
                    feasible = False
                    break
                chosen_r = ordered[0]
                assignment[a.appt_id] = chosen_r
                room_end_times[chosen_r] = a.end_min
                last_r = chosen_r
            if feasible:
                sig = frozenset(assignment.items())
                if sig not in seen:
                    seen.add(sig)
                    results.append(dict(assignment))

    return results


# ──────────────────────────────────────────────────────────────────────────────
# 4.  MASTER PROBLEM (MP)
# ──────────────────────────────────────────────────────────────────────────────

def print_cg_matrices(
    var_index, patterns_map, c, A, lb, ub, x,
    constraint_labels, iteration_label, max_cols_display=30,
) -> None:
    n_vars = len(var_index)
    n_rows = A.shape[0]
    W = 72

    print("\n" + "=" * W)
    print(f" CG MATRICES  –  {iteration_label} ".center(W))
    print("=" * W)

    print(f"\n{'─'*W}")
    print("  1. COLUMN INDEX TABLE")
    print(f"{'─'*W}")
    print(f"  {'Col j':>6}  {'Provider':<10}  {'Date':<12}  {'Pat ID':>6}  "
          f"{'cost':>10}")
    for j, (prov, date, pidx) in enumerate(var_index):
        pat = patterns_map[(prov, date)][pidx]
        print(f"  {j:>6}  {prov:<10}  {date:<12}  {pat.pattern_id:>6}  "
              f"{c[j]:>10.2f}")

    print(f"\n{'─'*W}")
    print("  2. OBJECTIVE COST VECTOR  c")
    print(f"{'─'*W}")
    row_size = 10
    for start in range(0, n_vars, row_size):
        end = min(start + row_size, n_vars)
        print("  " + "  ".join(f"j={j:>4}" for j in range(start, end)))
        print("  " + "  ".join(f"{c[j]:>6.2f}" for j in range(start, end)))
        print()

    print(f"{'─'*W}")
    print("  3. CONSTRAINT MATRIX  A")
    print(f"     Shape: {n_rows} rows × {n_vars} cols")
    print(f"{'─'*W}")

    col_blocks = list(range(0, n_vars, max_cols_display)) or [0]
    for block_start in col_blocks:
        block_end = min(block_start + max_cols_display, n_vars)
        block_cols = list(range(block_start, block_end))
        print(f"\n  Columns j = {block_start} … {block_end - 1}")
        hdr = (f"  {'Constraint':<36} {'lb':>4} {'ub':>4}  "
               + " ".join(f"{j:>3}" for j in block_cols))
        print(hdr)
        print(f"  {'-'*36} {'-'*4} {'-'*4}  "
              + " ".join("---" for _ in block_cols))
        for r in range(n_rows):
            label = constraint_labels[r] if r < len(constraint_labels) else f"row {r}"
            row_str = (f"  {label[:36]:<36} {lb[r]:>4.0f} {ub[r]:>4.0f}  "
                       + " ".join(f"{int(A[r, j]):>3}" for j in block_cols))
            print(row_str)

    print(f"\n{'─'*W}")
    print("  4. SOLUTION VECTOR  x")
    print(f"{'─'*W}")
    if x is None:
        print("  (No solution.)")
    else:
        for start in range(0, n_vars, row_size):
            end = min(start + row_size, n_vars)
            print("  " + "  ".join(f"j={j:>4}" for j in range(start, end)))
            print("  " + "  ".join(f"{x[j]:>6.3f}" for j in range(start, end)))
            print()

    print("\n" + "=" * W + "\n")


def compute_admin_overlap_minutes(a: Appointment, date: str) -> int:
    """Legacy helper kept for Policy D compatibility."""
    return admin_overlap_minutes(a.start_min, a.end_min, blocked_windows(date))


def _build_mp_arrays(
    groups,
    patterns_map,
    admin_budget=None,
    skip_conflict_constraints=False,
):
    var_index: List[Tuple[str, str, int]] = []
    for (prov, date), pats in patterns_map.items():
        for j in range(len(pats)):
            var_index.append((prov, date, j))

    n_vars = len(var_index)
    c = np.array(
        [patterns_map[(v[0], v[1])][v[2]].cost for v in var_index],
        dtype=float,
    )

    rows_A: List[np.ndarray] = []
    lb_list: List[float] = []
    ub_list: List[float] = []
    constraint_labels: List[str] = []

    group_var_indices: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for vi, (prov, date, pidx) in enumerate(var_index):
        group_var_indices[(prov, date)].append(vi)

    # Constraint (1): Coverage — each active appointment covered exactly once
    all_appt_ids: Dict[Tuple[str, str], List[int]] = {}
    for (prov, date), appts in groups.items():
        ids = [a.appt_id for a in appts if a.is_active]
        if ids:
            all_appt_ids[(prov, date)] = ids

    for (prov, date), appt_ids in all_appt_ids.items():
        pats = patterns_map.get((prov, date), [])
        if not pats:
            continue
        for aid in appt_ids:
            row = np.zeros(n_vars)
            for vi, (_p, _d, pidx) in enumerate(var_index):
                if _p == prov and _d == date and pats[pidx].covers(aid):
                    row[vi] = 1.0
            if row.sum() > 0:
                rows_A.append(row)
                lb_list.append(1.0)
                ub_list.append(1.0)
                constraint_labels.append(f"(1)COVER appt#{aid} {prov} {date}")

    # Constraint (2): Exactly one pattern per provider-day
    for (prov, date), vindices in group_var_indices.items():
        row = np.zeros(n_vars)
        for vi in vindices:
            row[vi] = 1.0
        rows_A.append(row)
        lb_list.append(1.0)
        ub_list.append(1.0)
        constraint_labels.append(f"(2)ASSIGN {prov} {date}")

    # Constraint (3): Cross-provider room conflict
    if not skip_conflict_constraints:
        date_provider_appts: Dict[str, Dict[str, List[Appointment]]] = defaultdict(dict)
        for (prov, date), appts in groups.items():
            active = [a for a in appts if a.is_active]
            if active:
                date_provider_appts[date][prov] = active

        for date, prov_appts in date_provider_appts.items():
            providers_today = list(prov_appts.keys())
            for idx_p, prov_p in enumerate(providers_today):
                for prov_pp in providers_today[idx_p + 1:]:
                    appts_p  = prov_appts[prov_p]
                    appts_pp = prov_appts[prov_pp]
                    pats_p   = patterns_map.get((prov_p,  date), [])
                    pats_pp  = patterns_map.get((prov_pp, date), [])
                    if not pats_p or not pats_pp:
                        continue
                    overlapping_pairs = [
                        (a.appt_id, b.appt_id)
                        for a in appts_p for b in appts_pp
                        if appointments_overlap(a, b)
                    ]
                    if not overlapping_pairs:
                        continue
                    for r in ROOMS:
                        for (aid_p, aid_pp) in overlapping_pairs:
                            row = np.zeros(n_vars)
                            for vi, (_p, _d, pidx) in enumerate(var_index):
                                if _p == prov_p and _d == date:
                                    if patterns_map[(prov_p, date)][pidx].assignment.get(aid_p) == r:
                                        row[vi] = 1.0
                                elif _p == prov_pp and _d == date:
                                    if patterns_map[(prov_pp, date)][pidx].assignment.get(aid_pp) == r:
                                        row[vi] = 1.0
                            if row.sum() > 0:
                                rows_A.append(row)
                                lb_list.append(0.0)
                                ub_list.append(1.0)
                                constraint_labels.append(
                                    f"(3)ROOM_CONFLICT r=ER{r} "
                                    f"appt#{aid_p}({prov_p}) vs "
                                    f"appt#{aid_pp}({prov_pp}) {date}"
                                )

    # Constraint (4): Admin budget (Policy D) — enforced per provider-day
    if admin_budget:
        for (prov, date), vindices in group_var_indices.items():
            B_d = admin_budget.get((prov, date))
            if B_d is None:
                continue
            row = np.zeros(n_vars)
            for vi in vindices:
                pidx = var_index[vi][2]
                pat  = patterns_map[(prov, date)][pidx]
                appts_pd = groups.get((prov, date), [])
                for a in appts_pd:
                    if not a.is_active:
                        continue
                    if pat.covers(a.appt_id):
                        tau_i = compute_admin_overlap_minutes(a, date)
                        row[vi] += tau_i
            if row.sum() > 0:
                rows_A.append(row)
                lb_list.append(0.0)
                ub_list.append(float(B_d))
                constraint_labels.append(
                    f"(4)ADMIN_BUDGET {prov} B={B_d}min {date}")

    A  = np.array(rows_A)
    lb = np.array(lb_list)
    ub = np.array(ub_list)
    return n_vars, var_index, A, lb, ub, c, group_var_indices, constraint_labels


def build_and_solve_master(
    groups,
    patterns_map,
    integer=True,
    print_matrices=False,
    iteration_label="MP",
    admin_budget=None,
    skip_conflict_constraints=False,
    return_duals=False,
) -> Tuple:
    """
    Solve the master problem.

    When return_duals=True (used by column_generation for the LP relaxation),
    returns (obj, chosen, duals_coverage, duals_assignment, constraint_labels)
    instead of the usual (obj, chosen).  This avoids a redundant second linprog
    call which was causing the CG loop to hang on large instances.
    """
    from scipy.optimize import linprog as _linprog

    n_vars, var_index, A, lb, ub, c, group_var_indices, constraint_labels = \
        _build_mp_arrays(groups, patterns_map,
                         admin_budget=admin_budget,
                         skip_conflict_constraints=skip_conflict_constraints)

    if n_vars == 0:
        print("  [MP] No variables – returning empty solution.")
        if return_duals:
            return 0.0, {}, {}, {}, []
        return 0.0, {}

    n_cov      = sum(1 for l in constraint_labels if l.startswith("(1)"))
    n_assign   = sum(1 for l in constraint_labels if l.startswith("(2)"))
    n_conflict = sum(1 for l in constraint_labels if l.startswith("(3)"))
    print(f"  [MP] Variables: {n_vars}, Constraints: {len(constraint_labels)} "
          f"(cov:{n_cov}, assign:{n_assign}, conflict:{n_conflict})")

    if return_duals:
        # ── LP relaxation path: use linprog so we can read dual variables ──────
        eq_mask   = (lb == ub)
        A_eq      = A[eq_mask]
        b_eq      = lb[eq_mask]
        ineq_rows = A[~eq_mask]

        A_ub_parts, b_ub_parts = [], []
        if ineq_rows.shape[0] > 0:
            A_ub_parts.append(ineq_rows)
            b_ub_parts.append(ub[~eq_mask])
            A_ub_parts.append(-ineq_rows)
            b_ub_parts.append(-lb[~eq_mask])

        A_ub_lp = np.vstack(A_ub_parts) if A_ub_parts else None
        b_ub_lp = np.concatenate(b_ub_parts) if b_ub_parts else None

        lp_result = _linprog(
            c=c,
            A_ub=A_ub_lp, b_ub=b_ub_lp,
            A_eq=A_eq,    b_eq=b_eq,
            bounds=[(0, 1)] * n_vars,
            method="highs",
        )

        obj = lp_result.fun if lp_result.success else float("inf")

        duals_eq   = (lp_result.eqlin.marginals
                      if lp_result.success else np.zeros(eq_mask.sum()))
        eq_labels  = [constraint_labels[i] for i, m in enumerate(eq_mask) if m]

        duals_assignment: Dict[Tuple[str, str], float] = {}
        duals_coverage:   Dict[int, float]             = defaultdict(float)

        for dual_val, label in zip(duals_eq, eq_labels):
            if label.startswith("(2)ASSIGN"):
                parts = label.split()
                if len(parts) >= 3:
                    duals_assignment[(parts[1], parts[2])] = float(dual_val)
            elif label.startswith("(1)COVER"):
                parts = label.split()
                if len(parts) >= 2:
                    try:
                        aid = int(parts[1].replace("appt#", ""))
                        duals_coverage[aid] = float(dual_val)
                    except ValueError:
                        pass

        return obj, {}, duals_coverage, duals_assignment, constraint_labels

    # ── Integer (or continuous) path via milp ─────────────────────────────────
    constraints = LinearConstraint(A, lb, ub)
    bounds      = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
    integrality = np.ones(n_vars) if integer else np.zeros(n_vars)

    result = milp(c=c, constraints=constraints,
                  integrality=integrality, bounds=bounds)

    x = result.x if result.success else None

    if print_matrices:
        print_cg_matrices(var_index, patterns_map, c, A, lb, ub, x,
                          constraint_labels, iteration_label)

    if not result.success:
        print(f"  [MP] Solver message: {result.message}")
        return float("inf"), {}

    obj = result.fun
    chosen: Dict[Tuple[str, str], Pattern] = {}
    for vi, (prov, date, pidx) in enumerate(var_index):
        if x[vi] > 0.5:
            chosen[(prov, date)] = patterns_map[(prov, date)][pidx]

    return obj, chosen


# ──────────────────────────────────────────────────────────────────────────────
# 5.  COLUMN GENERATION LOOP
# ──────────────────────────────────────────────────────────────────────────────

def column_generation(
    groups,
    patterns_map,
    max_iterations=10,
    print_matrices=False,
    single_room=False,
    penalty_per_min: float = ADMIN_PENALTY_PER_MIN,
) -> Dict[Tuple[str, str], List[Pattern]]:

    print("\n=== Column Generation Loop ===")
    EPSILON = 1e-6

    for iteration in range(1, max_iterations + 1):
        print(f"\n  Iteration {iteration}")

        # Skip conflict constraints (3) during LP relaxation — they cause
        # exponential matrix growth (tens of thousands of rows) and make the
        # LP solve hang.  Duals for coverage and assignment constraints are
        # sufficient to drive the pricing sub-problem correctly.
        obj, _, duals_coverage, duals_assignment, _ = build_and_solve_master(
            groups, patterns_map, integer=False,
            print_matrices=print_matrices,
            iteration_label=f"CG Iteration {iteration} – LP Relaxation",
            return_duals=True,
            skip_conflict_constraints=True,
        )
        if obj == float("inf"):
            print("  LP infeasible – stopping CG.")
            break
        print(f"  LP objective (relaxation): {obj:.4f}")

        new_cols_added = 0
        for (prov, date), appts in groups.items():
            active = [a for a in appts if a.is_active]
            if not active:
                continue

            result_pricing = solve_pricing_subproblem(
                appts=active, date=date, rooms=ROOMS,
                duals_coverage=duals_coverage,
                dual_assignment=duals_assignment.get((prov, date), 0.0),
                single_room=single_room,
                penalty_per_min=penalty_per_min,
            )
            if result_pricing is None:
                continue

            assignment, rc = result_pricing

            if rc < -EPSILON:
                sig = frozenset(assignment.items())
                existing_sigs = {
                    frozenset(p.assignment.items())
                    for p in patterns_map.get((prov, date), [])
                }
                if sig not in existing_sigs:
                    cost = compute_pattern_cost(
                        assignment, active,
                        date=date, penalty_per_min=penalty_per_min,
                    )
                    pid = len(patterns_map.get((prov, date), []))
                    new_pat = Pattern(pid, prov, date, assignment, cost)
                    patterns_map.setdefault((prov, date), []).append(new_pat)
                    new_cols_added += 1
                    print(f"    Added column for ({prov}, {date}): "
                          f"RC={rc:.4f}, cost={cost:.2f}")

        print(f"  New columns added this iteration: {new_cols_added}")
        if new_cols_added == 0:
            print("  No improving columns found – column generation converged.")
            break

    return patterns_map


# ──────────────────────────────────────────────────────────────────────────────
# 6.  KPI CALCULATION  (includes room-utilization metrics)
# ──────────────────────────────────────────────────────────────────────────────

def compute_room_utilization(
    chosen: Dict[Tuple[str, str], "Pattern"],
    groups: Dict[Tuple[str, str], List[Appointment]],
) -> Dict:
    """
    Room-utilization KPIs.

    Methodology
    -----------
    For each (date, room) pair that appears in the chosen schedule we compute:

        booked_minutes(date, room) =
            Σ_{appts assigned to room on date} min(end_min, DAY_END_MIN)
                                              - max(start_min, DAY_START_MIN)
            clipped to [0, appointment_duration]

    The denominator (available minutes) for a given room on a given day is:

        available_minutes(date, room) =
            schedulable_minutes_per_day(date)   [gross day minus blocked windows]

    This is the same for every room on the same date (rooms share the same
    operating window), so the denominator is simply:

        total_available_room_minutes =
            Σ_{distinct (date, room) pairs used} schedulable_minutes_per_day(date)

    KPIs returned
    -------------
    room_utilization_pct         : booked / available across all used rooms (%)
    rooms_used_count             : number of distinct rooms used
    total_booked_room_minutes    : total appointment-minutes across all rooms
    total_available_room_minutes : denominator (schedulable minutes × rooms used)
    avg_appts_per_room_used      : average appointment count per room used
    avg_booked_min_per_room      : average booked minutes per room used
    room_utilization_by_room     : dict {room: {"booked_min", "available_min",
                                                "utilization_pct", "appt_count"}}
    """
    # Accumulate booked minutes and appointment count per (date, room)
    room_date_booked: Dict[Tuple[int, str], int] = defaultdict(int)   # (room, date) -> booked min
    room_date_count:  Dict[Tuple[int, str], int] = defaultdict(int)   # (room, date) -> # appts
    room_date_available: Dict[Tuple[int, str], int] = {}              # (room, date) -> avail min

    for (prov, date), pat in chosen.items():
        appts = {a.appt_id: a
                 for a in groups.get((prov, date), [])
                 if a.is_active}
        sched_min = schedulable_minutes_per_day(date)

        for aid, room in pat.assignment.items():
            a = appts.get(aid)
            if a is None:
                continue
            # Clip appointment to operating window
            clipped_start = max(a.start_min, DAY_START_MIN)
            clipped_end   = min(a.end_min,   DAY_END_MIN)
            booked = max(0, clipped_end - clipped_start)

            key = (room, date)
            room_date_booked[key]  += booked
            room_date_count[key]   += 1
            room_date_available[key] = sched_min   # same for all providers on date

    if not room_date_booked:
        return {
            "room_utilization_pct":          0.0,
            "rooms_used_count":              0,
            "total_booked_room_minutes":     0,
            "total_available_room_minutes":  0,
            "avg_appts_per_room_used":       0.0,
            "avg_booked_min_per_room":       0.0,
            "room_utilization_by_room":      {},
        }

    total_booked    = sum(room_date_booked.values())
    total_available = sum(room_date_available.values())
    n_room_days     = len(room_date_booked)
    total_appts     = sum(room_date_count.values())

    util_pct = 100.0 * total_booked / total_available if total_available > 0 else 0.0

    # Per-room rollup (summed across all dates that room appears)
    room_booked:    Dict[int, int] = defaultdict(int)
    room_available: Dict[int, int] = defaultdict(int)
    room_appts:     Dict[int, int] = defaultdict(int)

    for (room, date), bmin in room_date_booked.items():
        room_booked[room]    += bmin
        room_available[room] += room_date_available[(room, date)]
        room_appts[room]     += room_date_count[(room, date)]

    room_util_by_room: Dict[int, Dict] = {}
    for room in sorted(room_booked.keys()):
        avail = room_available[room]
        booked = room_booked[room]
        room_util_by_room[room] = {
            "booked_min":      booked,
            "available_min":   avail,
            "utilization_pct": round(100.0 * booked / avail, 1) if avail > 0 else 0.0,
            "appt_count":      room_appts[room],
        }

    return {
        "room_utilization_pct":         round(util_pct, 1),
        "rooms_used_count":             len(room_booked),
        "total_booked_room_minutes":    total_booked,
        "total_available_room_minutes": total_available,
        "avg_appts_per_room_used":      round(total_appts / len(room_booked), 2),
        "avg_booked_min_per_room":      round(total_booked / len(room_booked), 1),
        "room_utilization_by_room":     room_util_by_room,
    }


def compute_kpis(
    chosen: Dict[Tuple[str, str], "Pattern"],
    groups: Dict[Tuple[str, str], List[Appointment]],
) -> Dict:
    """
    Compute all KPIs including room utilization.

    Room-utilization metrics (from compute_room_utilization):
        room_utilization_pct         – % of available room-minutes that are booked
        rooms_used_count             – distinct rooms used across chosen schedule
        total_booked_room_minutes    – sum of appointment-minutes in assigned rooms
        total_available_room_minutes – schedulable room-minutes (denominator)
        avg_appts_per_room_used      – average # appointments per room used
        avg_booked_min_per_room      – average booked minutes per room used
    """
    total_travel   = 0.0
    total_penalty  = 0.0
    total_switches = 0
    unassigned     = 0
    single_room_pd = 0
    provider_days  = 0
    appts_blocked  = 0

    for (prov, date), pat in chosen.items():
        appts = sorted(
            [a for a in groups[(prov, date)] if a.is_active],
            key=lambda a: a.start_min,
        )
        blocks = blocked_windows(date)
        rooms_seq = [pat.assignment.get(a.appt_id)
                     for a in appts if a.appt_id in pat.assignment]

        unassigned += sum(1 for r in rooms_seq if r is None)
        total_switches += sum(
            1 for r1, r2 in zip(rooms_seq, rooms_seq[1:]) if r1 != r2
        )

        appt_map = {a.appt_id: a for a in appts}
        for aid in pat.assignment:
            a = appt_map.get(aid)
            if a:
                ov = admin_overlap_minutes(a.start_min, a.end_min, blocks)
                if ov > 0:
                    appts_blocked += 1
                total_penalty += ADMIN_PENALTY_PER_MIN * ov

        pi_dp = compute_Pi_dp(appts)
        travel = sum(
            room_distance(
                pat.assignment.get(i, 0),
                pat.assignment.get(i_prime, 0),
            )
            for (i, i_prime) in pi_dp
            if i in pat.assignment and i_prime in pat.assignment
        )
        total_travel += travel
        provider_days += 1

        distinct_rooms = {r for r in rooms_seq if r is not None}
        if len(distinct_rooms) == 1:
            single_room_pd += 1

    # Room-utilization sub-KPIs
    util = compute_room_utilization(chosen, groups)

    return {
        # ── Travel & penalty ──────────────────────────────────────────────────
        "total_travel_m":                        total_travel,
        "total_admin_penalty":                   total_penalty,
        "total_cost_with_penalty":               total_travel + total_penalty,
        "appts_scheduled_in_blocked_window":     appts_blocked,
        # ── Scheduling counts ─────────────────────────────────────────────────
        "total_room_switches":                   total_switches,
        "unassigned_appointments":               unassigned,
        "provider_days_solved":                  provider_days,
        "provider_days_single_room":             single_room_pd,
        "avg_travel_per_provider_day":           (
            total_travel / provider_days if provider_days else 0.0
        ),
        # ── Room utilization ──────────────────────────────────────────────────
        "room_utilization_pct":                  util["room_utilization_pct"],
        "rooms_used_count":                      util["rooms_used_count"],
        "total_booked_room_minutes":             util["total_booked_room_minutes"],
        "total_available_room_minutes":          util["total_available_room_minutes"],
        "avg_appts_per_room_used":               util["avg_appts_per_room_used"],
        "avg_booked_min_per_room":               util["avg_booked_min_per_room"],
        # stored separately for display_room_utilization()
        "_room_utilization_by_room":             util["room_utilization_by_room"],
    }


def display_room_utilization(kpis: Dict) -> None:
    """Print the per-room utilization breakdown table."""
    by_room = kpis.get("_room_utilization_by_room", {})
    if not by_room:
        print("  (No room-utilization data.)")
        return

    print("\n" + "=" * 72)
    print("ROOM UTILIZATION BREAKDOWN")
    print(f"  Overall utilization : {kpis['room_utilization_pct']:.1f}%  "
          f"({kpis['total_booked_room_minutes']} booked / "
          f"{kpis['total_available_room_minutes']} available minutes)")
    print(f"  Rooms used          : {kpis['rooms_used_count']} of {NUM_ROOMS}")
    print(f"  Avg booked / room   : {kpis['avg_booked_min_per_room']:.0f} min")
    print(f"  Avg appts / room    : {kpis['avg_appts_per_room_used']:.1f}")
    print("=" * 72)
    print(f"  {'Room':<8}  {'Booked min':>10}  {'Avail min':>10}  "
          f"{'Util %':>8}  {'# Appts':>8}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")
    for room, info in sorted(by_room.items()):
        util_bar = "█" * int(info["utilization_pct"] / 5)  # 1 block per 5 %
        print(f"  {'ER'+str(room):<8}  {info['booked_min']:>10}  "
              f"{info['available_min']:>10}  {info['utilization_pct']:>7.1f}%  "
              f"{info['appt_count']:>8}  {util_bar}")
    print("=" * 72)


# ──────────────────────────────────────────────────────────────────────────────
# 7.  SCHEDULE DISPLAY
# ──────────────────────────────────────────────────────────────────────────────

def display_schedule(chosen, groups, date_filter=None) -> None:
    print("\n" + "=" * 72)
    print("EXAMINATION ROOM SCHEDULE")
    print("=" * 72)

    for (prov, date), pat in sorted(chosen.items()):
        if date_filter and date != date_filter:
            continue

        appts = sorted(
            [a for a in groups[(prov, date)] if a.is_active],
            key=lambda a: a.start_min,
        )
        print(f"\nProvider: {prov}   Date: {date}   "
              f"Pattern cost: {pat.cost:.1f}")
        print(f"  {'Appt':>6}  {'Patient':<12}  {'Time':>5}  "
              f"{'Dur':>4}  {'Room':<10}  {'Penalty':>8}  {'Type':<20}")
        print(f"  {'-'*6}  {'-'*12}  {'-'*5}  {'-'*4}  {'-'*10}  {'-'*8}  {'-'*20}")

        blocks = blocked_windows(date)
        for a in appts:
            room = pat.assignment.get(a.appt_id)
            room_str = f"ER{room}" if room is not None else "UNASSIGNED"
            overlap  = admin_overlap_minutes(a.start_min, a.end_min, blocks)
            pen_str  = f"{ADMIN_PENALTY_PER_MIN * overlap:.1f}" if overlap > 0 else "-"
            h, m = divmod(a.start_min, 60)
            ns_flag   = " [NS]"      if a.no_show else ""
            blk_flag  = " [BLOCKED]" if overlap > 0 else ""
            print(f"  {a.appt_id:>6}  {a.patient_id:<12}  "
                  f"{h:02d}:{m:02d}  {a.duration:>4}  "
                  f"{room_str:<10}  {pen_str:>8}  "
                  f"{a.appt_type:<20}{ns_flag}{blk_flag}")

    print("\n" + "=" * 72)


# ──────────────────────────────────────────────────────────────────────────────
# 8.  POLICIES A–F
# ──────────────────────────────────────────────────────────────────────────────

def policy_a_single_room(
    groups,
    week_lock: bool = False,
    cg_iters: int = 3,
) -> Tuple[float, Dict[Tuple[str, str], "Pattern"]]:
    """
    Policy A – Day variant: each provider uses a single room for the day.
    Week variant (week_lock=True): lock to the dominant room across all days.
    """
    patterns_map: Dict[Tuple[str, str], List[Pattern]] = {}
    for (prov, date), appts in groups.items():
        active = [a for a in appts if a.is_active]
        if active:
            patterns_map[(prov, date)] = generate_initial_patterns(
                active, prov, date, single_room=True
            )

    patterns_map = column_generation(
        groups, patterns_map, max_iterations=cg_iters, single_room=True
    )

    _, chosen = build_and_solve_master(groups, patterns_map, integer=True)

    if week_lock:
        from collections import Counter
        provider_rooms: Dict[str, Counter] = defaultdict(Counter)
        for (prov, date), pat in chosen.items():
            provider_rooms[prov].update(pat.assignment.values())

        pid = max((p.pattern_id for p in chosen.values()), default=0) + 1
        for (prov, date), pat in list(chosen.items()):
            dominant_r = provider_rooms[prov].most_common(1)[0][0]
            new_assignment = {aid: dominant_r for aid in pat.assignment}
            new_cost = compute_pattern_cost(
                new_assignment,
                [a for a in groups[(prov, date)] if a.is_active],
                date=date,
            )
            chosen[(prov, date)] = Pattern(
                pid, prov, date, new_assignment, new_cost)
            pid += 1

    total_cost = sum(p.cost for p in chosen.values())
    return total_cost, chosen


def policy_b_cluster(
    groups,
    proximity_threshold: float = 3.0,
    anchor_room: int = 1,
) -> Tuple[float, Dict[Tuple[str, str], "Pattern"]]:
    """Policy B: restrict each provider-day to rooms within proximity_threshold of anchor."""
    cluster = [r for r in ROOMS
               if room_distance(anchor_room, r) <= proximity_threshold]
    if not cluster:
        cluster = ROOMS[:4]

    chosen: Dict[Tuple[str, str], Pattern] = {}
    total_cost = 0.0
    pid = 0

    for (prov, date), appts in groups.items():
        active = [a for a in appts if a.is_active]
        if not active:
            continue

        pats = feasible_room_assignment(active, date, cluster)
        if not pats:
            pats = feasible_room_assignment(active, date, ROOMS)

        if pats:
            costs = [compute_pattern_cost(p, active, date=date) for p in pats]
            best_idx = int(np.argmin(costs))
            cost = costs[best_idx]
            pat = Pattern(pid, prov, date, pats[best_idx], cost)
            chosen[(prov, date)] = pat
            total_cost += cost
            pid += 1

    return total_cost, chosen


def policy_c_blocked_days(
    groups,
    blocked: Optional[Dict[str, List[str]]] = None,
) -> Tuple[float, Dict[Tuple[str, str], "Pattern"]]:
    """Policy C: block certain provider-days entirely (admin/leave days)."""
    if blocked is None:
        blocked = {"HPW114": ["11-11-2025"]}

    chosen: Dict[Tuple[str, str], Pattern] = {}
    total_cost = 0.0
    pid = 0
    blocked_count = 0

    for (prov, date), appts in groups.items():
        if prov in blocked and date in blocked[prov]:
            blocked_count += 1
            print(f"  [Policy C] Blocking {prov} on {date} "
                  f"({len(appts)} appointments dropped).")
            continue

        active = [a for a in appts if a.is_active]
        if not active:
            continue

        pats = feasible_room_assignment(active, date, ROOMS)
        if pats:
            costs = [compute_pattern_cost(p, active, date=date) for p in pats]
            best_idx = int(np.argmin(costs))
            cost = costs[best_idx]
            pat = Pattern(pid, prov, date, pats[best_idx], cost)
            chosen[(prov, date)] = pat
            total_cost += cost
            pid += 1

    print(f"  [Policy C] {blocked_count} provider-day(s) blocked.")
    return total_cost, chosen


def policy_d_admin_overflow(
    groups,
    allow_admin_overflow: bool = True,
    admin_budget_minutes: float = 30.0,
) -> Tuple[float, Dict[Tuple[str, str], "Pattern"]]:
    """
    Policy D: Administrative Buffer Time.

    ON  (allow_admin_overflow=True):
        Appointments in blocked windows ARE schedulable. Each provider-day is
        capped at admin_budget_minutes of total blocked-window overlap via
        Constraint 4. The budget is applied per provider-day. If a provider-day's
        minimum possible overlap exceeds the budget (i.e. the constraint would be
        infeasible), the budget is raised to that provider-day's actual minimum
        so the solver always has a feasible solution.

    OFF (allow_admin_overflow=False):
        Appointments that overlap any blocked window are excluded from pattern
        generation entirely — hard filter, no Constraint 4 needed.
    """
    patterns_map: Dict[Tuple[str, str], List[Pattern]] = {}

    for (prov, date), appts in groups.items():
        if allow_admin_overflow:
            active = [a for a in appts if a.is_active]
        else:
            blocks = blocked_windows(date)
            active = [
                a for a in appts
                if a.is_active and
                   admin_overlap_minutes(a.start_min, a.end_min, blocks) == 0
            ]
        if active:
            patterns_map[(prov, date)] = generate_initial_patterns(
                active, prov, date)

    if allow_admin_overflow:
        # Compute per-provider-day budget: at least the minimum overlap across
        # all patterns for that provider-day, so the constraint is always feasible.
        pd_budget: Dict[Tuple[str, str], float] = {}
        for (prov, date), pats in patterns_map.items():
            appts_pd = [a for a in groups.get((prov, date), []) if a.is_active]
            blocks    = blocked_windows(date)
            min_overlap = float("inf")
            for pat in pats:
                pat_overlap = sum(
                    admin_overlap_minutes(a.start_min, a.end_min, blocks)
                    for a in appts_pd if pat.covers(a.appt_id)
                )
                min_overlap = min(min_overlap, pat_overlap)
            if min_overlap == float("inf"):
                min_overlap = 0.0
            # Use the requested budget, but never below the minimum achievable
            pd_budget[(prov, date)] = max(admin_budget_minutes, min_overlap)

        _, chosen = build_and_solve_master(
            groups, patterns_map, integer=True, admin_budget=pd_budget,
        )
    else:
        _, chosen = build_and_solve_master(
            groups, patterns_map, integer=True,
        )

    total_cost = sum(p.cost for p in chosen.values())
    return total_cost, chosen


def policy_e_overbook(
    groups,
    no_show_rate: float = 0.10,
    overbook_factor: float = 1.15,
) -> Tuple[float, Dict[Tuple[str, str], "Pattern"]]:
    """Policy E: overbook appointments to account for expected no-shows."""
    chosen: Dict[Tuple[str, str], Pattern] = {}
    total_cost = 0.0
    pid = 0
    extra_total = 0

    for (prov, date), appts in groups.items():
        all_active = [a for a in appts if not a.cancelled and not a.deleted]
        no_shows   = [a for a in all_active if a.no_show]
        real_appts = [a for a in all_active if not a.no_show]

        obs_rate = (len(no_shows) / len(all_active)
                    if all_active else no_show_rate)
        expected_no_shows = max(obs_rate * len(all_active), len(no_shows))
        extra_slots = int(np.floor(expected_no_shows * overbook_factor))
        extra_total += extra_slots

        schedule_appts = real_appts + no_shows
        if not schedule_appts:
            continue

        pats = feasible_room_assignment(schedule_appts, date, ROOMS)
        if pats:
            costs = [compute_pattern_cost(p, schedule_appts, date=date)
                     for p in pats]
            best_idx = int(np.argmin(costs))
            cost = costs[best_idx]
            pat = Pattern(pid, prov, date, pats[best_idx], cost)
            chosen[(prov, date)] = pat
            total_cost += cost
            pid += 1

    print(f"  [Policy E] Total extra overbook slots added: {extra_total}")
    return total_cost, chosen


def policy_f_uncertainty(
    groups,
    buffer_factor: float = 0.20,
    cg_iters: int = 3,
) -> Tuple[float, Dict[Tuple[str, str], "Pattern"]]:
    """
    Policy F: inflate appointment durations by buffer_factor (d'_i = d_i(1+μ))
    to account for uncertainty, then solve and remap to nominal durations.
    """
    buffered_groups: Dict[Tuple[str, str], List[Appointment]] = {}
    for (prov, date), appts in groups.items():
        buffered = []
        for a in appts:
            if not a.is_active:
                continue
            buffered_dur = int(np.ceil(a.duration * (1 + buffer_factor)))
            buffered.append(Appointment(
                appt_id=a.appt_id, patient_id=a.patient_id,
                provider=a.provider, date=a.date,
                start_min=a.start_min, duration=buffered_dur,
                appt_type=a.appt_type, no_show=a.no_show,
                cancelled=a.cancelled, deleted=a.deleted,
            ))
        if buffered:
            buffered_groups[(prov, date)] = buffered

    patterns_map: Dict[Tuple[str, str], List[Pattern]] = {}
    for (prov, date), buffered_appts in buffered_groups.items():
        patterns_map[(prov, date)] = generate_initial_patterns(
            buffered_appts, prov, date)

    patterns_map = column_generation(buffered_groups, patterns_map, cg_iters)

    _, chosen_buffered = build_and_solve_master(
        buffered_groups, patterns_map, integer=True,
        skip_conflict_constraints=True,
    )

    chosen: Dict[Tuple[str, str], Pattern] = {}
    pid = 0
    for (prov, date), pat in chosen_buffered.items():
        orig_appts = [a for a in groups.get((prov, date), []) if a.is_active]
        nominal_cost = compute_pattern_cost(pat.assignment, orig_appts, date=date)
        chosen[(prov, date)] = Pattern(
            pid, prov, date, pat.assignment, nominal_cost)
        pid += 1

    total_cost = sum(p.cost for p in chosen.values())
    return total_cost, chosen


# ──────────────────────────────────────────────────────────────────────────────
# 9.  POLICY COMPARISON TABLE
# ──────────────────────────────────────────────────────────────────────────────

def run_all_policies(groups, optimal_kpis) -> None:
    print("\n" + "=" * 80)
    print("POLICY COMPARISON  (Policies A – F  vs  Column-Generation Optimal)")
    print("=" * 80)

    policies = [
        ("A – Single Room (day)",
         lambda g: policy_a_single_room(g, week_lock=False)),
        ("A – Single Room (week)",
         lambda g: policy_a_single_room(g, week_lock=True)),
        ("B – Cluster (≤3 m)",
         lambda g: policy_b_cluster(g, proximity_threshold=3.0)),
        ("B – Cluster (≤5 m)",
         lambda g: policy_b_cluster(g, proximity_threshold=5.0)),
        ("C – Blocked Days",
         lambda g: policy_c_blocked_days(g)),
        ("D – Admin Overflow ON",
         lambda g: policy_d_admin_overflow(g, allow_admin_overflow=True)),
        ("D – Admin Overflow OFF",
         lambda g: policy_d_admin_overflow(g, allow_admin_overflow=False)),
        ("E – Overbook (10% NS)",
         lambda g: policy_e_overbook(g, no_show_rate=0.10)),
        ("F – Uncertainty (σ=20%)",
         lambda g: policy_f_uncertainty(g, buffer_factor=0.20)),
    ]

    # Metrics shown in comparison table
    metrics = [
        ("total_travel_m",              "Travel (m)"),
        ("total_admin_penalty",         "Adm Penalty"),
        ("appts_scheduled_in_blocked_window", "Blk Appts"),
        ("total_room_switches",         "Rm Switch"),
        ("room_utilization_pct",        "Rm Util %"),
        ("rooms_used_count",            "Rms Used"),
        ("avg_appts_per_room_used",     "Appt/Rm"),
        ("provider_days_solved",        "PD Solved"),
    ]

    results = []
    for label, fn in policies:
        print(f"\n  Running {label} ...")
        _, chosen = fn(groups)
        kpi = compute_kpis(chosen, groups)
        results.append((label, kpi))

    col_w   = 11
    label_w = 26
    print(f"\n  {'Policy':<{label_w}}", end="")
    for _, header in metrics:
        print(f"  {header:>{col_w}}", end="")
    print()
    print(f"  {'-'*label_w}", end="")
    for _ in metrics:
        print(f"  {'-'*col_w}", end="")
    print()

    def fmt(v) -> str:
        if isinstance(v, float):
            return f"{v:.1f}"
        return str(v)

    # Optimal row
    print(f"  {'Optimal (CG)':<{label_w}}", end="")
    for key, _ in metrics:
        print(f"  {fmt(optimal_kpis.get(key, 0)):>{col_w}}", end="")
    print()

    # Policy rows
    for label, kpi in results:
        print(f"  {label:<{label_w}}", end="")
        for key, _ in metrics:
            print(f"  {fmt(kpi.get(key, 0)):>{col_w}}", end="")
        print()

    print()


# ──────────────────────────────────────────────────────────────────────────────
# 10. GANTT EXPORT
# ──────────────────────────────────────────────────────────────────────────────

def export_gantt(
    chosen,
    groups,
    date_filter=None,
    output_path="schedule_gantt.png",
):
    """Export schedule as a Gantt chart PNG.  Blocked-window appointments get a red border."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch

    dates = sorted({date for (_, date) in chosen.keys()
                    if date_filter is None or date == date_filter})
    if not dates:
        print("  [Gantt] No schedule data to plot.")
        return

    all_providers = sorted({prov for (prov, _) in chosen.keys()})
    cmap = plt.cm.get_cmap("tab20", max(len(all_providers), 1))
    prov_colour = {p: cmap(i) for i, p in enumerate(all_providers)}
    saved_files = []

    for date in dates:
        blocks    = blocked_windows(date)
        day_start = DAY_START_MIN
        day_end   = DAY_END_MIN

        bars = []
        for (prov, d), pat in chosen.items():
            if d != date:
                continue
            for a in groups.get((prov, d), []):
                room = pat.assignment.get(a.appt_id)
                if room is None:
                    continue
                overlap = admin_overlap_minutes(a.start_min, a.end_min, blocks)
                bars.append((room, a.start_min, a.duration,
                             prov, a.patient_id, a.no_show, overlap > 0))

        if not bars:
            continue

        rooms_used = sorted({b[0] for b in bars})
        n_rooms    = len(rooms_used)
        room_to_y  = {r: i for i, r in enumerate(rooms_used)}

        fig_w = max(18, (day_end - day_start) / 30)
        fig_h = max(6, n_rooms * 0.55 + 2)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        for (bs, be) in blocks:
            ax.axvspan(bs, be, color="lightgrey", alpha=0.5,
                       hatch="//", linewidth=0, zorder=0)
            mid   = (bs + be) / 2
            label = "Admin" if (be - bs) <= 30 else "Lunch"
            ax.text(mid, n_rooms - 0.3, label, ha="center", va="top",
                    fontsize=6, color="grey", style="italic")

        bar_h = 0.7
        for (room, start, dur, prov, patient, no_show, in_block) in bars:
            y          = room_to_y[room]
            colour     = prov_colour[prov]
            edge_color = "red" if in_block else "white"
            edge_width = 1.5  if in_block else 0.6

            rect = FancyBboxPatch(
                (start, y - bar_h / 2), dur, bar_h,
                boxstyle="round,pad=1",
                linewidth=edge_width,
                edgecolor=edge_color,
                facecolor=colour,
                alpha=0.5 if no_show else 0.88,
                zorder=2,
            )
            ax.add_patch(rect)

            if no_show:
                ax.add_patch(FancyBboxPatch(
                    (start, y - bar_h / 2), dur, bar_h,
                    boxstyle="round,pad=1",
                    linewidth=0, edgecolor=colour,
                    facecolor="none", hatch="xxx", zorder=3,
                ))

            if dur >= 8:
                ax.text(start + dur / 2, y,
                        patient.replace("Patient ", "P"),
                        ha="center", va="center",
                        fontsize=5.5, fontweight="bold",
                        color="white", zorder=4, clip_on=True)

        ax.set_yticks(range(n_rooms))
        ax.set_yticklabels([f"ER{r}" for r in rooms_used], fontsize=8)
        ax.set_ylim(-0.7, n_rooms - 0.3)

        tick_mins = list(range(day_start, day_end + 1, 30))
        ax.set_xticks(tick_mins)
        ax.set_xticklabels(
            [f"{m // 60:02d}:{m % 60:02d}" for m in tick_mins],
            rotation=45, ha="right", fontsize=7,
        )
        ax.set_xlim(day_start, day_end)
        ax.set_xlabel("Time of Day", fontsize=9)
        ax.set_ylabel("Exam Room", fontsize=9)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, zorder=1)
        ax.yaxis.grid(True, linestyle="-",  linewidth=0.3, alpha=0.3, zorder=1)
        ax.set_title(
            f"Examination Room Schedule — {date}  "
            f"(red border = admin-window penalty)",
            fontsize=12, fontweight="bold", pad=10,
        )

        providers_today = sorted({b[3] for b in bars})
        legend_patches = [
            mpatches.Patch(facecolor=prov_colour[p], label=p, alpha=0.88)
            for p in providers_today
        ]
        legend_patches += [
            mpatches.Patch(facecolor="grey", hatch="xxx",
                           label="No-Show", alpha=0.5),
            mpatches.Patch(facecolor="white", edgecolor="red",
                           linewidth=1.5, label="Admin-window penalty"),
        ]
        ax.legend(handles=legend_patches, loc="upper right",
                  fontsize=7,
                  ncol=max(1, len(providers_today) // 6 + 1),
                  framealpha=0.9, title="Provider", title_fontsize=7)

        plt.tight_layout()

        if len(dates) == 1:
            fname = output_path
        else:
            base, ext = (output_path.rsplit(".", 1)
                         if "." in output_path else (output_path, "png"))
            fname = f"{base}_{date}.{ext}"

        fig.savefig(fname, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(fname)
        print(f"  [Gantt] Saved: {fname}")

    return saved_files


# ──────────────────────────────────────────────────────────────────────────────
# 11. MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def main(
    csv_path: str,
    day_filter: Optional[str] = None,
    max_cg_iters: int = 3,
    print_matrices: bool = False,
    gantt: bool = False,
    gantt_path: str = "schedule_gantt.png",
    penalty_per_min: float = ADMIN_PENALTY_PER_MIN,
) -> None:
    print(f"\nLoading appointments from: {csv_path}")
    print(f"Admin-window soft penalty : {penalty_per_min:.1f} per overlapping minute")
    all_appts = load_appointments(csv_path)
    print(f"  Loaded {len(all_appts)} appointments (after filtering 0-duration).")

    groups: Dict[Tuple[str, str], List[Appointment]] = defaultdict(list)
    for a in all_appts:
        groups[(a.provider, a.date)].append(a)

    if day_filter:
        groups = {k: v for k, v in groups.items() if k[1] == day_filter}
        print(f"  Filtered to date: {day_filter}  "
              f"({len(groups)} provider-day groups)")

    # ── Initial pattern generation ────────────────────────────────────────────
    print("\n=== Generating Initial Patterns ===")
    patterns_map: Dict[Tuple[str, str], List[Pattern]] = {}
    total_initial = 0
    for (prov, date), appts in groups.items():
        active = [a for a in appts if a.is_active]
        if not active:
            continue
        pats = generate_patterns(active, prov, date)
        patterns_map[(prov, date)] = pats
        total_initial += len(pats)
        print(f"  ({prov}, {date}): {len(active)} active appts → "
              f"{len(pats)} patterns")
    print(f"  Total initial patterns: {total_initial}")

    # ── Column generation ─────────────────────────────────────────────────────
    if max_cg_iters > 0:
        patterns_map = column_generation(
            groups, patterns_map, max_cg_iters,
            print_matrices=print_matrices,
            penalty_per_min=penalty_per_min,
        )

    # ── Solve integer Master Problem ──────────────────────────────────────────
    print("\n=== Solving Integer Master Problem (MIP) ===")
    obj, chosen_opt = build_and_solve_master(
        groups, patterns_map, integer=True,
        print_matrices=print_matrices,
        iteration_label="Final MIP Solve",
    )
    print(f"  Optimal total cost (travel + penalty): {obj:.2f}")

    # ── Display schedule ──────────────────────────────────────────────────────
    display_schedule(chosen_opt, groups, date_filter=day_filter)

    # ── Gantt export ──────────────────────────────────────────────────────────
    if gantt:
        print("\n=== Exporting Gantt Chart ===")
        export_gantt(chosen_opt, groups,
                     date_filter=day_filter, output_path=gantt_path)

    # ── KPIs ──────────────────────────────────────────────────────────────────
    kpis_opt = compute_kpis(chosen_opt, groups)

    print("\n=== Optimal Schedule KPIs ===")
    skip = {"_room_utilization_by_room"}
    for k, v in kpis_opt.items():
        if k in skip:
            continue
        if isinstance(v, float):
            print(f"  {k:<45}: {v:.2f}")
        else:
            print(f"  {k:<45}: {v}")

    # ── Room-utilization breakdown ────────────────────────────────────────────
    display_room_utilization(kpis_opt)

    # ── Policy comparison ─────────────────────────────────────────────────────
    run_all_policies(groups, kpis_opt)

    print("Done.")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MSE 435 Examination Room Scheduler – soft admin-window constraints"
    )
    parser.add_argument("--csv",  default="AppointmentDataWeek1.csv",
                        help="Path to appointment CSV file")
    parser.add_argument("--day",  default=None,
                        help="Filter to a single date (MM-DD-YYYY)")
    parser.add_argument("--cg-iters", type=int, default=3,
                        help="Max column generation iterations (0 to skip CG)")
    parser.add_argument("--print-matrices", action="store_true",
                        help="Print all CG matrices at each iteration")
    parser.add_argument("--gantt", action="store_true",
                        help="Export schedule as a Gantt chart PNG")
    parser.add_argument("--gantt-path", default="schedule_gantt.png",
                        help="Output path for Gantt PNG")
    parser.add_argument(
        "--penalty", type=float, default=ADMIN_PENALTY_PER_MIN,
        help=f"Admin-window penalty per overlapping minute "
             f"(default: {ADMIN_PENALTY_PER_MIN})",
    )
    args = parser.parse_args()
    main(
        csv_path=args.csv,
        day_filter=args.day,
        max_cg_iters=args.cg_iters,
        print_matrices=args.print_matrices,
        gantt=args.gantt,
        gantt_path=args.gantt_path,
        penalty_per_min=args.penalty,
    )

# The use of GEN-AI was used to assist in coding up the column generation
# formulation, MP, the policies A-F, and the room-utilization KPI.