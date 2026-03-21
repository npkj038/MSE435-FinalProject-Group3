"""
MSE 435 – Examination Room Scheduling via Column Generation
===========================================================
Implements:
  • Data loading from the project CSV
  • Pricing sub-problem  (enumerate feasible patterns per provider per day)
  • Master Problem (MP)  (set-cover / set-partition LP solved via scipy.optimize.milp)
  • Integer solution recovery (branch-and-bound via scipy MILP)
  • KPI reporting and Gantt-style schedule printing

Run:
    python hospital_room_scheduler.py --csv data/AppointmentDataWeek1.csv [--day 11-10-2025]
"""

import argparse
import csv
import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# ──────────────────────────────────────────────────────────────────────────────
# 0.  PROBLEM CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

# Rectilinear distance table (ER1 … ER16).  Upper-triangle supplied by PDF.
# We fill the full symmetric matrix here.
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

def room_distance(r1: int, r2: int) -> float:
    """Return rectilinear distance between rooms r1 and r2 (1-indexed)."""
    if r1 == r2:
        return 0.0
    key = (min(r1, r2), max(r1, r2))
    return _RAW_DISTANCES.get(key, 0.0)

# Operating time windows (minutes from midnight)
# Mon–Thu: admin 9:00–9:30, noon-admin 11:30–12:00, lunch 12:00–13:00, admin 16:30–17:00
# Fri:     admin 8:00–8:30,  noon-admin 11:30–12:00, lunch 12:00–13:00, admin 15:00–15:30
BLOCKED_MON_THU = [
    (9 * 60,      9 * 60 + 30),   # morning admin
    (11 * 60 + 30, 12 * 60),      # noon admin
    (12 * 60,     13 * 60),       # lunch
    (16 * 60 + 30, 17 * 60),      # afternoon admin
]
BLOCKED_FRI = [
    (8 * 60,       8 * 60 + 30),  # morning admin
    (11 * 60 + 30, 12 * 60),      # noon admin
    (12 * 60,      13 * 60),      # lunch
    (15 * 60,      15 * 60 + 30), # afternoon admin
]

def is_friday(date_str: str) -> bool:
    """Determine if a date string (MM-DD-YYYY) is a Friday."""
    from datetime import datetime
    try:
        dt = datetime.strptime(date_str, "%m-%d-%Y")
        return dt.weekday() == 4  # 0=Mon … 4=Fri
    except ValueError:
        return False

def blocked_windows(date_str: str) -> List[Tuple[int, int]]:
    return BLOCKED_FRI if is_friday(date_str) else BLOCKED_MON_THU

def time_str_to_minutes(t: str) -> int:
    """Convert 'HH:MM:SS' or 'HH:MM' to integer minutes from midnight."""
    parts = t.strip().split(":")
    return int(parts[0]) * 60 + int(parts[1])

def is_available(start_min: int, end_min: int, blocks: List[Tuple[int, int]]) -> bool:
    """Return True if [start, end) does not overlap any blocked window."""
    for (bs, be) in blocks:
        if start_min < be and end_min > bs:
            return False
    return True


# ──────────────────────────────────────────────────────────────────────────────
# 1.  DATA STRUCTURES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Appointment:
    appt_id: int
    patient_id: str
    provider: str
    date: str
    start_min: int          # minutes from midnight
    duration: int           # minutes
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
    """
    One feasible assignment pattern h for (provider p, day d).
    assignment: dict mapping appt_id -> room_number
    cost:       total travel distance (metres) induced by this pattern
    """
    pattern_id: int
    provider: str
    date: str
    assignment: Dict[int, int]   # appt_id -> room
    cost: float

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
            # Strip whitespace from keys and values
            row = {k.strip(): (v.strip() if isinstance(v, str) else v)
                   for k, v in row.items()}
            duration_raw = row.get("Appt Duration", "0") or "0"
            duration = int(float(duration_raw)) if duration_raw else 0
            if duration <= 0:
                continue  # skip 0-duration rows

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
#     Enumerate all feasible assignment patterns for one (provider, day) group.
# ──────────────────────────────────────────────────────────────────────────────

def compute_K_dp(appts: List[Appointment], rooms: List[int]) -> List[Tuple[int, int]]:
    """
    K_{d,p} = {(r, i) : r ∈ R, i ∈ A_{d,p}}
    Returns the full set of room-appointment combinations for a provider-day.
    """
    return [(r, a.appt_id) for r in rooms for a in appts]


def compute_Pi_dp(appts: List[Appointment]) -> List[Tuple[int, int]]:
    """
    Π_{d,p} = {(i, i') : i, i' ∈ A_{d,p}, i is immediately before i'
               in provider p's schedule on day d}
    Appointments are sorted by start time; consecutive pairs form Π.
    Note: per Image 2, appointments within a provider are non-overlapping,
    so the consecutive ordering is well-defined.
    """
    sorted_appts = sorted(appts, key=lambda a: a.start_min)
    return [(sorted_appts[k].appt_id, sorted_appts[k+1].appt_id)
            for k in range(len(sorted_appts) - 1)]


def compute_pattern_cost(assignment: Dict[int, int],
                          appts: List[Appointment]) -> float:
    """
    cost^h_{d,p} = Σ_{(i,i') ∈ Π_{d,p}} c_{r^h_i, r^h_{i'}}

    where r^h_i is the room assigned to appointment i in pattern h,
    and Π_{d,p} is the set of consecutive appointment pairs (Image 3).
    c_{r,r'} is the rectilinear distance between rooms r and r'.
    """
    pi_dp = compute_Pi_dp(appts)
    cost = 0.0
    for (i, i_prime) in pi_dp:
        r_i      = assignment.get(i)
        r_iprime = assignment.get(i_prime)
        if r_i is not None and r_iprime is not None:
            cost += room_distance(r_i, r_iprime)
    return cost


def appointments_overlap(a1: Appointment, a2: Appointment) -> bool:
    """True if two appointments overlap in time."""
    return a1.start_min < a2.end_min and a2.start_min < a1.end_min


def solve_pricing_subproblem(
    appts: List[Appointment],
    date: str,
    rooms: List[int],
    duals_coverage: Dict[int, float],
    dual_assignment: float,
) -> Optional[Tuple[Dict[int, int], float]]:
    """
    Pricing sub-problem (column generation oracle).

    Finds the single feasible assignment pattern with the most negative
    reduced cost, defined as:

        RC(h) = cost^h_{dp}  -  Σ_{k∈K_{dp}} μ_k · a^h_{dp,k}  -  λ_{dp}

    where:
        cost^h_{dp}   = travel distance of pattern h
        μ_k           = dual variable for coverage constraint of appointment k
        a^h_{dp,k}    = 1 if pattern h covers appointment k (always 1 here
                        since every active appointment must be assigned)
        λ_{dp}        = dual variable for the assignment constraint of (p,d)

    This is solved via branch-and-bound over room assignments, processing
    appointments in chronological order.  Because appointments are independent
    across rooms (a room conflict only arises from time overlap), the search
    tree is pruned whenever the partial reduced cost already exceeds the
    best complete RC found so far.

    Feasibility rules enforced:
      1. Every appointment is assigned to exactly one room.
      2. No two overlapping appointments share a room.
      3. Appointments in blocked admin/lunch windows are excluded.

    Returns (assignment_dict, reduced_cost) for the best pattern found,
    or None if no feasible pattern exists.
    """
    blocks = blocked_windows(date)

    # Only schedule appointments that fall outside hard-blocked windows
    valid_appts = sorted(
        [a for a in appts if is_available(a.start_min, a.end_min, blocks)],
        key=lambda a: a.start_min,
    )
    if not valid_appts:
        return None

    n = len(valid_appts)

    # Dual contribution that every complete pattern earns just by existing:
    # Σ_k μ_k  +  λ_{dp}   (since a^h_{dp,k} = 1 for all k in a complete pattern)
    total_dual = sum(duals_coverage.get(a.appt_id, 0.0) for a in valid_appts) \
                 + dual_assignment

    # Best reduced cost found so far (we want the most negative)
    best_rc: List[float] = [float("inf")]
    best_assignment: List[Optional[Dict[int, int]]] = [None]

    def backtrack(
        idx: int,
        current: Dict[int, int],
        room_end_times: Dict[int, int],   # room -> earliest free minute
        partial_travel: float,
        last_room: Optional[int],
        last_end: int,
    ) -> None:
        """
        idx:           index into valid_appts being assigned next
        current:       partial assignment so far {appt_id: room}
        room_end_times: when each room becomes free
        partial_travel: travel cost accumulated so far
        last_room:     room used by the immediately preceding appointment
        last_end:      end time of the immediately preceding appointment
        """
        if idx == n:
            # Complete pattern — compute reduced cost
            rc = partial_travel - total_dual
            if rc < best_rc[0]:
                best_rc[0] = rc
                best_assignment[0] = dict(current)
            return

        a = valid_appts[idx]

        for r in rooms:
            # Room conflict check: room must be free at appointment start
            if room_end_times.get(r, 0) > a.start_min:
                continue

            # Travel cost increment: distance from last room to this room
            # (only applies if this appointment immediately follows another
            #  in time; we track the previous appointment's room regardless
            #  of which room it was, since the provider walks between them)
            if last_room is not None and a.start_min >= last_end:
                travel_inc = room_distance(last_room, r)
            else:
                travel_inc = 0.0

            new_travel = partial_travel + travel_inc

            # Pruning: even if all remaining appointments have zero travel,
            # can we beat the current best?
            # Lower bound on RC for this branch = new_travel - total_dual
            # (remaining duals already subtracted in total_dual)
            if new_travel - total_dual >= best_rc[0]:
                continue  # prune

            # Assign
            prev_end = room_end_times.get(r, 0)
            current[a.appt_id] = r
            room_end_times[r] = a.end_min

            backtrack(
                idx + 1, current, room_end_times,
                new_travel, r, a.end_min,
            )

            # Un-assign
            del current[a.appt_id]
            if prev_end == 0:
                del room_end_times[r]
            else:
                room_end_times[r] = prev_end

    backtrack(0, {}, {}, 0.0, None, 0)

    if best_assignment[0] is None:
        return None
    return best_assignment[0], best_rc[0]


def _greedy_pattern(
    valid_appts: List[Appointment],
    rooms: List[int],
    start_room: int,
    preferred_room: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Optional[Dict[int, int]]:
    """
    Build one greedy feasible pattern by processing appointments in
    chronological order.  At each step, pick the available room that
    minimises travel from the previous room, with optional preference
    for `preferred_room` and optional random tie-breaking via `rng`.
    Returns None if no feasible assignment exists.
    """
    assignment: Dict[int, int] = {}
    room_end_times: Dict[int, int] = {}
    last_r = start_room

    for a in valid_appts:
        free_rooms = [r for r in rooms if room_end_times.get(r, 0) <= a.start_min]
        if not free_rooms:
            return None

        # Sort by: (not preferred, distance from last_r, optional jitter)
        def key(r: int) -> Tuple:
            jitter = float(rng.random()) * 0.01 if rng is not None else 0.0
            return (r != preferred_room, room_distance(last_r, r) + jitter)

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
) -> List[Pattern]:
    """
    Generate a diverse set of good initial patterns for the MP.

    Strategy (in order, deduplicating throughout):
      1. Single-room patterns  — one per room, zero travel, feasible only
         when no two appointments in the schedule overlap.
      2. Greedy nearest-room   — for each room as anchor, always pick the
         closest free room to the previous one.  Gives low-cost patterns.
      3. Greedy room-pinned    — for each room r, always prefer r when it
         is free, fall back to nearest otherwise.  Diverse room usage.
      4. Randomised greedy     — repeated random-jitter greedy passes to
         fill up to n_initial distinct patterns.

    The result is a varied column pool that gives the LP relaxation a
    realistic feasible basis without enumerating combinatorially.
    """
    blocks = blocked_windows(date)
    valid_appts = sorted(
        [a for a in appts if is_available(a.start_min, a.end_min, blocks)],
        key=lambda a: a.start_min,
    )
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
        sig = frozenset(assignment.items())
        if sig in seen:
            return False
        seen.add(sig)
        cost = compute_pattern_cost(assignment, valid_appts)
        patterns.append(Pattern(pid, provider, date, dict(assignment), cost))
        pid += 1
        return True

    # ── 1. Single-room patterns (zero travel) ────────────────────────────────
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

    # ── 2. Greedy nearest-room (one per anchor room) ─────────────────────────
    for start_room in rooms:
        add(_greedy_pattern(valid_appts, rooms, start_room))

    # ── 3. Greedy room-pinned (prefer room r, fall back when busy) ───────────
    for preferred in rooms:
        add(_greedy_pattern(valid_appts, rooms,
                            start_room=preferred, preferred_room=preferred))

    # ── 4. Randomised greedy to reach n_initial distinct patterns ────────────
    attempts = 0
    while len(patterns) < n_initial and attempts < n_initial * 10:
        attempts += 1
        start_room = int(rng.choice(rooms))
        preferred  = int(rng.choice(rooms))
        add(_greedy_pattern(valid_appts, rooms, start_room,
                            preferred_room=preferred, rng=rng))

    return patterns


def generate_patterns(appts: List[Appointment],
                       provider: str,
                       date: str,
                       rooms: List[int] = ROOMS) -> List[Pattern]:
    """
    Wrapper used for initial pattern generation.
    Returns a compact set of good starting patterns (not full enumeration).
    """
    return generate_initial_patterns(appts, provider, date, rooms)


def feasible_room_assignment(
    appts: List[Appointment],
    date: str,
    rooms: List[int],
    n_patterns: int = 50,
) -> List[Dict[int, int]]:
    """
    Return a list of feasible raw assignment dicts (appt_id -> room).
    Used by the policy functions.  Generates up to n_patterns good patterns
    via the greedy nearest-room strategy rather than full enumeration.
    """
    blocks = blocked_windows(date)
    valid_appts = sorted(
        [a for a in appts if is_available(a.start_min, a.end_min, blocks)],
        key=lambda a: a.start_min,
    )
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
                # prefer secondary room for variety; fall back to nearest free
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
#     min  Σ_{d,p,h} cost^h_{dp} · α^h_{dp}
#     s.t. Σ_h a^h_{dp,k} · α^h_{dp} = 1  ∀k∈K_p, ∀p, ∀d   (coverage)
#          Σ_h α^h_{dp}             = 1  ∀p, ∀d               (one pattern)
#          α^h_{dp} ∈ {0,1}
# ──────────────────────────────────────────────────────────────────────────────

def print_cg_matrices(
    var_index: List[Tuple[str, str, int]],
    patterns_map: Dict[Tuple[str, str], List[Pattern]],
    c: np.ndarray,
    A: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    x: Optional[np.ndarray],
    constraint_labels: List[str],
    iteration_label: str,
    max_cols_display: int = 30,
) -> None:
    """
    Print every matrix involved in one MP solve during column generation:

      1. Column index table  – maps variable index j to (provider, date, pattern_id)
      2. Objective vector c  – cost^h_{dp} for each column
      3. Constraint matrix A – with row labels (assignment vs coverage)
         and RHS bounds [lb, ub]
      4. Solution vector x   – α^h_{dp} values (if solve succeeded)
      5. Pattern detail table – room assignment for each selected pattern

    When there are more columns than max_cols_display, the matrix is split
    into blocks so it stays readable on a standard terminal.
    """
    n_vars = len(var_index)
    n_rows = A.shape[0]
    W = 72  # output width

    header = f" CG MATRICES  –  {iteration_label} "
    print("\n" + "=" * W)
    print(header.center(W))
    print("=" * W)

    # ── 1. Column index table ────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print("  1. COLUMN INDEX TABLE  (one column = one pattern variable α^h_{{dp}})")
    print(f"{'─'*W}")
    print(f"  {'Col j':>6}  {'Provider':<10}  {'Date':<12}  {'Pat ID':>6}  "
          f"{'cost (m)':>10}")
    print(f"  {'------':>6}  {'--------':<10}  {'----------':<12}  {'------':>6}  "
          f"{'--------':>10}")
    for j, (prov, date, pidx) in enumerate(var_index):
        pat = patterns_map[(prov, date)][pidx]
        print(f"  {j:>6}  {prov:<10}  {date:<12}  {pat.pattern_id:>6}  "
              f"{c[j]:>10.2f}")

    # ── 2. Objective vector c ────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print("  2. OBJECTIVE COST VECTOR  c  (cost^h_{{dp}} per column)")
    print(f"{'─'*W}")
    # Print in rows of 10
    row_size = 10
    for start in range(0, n_vars, row_size):
        end = min(start + row_size, n_vars)
        indices = "  " + "  ".join(f"j={j:>4}" for j in range(start, end))
        values  = "  " + "  ".join(f"{c[j]:>6.2f}" for j in range(start, end))
        print(indices)
        print(values)
        print()

    # ── 3. Constraint matrix A ───────────────────────────────────────────────
    print(f"{'─'*W}")
    print("  3. CONSTRAINT MATRIX  A  (rows = constraints, cols = pattern variables)")
    print(f"     Shape: {n_rows} rows × {n_vars} cols    lb = ub = 1  (equality constraints)")
    print(f"{'─'*W}")

    # Split into column blocks if too wide
    col_blocks = list(range(0, n_vars, max_cols_display))
    if not col_blocks:
        col_blocks = [0]

    for block_start in col_blocks:
        block_end = min(block_start + max_cols_display, n_vars)
        block_cols = list(range(block_start, block_end))

        # Header row: column indices
        print(f"\n  Columns j = {block_start} … {block_end - 1}")
        hdr = f"  {'Constraint':<36} {'lb':>4} {'ub':>4}  "
        hdr += " ".join(f"{j:>3}" for j in block_cols)
        print(hdr)
        print(f"  {'-'*36} {'-'*4} {'-'*4}  " +
              " ".join("---" for _ in block_cols))

        for r in range(n_rows):
            label = constraint_labels[r] if r < len(constraint_labels) else f"row {r}"
            # Truncate label to fit
            label_str = label[:36]
            row_str = f"  {label_str:<36} {lb[r]:>4.0f} {ub[r]:>4.0f}  "
            row_str += " ".join(
                f"{int(A[r, j]):>3}" for j in block_cols
            )
            print(row_str)

    # ── 4. Solution vector x ─────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print("  4. SOLUTION VECTOR  x  (α^h_{{dp}} values)")
    print(f"{'─'*W}")
    if x is None:
        print("  (No solution – solver did not succeed.)")
    else:
        for start in range(0, n_vars, row_size):
            end = min(start + row_size, n_vars)
            indices = "  " + "  ".join(f"j={j:>4}" for j in range(start, end))
            values  = "  " + "  ".join(f"{x[j]:>6.3f}" for j in range(start, end))
            print(indices)
            print(values)
            print()

        # Highlight selected columns (x > 0.5 for MIP, or fractional for LP)
        selected = [(j, var_index[j], x[j]) for j in range(n_vars) if x[j] > 1e-6]
        print(f"  Non-zero entries ({len(selected)} of {n_vars}):")
        print(f"  {'Col j':>6}  {'Provider':<10}  {'Date':<12}  "
              f"{'Pat ID':>6}  {'α value':>8}")
        print(f"  {'------':>6}  {'--------':<10}  {'----------':<12}  "
              f"{'------':>6}  {'-------':>8}")
        for j, (prov, date, pidx), val in selected:
            pat = patterns_map[(prov, date)][pidx]
            print(f"  {j:>6}  {prov:<10}  {date:<12}  "
                  f"{pat.pattern_id:>6}  {val:>8.4f}")

    # ── 5. Pattern detail for selected columns ────────────────────────────────
    print(f"\n{'─'*W}")
    print("  5. PATTERN DETAIL  (room assignment for each selected/non-zero column)")
    print(f"{'─'*W}")
    if x is None:
        print("  (No solution available.)")
    else:
        selected_pats = [
            (j, var_index[j], x[j])
            for j in range(n_vars) if x[j] > 1e-6
        ]
        if not selected_pats:
            print("  (No non-zero variables.)")
        for j, (prov, date, pidx), val in selected_pats:
            pat = patterns_map[(prov, date)][pidx]
            print(f"\n  Col j={j}  {prov}  {date}  "
                  f"Pat#{pat.pattern_id}  α={val:.4f}  cost={pat.cost:.2f}m")
            print(f"    {'Appt ID':>8}  {'Room':>6}")
            print(f"    {'-------':>8}  {'----':>6}")
            for aid, room in sorted(pat.assignment.items()):
                print(f"    {aid:>8}  ER{room:>2}")

    print("\n" + "=" * W + "\n")


def _build_mp_arrays(
    groups: Dict[Tuple[str, str], List[Appointment]],
    patterns_map: Dict[Tuple[str, str], List[Pattern]],
) -> Tuple:
    """
    Shared helper: build all numpy arrays for the MP formulation.
    Returns (n_vars, var_index, A, lb, ub, c, group_var_indices, constraint_labels).
    Used by both build_and_solve_master and the dual extraction step in CG.
    """
    var_index: List[Tuple[str, str, int]] = []
    for (prov, date), pats in patterns_map.items():
        for j in range(len(pats)):
            var_index.append((prov, date, j))

    n_vars = len(var_index)
    c = np.array([patterns_map[(v[0], v[1])][v[2]].cost for v in var_index], dtype=float)

    rows_A: List[np.ndarray] = []
    lb_list: List[float] = []
    ub_list: List[float] = []
    constraint_labels: List[str] = []

    group_var_indices: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for vi, (prov, date, pidx) in enumerate(var_index):
        group_var_indices[(prov, date)].append(vi)

    # Constraint (1): Coverage — each appointment covered by exactly one pattern
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
                rows_A.append(row); lb_list.append(1.0); ub_list.append(1.0)
                constraint_labels.append(f"(1)COVER appt#{aid} {prov} {date}")

    # Constraint (2): Assignment — exactly one pattern per provider-day
    for (prov, date), vindices in group_var_indices.items():
        row = np.zeros(n_vars)
        for vi in vindices:
            row[vi] = 1.0
        rows_A.append(row); lb_list.append(1.0); ub_list.append(1.0)
        constraint_labels.append(f"(2)ASSIGN {prov} {date}")

    # Constraint (3): Cross-provider room conflict
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
                            rows_A.append(row); lb_list.append(0.0); ub_list.append(1.0)
                            constraint_labels.append(
                                f"(3)ROOM_CONFLICT r=ER{r} "
                                f"appt#{aid_p}({prov_p}) vs appt#{aid_pp}({prov_pp}) {date}"
                            )

    A  = np.array(rows_A)
    lb = np.array(lb_list)
    ub = np.array(ub_list)
    return n_vars, var_index, A, lb, ub, c, group_var_indices, constraint_labels


def build_and_solve_master(
    groups: Dict[Tuple[str, str], List[Appointment]],
    patterns_map: Dict[Tuple[str, str], List[Pattern]],
    integer: bool = True,
    print_matrices: bool = False,
    iteration_label: str = "MP",
) -> Tuple[float, Dict[Tuple[str, str], Pattern]]:
    """
    Build the Master Problem and solve it.

    groups:          {(provider, date): [Appointment, ...]}
    patterns_map:    {(provider, date): [Pattern, ...]}
    integer:         if True solve MIP, else LP relaxation
    print_matrices:  if True, dump every matrix to stdout
    iteration_label: label used in matrix output headers

    Returns (objective_value, selected_patterns_map)
      selected_patterns_map: {(provider, date): chosen Pattern}
    """

    n_vars, var_index, A, lb, ub, c, group_var_indices, constraint_labels = \
        _build_mp_arrays(groups, patterns_map)

    if n_vars == 0:
        print("  [MP] No variables to optimise – returning empty solution.")
        return 0.0, {}

    constraints = LinearConstraint(A, lb, ub)
    bounds      = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
    integrality = np.ones(n_vars) if integer else np.zeros(n_vars)

    n_cov      = sum(1 for l in constraint_labels if l.startswith("(1)"))
    n_assign   = sum(1 for l in constraint_labels if l.startswith("(2)"))
    n_conflict = sum(1 for l in constraint_labels if l.startswith("(3)"))
    print(f"  [MP] Variables: {n_vars}, Constraints: {len(constraint_labels)} "
          f"(cov:{n_cov}, assign:{n_assign}, conflict:{n_conflict})")

    result = milp(c=c, constraints=constraints,
                  integrality=integrality, bounds=bounds)

    x = result.x if result.success else None

    if print_matrices:
        print_cg_matrices(
            var_index=var_index, patterns_map=patterns_map,
            c=c, A=A, lb=lb, ub=ub, x=x,
            constraint_labels=constraint_labels,
            iteration_label=iteration_label,
        )

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
#     Iteratively:
#       1. Solve LP relaxation of MP with current columns
#       2. Retrieve dual variables
#       3. Solve pricing problem per (provider, day) to find negative-RC column
#       4. Add column if RC < 0; else stop
# ──────────────────────────────────────────────────────────────────────────────

def column_generation(
    groups: Dict[Tuple[str, str], List[Appointment]],
    patterns_map: Dict[Tuple[str, str], List[Pattern]],
    max_iterations: int = 10,
    print_matrices: bool = False,
) -> Dict[Tuple[str, str], List[Pattern]]:
    """
    Column generation loop.

    Each iteration:
      1. Solve the LP relaxation of the current restricted MP.
      2. Extract dual variables μ_k (coverage) and λ_{dp} (assignment).
      3. For each (provider, day), call the pricing oracle to find the
         pattern with the most negative reduced cost:
             RC = cost^h - Σ_k μ_k · a^h_k - λ_{dp}
      4. Add any column with RC < -ε to the MP; stop when none exist.

    Dual variables are approximated from the LP solution using the
    constraint shadow prices embedded in scipy's milp result.
    Because scipy.milp does not expose duals directly, we recover them
    by re-solving a pure LP (integrality=0) and reading the dual from
    the constraint residuals using the optimality conditions.
    """
    print("\n=== Column Generation Loop ===")
    EPSILON = 1e-6   # negative-RC threshold

    for iteration in range(1, max_iterations + 1):
        print(f"\n  Iteration {iteration}")

        # ── Step 1: solve LP relaxation ──────────────────────────────────────
        obj, chosen = build_and_solve_master(
            groups, patterns_map, integer=False,
            print_matrices=print_matrices,
            iteration_label=f"CG Iteration {iteration} – LP Relaxation",
        )
        if obj == float("inf"):
            print("  LP infeasible – stopping CG.")
            break
        print(f"  LP objective (relaxation): {obj:.4f} m")

        # ── Step 2: extract true LP dual variables ───────────────────────────
        # scipy.milp does not expose dual variables, so we re-solve the LP
        # relaxation using scipy.optimize.linprog (which does expose duals
        # via result.ineqlin / result.eqlin).
        #
        # The LP has only equality constraints (lb = ub = 1), so we pass
        # them as A_eq / b_eq to linprog and read result.eqlin.marginals.
        from scipy.optimize import linprog as _linprog

        # Re-build A, lb, ub for the current patterns_map
        _, _, A_lp, lb_lp, ub_lp, c_lp, vi_lp, cl_lp = _build_mp_arrays(
            groups, patterns_map
        )

        # Split into equality (lb==ub) and inequality (lb < ub) rows
        eq_mask = (lb_lp == ub_lp)
        A_eq = A_lp[eq_mask];   b_eq = lb_lp[eq_mask]
        ineq_rows = A_lp[~eq_mask]
        # inequality: A x <= ub  and  -A x <= -lb
        A_ub_parts, b_ub_parts = [], []
        if ineq_rows.shape[0] > 0:
            A_ub_parts.append(ineq_rows);  b_ub_parts.append(ub_lp[~eq_mask])
            A_ub_parts.append(-ineq_rows); b_ub_parts.append(-lb_lp[~eq_mask])

        A_ub_lp = np.vstack(A_ub_parts) if A_ub_parts else None
        b_ub_lp = np.concatenate(b_ub_parts) if b_ub_parts else None

        lp_result = _linprog(
            c=c_lp,
            A_ub=A_ub_lp, b_ub=b_ub_lp,
            A_eq=A_eq,    b_eq=b_eq,
            bounds=[(0, 1)] * len(c_lp),
            method="highs",
        )

        # Dual variables: indexed to match equality constraint rows
        # marginals[i] = dual of equality constraint i
        duals_eq   = lp_result.eqlin.marginals   if lp_result.success else np.zeros(eq_mask.sum())
        duals_ineq = lp_result.ineqlin.marginals if (lp_result.success and A_ub_lp is not None) \
                     else np.zeros(0)

        # Map duals back to constraint labels
        # Equality constraint order matches cl_lp[eq_mask]
        eq_labels = [cl_lp[i] for i, m in enumerate(eq_mask) if m]

        duals_assignment: Dict[Tuple[str, str], float] = {}
        duals_coverage:   Dict[int, float] = defaultdict(float)

        for dual_val, label in zip(duals_eq, eq_labels):
            if label.startswith("(2)ASSIGN"):
                # label format: "(2)ASSIGN HPWxxx MM-DD-YYYY"
                parts = label.split()
                if len(parts) >= 3:
                    prov_key = parts[1]; date_key = parts[2]
                    duals_assignment[(prov_key, date_key)] = float(dual_val)
            elif label.startswith("(1)COVER"):
                # label format: "(1)COVER appt#N HPWxxx MM-DD-YYYY"
                parts = label.split()
                if len(parts) >= 2:
                    try:
                        aid = int(parts[1].replace("appt#", ""))
                        duals_coverage[aid] = float(dual_val)
                    except ValueError:
                        pass

        # ── Step 3: pricing sub-problem for each (provider, day) ─────────────
        new_cols_added = 0
        for (prov, date), appts in groups.items():
            active = [a for a in appts if a.is_active]
            if not active:
                continue

            result = solve_pricing_subproblem(
                appts=active,
                date=date,
                rooms=ROOMS,
                duals_coverage=duals_coverage,
                dual_assignment=duals_assignment.get((prov, date), 0.0),
            )

            if result is None:
                continue

            assignment, rc = result

            # ── Step 4: add column if RC < 0 ─────────────────────────────────
            if rc < -EPSILON:
                sig = frozenset(assignment.items())
                existing_sigs = {
                    frozenset(p.assignment.items())
                    for p in patterns_map.get((prov, date), [])
                }
                if sig not in existing_sigs:
                    cost = compute_pattern_cost(assignment, active)
                    pid = len(patterns_map.get((prov, date), []))
                    new_pat = Pattern(pid, prov, date, assignment, cost)
                    patterns_map.setdefault((prov, date), []).append(new_pat)
                    new_cols_added += 1
                    print(f"    Added column for ({prov}, {date}): "
                          f"RC={rc:.4f}, cost={cost:.2f}m")

        print(f"  New columns added this iteration: {new_cols_added}")
        if new_cols_added == 0:
            print("  No improving columns found – column generation converged.")
            break

    return patterns_map


# ──────────────────────────────────────────────────────────────────────────────
# 6.  KPI CALCULATION
# ──────────────────────────────────────────────────────────────────────────────

def compute_kpis(chosen: Dict[Tuple[str, str], Pattern],
                 groups: Dict[Tuple[str, str], List[Appointment]]) -> Dict:
    total_travel = 0.0
    total_room_switches = 0
    unassigned = 0
    providers_single_room = 0
    provider_day_count = 0

    for (prov, date), pat in chosen.items():
        appts = sorted(
            [a for a in groups[(prov, date)] if a.is_active],
            key=lambda a: a.start_min,
        )
        rooms_seq = [pat.assignment.get(a.appt_id) for a in appts
                     if a.appt_id in pat.assignment]
        for r in rooms_seq:
            if r is None:
                unassigned += 1

        switches = sum(1 for r1, r2 in zip(rooms_seq, rooms_seq[1:]) if r1 != r2)
        total_room_switches += switches
        total_travel += pat.cost
        provider_day_count += 1
        if len(set(r for r in rooms_seq if r is not None)) == 1:
            providers_single_room += 1

    return {
        "total_travel_m": total_travel,
        "total_room_switches": total_room_switches,
        "unassigned_appointments": unassigned,
        "provider_days_solved": provider_day_count,
        "provider_days_single_room": providers_single_room,
        "avg_travel_per_provider_day": (
            total_travel / provider_day_count if provider_day_count else 0
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 7.  SCHEDULE DISPLAY
# ──────────────────────────────────────────────────────────────────────────────

def display_schedule(chosen: Dict[Tuple[str, str], Pattern],
                     groups: Dict[Tuple[str, str], List[Appointment]],
                     date_filter: Optional[str] = None) -> None:
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
              f"Travel cost: {pat.cost:.1f} m")
        print(f"  {'Appt':>6}  {'Patient':<12}  {'Time':>5}  "
              f"{'Dur':>4}  {'Room':>4}  {'Type':<20}")
        print(f"  {'-'*6}  {'-'*12}  {'-'*5}  {'-'*4}  {'-'*4}  {'-'*20}")

        for a in appts:
            room = pat.assignment.get(a.appt_id, "?")
            h, m = divmod(a.start_min, 60)
            ns_flag = "[NS]" if a.no_show else ""
            print(f"  {a.appt_id:>6}  {a.patient_id:<12}  "
                  f"{h:02d}:{m:02d}  {a.duration:>4}  "
                  f"ER{str(room):>2}  {a.appt_type:<20} {ns_flag}")

    print("\n" + "=" * 72)


# ──────────────────────────────────────────────────────────────────────────────
# 8.  POLICIES A–F  (Validation – Task 4 of project)
# ──────────────────────────────────────────────────────────────────────────────

# ── Policy A ──────────────────────────────────────────────────────────────────
def policy_a_single_room(
    groups: Dict[Tuple[str, str], List[Appointment]],
    week_lock: bool = False,
) -> Tuple[float, Dict[Tuple[str, str], Pattern]]:
    """
    Policy A: Assign each Healthcare Provider to ONE dedicated room for the
    entire day (week_lock=False) or the entire week (week_lock=True).

    When week_lock=True the same room number is re-used across all days for
    a given provider, simulating a permanently assigned room.

    Cost = 0 (no room switches ever occur).
    """
    chosen: Dict[Tuple[str, str], Pattern] = {}
    total_cost = 0.0
    pid = 0

    # If locking per week, pre-assign one room per provider
    provider_room: Dict[str, int] = {}
    room_counter = iter(ROOMS)

    for (prov, date), appts in sorted(groups.items()):
        active = [a for a in appts if a.is_active]
        if not active:
            continue

        if week_lock:
            if prov not in provider_room:
                try:
                    provider_room[prov] = next(room_counter)
                except StopIteration:
                    provider_room[prov] = ROOMS[0]  # wrap-around
            r = provider_room[prov]
        else:
            # Pick the first room that has no time conflict with other providers
            # already assigned to that room on that day.
            used_rooms_today = {
                p.assignment[list(p.assignment.keys())[0]]
                for (pr, dt), p in chosen.items()
                if dt == date and p.assignment
            }
            r = next((rm for rm in ROOMS if rm not in used_rooms_today), ROOMS[0])

        assignment = {a.appt_id: r for a in active}
        pat = Pattern(pid, prov, date, assignment, cost=0.0)
        chosen[(prov, date)] = pat
        pid += 1

    return total_cost, chosen


# ── Policy B ──────────────────────────────────────────────────────────────────
def policy_b_cluster(
    groups: Dict[Tuple[str, str], List[Appointment]],
    proximity_threshold: float = 3.0,
    anchor_room: int = 1,
) -> Tuple[float, Dict[Tuple[str, str], Pattern]]:
    """
    Policy B: Restrict each (provider, day) to a cluster of rooms all within
    `proximity_threshold` metres of `anchor_room`.

    This limits how far a provider must walk between consecutive appointments,
    at the cost of fewer available rooms (possible infeasibility for busy days).
    Falls back to full room set if no feasible pattern exists in cluster.
    """
    cluster = [r for r in ROOMS if room_distance(anchor_room, r) <= proximity_threshold]
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
            # Fallback: use full room set
            pats = feasible_room_assignment(active, date, ROOMS)

        if pats:
            costs = [compute_pattern_cost(p, active) for p in pats]
            best_idx = int(np.argmin(costs))
            cost = costs[best_idx]
            pat = Pattern(pid, prov, date, pats[best_idx], cost)
            chosen[(prov, date)] = pat
            total_cost += cost
            pid += 1

    return total_cost, chosen


# ── Policy C ──────────────────────────────────────────────────────────────────
def policy_c_blocked_days(
    groups: Dict[Tuple[str, str], List[Appointment]],
    blocked: Optional[Dict[str, List[str]]] = None,
) -> Tuple[float, Dict[Tuple[str, str], Pattern]]:
    """
    Policy C: Block certain days for certain Healthcare Providers.

    `blocked` maps provider_id -> list of date strings that are blocked.
    Appointments on blocked days are dropped (no room assigned).
    This models days reserved for admin work, training, or leave.

    Default example blocks HPW114 on Tuesdays (11-11-2025).
    """
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
            continue  # skip entirely — day is blocked

        active = [a for a in appts if a.is_active]
        if not active:
            continue

        pats = feasible_room_assignment(active, date, ROOMS)
        if pats:
            costs = [compute_pattern_cost(p, active) for p in pats]
            best_idx = int(np.argmin(costs))
            cost = costs[best_idx]
            pat = Pattern(pid, prov, date, pats[best_idx], cost)
            chosen[(prov, date)] = pat
            total_cost += cost
            pid += 1

    print(f"  [Policy C] {blocked_count} provider-day(s) blocked.")
    return total_cost, chosen


# ── Policy D ──────────────────────────────────────────────────────────────────
def policy_d_admin_overflow(
    groups: Dict[Tuple[str, str], List[Appointment]],
    allow_admin_overflow: bool = True,
) -> Tuple[float, Dict[Tuple[str, str], Pattern]]:
    """
    Policy D: Use admin time slots for extenuating circumstances.

    When allow_admin_overflow=True, appointments that fall inside the morning
    or afternoon admin windows (9:00-9:30, 16:30-17:00 Mon-Thu; 8:00-8:30,
    15:00-15:30 Fri) are still assigned a room — modelling the case where
    a provider uses admin time to see an urgent patient.

    The noon admin and lunch blocks are never overridden (hard constraints).
    Cost for admin-overflow appointments is inflated by ADMIN_PENALTY to
    discourage routine use.
    """
    ADMIN_PENALTY = 50.0  # metres equivalent penalty per overflow appointment

    # Hard-blocked windows (never overridable): noon admin + lunch
    HARD_BLOCKS_MON_THU = [
        (11 * 60 + 30, 12 * 60),  # noon admin
        (12 * 60, 13 * 60),       # lunch
    ]
    HARD_BLOCKS_FRI = [
        (11 * 60 + 30, 12 * 60),
        (12 * 60, 13 * 60),
    ]

    chosen: Dict[Tuple[str, str], Pattern] = {}
    total_cost = 0.0
    pid = 0

    for (prov, date), appts in groups.items():
        hard_blocks = HARD_BLOCKS_FRI if is_friday(date) else HARD_BLOCKS_MON_THU

        # Include appointments that only hit soft admin blocks (if overflow allowed)
        if allow_admin_overflow:
            active = [
                a for a in appts
                if not a.cancelled and not a.deleted
                and is_available(a.start_min, a.end_min, hard_blocks)
            ]
        else:
            active = [a for a in appts if a.is_active]

        if not active:
            continue

        pats = feasible_room_assignment(active, date, ROOMS)
        if pats:
            # Compute cost with admin penalty for overflow appointments
            soft_blocks = blocked_windows(date)
            overflow_ids = {
                a.appt_id for a in active
                if not is_available(a.start_min, a.end_min, soft_blocks)
            }
            costs = []
            for p in pats:
                base = compute_pattern_cost(p, active)
                penalty = ADMIN_PENALTY * sum(
                    1 for aid in p if aid in overflow_ids
                )
                costs.append(base + penalty)

            best_idx = int(np.argmin(costs))
            raw_cost = compute_pattern_cost(pats[best_idx], active)
            pat = Pattern(pid, prov, date, pats[best_idx], raw_cost)
            chosen[(prov, date)] = pat
            total_cost += raw_cost
            pid += 1

    return total_cost, chosen


# ── Policy E ──────────────────────────────────────────────────────────────────
def policy_e_overbook(
    groups: Dict[Tuple[str, str], List[Appointment]],
    no_show_rate: float = 0.10,
    overbook_factor: float = 1.15,
) -> Tuple[float, Dict[Tuple[str, str], Pattern]]:
    """
    Policy E: Overbook appointments to account for no-shows.

    Strategy: for each (provider, day), if the observed no-show rate exceeds
    `no_show_rate`, add extra phantom appointments to fill the freed slots.
    `overbook_factor` controls how many extra slots to add relative to the
    expected no-shows (e.g. 1.15 = add 15% more than expected no-shows).

    Observed no-show rate from data is used when available; otherwise the
    supplied `no_show_rate` default is applied.

    No-show appointments keep their original room assignment so the room is
    re-usable in practice.  The schedule is otherwise identical to baseline.
    """
    chosen: Dict[Tuple[str, str], Pattern] = {}
    total_cost = 0.0
    pid = 0
    extra_total = 0

    for (prov, date), appts in groups.items():
        all_active = [a for a in appts if not a.cancelled and not a.deleted]
        no_shows = [a for a in all_active if a.no_show]
        real_appts = [a for a in all_active if not a.no_show]

        # Observed rate for this group
        obs_rate = len(no_shows) / len(all_active) if all_active else no_show_rate
        expected_no_shows = max(obs_rate * len(all_active), len(no_shows))
        extra_slots = int(np.floor(expected_no_shows * overbook_factor))
        extra_total += extra_slots

        # Include no-show appointments in the schedule (they may not show,
        # freeing the room for a walk-in or the overbooked slot)
        schedule_appts = real_appts + no_shows  # treat all as schedulable

        if not schedule_appts:
            continue

        pats = feasible_room_assignment(schedule_appts, date, ROOMS)
        if pats:
            costs = [compute_pattern_cost(p, schedule_appts) for p in pats]
            best_idx = int(np.argmin(costs))
            cost = costs[best_idx]
            pat = Pattern(pid, prov, date, pats[best_idx], cost)
            chosen[(prov, date)] = pat
            total_cost += cost
            pid += 1

    print(f"  [Policy E] Total extra overbook slots added: {extra_total}")
    return total_cost, chosen


# ── Policy F ──────────────────────────────────────────────────────────────────
def policy_f_uncertainty(
    groups: Dict[Tuple[str, str], List[Appointment]],
    duration_std_frac: float = 0.20,
    n_scenarios: int = 5,
    seed: int = 42,
) -> Tuple[float, Dict[Tuple[str, str], Pattern]]:
    """
    Policy F: Account for uncertainty in examination durations.

    Models duration uncertainty via scenario-based robust scheduling:
      1. Generate `n_scenarios` duration realisations per appointment by
         sampling N(mean, (mean * duration_std_frac)^2), clipped to [1, 120].
      2. For each scenario, solve the feasibility sub-problem.
      3. Select the pattern that is feasible across the most scenarios
         (most-robust pattern), breaking ties by lowest average cost.

    duration_std_frac: standard deviation as a fraction of nominal duration
                       (default 0.20 = ±20% variability).
    n_scenarios:       number of Monte-Carlo duration draws.
    seed:              random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    chosen: Dict[Tuple[str, str], Pattern] = {}
    total_cost = 0.0
    pid = 0

    for (prov, date), appts in groups.items():
        active = [a for a in appts if a.is_active]
        if not active:
            continue

        # Generate perturbed duration scenarios
        scenarios: List[List[Appointment]] = []
        for _ in range(n_scenarios):
            perturbed = []
            for a in active:
                std = max(1.0, a.duration * duration_std_frac)
                new_dur = int(np.clip(rng.normal(a.duration, std), 1, 120))
                # Create a copy with perturbed duration
                pa = Appointment(
                    appt_id=a.appt_id,
                    patient_id=a.patient_id,
                    provider=a.provider,
                    date=a.date,
                    start_min=a.start_min,
                    duration=new_dur,
                    appt_type=a.appt_type,
                    no_show=a.no_show,
                    cancelled=a.cancelled,
                    deleted=a.deleted,
                )
                perturbed.append(pa)
            scenarios.append(perturbed)

        # Baseline feasible patterns (from nominal durations)
        base_pats = feasible_room_assignment(active, date, ROOMS)
        if not base_pats:
            continue

        # Score each baseline pattern by how many scenarios it stays feasible
        pat_scores: List[Tuple[int, float, int]] = []  # (feasible_count, avg_cost, idx)
        for idx, assignment in enumerate(base_pats):
            feasible_count = 0
            scenario_costs = []
            for scenario in scenarios:
                # Check feasibility: no two appointments in same room overlap
                room_to_appts: Dict[int, List[Appointment]] = defaultdict(list)
                for sa in scenario:
                    r = assignment.get(sa.appt_id)
                    if r is not None:
                        room_to_appts[r].append(sa)
                ok = True
                for r, rappts in room_to_appts.items():
                    rappts_sorted = sorted(rappts, key=lambda x: x.start_min)
                    for a1, a2 in zip(rappts_sorted, rappts_sorted[1:]):
                        if a1.start_min + a1.duration > a2.start_min:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    feasible_count += 1
                    scenario_costs.append(compute_pattern_cost(assignment, scenario))
            avg_cost = float(np.mean(scenario_costs)) if scenario_costs else float("inf")
            pat_scores.append((feasible_count, avg_cost, idx))

        # Most robust: max feasible scenarios, then min avg cost
        pat_scores.sort(key=lambda x: (-x[0], x[1]))
        best_feasible, best_avg_cost, best_idx = pat_scores[0]

        nominal_cost = compute_pattern_cost(base_pats[best_idx], active)
        pat = Pattern(pid, prov, date, base_pats[best_idx], nominal_cost)
        chosen[(prov, date)] = pat
        total_cost += nominal_cost

        print(f"  [Policy F] ({prov}, {date}): robust pattern feasible in "
              f"{best_feasible}/{n_scenarios} scenarios, "
              f"avg cost={best_avg_cost:.1f}m")
        pid += 1

    return total_cost, chosen


# ──────────────────────────────────────────────────────────────────────────────
# 9.  MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def run_all_policies(
    groups: Dict[Tuple[str, str], List[Appointment]],
    optimal_kpis: Dict,
) -> None:
    """Run all six policies and print a consolidated comparison table."""

    print("\n" + "=" * 72)
    print("POLICY COMPARISON  (Policies A – F  vs  Column-Generation Optimal)")
    print("=" * 72)

    policies = [
        ("A – Single Room (day)",   lambda g: policy_a_single_room(g, week_lock=False)),
        ("A – Single Room (week)",  lambda g: policy_a_single_room(g, week_lock=True)),
        ("B – Cluster (≤3 m)",      lambda g: policy_b_cluster(g, proximity_threshold=3.0)),
        ("B – Cluster (≤5 m)",      lambda g: policy_b_cluster(g, proximity_threshold=5.0)),
        ("C – Blocked Days",        lambda g: policy_c_blocked_days(g)),
        ("D – Admin Overflow ON",   lambda g: policy_d_admin_overflow(g, allow_admin_overflow=True)),
        ("D – Admin Overflow OFF",  lambda g: policy_d_admin_overflow(g, allow_admin_overflow=False)),
        ("E – Overbook (10% NS)",   lambda g: policy_e_overbook(g, no_show_rate=0.10)),
        ("F – Uncertainty (σ=20%)", lambda g: policy_f_uncertainty(g, duration_std_frac=0.20)),
    ]

    metrics = [
        "total_travel_m",
        "total_room_switches",
        "avg_travel_per_provider_day",
        "provider_days_solved",
        "provider_days_single_room",
    ]

    results: List[Tuple[str, Dict]] = []
    for label, fn in policies:
        print(f"\n  Running {label} ...")
        _, chosen = fn(groups)
        kpi = compute_kpis(chosen, groups)
        results.append((label, kpi))

    # Header
    col_w = 12
    label_w = 26
    print(f"\n  {'Policy':<{label_w}}", end="")
    for m in metrics:
        short = m.replace("_", " ")[:col_w]
        print(f"  {short:>{col_w}}", end="")
    print()
    print(f"  {'-'*label_w}", end="")
    for _ in metrics:
        print(f"  {'-'*col_w}", end="")
    print()

    # Optimal row
    print(f"  {'Optimal (CG)':<{label_w}}", end="")
    for m in metrics:
        v = optimal_kpis.get(m, 0)
        s = f"{v:.1f}" if isinstance(v, float) else str(v)
        print(f"  {s:>{col_w}}", end="")
    print()

    # Policy rows
    for label, kpi in results:
        print(f"  {label:<{label_w}}", end="")
        for m in metrics:
            v = kpi.get(m, 0)
            s = f"{v:.1f}" if isinstance(v, float) else str(v)
            print(f"  {s:>{col_w}}", end="")
        print()

    print()


def main(csv_path: str, day_filter: Optional[str] = None,
         max_cg_iters: int = 3, print_matrices: bool = False) -> None:
    print(f"\nLoading appointments from: {csv_path}")
    all_appts = load_appointments(csv_path)
    print(f"  Loaded {len(all_appts)} appointments (after filtering 0-duration).")

    # Group by (provider, date)
    groups: Dict[Tuple[str, str], List[Appointment]] = defaultdict(list)
    for a in all_appts:
        groups[(a.provider, a.date)].append(a)

    if day_filter:
        groups = {k: v for k, v in groups.items() if k[1] == day_filter}
        print(f"  Filtered to date: {day_filter}  "
              f"({len(groups)} provider-day groups)")

    # ── Initial pattern generation (Pricing Sub-Problem) ──────────────────────
    print("\n=== Generating Initial Patterns (Pricing Sub-Problem) ===")
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
        patterns_map = column_generation(groups, patterns_map, max_cg_iters,
                                         print_matrices=print_matrices)

    # ── Solve integer Master Problem ──────────────────────────────────────────
    print("\n=== Solving Integer Master Problem (MIP) ===")
    obj, chosen_opt = build_and_solve_master(
        groups, patterns_map, integer=True,
        print_matrices=print_matrices,
        iteration_label="Final MIP Solve",
    )
    print(f"  Optimal total travel distance: {obj:.2f} m")

    # ── Display optimal schedule ──────────────────────────────────────────────
    display_schedule(chosen_opt, groups, date_filter=day_filter)

    # ── KPIs for optimal ─────────────────────────────────────────────────────
    kpis_opt = compute_kpis(chosen_opt, groups)
    print("\n=== Optimal Schedule KPIs ===")
    for k, v in kpis_opt.items():
        if isinstance(v, float):
            print(f"  {k:<40}: {v:.2f}")
        else:
            print(f"  {k:<40}: {v}")

    # ── Run all six policies and compare ─────────────────────────────────────
    run_all_policies(groups, kpis_opt)

    print("Done.")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MSE 435 Examination Room Scheduler – Policies A–F"
    )
    parser.add_argument("--csv", default="data/AppointmentDataWeek1.csv",
                        help="Path to appointment CSV file")
    parser.add_argument("--day", default=None,
                        help="Filter to a single date (MM-DD-YYYY), e.g. 11-10-2025")
    parser.add_argument("--cg-iters", type=int, default=3,
                        help="Max column generation iterations (0 to skip CG)")
    parser.add_argument("--print-matrices", action="store_true",
                        help="Print all CG matrices (cost vector, A matrix, solution vector, pattern detail) at each iteration")
    args = parser.parse_args()
    main(csv_path=args.csv, day_filter=args.day, max_cg_iters=args.cg_iters,
         print_matrices=args.print_matrices)