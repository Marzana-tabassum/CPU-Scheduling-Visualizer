"""
CPU Scheduling Visualizer — ULTIMATE Edition v3
Author: Marzana Tabassum (2232327642)
North South University

New in v3:
  ● Waiting Queue visible in simulation (animated entry/exit)
  ● Avg Response, Completion, Turnaround & Waiting Time chart in simulation
  ● Prediction scoring breakdown — visual rule-by-rule diagram/chart
  ● Live Process Injection (add while simulation plays)
  ● Drag & Drop priority reordering in queue
  ● Step-by-step mode with decision explanation
  ● CPU Utilization % timeline (idle vs busy)
  ● Starvation Detection with warnings
  ● Throughput & Efficiency Gauge
  ● Adaptive Quantum Tuning (RR)
  ● Workload Classifier (sklearn ML model)
  ● Rule-Based scoring kept + clearly visualized
  ● OS Chatbot (answers OS/scheduling questions)
  ● NEW ALGORITHM: SMART-SJF (Shortest Job with Fair Round-Robin) = SJF + RR hybrid
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import threading
import time
import random
from collections import deque

# sklearn for ML classifier
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

# ─────────────────────────────────────────────
#  THEME  — Soft, eye-friendly palette
# ─────────────────────────────────────────────
BG      = '#1e2028'
PANEL   = '#252830'
CARD    = '#2a2d38'
ACCENT  = '#353848'
BLUE    = '#7eb8d4'
PURPLE  = '#a98fd4'
GREEN   = '#7ec8a0'
PINK    = '#d47eb0'
YELLOW  = '#d4c07e'
ORANGE  = '#d4a07e'
RED     = '#d47e7e'
TEXT    = '#d4d8e0'
SUBTEXT = '#8a909e'
TEAL    = '#7ec8c8'
LIME    = '#a8d47e'

PROC_COLORS = [
    '#7eb8d4','#a98fd4','#7ec8a0','#d47eb0',
    '#d4c07e','#d4a07e','#7ec8c8','#c87eb0',
    '#a8d47e','#d49090','#90c8d4','#b490c8'
]

ALGO_COLORS = {
    'FCFS':        '#7eb8d4',
    'SJF':         '#7ec8a0',
    'Round Robin': '#d4a07e',
    'Priority':    '#d47eb0',
    'SMART-SJF':   '#d4c07e',
}

# ─────────────────────────────────────────────
#  ALGORITHMS
# ─────────────────────────────────────────────

def fcfs(processes):
    procs = sorted(processes, key=lambda p: p['arrival'])
    timeline, current = [], 0
    for p in procs:
        start = max(current, p['arrival'])
        end   = start + p['burst']
        timeline.append({'pid': p['pid'], 'start': start, 'end': end})
        current = end
    return timeline


def sjf(processes):
    procs     = sorted(processes, key=lambda p: p['arrival'])
    timeline, current = [], 0
    remaining = list(procs)
    while remaining:
        available = [p for p in remaining if p['arrival'] <= current]
        if not available:
            current   = remaining[0]['arrival']
            available = [remaining[0]]
        shortest = min(available, key=lambda p: p['burst'])
        start    = current
        end      = start + shortest['burst']
        timeline.append({'pid': shortest['pid'], 'start': start, 'end': end})
        current = end
        remaining.remove(shortest)
    return timeline


def round_robin(processes, quantum):
    procs           = sorted(processes, key=lambda p: p['arrival'])
    remaining_burst = {p['pid']: p['burst']   for p in procs}
    arrival         = {p['pid']: p['arrival'] for p in procs}
    queue           = deque()
    timeline, current = [], 0
    procs_list = list(procs)
    added      = set()
    for p in procs_list:
        if p['arrival'] <= current and p['pid'] not in added:
            queue.append(p['pid']); added.add(p['pid'])
    while queue or any(remaining_burst[p['pid']] > 0 for p in procs_list):
        if not queue:
            next_arr = min(arrival[p['pid']] for p in procs_list if remaining_burst[p['pid']] > 0)
            current  = next_arr
            for p in procs_list:
                if p['arrival'] <= current and p['pid'] not in added and remaining_burst[p['pid']] > 0:
                    queue.append(p['pid']); added.add(p['pid'])
            if not queue:
                break
        pid = queue.popleft()
        if remaining_burst[pid] <= 0:
            continue
        run = min(quantum, remaining_burst[pid])
        timeline.append({'pid': pid, 'start': current, 'end': current + run})
        current += run
        remaining_burst[pid] -= run
        for p in procs_list:
            if p['arrival'] <= current and p['pid'] not in added and remaining_burst[p['pid']] > 0:
                queue.append(p['pid']); added.add(p['pid'])
        if remaining_burst[pid] > 0:
            queue.append(pid)
    return timeline


def preemptive_priority(processes):
    if not processes:
        return []
    procs           = sorted(processes, key=lambda p: p['arrival'])
    remaining_burst = {p['pid']: p['burst']   for p in procs}
    priority        = {p['pid']: p.get('priority', 1) for p in procs}
    max_time  = sum(p['burst'] for p in procs) + max(p['arrival'] for p in procs) + 1
    timeline  = []
    current   = 0
    last_pid  = None
    last_start = 0
    while current < max_time:
        available = [p for p in procs
                     if p['arrival'] <= current and remaining_burst[p['pid']] > 0]
        if not available:
            current += 1
            continue
        chosen = min(available, key=lambda p: (priority[p['pid']], p['arrival']))
        pid    = chosen['pid']
        if pid != last_pid:
            if last_pid is not None and last_start < current:
                timeline.append({'pid': last_pid, 'start': last_start, 'end': current})
            last_pid   = pid
            last_start = current
        remaining_burst[pid] -= 1
        current += 1
        if remaining_burst[pid] == 0:
            timeline.append({'pid': pid, 'start': last_start, 'end': current})
            last_pid   = None
            last_start = current
        if all(remaining_burst[p['pid']] == 0 for p in procs):
            break
    return timeline


def smart_sjf(processes, quantum=None):
    """
    SMART-SJF — Shortest Job with Adaptive Fair Round-Robin
    ─────────────────────────────────────────────────────────
    Novel algorithm by Marzana Tabassum / NSU CPU Scheduler v3.

    Idea:
      • Like SJF, it always prefers the shortest remaining job.
      • BUT — if a process has been waiting more than PATIENCE units
        while longer jobs execute (starvation risk), it gets promoted
        into a mini Round-Robin window of size 'fair_q'.
      • This eliminates the core SJF flaw (starvation) while keeping
        near-optimal average waiting times for short jobs.

    Why it matters:
      • Pure SJF starves long processes indefinitely.
      • Pure RR treats all processes equally, missing short-job optimality.
      • SMART-SJF is a middle ground: short jobs get priority, but no
        process waits longer than PATIENCE time units without guaranteed CPU
        time, making it both efficient AND fair.
    """
    if not processes:
        return []
    procs = sorted(processes, key=lambda p: p['arrival'])
    n     = len(procs)
    avg_b = np.mean([p['burst'] for p in procs])
    PATIENCE = max(3, int(avg_b * 1.5))  # adaptive starvation threshold
    fair_q   = max(2, int(avg_b * 0.6))  # fair round-robin slice

    remaining   = {p['pid']: p['burst']   for p in procs}
    last_served = {p['pid']: p['arrival'] for p in procs}
    timeline    = []
    current     = 0

    while any(remaining[p['pid']] > 0 for p in procs):
        available = [p for p in procs
                     if p['arrival'] <= current and remaining[p['pid']] > 0]
        if not available:
            current += 1
            continue

        # Check for starving processes (waiting > PATIENCE)
        starving = [p for p in available
                    if (current - last_served[p['pid']]) >= PATIENCE]

        if starving:
            # Fair mini-RR for all starving processes
            rr_q = [p for p in starving]
            for sp in rr_q:
                run = min(fair_q, remaining[sp['pid']])
                timeline.append({'pid': sp['pid'], 'start': current, 'end': current + run})
                last_served[sp['pid']] = current + run
                remaining[sp['pid']] -= run
                current += run
        else:
            # Normal SJF step
            chosen = min(available, key=lambda p: remaining[p['pid']])
            run    = min(fair_q, remaining[chosen['pid']])
            timeline.append({'pid': chosen['pid'], 'start': current, 'end': current + run})
            last_served[chosen['pid']] = current + run
            remaining[chosen['pid']] -= run
            current += run

    return timeline


# ─────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────

def calculate_metrics(processes, timeline):
    arrival = {p['pid']: p['arrival'] for p in processes}
    burst   = {p['pid']: p['burst']   for p in processes}
    finish  = {}
    for seg in timeline:
        finish[seg['pid']] = max(finish.get(seg['pid'], 0), seg['end'])
    first_cpu = {}
    for seg in timeline:
        if seg['pid'] not in first_cpu:
            first_cpu[seg['pid']] = seg['start']
    metrics = []
    for p in processes:
        pid = p['pid']
        ct  = finish.get(pid, 0)
        tat = ct - arrival[pid]
        wt  = tat - burst[pid]
        rt  = first_cpu.get(pid, 0) - arrival[pid]
        metrics.append({
            'pid': pid, 'arrival': arrival[pid], 'burst': burst[pid],
            'completion': ct, 'turnaround': tat, 'waiting': wt, 'response': rt
        })
    avg_ct  = np.mean([m['completion']  for m in metrics])
    avg_tat = np.mean([m['turnaround']  for m in metrics])
    avg_wt  = np.mean([m['waiting']     for m in metrics])
    avg_rt  = np.mean([m['response']    for m in metrics])
    return metrics, avg_ct, avg_tat, avg_wt, avg_rt


def cpu_utilization(timeline):
    """Returns fraction of time CPU is busy."""
    if not timeline:
        return 0.0
    total_time = timeline[-1]['end']
    busy = sum(seg['end'] - seg['start'] for seg in timeline)
    return busy / total_time if total_time > 0 else 0.0


def detect_starvation(processes, timeline, threshold=20):
    """Returns list of (pid, wait_time) where wait > threshold."""
    _, _, _, avg_wt, _ = calculate_metrics(processes, timeline)
    arrival = {p['pid']: p['arrival'] for p in processes}
    first_cpu = {}
    for seg in timeline:
        if seg['pid'] not in first_cpu:
            first_cpu[seg['pid']] = seg['start']
    starved = []
    for p in processes:
        pid = p['pid']
        wait = first_cpu.get(pid, 0) - arrival[pid]
        if wait >= threshold:
            starved.append((pid, wait))
    return starved


def adaptive_quantum(processes):
    """Calculate optimal RR quantum based on burst distribution."""
    bursts = [p['burst'] for p in processes]
    avg    = np.mean(bursts)
    std    = np.std(bursts)
    # Heuristic: quantum = ceil(avg * 0.6), clamped to [2, 10]
    q = max(2, min(10, int(avg * 0.6)))
    return q


# ─────────────────────────────────────────────
#  RULE-BASED PREDICTION (ENHANCED)
# ─────────────────────────────────────────────

RULE_DEFINITIONS = {
    'FCFS': [
        {'name': 'Burst Variance LOW',  'condition': lambda v: v['var_burst'] < 5,    'weight': 20, 'desc': 'Low burst variance → FCFS is fair'},
        {'name': 'Arrival Density HIGH','condition': lambda v: v['density'] > 0.8,    'weight': 15, 'desc': 'Processes arrive close together'},
        {'name': 'Burst Range SMALL',   'condition': lambda v: v['range_burst'] < 3,  'weight': 15, 'desc': 'Similar burst lengths → no starvation risk'},
    ],
    'SJF': [
        {'name': 'Burst Variance HIGH', 'condition': lambda v: v['var_burst'] > 10,   'weight': 30, 'desc': 'High burst variance → SJF picks shortest'},
        {'name': 'Burst Range LARGE',   'condition': lambda v: v['range_burst'] > 5,  'weight': 20, 'desc': 'Wide burst spread → big savings from ordering'},
        {'name': 'Avg Burst > 5',       'condition': lambda v: v['avg_burst'] > 5,    'weight': 10, 'desc': 'Longer jobs make shortest-first more impactful'},
    ],
    'Round Robin': [
        {'name': 'Many Processes (>4)', 'condition': lambda v: v['n'] > 4,            'weight': 20, 'desc': 'More processes benefit from time-sharing'},
        {'name': 'Burst Variance MIX',  'condition': lambda v: v['var_burst'] > 5,    'weight': 15, 'desc': 'Mixed bursts need fair turn allocation'},
        {'name': 'Quantum ≤ Avg Burst', 'condition': lambda v: v['quantum'] <= v['avg_burst'], 'weight': 15, 'desc': 'Good slicing prevents monopolisation'},
    ],
    'Priority': [
        {'name': 'Priority Variance HIGH','condition': lambda v: v['pri_var'] > 2,    'weight': 30, 'desc': 'Diverse priorities → critical tasks must run first'},
        {'name': 'Many Priority Levels', 'condition': lambda v: v['unique_pri'] > 2,  'weight': 20, 'desc': 'Multiple priority tiers need preemptive ordering'},
    ],
    'SMART-SJF': [
        {'name': 'Starvation Risk',     'condition': lambda v: v['var_burst'] > 8 and v['n'] > 3, 'weight': 25, 'desc': 'High variance + many processes → starvation risk SJF solves'},
        {'name': 'Fairness Needed',     'condition': lambda v: v['range_burst'] > 6,  'weight': 20, 'desc': 'Wide burst range → fairness + efficiency needed'},
        {'name': 'Medium Process Count','condition': lambda v: 3 <= v['n'] <= 8,       'weight': 15, 'desc': 'Mid-sized workloads benefit from SMART-SJF hybrid'},
    ],
}

def predict_best_algorithm(processes, quantum):
    bursts    = [p['burst']           for p in processes]
    arrivals  = [p['arrival']         for p in processes]
    priorities= [p.get('priority', 1) for p in processes]
    n = len(processes)
    avg_burst   = np.mean(bursts)
    var_burst   = np.var(bursts)
    max_burst   = max(bursts)
    min_burst   = min(bursts)
    range_burst = max_burst - min_burst
    arrival_span= max(arrivals) - min(arrivals) if len(arrivals) > 1 else 1
    density     = n / (arrival_span + 1)
    pri_var     = np.var(priorities)
    unique_pri  = len(set(priorities))

    vals = {
        'var_burst': var_burst, 'density': density, 'range_burst': range_burst,
        'avg_burst': avg_burst, 'n': n, 'quantum': quantum,
        'pri_var': pri_var, 'unique_pri': unique_pri,
    }

    scores     = {}
    reasons    = {}
    rule_breakdown = {}  # per-algo, per-rule: {triggered, weight, name, desc}

    BASE = 50
    for algo, rules in RULE_DEFINITIONS.items():
        s = BASE
        triggered = []
        missed    = []
        for rule in rules:
            if rule['condition'](vals):
                s += rule['weight']
                triggered.append(rule)
            else:
                missed.append(rule)
        scores[algo]        = s
        rule_breakdown[algo] = {'triggered': triggered, 'missed': missed}

        det = [f"✅ +{r['weight']}  {r['name']}: {r['desc']}" for r in triggered]
        det+= [f"✗   {r['name']}: not met (0 pts)" for r in missed]
        reasons[algo] = {
            'summary': _algo_summary(algo, s, triggered),
            'details': det,
            'triggered': triggered,
            'missed': missed,
            'base': BASE,
        }

    best = max(scores, key=scores.get)
    return best, scores, reasons, rule_breakdown, vals


def _algo_summary(algo, score, triggered):
    n = len(triggered)
    if algo == 'FCFS':
        return f"Works best when all jobs are similar in length. {n} rule(s) matched → score {score}."
    if algo == 'SJF':
        return f"Minimises average wait by always picking the shortest job. {n} rule(s) matched → score {score}."
    if algo == 'Round Robin':
        return f"Gives every process a fair slice. Best for interactive, mixed workloads. {n} rule(s) matched → score {score}."
    if algo == 'Priority':
        return f"Critical tasks run first (preemptive). Best when urgency varies. {n} rule(s) matched → score {score}."
    if algo == 'SMART-SJF':
        return f"SJF + fairness hybrid. Eliminates starvation while keeping near-optimal WT. {n} rule(s) matched → score {score}."
    return f"Score {score}."


# ─────────────────────────────────────────────
#  ML WORKLOAD CLASSIFIER
# ─────────────────────────────────────────────

def _generate_training_data(n=1000):
    """Generate synthetic scheduling data with ground-truth labels."""
    X, y = [], []
    rng = np.random.default_rng(42)
    for _ in range(n):
        num_p   = rng.integers(2, 12)
        bursts  = rng.integers(1, 25, size=num_p).tolist()
        arrivals= rng.integers(0, 10, size=num_p).tolist()
        pris    = rng.integers(1, 5, size=num_p).tolist()
        q       = rng.integers(2, 8)

        avg_b  = np.mean(bursts)
        var_b  = np.var(bursts)
        rng_b  = max(bursts) - min(bursts)
        arr_s  = max(arrivals) - min(arrivals) + 1
        dens   = num_p / arr_s
        pvar   = np.var(pris)
        upri   = len(set(pris))

        # Assign ground truth using rule scores
        _, scores, _, _, _ = predict_best_algorithm(
            [{'pid': f'P{i}', 'burst': b, 'arrival': a, 'priority': p}
             for i, (b, a, p) in enumerate(zip(bursts, arrivals, pris))], q)
        best = max(scores, key=scores.get)

        X.append([avg_b, var_b, rng_b, dens, pvar, upri, num_p, q])
        y.append(best)
    return np.array(X), y


_ML_MODEL = None
_ML_LABEL_ENC = None

def train_ml_model():
    global _ML_MODEL, _ML_LABEL_ENC
    if not SKLEARN_OK:
        return False
    X, y = _generate_training_data(1200)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    clf = RandomForestClassifier(n_estimators=60, random_state=42)
    clf.fit(X, y_enc)
    _ML_MODEL     = clf
    _ML_LABEL_ENC = le
    return True


def ml_predict(processes, quantum):
    if _ML_MODEL is None or not SKLEARN_OK:
        return None, None
    bursts  = [p['burst']           for p in processes]
    pris    = [p.get('priority', 1) for p in processes]
    arrives = [p['arrival']         for p in processes]
    feats = np.array([[
        np.mean(bursts), np.var(bursts), max(bursts)-min(bursts),
        len(processes) / (max(arrives)-min(arrives)+1),
        np.var(pris), len(set(pris)), len(processes), quantum
    ]])
    proba  = _ML_MODEL.predict_proba(feats)[0]
    labels = _ML_LABEL_ENC.classes_
    return labels[np.argmax(proba)], dict(zip(labels, proba))


# ─────────────────────────────────────────────
#  CHATBOT KNOWLEDGE BASE
# ─────────────────────────────────────────────

CHATBOT_KB = {
    # Algorithms
    'fcfs': "FCFS (First Come First Served) is the simplest CPU scheduling algorithm. Processes execute in arrival order. It is non-preemptive and easy to implement, but suffers from the Convoy Effect — short jobs stuck behind long ones.",
    'sjf': "SJF (Shortest Job First) picks the process with the smallest burst time. It minimises average waiting time and is optimal in that sense. However, it can cause starvation for long processes, and requires knowing burst time in advance.",
    'round robin': "Round Robin (RR) assigns each process a fixed time quantum in cyclic order. It is preemptive and ensures fairness. Performance depends heavily on the quantum size — too small causes excess context switching; too large degrades to FCFS.",
    'priority': "Priority Scheduling runs the process with the highest priority (lowest number) first. It is preemptive in this visualiser. Main issue: starvation of low-priority processes, which can be solved by aging (gradually increasing priority over time).",
    'smart-sjf': "SMART-SJF is a novel hybrid algorithm designed in this visualiser. It combines SJF's efficiency with a fairness mechanism: if any process waits more than a patience threshold, it gets a mini Round-Robin burst. This eliminates starvation while keeping near-optimal average waiting time.",
    # Metrics
    'completion time': "Completion Time (CT) is when a process finishes execution. Formula: CT = time at which the last CPU burst ends.",
    'turnaround time': "Turnaround Time (TAT) = Completion Time − Arrival Time. It measures total time a process spends in the system, from arrival to completion.",
    'waiting time': "Waiting Time (WT) = Turnaround Time − Burst Time. It measures how long a process spent waiting in the ready queue rather than executing.",
    'response time': "Response Time (RT) = First CPU Time − Arrival Time. Especially important for interactive systems — the time until a process first gets the CPU.",
    # OS Concepts
    'context switch': "A context switch is the OS saving the state (registers, program counter) of the currently running process and loading another process's state. It has overhead — excessive context switches slow down the system.",
    'starvation': "Starvation occurs when a process never gets CPU time because higher-priority or shorter processes keep arriving. Solution: Aging (increase priority of waiting processes over time) or SMART-SJF's patience mechanism.",
    'convoy effect': "The Convoy Effect happens in FCFS when many short processes get stuck behind one long process. It increases average waiting time significantly.",
    'preemptive': "Preemptive scheduling allows the OS to forcibly take the CPU away from a running process when a higher-priority process arrives. Examples: Preemptive Priority, Round Robin. Opposite is Non-preemptive (FCFS, SJF).",
    'non-preemptive': "Non-preemptive scheduling means once a process starts executing, it runs to completion (or until it blocks for I/O). Examples: FCFS, SJF. Simpler but less responsive.",
    'cpu burst': "A CPU burst is the period during which a process actively uses the CPU. After a CPU burst, a process might do I/O (I/O burst). Scheduling decisions happen between CPU bursts.",
    'ready queue': "The Ready Queue holds all processes that are loaded in memory, ready to execute, waiting for CPU time. The scheduler picks from this queue each time a CPU becomes free.",
    'throughput': "Throughput is the number of processes completed per unit time. Higher is better. It is a key metric for batch systems.",
    'aging': "Aging is a technique to prevent starvation: gradually increase the priority of processes that have been waiting for a long time. This ensures every process eventually runs.",
    'quantum': "The time quantum (or time slice) in Round Robin is the maximum time a process can run before being preempted. Choosing it wisely is crucial: small quantum → many context switches; large quantum → FCFS behaviour.",
    'os': "An Operating System (OS) is system software that manages hardware resources and provides services to applications. Key functions include process management, memory management, file systems, and I/O handling.",
    'process': "A process is a program in execution. It includes the program code, current activity (program counter, registers), and allocated resources. The OS manages process creation, scheduling, synchronisation, and termination.",
    'scheduler': "The CPU Scheduler (short-term scheduler) selects which process from the ready queue to execute next. It aims to maximise CPU utilisation, throughput, and fairness while minimising waiting and response times.",
    'dispatcher': "The Dispatcher is the module that gives control of the CPU to the selected process. It handles context switches, switching to user mode, and jumping to the right location in the program. Dispatch latency = time taken to do all this.",
    'multilevel queue': "Multilevel Queue Scheduling divides the ready queue into several queues (e.g., foreground/background), each with its own algorithm. Processes are permanently assigned to a queue based on properties like priority or type.",
    'default': "I'm your OS & CPU Scheduling assistant! Ask me about algorithms (FCFS, SJF, RR, Priority, SMART-SJF), metrics (WT, TAT, RT, CT), or OS concepts like context switch, starvation, quantum, ready queue, aging, throughput, and more.",
}

def chatbot_respond(user_input):
    q = user_input.lower().strip()
    # Direct key match
    for key, answer in CHATBOT_KB.items():
        if key in q:
            return answer
    # Fallback
    return (CHATBOT_KB['default'] +
            "\n\nYou asked: \"" + user_input + "\"\n"
            "Try asking about: fcfs, sjf, round robin, priority, smart-sjf, "
            "waiting time, turnaround time, response time, starvation, context switch, quantum, throughput, aging, os, process, scheduler.")


# ─────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────

class CPUSchedulerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CPU Scheduling Visualizer | Marzana Tabassum | NSU")
        self.root.configure(bg=BG)
        self.root.state('zoomed')

        self.processes       = []
        self.pid_counter     = 1
        self.current_algo    = tk.StringVar(value='FCFS')
        self.quantum_var     = tk.IntVar(value=2)
        self._anim_job       = None
        self._sim_running    = False
        self._sim_paused     = False
        self._editing_pid    = None
        self._step_mode      = tk.BooleanVar(value=False)
        self._starvation_thr = tk.IntVar(value=15)
        self._drag_data      = {}

        # ML training in background
        self._ml_ready = False
        threading.Thread(target=self._train_ml_bg, daemon=True).start()

        self._build_ui()
        self._start_pulse()

    def _train_ml_bg(self):
        self._ml_ready = train_ml_model()

    # ── PULSE ─────────────────────────────────
    def _start_pulse(self):
        colors = [BLUE, PURPLE, GREEN, PINK, YELLOW, ORANGE]
        self._pulse_idx = 0
        def pulse():
            c = colors[self._pulse_idx % len(colors)]
            try: self.title_label.config(fg=c)
            except: return
            self._pulse_idx += 1
            self.root.after(600, pulse)
        pulse()

    # ── UI BUILDER ────────────────────────────
    def _build_ui(self):
        topbar = tk.Frame(self.root, bg=PANEL, pady=12)
        topbar.pack(fill='x')
        self.title_label = tk.Label(
            topbar, text="⚙  CPU Scheduling Visualizer ",
            font=('Segoe UI', 18, 'bold'), bg=PANEL, fg=BLUE)
        self.title_label.pack(side='left', padx=20)
        tk.Label(topbar,
            text="Marzana Tabassum  |  2232327642  |  North South University",
            font=('Segoe UI', 10), bg=PANEL, fg=SUBTEXT).pack(side='right', padx=20)

        main  = tk.Frame(self.root, bg=BG)
        main.pack(fill='both', expand=True, padx=10, pady=6)
        left  = tk.Frame(main, bg=BG, width=380)
        right = tk.Frame(main, bg=BG)
        left.pack(side='left', fill='y', padx=(0, 8))
        right.pack(side='left', fill='both', expand=True)
        left.pack_propagate(False)
        self._build_left(left)
        self._build_right(right)

    # ── LEFT PANEL ────────────────────────────
    def _build_left(self, parent):
        canvas = tk.Canvas(parent, bg=BG, highlightthickness=0)
        sb     = tk.Scrollbar(parent, orient='vertical', command=canvas.yview, bg=PANEL)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)
        inner = tk.Frame(canvas, bg=BG)
        cw    = canvas.create_window((0, 0), window=inner, anchor='nw')
        inner.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',  lambda e: canvas.itemconfig(cw, width=e.width))
        canvas.bind_all('<MouseWheel>',
            lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), 'units'))

        p = inner

        # ── Add / Edit Process card
        self._card_title(p, "➕  Add / Edit Process", BLUE)
        card = tk.Frame(p, bg=CARD, bd=0)
        card.pack(fill='x', padx=8, pady=4)

        fields = [("PID",      'pid_var',      'P1'),
                  ("Arrival",  'arrival_var',  '0'),
                  ("Burst",    'burst_var',    '5'),
                  ("Priority", 'priority_var', '1')]
        for label, attr, default in fields:
            row = tk.Frame(card, bg=CARD)
            row.pack(fill='x', padx=10, pady=3)
            tk.Label(row, text=label, width=9, anchor='w',
                     font=('Segoe UI', 9), bg=CARD, fg=TEXT).pack(side='left')
            var = tk.StringVar(value=default)
            setattr(self, attr, var)
            e = tk.Entry(row, textvariable=var, width=10,
                         bg=ACCENT, fg=TEXT, insertbackground=BLUE,
                         relief='flat', font=('Segoe UI', 9), bd=4)
            e.pack(side='left', padx=4)

        btn_row = tk.Frame(card, bg=CARD)
        btn_row.pack(pady=8)
        self._btn(btn_row, "＋ Add",           self._add_process,       BLUE).pack(side='left', padx=3)
        self._btn(btn_row, "💾 Save Edit",      self._save_edit,         GREEN).pack(side='left', padx=3)
        self._btn(btn_row, "✕ Clear All",       self._clear_all,         PINK).pack(side='left', padx=3)

        # ── Live inject button (visible when sim is running)
        self._inject_btn = self._btn(card, "⚡ Inject Live", self._inject_live, YELLOW)
        self._inject_btn.pack(pady=(0, 6))

        # ── Process table (drag & drop)
        self._card_title(p, "📋  Process Queue (Drag to Reorder Priority)", PURPLE)
        tbl = tk.Frame(p, bg=CARD)
        tbl.pack(fill='x', padx=8, pady=4)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Dark.Treeview', background=CARD, foreground=TEXT,
                        fieldbackground=CARD, rowheight=24, font=('Segoe UI', 9))
        style.configure('Dark.Treeview.Heading', background=ACCENT, foreground=BLUE,
                        font=('Segoe UI', 9, 'bold'))
        style.map('Dark.Treeview', background=[('selected', BLUE)],
                  foreground=[('selected', BG)])

        cols = ('PID', 'Arrival', 'Burst', 'Priority')
        self.tree = ttk.Treeview(tbl, columns=cols, show='headings',
                                 height=6, style='Dark.Treeview')
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=72, anchor='center')
        self.tree.pack(fill='x', padx=6, pady=6)

        # Drag & Drop bindings
        self.tree.bind('<ButtonPress-1>',   self._dnd_start)
        self.tree.bind('<B1-Motion>',       self._dnd_motion)
        self.tree.bind('<ButtonRelease-1>', self._dnd_drop)

        tbl_btns = tk.Frame(tbl, bg=CARD)
        tbl_btns.pack(pady=4)
        self._btn(tbl_btns, "✏ Edit",   self._edit_selected,   YELLOW).pack(side='left', padx=4)
        self._btn(tbl_btns, "− Remove", self._remove_selected, ORANGE).pack(side='left', padx=4)
        self._btn(tbl_btns, "↑ Move Up",self._move_up,         TEAL).pack(side='left', padx=4)
        self._btn(tbl_btns, "↓ Move Dn",self._move_down,       PURPLE).pack(side='left', padx=4)

        # ── Algorithm
        self._card_title(p, "🧮  Algorithm", GREEN)
        acard = tk.Frame(p, bg=CARD)
        acard.pack(fill='x', padx=8, pady=4)

        algos = [
            ('FCFS',        'First Come First Served',           BLUE),
            ('SJF',         'Shortest Job First',                GREEN),
            ('Round Robin', 'Round Robin',                       ORANGE),
            ('Priority',    'Preemptive Priority',               PINK),
            ('SMART-SJF',   'SMART-SJF (Novel Hybrid)',          YELLOW),
            ('Compare All', 'Compare All Algorithms',            SUBTEXT),
        ]
        for val, label, color in algos:
            rb = tk.Radiobutton(acard, text=f"  {val}  —  {label}",
                                variable=self.current_algo, value=val,
                                bg=CARD, fg=color, selectcolor=ACCENT,
                                activebackground=CARD, activeforeground=color,
                                font=('Segoe UI', 9, 'bold'))
            rb.pack(anchor='w', padx=10, pady=2)

        rr_row = tk.Frame(acard, bg=CARD); rr_row.pack(fill='x', padx=10, pady=4)
        tk.Label(rr_row, text="⏱ Time Quantum:", bg=CARD, fg=TEXT,
                 font=('Segoe UI', 9)).pack(side='left')
        tk.Spinbox(rr_row, from_=1, to=20, textvariable=self.quantum_var,
                   width=5, bg=ACCENT, fg=BLUE, buttonbackground=ACCENT,
                   font=('Segoe UI', 9, 'bold'), relief='flat').pack(side='left', padx=6)
        self._btn(rr_row, "🧠 Auto-Tune", self._auto_tune_quantum, TEAL).pack(side='left', padx=6)

        thr_row = tk.Frame(acard, bg=CARD); thr_row.pack(fill='x', padx=10, pady=4)
        tk.Label(thr_row, text="⚠ Starvation Threshold:", bg=CARD, fg=TEXT,
                 font=('Segoe UI', 9)).pack(side='left')
        tk.Spinbox(thr_row, from_=5, to=100, increment=5, textvariable=self._starvation_thr,
                   width=5, bg=ACCENT, fg=ORANGE, buttonbackground=ACCENT,
                   font=('Segoe UI', 9, 'bold'), relief='flat').pack(side='left', padx=6)

        step_row = tk.Frame(acard, bg=CARD); step_row.pack(fill='x', padx=10, pady=4)
        tk.Checkbutton(step_row, text="🔬 Step-by-Step Mode",
                       variable=self._step_mode,
                       bg=CARD, fg=LIME, selectcolor=ACCENT,
                       activebackground=CARD, activeforeground=LIME,
                       font=('Segoe UI', 9, 'bold')).pack(side='left')

        # ── Action buttons
        self._card_title(p, "▶  Actions", YELLOW)
        bcard = tk.Frame(p, bg=BG); bcard.pack(fill='x', padx=8, pady=4)
        self._big_btn(bcard, "▶  Run Simulation",          self._run_simulation,  GREEN).pack(fill='x', pady=3)
        self._big_btn(bcard, "🔮  Predict Best Algorithm", self._run_prediction,  PURPLE).pack(fill='x', pady=3)
        self._big_btn(bcard, "📊  Load Sample Data",       self._load_sample,     BLUE).pack(fill='x', pady=3)
        self._big_btn(bcard, "⏸  Pause / Resume Sim",     self._toggle_pause,    ORANGE).pack(fill='x', pady=3)

    # ── RIGHT PANEL ───────────────────────────
    def _build_right(self, parent):
        nb = ttk.Notebook(parent)
        nb.pack(fill='both', expand=True)
        self.notebook = nb

        style = ttk.Style()
        style.configure('TNotebook',     background=BG, borderwidth=0)
        style.configure('TNotebook.Tab', background=PANEL, foreground=SUBTEXT,
                        padding=[12, 6], font=('Segoe UI', 9))
        style.map('TNotebook.Tab', background=[('selected', BLUE)],
                  foreground=[('selected', BG)])

        # Tab 0 — Gantt
        t0 = tk.Frame(nb, bg=BG); nb.add(t0, text='📊  Gantt')
        self.fig_gantt, self.ax_gantt = plt.subplots(figsize=(10, 4))
        self.fig_gantt.patch.set_facecolor(BG)
        self.canvas_gantt = FigureCanvasTkAgg(self.fig_gantt, t0)
        self.canvas_gantt.get_tk_widget().pack(fill='both', expand=True)

        # Tab 1 — Metrics
        t1 = tk.Frame(nb, bg=BG); nb.add(t1, text='📈  Metrics')
        self._build_metrics_tab(t1)

        # Tab 2 — Compare
        t2 = tk.Frame(nb, bg=BG); nb.add(t2, text='🔁  Compare All')
        self.fig_compare, self.axes_compare = plt.subplots(5, 1, figsize=(10, 14))
        self.fig_compare.patch.set_facecolor(BG)
        self.canvas_compare = FigureCanvasTkAgg(self.fig_compare, t2)
        self.canvas_compare.get_tk_widget().pack(fill='both', expand=True)

        # Tab 3 — Process Simulation
        t3 = tk.Frame(nb, bg=BG); nb.add(t3, text='🎬  Simulation')
        self._build_simulation_tab(t3)

        # Tab 4 — Prediction + ML
        t4 = tk.Frame(nb, bg=BG); nb.add(t4, text='🔮  Prediction')
        self._build_prediction_tab(t4)

        # Tab 5 — Analytics (CPU util, starvation, throughput)
        t5 = tk.Frame(nb, bg=BG); nb.add(t5, text='📡  Analytics')
        self._build_analytics_tab(t5)

        # Tab 6 — SMART-SJF Explainer
        t6 = tk.Frame(nb, bg=BG); nb.add(t6, text='🧬  SMART-SJF')
        self._build_smart_sjf_tab(t6)

        # Tab 7 — Chatbot
        t7 = tk.Frame(nb, bg=BG); nb.add(t7, text='🤖  OS Chatbot')
        self._build_chatbot_tab(t7)

        # Status bar
        self.status_var = tk.StringVar(value="Ready — Add processes and click ▶ Run Simulation")
        tk.Label(parent, textvariable=self.status_var,
                 bg=PANEL, fg=GREEN, font=('Segoe UI', 9), anchor='w', padx=12,
                 pady=4).pack(fill='x', side='bottom')

    # ── METRICS TAB ───────────────────────────
    def _build_metrics_tab(self, parent):
        formula_frame = tk.Frame(parent, bg=PANEL, pady=6)
        formula_frame.pack(fill='x', padx=10, pady=(8, 0))
        tk.Label(formula_frame, text="📐  Formulas:",
                 font=('Segoe UI', 10, 'bold'), bg=PANEL, fg=YELLOW).pack(side='left', padx=10)
        for f in ["CT = Last CPU finish", "TAT = CT−Arrival", "WT = TAT−Burst", "RT = FirstCPU−Arrival"]:
            tk.Label(formula_frame, text=f"  |  {f}",
                     font=('Segoe UI', 9), bg=PANEL, fg=SUBTEXT).pack(side='left')

        tbl_outer = tk.Frame(parent, bg=BG)
        tbl_outer.pack(fill='x', padx=10, pady=6)

        style = ttk.Style()
        style.configure('Metrics.Treeview', background=CARD, foreground=TEXT,
                        fieldbackground=CARD, rowheight=26, font=('Segoe UI', 9))
        style.configure('Metrics.Treeview.Heading', background=ACCENT, foreground=GREEN,
                        font=('Segoe UI', 9, 'bold'))
        style.map('Metrics.Treeview', background=[('selected', GREEN)], foreground=[('selected', BG)])

        m_cols = ('PID','Arrival','Burst','CT','TAT','WT','RT')
        self.metrics_tree = ttk.Treeview(tbl_outer, columns=m_cols, show='headings',
                                         height=8, style='Metrics.Treeview')
        col_widths = {'PID': 60, 'Arrival': 70, 'Burst': 60, 'CT': 80, 'TAT': 80, 'WT': 80, 'RT': 80}
        for c in m_cols:
            self.metrics_tree.heading(c, text=c)
            self.metrics_tree.column(c, width=col_widths.get(c, 80), anchor='center')
        self.metrics_tree.pack(fill='x', padx=4, pady=4)

        self.avg_label = tk.Label(tbl_outer, text="", font=('Segoe UI', 10, 'bold'),
                                  bg=BG, fg=YELLOW, anchor='w', padx=4)
        self.avg_label.pack(fill='x', pady=4)

        self.fig_metrics, self.axes_metrics = plt.subplots(1, 4, figsize=(12, 3.5))
        self.fig_metrics.patch.set_facecolor(BG)
        self.canvas_metrics = FigureCanvasTkAgg(self.fig_metrics, parent)
        self.canvas_metrics.get_tk_widget().pack(fill='both', expand=True, padx=6, pady=4)

    # ── SIMULATION TAB (ENHANCED) ─────────────
    def _build_simulation_tab(self, parent):
        ctrl = tk.Frame(parent, bg=PANEL, pady=6)
        ctrl.pack(fill='x', padx=10, pady=(8, 0))

        tk.Label(ctrl, text="🎬  Animated Queue, CPU & Waiting Queue",
                 font=('Segoe UI', 11, 'bold'), bg=PANEL, fg=ORANGE).pack(side='left', padx=10)

        self._btn(ctrl, "▶ Play",     self._play_sim,          GREEN).pack(side='right', padx=4)
        self._btn(ctrl, "⏹ Stop",    self._stop_sim,           PINK).pack(side='right', padx=4)
        self._btn(ctrl, "⏭ Step",    self._step_once,          YELLOW).pack(side='right', padx=4)

        self.sim_speed = tk.IntVar(value=700)
        tk.Label(ctrl, text="Speed:", bg=PANEL, fg=SUBTEXT,
                 font=('Segoe UI', 9)).pack(side='right', padx=(10, 2))
        tk.Spinbox(ctrl, from_=100, to=2000, increment=100, textvariable=self.sim_speed,
                   width=6, bg=ACCENT, fg=BLUE, buttonbackground=ACCENT,
                   font=('Segoe UI', 9, 'bold'), relief='flat').pack(side='right')

        # Step explanation label
        self.step_label = tk.Label(parent, text="",
                                   font=('Segoe UI', 10, 'italic'), bg=BG, fg=LIME,
                                   wraplength=900, justify='left', padx=14)
        self.step_label.pack(fill='x')

        # Main simulation canvas
        self.sim_canvas = tk.Canvas(parent, bg='#0a0a14', highlightthickness=0, height=380)
        self.sim_canvas.pack(fill='both', expand=True, padx=10, pady=4)

        # Avg metrics chart below simulation
        self._card_title(parent, "📊  Average Metrics — Response | Completion | Turnaround | Waiting", TEAL)
        self.fig_sim_metrics, self.ax_sim_metrics = plt.subplots(figsize=(10, 2.4))
        self.fig_sim_metrics.patch.set_facecolor(BG)
        self.canvas_sim_metrics = FigureCanvasTkAgg(self.fig_sim_metrics, parent)
        self.canvas_sim_metrics.get_tk_widget().pack(fill='x', padx=10, pady=4)

        self._sim_step     = 0
        self._sim_timeline = []
        self._sim_after    = None

    # ── ANALYTICS TAB ─────────────────────────
    def _build_analytics_tab(self, parent):
        canvas = tk.Canvas(parent, bg=BG, highlightthickness=0)
        sb     = tk.Scrollbar(parent, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)
        inner = tk.Frame(canvas, bg=BG)
        cw    = canvas.create_window((0, 0), window=inner, anchor='nw')
        inner.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',  lambda e: canvas.itemconfig(cw, width=e.width))
        self.analytics_inner = inner

        tk.Label(inner, text="📡  CPU Analytics Dashboard",
                 font=('Segoe UI', 14, 'bold'), bg=BG, fg=TEAL).pack(pady=(16, 4))
        tk.Label(inner, text="Run a simulation first, then check here for detailed analytics.",
                 font=('Segoe UI', 10), bg=BG, fg=SUBTEXT).pack()

        # Utilization timeline
        self._card_title(inner, "⚡ CPU Utilization Over Time", GREEN)
        self.fig_util, self.ax_util = plt.subplots(figsize=(10, 2.5))
        self.fig_util.patch.set_facecolor(BG)
        self.canvas_util = FigureCanvasTkAgg(self.fig_util, inner)
        self.canvas_util.get_tk_widget().pack(fill='x', padx=10, pady=4)

        # Starvation
        self._card_title(inner, "😴 Starvation Detection", RED)
        self.starvation_label = tk.Label(inner, text="No data yet.",
                                         font=('Segoe UI', 10), bg=BG, fg=SUBTEXT,
                                         anchor='w', padx=16)
        self.starvation_label.pack(fill='x', pady=4)

        # Throughput gauge
        self._card_title(inner, "🚀 Throughput & Efficiency", YELLOW)
        self.fig_gauge, self.ax_gauge = plt.subplots(figsize=(6, 3), subplot_kw=dict(polar=False))
        self.fig_gauge.patch.set_facecolor(BG)
        self.canvas_gauge = FigureCanvasTkAgg(self.fig_gauge, inner)
        self.canvas_gauge.get_tk_widget().pack(fill='x', padx=10, pady=4)

    # ── SMART-SJF EXPLAINER TAB ───────────────
    def _build_smart_sjf_tab(self, parent):
        canvas = tk.Canvas(parent, bg=BG, highlightthickness=0)
        sb     = tk.Scrollbar(parent, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)
        inner = tk.Frame(canvas, bg=BG)
        cw    = canvas.create_window((0, 0), window=inner, anchor='nw')
        inner.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',  lambda e: canvas.itemconfig(cw, width=e.width))

        tk.Label(inner, text="🧬  SMART-SJF — A Novel Scheduling Algorithm",
                 font=('Segoe UI', 15, 'bold'), bg=BG, fg=YELLOW).pack(pady=(20, 6))

        desc = (
            "SMART-SJF (Shortest Job with Adaptive Fair Round-Robin) is a hybrid CPU "
            "scheduling algorithm that addresses the core weakness of SJF: starvation.\n\n"
            "How it works:\n"
            "  1. At each scheduling decision, check if any ready process has been waiting\n"
            "     longer than a PATIENCE threshold (adaptive, based on average burst time).\n"
            "  2. If starving processes exist → give ALL of them a mini Round-Robin burst\n"
            "     (fair_q = ~60% of avg burst), ensuring progress for long-waiting jobs.\n"
            "  3. If no starvation → pick the process with the shortest remaining burst\n"
            "     (like SRTF/SJF), maintaining near-optimal average waiting time.\n\n"
            "Why it matters:\n"
            "  • Pure SJF: optimal WT but starves long processes indefinitely.\n"
            "  • Pure RR: perfectly fair but suboptimal WT (treats all jobs equally).\n"
            "  • SMART-SJF: short jobs still get priority most of the time,\n"
            "    but guaranteed CPU time for every process via the patience mechanism.\n"
            "  • Adaptive parameters: PATIENCE and fair_q auto-tune to your workload.\n\n"
            "Where it excels:\n"
            "  • Mixed workloads with both short (I/O-bound) and long (CPU-bound) processes.\n"
            "  • Interactive systems where responsiveness AND fairness both matter.\n"
            "  • Systems where burst times vary widely and starvation risk is high."
        )
        tk.Label(inner, text=desc, font=('Segoe UI', 10), bg=BG, fg=TEXT,
                 justify='left', wraplength=820, padx=20).pack(anchor='w', pady=8)

        # Flowchart diagram
        self._card_title(inner, "🔄 SMART-SJF Decision Flowchart", YELLOW)
        fig_flow, ax_flow = plt.subplots(figsize=(9, 5))
        fig_flow.patch.set_facecolor(BG)
        self._draw_smart_sjf_flowchart(ax_flow)
        FigureCanvasTkAgg(fig_flow, inner).get_tk_widget().pack(fill='x', padx=10, pady=8)

        # Comparison chart
        self._card_title(inner, "📊 SJF vs RR vs SMART-SJF — Conceptual Comparison", LIME)
        fig_cmp, ax_cmp = plt.subplots(figsize=(9, 3.5))
        fig_cmp.patch.set_facecolor(BG)
        self._draw_algo_comparison(ax_cmp)
        FigureCanvasTkAgg(fig_cmp, inner).get_tk_widget().pack(fill='x', padx=10, pady=8)

    def _draw_smart_sjf_flowchart(self, ax):
        ax.set_xlim(0, 10); ax.set_ylim(0, 10)
        ax.axis('off'); ax.set_facecolor(BG)

        def box(x, y, w, h, text, color, fontsize=9):
            ax.add_patch(plt.Rectangle((x-w/2, y-h/2), w, h,
                                       facecolor=color, edgecolor='white', linewidth=1.5,
                                       alpha=0.85, zorder=2))
            ax.text(x, y, text, ha='center', va='center',
                    fontsize=fontsize, color='black' if color in [YELLOW, LIME, GREEN] else TEXT,
                    fontweight='bold', zorder=3, wrap=True)

        def diamond(x, y, w, h, text, color):
            pts = np.array([[x, y+h/2],[x+w/2, y],[x, y-h/2],[x-w/2, y]])
            ax.add_patch(plt.Polygon(pts, facecolor=color, edgecolor='white', linewidth=1.5,
                                     alpha=0.85, zorder=2))
            ax.text(x, y, text, ha='center', va='center',
                    fontsize=8, color=BG, fontweight='bold', zorder=3)

        def arrow(x1, y1, x2, y2, label='', color=SUBTEXT):
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.8))
            if label:
                mx, my = (x1+x2)/2, (y1+y2)/2
                ax.text(mx+0.12, my, label, fontsize=8, color=color, fontweight='bold')

        box(5, 9.2, 4, 0.8, "START: New time unit", BLUE, 10)
        arrow(5, 8.8, 5, 8.0)
        diamond(5, 7.4, 4, 1.0, "Any process\narrived & ready?", PURPLE)
        arrow(5, 6.9, 5, 6.1)
        ax.text(5.15, 6.5, 'YES', fontsize=8, color=GREEN, fontweight='bold')
        arrow(7, 7.4, 8.5, 7.4, 'NO')
        box(8.5, 7.4, 1.8, 0.7, "CPU IDLE\n+1 tick", ACCENT, 8)

        diamond(5, 5.5, 4.5, 1.0, "Any process waited\n> PATIENCE units?", RED)
        arrow(5, 5.0, 5, 4.2); ax.text(5.15, 4.6, 'NO', fontsize=8, color=SUBTEXT)
        arrow(7.25, 5.5, 8.5, 5.5, 'YES')

        box(5, 3.6, 4, 0.8, "Pick SHORTEST remaining burst\n(SJF step, run for fair_q ticks)", GREEN, 8)
        arrow(5, 3.2, 5, 2.4)

        box(8.5, 5.5, 1.8, 0.8, "Give starving\nprocesses RR\nslice (fair_q)", ORANGE, 8)

        box(5, 1.9, 4, 0.9, "Update last_served time\nDecrement remaining burst", TEAL, 8)
        arrow(5, 1.45, 5, 0.7)
        box(5, 0.4, 3.5, 0.6, "All done? → END", PINK, 9)

        ax.set_title("SMART-SJF Decision Logic", color=YELLOW, fontsize=12, fontweight='bold')

    def _draw_algo_comparison(self, ax):
        ax.set_facecolor('#0d0d1a')
        algos   = ['SJF', 'Round Robin', 'SMART-SJF']
        metrics = ['Avg WT (lower=better)', 'Fairness', 'Starvation-free', 'Implementation']
        scores  = {
            'SJF':         [95, 30, 10, 80],
            'Round Robin': [55, 95, 95, 90],
            'SMART-SJF':   [82, 80, 88, 70],
        }
        colors  = [GREEN, ORANGE, YELLOW]
        x       = np.arange(len(metrics))
        w       = 0.25
        for i, (algo, col) in enumerate(zip(algos, colors)):
            bars = ax.bar(x + (i-1)*w, scores[algo], w, label=algo, color=col, alpha=0.85, edgecolor=BG)
        ax.set_xticks(x); ax.set_xticklabels(metrics, color=TEXT, fontsize=9)
        ax.tick_params(colors=SUBTEXT)
        ax.spines[:].set_color(ACCENT)
        ax.set_ylabel('Score (0–100)', color=SUBTEXT, fontsize=9)
        ax.set_title('Algorithm Comparison: SJF vs RR vs SMART-SJF', color=LIME, fontsize=11, fontweight='bold')
        ax.legend(facecolor=CARD, edgecolor=ACCENT, labelcolor=TEXT, fontsize=9)
        ax.set_ylim(0, 110)

    # ── PREDICTION TAB (ENHANCED) ─────────────
    def _build_prediction_tab(self, parent):
        pcanvas = tk.Canvas(parent, bg=BG, highlightthickness=0)
        psb     = tk.Scrollbar(parent, orient='vertical', command=pcanvas.yview)
        pcanvas.configure(yscrollcommand=psb.set)
        psb.pack(side='right', fill='y')
        pcanvas.pack(side='left', fill='both', expand=True)
        pinner = tk.Frame(pcanvas, bg=BG)
        pcw    = pcanvas.create_window((0, 0), window=pinner, anchor='nw')
        pinner.bind('<Configure>', lambda e: pcanvas.configure(scrollregion=pcanvas.bbox('all')))
        pcanvas.bind('<Configure>',  lambda e: pcanvas.itemconfig(pcw, width=e.width))
        self.pred_inner = pinner

        tk.Label(pinner, text="🔮  Smart Algorithm Recommendation Engine",
                 font=('Segoe UI', 15, 'bold'), bg=BG, fg=PURPLE).pack(pady=(20, 4))
        tk.Label(pinner, text="Rule-Based + ML Classifier. Click 🔮 Predict to analyse your workload.",
                 font=('Segoe UI', 10), bg=BG, fg=SUBTEXT).pack()

        self.pred_winner = tk.Label(pinner, text="Click '🔮 Predict Best Algorithm' to begin",
                                    font=('Segoe UI', 13, 'bold'), bg=CARD, fg=SUBTEXT,
                                    pady=16, padx=20, relief='flat')
        self.pred_winner.pack(fill='x', padx=20, pady=16)

        # ML prediction label
        self.ml_pred_label = tk.Label(pinner, text="🧠 ML Prediction: (training in background...)",
                                       font=('Segoe UI', 11, 'bold'), bg=CARD, fg=TEAL,
                                       pady=8, padx=20)
        self.ml_pred_label.pack(fill='x', padx=20, pady=4)

        # Scores bar chart
        self._card_title(pinner, "📊 Algorithm Suitability Scores", BLUE)
        self.fig_pred, self.ax_pred = plt.subplots(figsize=(9, 3))
        self.fig_pred.patch.set_facecolor(BG)
        self.canvas_pred = FigureCanvasTkAgg(self.fig_pred, pinner)
        self.canvas_pred.get_tk_widget().pack(fill='x', padx=20, pady=4)

        # NEW: Rule breakdown chart per algorithm
        self._card_title(pinner, "🔬 Rule-by-Rule Scoring Breakdown (How Each Score Was Computed)", ORANGE)
        self.fig_rules, self.axes_rules = plt.subplots(1, 5, figsize=(14, 3.5))
        self.fig_rules.patch.set_facecolor(BG)
        self.canvas_rules = FigureCanvasTkAgg(self.fig_rules, pinner)
        self.canvas_rules.get_tk_widget().pack(fill='x', padx=20, pady=4)

        # Feature values radar/bar
        self._card_title(pinner, "📐 Your Workload Feature Values", LIME)
        self.fig_features, self.ax_features = plt.subplots(figsize=(9, 2.8))
        self.fig_features.patch.set_facecolor(BG)
        self.canvas_features = FigureCanvasTkAgg(self.fig_features, pinner)
        self.canvas_features.get_tk_widget().pack(fill='x', padx=20, pady=4)

        # Detail cards
        self.pred_cards_frame = tk.Frame(pinner, bg=BG)
        self.pred_cards_frame.pack(fill='both', expand=True, padx=20, pady=10)

    # ── CHATBOT TAB ───────────────────────────
    def _build_chatbot_tab(self, parent):
        tk.Label(parent, text="🤖  OS & CPU Scheduling Assistant",
                 font=('Segoe UI', 14, 'bold'), bg=BG, fg=BLUE).pack(pady=(16, 4))
        tk.Label(parent,
                 text="Ask me anything about CPU scheduling algorithms, OS concepts, metrics, or this visualiser!",
                 font=('Segoe UI', 10), bg=BG, fg=SUBTEXT).pack()

        chat_frame = tk.Frame(parent, bg=BG)
        chat_frame.pack(fill='both', expand=True, padx=16, pady=8)

        self.chat_display = scrolledtext.ScrolledText(
            chat_frame, bg=CARD, fg=TEXT, insertbackground=BLUE,
            font=('Segoe UI', 10), relief='flat', wrap='word',
            state='disabled', height=22)
        self.chat_display.pack(fill='both', expand=True, pady=(0, 8))

        # Configure tags
        self.chat_display.tag_config('user',  foreground=BLUE,   font=('Segoe UI', 10, 'bold'))
        self.chat_display.tag_config('bot',   foreground=GREEN,  font=('Segoe UI', 10))
        self.chat_display.tag_config('label', foreground=YELLOW, font=('Segoe UI', 10, 'bold'))

        inp_frame = tk.Frame(chat_frame, bg=BG)
        inp_frame.pack(fill='x')
        self.chat_input = tk.Entry(inp_frame, bg=ACCENT, fg=TEXT, insertbackground=BLUE,
                                   font=('Segoe UI', 11), relief='flat', bd=6)
        self.chat_input.pack(side='left', fill='x', expand=True, padx=(0, 8))
        self.chat_input.bind('<Return>', lambda e: self._chat_send())
        self._btn(inp_frame, "Send 📤", self._chat_send, BLUE).pack(side='right')

        # Suggestion buttons
        sug_frame = tk.Frame(parent, bg=BG)
        sug_frame.pack(fill='x', padx=16, pady=(0, 12))
        suggestions = ["What is FCFS?","Explain SJF","What is SMART-SJF?",
                       "What is starvation?","Round Robin quantum","Turnaround time formula",
                       "Explain context switch","What is aging?"]
        for s in suggestions:
            self._btn(sug_frame, s, lambda txt=s: self._chat_quick(txt), ACCENT).pack(
                side='left', padx=3, pady=2)

        # Welcome message
        self._chat_append("🤖 Bot", chatbot_respond("hello"), 'bot')

    def _chat_append(self, who, text, tag):
        self.chat_display.config(state='normal')
        self.chat_display.insert('end', f"\n{who}:\n", 'label')
        self.chat_display.insert('end', text + "\n", tag)
        self.chat_display.config(state='disabled')
        self.chat_display.see('end')

    def _chat_send(self):
        q = self.chat_input.get().strip()
        if not q: return
        self.chat_input.delete(0, 'end')
        self._chat_append("👤 You", q, 'user')
        resp = chatbot_respond(q)
        self.root.after(200, lambda: self._chat_append("🤖 Bot", resp, 'bot'))

    def _chat_quick(self, txt):
        self.chat_input.delete(0, 'end')
        self.chat_input.insert(0, txt)
        self._chat_send()

    # ── HELPERS ───────────────────────────────
    def _card_title(self, parent, text, color):
        f = tk.Frame(parent, bg=BG)
        f.pack(fill='x', padx=8, pady=(12, 2))
        tk.Label(f, text=text, font=('Segoe UI', 10, 'bold'),
                 bg=BG, fg=color).pack(anchor='w')
        tk.Frame(f, bg=color, height=2).pack(fill='x', pady=2)

    def _btn(self, parent, text, cmd, color):
        return tk.Button(parent, text=text, command=cmd,
                         bg=color, fg=BG, activebackground=color,
                         font=('Segoe UI', 9, 'bold'),
                         relief='flat', cursor='hand2', pady=3, padx=8)

    def _big_btn(self, parent, text, cmd, color):
        return tk.Button(parent, text=text, command=cmd,
                         bg=color, fg=BG, activebackground=color,
                         font=('Segoe UI', 11, 'bold'),
                         relief='flat', cursor='hand2', pady=8)

    # ── DRAG & DROP ───────────────────────────
    def _dnd_start(self, event):
        row = self.tree.identify_row(event.y)
        if row:
            self._drag_data = {'item': row}

    def _dnd_motion(self, event):
        if not self._drag_data.get('item'): return
        self.tree.selection_set(self._drag_data['item'])

    def _dnd_drop(self, event):
        if not self._drag_data.get('item'): return
        src  = self._drag_data['item']
        dest = self.tree.identify_row(event.y)
        if dest and dest != src:
            # Get indices
            items = self.tree.get_children()
            si    = list(items).index(src)
            di    = list(items).index(dest)
            # Swap in processes list
            self.processes[si], self.processes[di] = self.processes[di], self.processes[si]
            # Rebuild tree
            for item in self.tree.get_children():
                self.tree.delete(item)
            for proc in self.processes:
                self.tree.insert('', 'end', values=(proc['pid'], proc['arrival'],
                                                    proc['burst'], proc['priority']))
            self.status_var.set(f"🔀 Reordered: swapped positions {si+1} ↔ {di+1}")
        self._drag_data = {}

    def _move_up(self):
        sel = self.tree.selection()
        if not sel: return
        items = list(self.tree.get_children())
        idx   = items.index(sel[0])
        if idx == 0: return
        self.processes[idx], self.processes[idx-1] = self.processes[idx-1], self.processes[idx]
        self._rebuild_tree()

    def _move_down(self):
        sel = self.tree.selection()
        if not sel: return
        items = list(self.tree.get_children())
        idx   = items.index(sel[0])
        if idx >= len(self.processes)-1: return
        self.processes[idx], self.processes[idx+1] = self.processes[idx+1], self.processes[idx]
        self._rebuild_tree()

    def _rebuild_tree(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for proc in self.processes:
            self.tree.insert('', 'end', values=(proc['pid'], proc['arrival'],
                                                proc['burst'], proc['priority']))

    # ── PROCESS MANAGEMENT ────────────────────
    def _add_process(self):
        try:
            pid      = self.pid_var.get().strip() or f"P{self.pid_counter}"
            arrival  = int(self.arrival_var.get())
            burst    = int(self.burst_var.get())
            priority = int(self.priority_var.get())
            if burst <= 0:
                messagebox.showerror("Error", "Burst time must be > 0"); return
            if any(p['pid'] == pid for p in self.processes):
                messagebox.showerror("Error", f"PID '{pid}' already exists"); return
            self.processes.append({'pid': pid, 'arrival': arrival,
                                   'burst': burst, 'priority': priority})
            self.tree.insert('', 'end', values=(pid, arrival, burst, priority))
            self.pid_counter += 1
            self.pid_var.set(f"P{self.pid_counter}")
            self.arrival_var.set('0'); self.burst_var.set('5'); self.priority_var.set('1')
            self._editing_pid = None
            self.status_var.set(f"✅ Added {pid}  |  Total: {len(self.processes)}")
        except ValueError:
            messagebox.showerror("Error", "Please enter valid integer values")

    def _inject_live(self):
        """Add a process while simulation is running."""
        try:
            pid      = self.pid_var.get().strip() or f"P{self.pid_counter}"
            arrival  = int(self.arrival_var.get())
            burst    = int(self.burst_var.get())
            priority = int(self.priority_var.get())
            if burst <= 0:
                messagebox.showerror("Error", "Burst time must be > 0"); return
            if any(p['pid'] == pid for p in self.processes):
                messagebox.showerror("Error", f"PID '{pid}' already exists"); return
            self.processes.append({'pid': pid, 'arrival': arrival,
                                   'burst': burst, 'priority': priority})
            self.tree.insert('', 'end', values=(pid, arrival, burst, priority))
            self.pid_counter += 1
            self.pid_var.set(f"P{self.pid_counter}")
            self.status_var.set(f"⚡ LIVE INJECT: {pid} added — restart sim to include in schedule")
        except ValueError:
            messagebox.showerror("Error", "Please enter valid integer values")

    def _edit_selected(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Info", "Select a process to edit"); return
        item = sel[0]
        vals = self.tree.item(item, 'values')
        pid, arrival, burst, priority = vals
        self.pid_var.set(pid); self.arrival_var.set(arrival)
        self.burst_var.set(burst); self.priority_var.set(priority)
        self._editing_pid  = pid
        self._editing_item = item
        self.status_var.set(f"✏ Editing {pid} — change values then click 💾 Save Edit")

    def _save_edit(self):
        if not self._editing_pid:
            messagebox.showinfo("Info", "Select a process and click ✏ Edit first"); return
        try:
            new_pid      = self.pid_var.get().strip()
            new_arrival  = int(self.arrival_var.get())
            new_burst    = int(self.burst_var.get())
            new_priority = int(self.priority_var.get())
            if new_burst <= 0:
                messagebox.showerror("Error", "Burst time must be > 0"); return
            if new_pid != self._editing_pid and any(p['pid'] == new_pid for p in self.processes):
                messagebox.showerror("Error", f"PID '{new_pid}' already exists"); return
            for p in self.processes:
                if p['pid'] == self._editing_pid:
                    p['pid'] = new_pid; p['arrival'] = new_arrival
                    p['burst'] = new_burst; p['priority'] = new_priority
                    break
            self.tree.item(self._editing_item, values=(new_pid, new_arrival, new_burst, new_priority))
            self._editing_pid = None
            self.pid_var.set(f"P{self.pid_counter}")
            self.arrival_var.set('0'); self.burst_var.set('5'); self.priority_var.set('1')
            self.status_var.set(f"✅ Saved edit for {new_pid}")
        except ValueError:
            messagebox.showerror("Error", "Please enter valid integer values")

    def _remove_selected(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Info", "Select a process to remove"); return
        for item in sel:
            vals = self.tree.item(item, 'values')
            self.processes = [p for p in self.processes if p['pid'] != vals[0]]
            self.tree.delete(item)
        self.status_var.set(f"Removed  |  Remaining: {len(self.processes)}")

    def _clear_all(self):
        self.processes.clear(); self.pid_counter = 1; self._editing_pid = None
        for item in self.tree.get_children(): self.tree.delete(item)
        self.pid_var.set('P1')
        self.status_var.set("Cleared all processes")

    def _load_sample(self):
        self._clear_all()
        samples = [
            {'pid':'P1','arrival':0,'burst':8,'priority':2},
            {'pid':'P2','arrival':1,'burst':4,'priority':1},
            {'pid':'P3','arrival':2,'burst':9,'priority':3},
            {'pid':'P4','arrival':3,'burst':5,'priority':2},
            {'pid':'P5','arrival':4,'burst':2,'priority':1},
        ]
        for s in samples:
            self.processes.append(s)
            self.tree.insert('', 'end', values=(s['pid'],s['arrival'],s['burst'],s['priority']))
        self.pid_counter = 6; self.pid_var.set('P6')
        self.status_var.set("✅ Sample loaded — 5 processes ready")

    def _auto_tune_quantum(self):
        if not self.processes:
            messagebox.showinfo("Info", "Add processes first"); return
        q = adaptive_quantum(self.processes)
        self.quantum_var.set(q)
        self.status_var.set(f"🧠 Auto-tuned quantum = {q}  (based on burst distribution)")

    def _toggle_pause(self):
        self._sim_paused = not self._sim_paused
        if self._sim_paused:
            self.status_var.set("⏸ Simulation PAUSED — click Pause/Resume to continue")
        else:
            self.status_var.set("▶ Simulation RESUMED")
            self._advance_sim()

    # ── SIMULATION ────────────────────────────
    def _run_simulation(self):
        if not self.processes:
            messagebox.showwarning("No Processes", "Please add at least one process first"); return
        algo = self.current_algo.get()
        if algo == 'Compare All':
            self._run_compare(); return
        self.status_var.set(f"⏳ Running {algo}...")
        self.root.update()
        threading.Thread(target=self._sim_thread, args=(algo,), daemon=True).start()

    def _sim_thread(self, algo):
        try:
            q = self.quantum_var.get()
            tl = self._get_timeline(algo, q)
            metrics, avg_ct, avg_tat, avg_wt, avg_rt = calculate_metrics(self.processes, tl)
            self.root.after(0, self._animate_gantt, tl, algo)
            self.root.after(500, self._update_metrics, metrics, avg_ct, avg_tat, avg_wt, avg_rt, algo)
            self.root.after(0, self._update_analytics, tl, metrics, avg_ct, avg_tat, avg_wt, avg_rt)
            self.root.after(0, lambda: self.status_var.set(
                f"✅ {algo}  |  Avg WT: {avg_wt:.2f}  |  Avg TAT: {avg_tat:.2f}  |  Avg RT: {avg_rt:.2f}"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))

    def _get_timeline(self, algo, q):
        if algo == 'FCFS':         return fcfs(self.processes)
        elif algo == 'SJF':        return sjf(self.processes)
        elif algo == 'Round Robin':return round_robin(self.processes, q)
        elif algo == 'Priority':   return preemptive_priority(self.processes)
        elif algo == 'SMART-SJF':  return smart_sjf(self.processes)
        return fcfs(self.processes)

    # ── ANIMATED GANTT ────────────────────────
    def _animate_gantt(self, timeline, algo, ax=None, fig=None, canvas=None, title=None):
        _ax     = ax     or self.ax_gantt
        _fig    = fig    or self.fig_gantt
        _canvas = canvas or self.canvas_gantt
        _ax.clear(); _ax.set_facecolor('#0d0d1a')
        _fig.patch.set_facecolor(BG)

        pids      = list(dict.fromkeys(seg['pid'] for seg in timeline))
        color_map = {pid: PROC_COLORS[i % len(PROC_COLORS)] for i, pid in enumerate(pids)}
        total_segs = len(timeline)
        drawn = [0]

        def draw_next():
            if drawn[0] >= total_segs:
                for seg in timeline:
                    _ax.text(seg['start'], -0.42, str(seg['start']),
                             ha='center', va='top', fontsize=7, color=SUBTEXT)
                if timeline:
                    _ax.text(timeline[-1]['end'], -0.42, str(timeline[-1]['end']),
                             ha='center', va='top', fontsize=7, color=SUBTEXT)
                _canvas.draw()
                if ax is None: self.notebook.select(0)
                return
            seg   = timeline[drawn[0]]
            width = seg['end'] - seg['start']
            color = color_map[seg['pid']]
            _ax.barh(0, width, left=seg['start'], height=0.6,
                     color=color, edgecolor='white', linewidth=0.5, alpha=0.25)
            _ax.barh(0, width, left=seg['start'], height=0.5,
                     color=color, edgecolor='white', linewidth=1.0)
            if width > 0.5:
                _ax.text(seg['start']+width/2, 0, seg['pid'],
                         ha='center', va='center', fontsize=9,
                         color='white', fontweight='bold')
            legend = [mpatches.Patch(color=color_map[p], label=p) for p in pids]
            _ax.legend(handles=legend, loc='upper right', fontsize=8,
                       facecolor=CARD, edgecolor=ACCENT, labelcolor=TEXT)
            _ax.set_yticks([])
            _ax.set_xlabel('Time (units)', color=SUBTEXT, fontsize=9)
            _ax.set_title(title or f'Gantt Chart — {algo}', color=BLUE, fontsize=12, fontweight='bold')
            _ax.tick_params(colors=SUBTEXT)
            _ax.spines[:].set_color(ACCENT)
            _fig.tight_layout()
            _canvas.draw()
            drawn[0] += 1
            delay = max(30, 300 // total_segs)
            self.root.after(delay, draw_next)

        draw_next()

    # ── METRICS ───────────────────────────────
    def _update_metrics(self, metrics, avg_ct, avg_tat, avg_wt, avg_rt, algo):
        for row in self.metrics_tree.get_children():
            self.metrics_tree.delete(row)
        for m in metrics:
            self.metrics_tree.insert('', 'end', values=(
                m['pid'], m['arrival'], m['burst'],
                m['completion'], m['turnaround'], m['waiting'], m['response']
            ))
        self.avg_label.config(
            text=(f"  Averages:   CT = {avg_ct:.2f}   |   TAT = {avg_tat:.2f}   "
                  f"|   WT = {avg_wt:.2f}   |   RT = {avg_rt:.2f}")
        )
        for ax in self.axes_metrics:
            ax.clear(); ax.set_facecolor('#0d0d1a')
        self.fig_metrics.patch.set_facecolor(BG)
        pids = [m['pid'] for m in metrics]
        x    = np.arange(len(pids))
        datasets = [
            (self.axes_metrics[0], [m['completion']  for m in metrics], BLUE,   f'CT  (Avg: {avg_ct:.2f})',  avg_ct),
            (self.axes_metrics[1], [m['turnaround']  for m in metrics], GREEN,  f'TAT (Avg: {avg_tat:.2f})', avg_tat),
            (self.axes_metrics[2], [m['waiting']     for m in metrics], ORANGE, f'WT  (Avg: {avg_wt:.2f})',  avg_wt),
            (self.axes_metrics[3], [m['response']    for m in metrics], PURPLE, f'RT  (Avg: {avg_rt:.2f})',  avg_rt),
        ]
        for ax, vals, col, title, avg in datasets:
            bars = ax.bar(x, vals, color=col, edgecolor=BG, width=0.6)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+0.05, str(val),
                        ha='center', va='bottom', color=TEXT, fontsize=8)
            ax.set_xticks(x); ax.set_xticklabels(pids)
            ax.tick_params(colors=SUBTEXT); ax.spines[:].set_color(ACCENT)
            ax.set_facecolor('#0d0d1a')
            ax.set_title(title, color=col, fontsize=9, fontweight='bold')
            ax.axhline(avg, color=YELLOW, linestyle='--', linewidth=1.2, alpha=0.7)
        self.fig_metrics.suptitle(f'Performance Metrics — {algo}', color=TEXT, fontsize=11)
        self.fig_metrics.tight_layout()
        self.canvas_metrics.draw()
        self.notebook.select(1)

    # ── ANALYTICS UPDATE ──────────────────────
    def _update_analytics(self, tl, metrics, avg_ct, avg_tat, avg_wt, avg_rt):
        if not tl: return

        # ── CPU Utilization Timeline
        self.ax_util.clear()
        self.ax_util.set_facecolor('#0d0d1a')
        self.fig_util.patch.set_facecolor(BG)
        max_t = tl[-1]['end']
        pids  = list(dict.fromkeys(seg['pid'] for seg in tl))
        cmap  = {pid: PROC_COLORS[i % len(PROC_COLORS)] for i, pid in enumerate(pids)}
        for seg in tl:
            self.ax_util.barh(0, seg['end']-seg['start'], left=seg['start'],
                              height=0.6, color=cmap[seg['pid']], edgecolor=BG, alpha=0.85)
        # Idle gaps
        prev = 0
        for seg in tl:
            if seg['start'] > prev:
                self.ax_util.barh(0, seg['start']-prev, left=prev,
                                  height=0.6, color='#333355', edgecolor=BG, alpha=0.6)
                self.ax_util.text(prev + (seg['start']-prev)/2, 0, 'IDLE',
                                  ha='center', va='center', fontsize=7, color=SUBTEXT)
            prev = max(prev, seg['end'])

        util_pct = cpu_utilization(tl) * 100
        self.ax_util.set_title(f"CPU Timeline — Utilization: {util_pct:.1f}%  |  "
                               f"Busy={int(util_pct*max_t/100)} / Total={max_t} units",
                               color=GREEN, fontsize=10, fontweight='bold')
        self.ax_util.set_yticks([]); self.ax_util.set_xlabel('Time', color=SUBTEXT)
        self.ax_util.tick_params(colors=SUBTEXT); self.ax_util.spines[:].set_color(ACCENT)
        self.fig_util.tight_layout()
        self.canvas_util.draw()

        # ── Starvation
        starved = detect_starvation(self.processes, tl, self._starvation_thr.get())
        if starved:
            txt = "⚠️  STARVATION DETECTED:\n"
            for pid, w in starved:
                txt += f"   • {pid} waited {w} units before first CPU time (threshold: {self._starvation_thr.get()})\n"
            txt += "\nSuggestion: Use Aging, Priority Boost, or try SMART-SJF."
            self.starvation_label.config(text=txt, fg=RED)
        else:
            self.starvation_label.config(
                text=f"✅ No starvation detected (threshold = {self._starvation_thr.get()} units)",
                fg=GREEN)

        # ── Throughput gauge (bar chart as gauge)
        self.ax_gauge.clear()
        self.ax_gauge.set_facecolor('#0d0d1a')
        self.fig_gauge.patch.set_facecolor(BG)
        total_t   = tl[-1]['end'] if tl else 1
        throughput= len(self.processes) / total_t
        efficiency= cpu_utilization(tl)
        labels    = ['Throughput\n(proc/unit × 10)', 'CPU Util %', 'Avg WT\n(inverted %)']
        # normalize to 0-100 for display
        tp_norm   = min(100, throughput * 100)
        util_norm = efficiency * 100
        # inverted waiting time (lower is better, show as % of max possible)
        max_possible_wt = total_t
        wt_inverted = max(0, 100 - (avg_wt / max_possible_wt * 100)) if max_possible_wt > 0 else 100
        values = [tp_norm, util_norm, wt_inverted]
        colors = [TEAL, GREEN, LIME]
        bars = self.ax_gauge.barh(labels, values, color=colors, edgecolor=BG, height=0.5)
        for bar, val in zip(bars, values):
            self.ax_gauge.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                               f"{val:.1f}%", va='center', color=TEXT, fontsize=10, fontweight='bold')
        self.ax_gauge.set_xlim(0, 115)
        self.ax_gauge.axvline(100, color=YELLOW, linestyle='--', alpha=0.5, linewidth=1)
        self.ax_gauge.set_title(
            f"Efficiency Gauge  |  Throughput: {throughput:.3f} proc/unit  |  CPU Util: {util_norm:.1f}%",
            color=YELLOW, fontsize=10, fontweight='bold')
        self.ax_gauge.tick_params(colors=TEXT); self.ax_gauge.spines[:].set_color(ACCENT)
        self.fig_gauge.tight_layout()
        self.canvas_gauge.draw()

    # ── COMPARE ALL ───────────────────────────
    def _run_compare(self):
        if not self.processes:
            messagebox.showwarning("No Processes", "Add processes first"); return
        q   = self.quantum_var.get()
        tls = {
            'FCFS':        fcfs(self.processes),
            'SJF':         sjf(self.processes),
            'Round Robin': round_robin(self.processes, q),
            'Priority':    preemptive_priority(self.processes),
            'SMART-SJF':   smart_sjf(self.processes),
        }
        for ax in self.axes_compare: ax.clear()
        self.fig_compare.patch.set_facecolor(BG)
        summary = []
        for i, (algo, tl) in enumerate(tls.items()):
            _, _, avg_tat, avg_wt, _ = calculate_metrics(self.processes, tl)
            self._animate_gantt(tl, algo,
                                ax=self.axes_compare[i],
                                fig=self.fig_compare,
                                canvas=self.canvas_compare,
                                title=f'{algo}  |  Avg WT={avg_wt:.1f}  |  Avg TAT={avg_tat:.1f}')
            summary.append((algo, avg_wt, avg_tat))
        self.fig_compare.tight_layout(pad=2.0)
        self.canvas_compare.draw()
        self.notebook.select(2)
        best = min(summary, key=lambda x: x[1])
        self.status_var.set(f"✅ Compare done  |  Best Avg WT: {best[0]} ({best[1]:.2f})")

    # ── PREDICTION ────────────────────────────
    def _run_prediction(self):
        if not self.processes:
            messagebox.showwarning("No Processes", "Add processes first"); return
        q = self.quantum_var.get()
        best, scores, reasons, rule_breakdown, feat_vals = predict_best_algorithm(self.processes, q)

        banner_color = ALGO_COLORS.get(best, GREEN)
        self.pred_winner.config(
            text=f"🏆  Rule-Based Recommendation:  {best}\n\n\"{reasons[best]['summary']}\"",
            fg=banner_color, bg=CARD, font=('Segoe UI', 13, 'bold'))

        # ML Prediction
        ml_best, ml_proba = ml_predict(self.processes, q)
        if ml_best:
            self.ml_pred_label.config(
                text=f"🧠 ML Prediction (Random Forest): {ml_best}  "
                     f"(confidence: {ml_proba[ml_best]*100:.1f}%)",
                fg=TEAL if ml_best == best else ORANGE)
        else:
            self.ml_pred_label.config(text="🧠 ML Prediction: Training in background…", fg=SUBTEXT)

        # ── Scores bar chart
        self.ax_pred.clear()
        self.ax_pred.set_facecolor('#0d0d1a')
        self.fig_pred.patch.set_facecolor(BG)
        algos  = list(scores.keys())
        vals   = list(scores.values())
        colors = [ALGO_COLORS.get(a, BLUE) for a in algos]
        bars   = self.ax_pred.bar(algos, vals, color=colors, edgecolor=BG, width=0.5)
        for bar, val, algo in zip(bars, vals, algos):
            self.ax_pred.text(bar.get_x()+bar.get_width()/2,
                              bar.get_height()+0.5, str(val),
                              ha='center', va='bottom', color=TEXT,
                              fontsize=10, fontweight='bold')
            if algo == best:
                bar.set_edgecolor(YELLOW); bar.set_linewidth(3)
                self.ax_pred.text(bar.get_x()+bar.get_width()/2,
                                  bar.get_height()/2, '★ BEST',
                                  ha='center', va='center', color=BG,
                                  fontsize=8, fontweight='bold')
        self.ax_pred.set_title('Algorithm Suitability Scores — Rule-Based',
                               color=PURPLE, fontsize=11, fontweight='bold')
        self.ax_pred.set_ylabel('Score', color=SUBTEXT)
        self.ax_pred.tick_params(colors=TEXT); self.ax_pred.spines[:].set_color(ACCENT)
        self.ax_pred.set_ylim(0, max(vals)+20)
        self.fig_pred.tight_layout()
        self.canvas_pred.draw()

        # ── Rule-by-rule breakdown chart (NEW)
        for ax in self.axes_rules:
            ax.clear(); ax.set_facecolor('#0d0d1a')
        self.fig_rules.patch.set_facecolor(BG)

        for i, algo in enumerate(algos):
            ax = self.axes_rules[i]
            color     = ALGO_COLORS.get(algo, BLUE)
            breakdown = rule_breakdown[algo]
            all_rules = RULE_DEFINITIONS[algo]
            rule_names= [r['name'] for r in all_rules]
            rule_wts  = [r['weight'] if r in breakdown['triggered'] else 0 for r in all_rules]
            max_wts   = [r['weight'] for r in all_rules]
            bar_colors= [color if r in breakdown['triggered'] else '#333355' for r in all_rules]

            y = np.arange(len(rule_names))
            # Background (max possible)
            ax.barh(y, max_wts, height=0.5, color='#1a1a2e', edgecolor='#333366')
            # Earned
            ax.barh(y, rule_wts, height=0.5, color=bar_colors, edgecolor=BG)
            # Labels
            short_names = [n[:18] for n in rule_names]
            ax.set_yticks(y); ax.set_yticklabels(short_names, fontsize=7, color=TEXT)
            ax.set_title(
                f"{algo}\nBase 50 + {scores[algo]-50} = {scores[algo]}",
                color=color, fontsize=8, fontweight='bold', pad=4)
            ax.tick_params(colors=SUBTEXT, labelsize=7)
            ax.spines[:].set_color(ACCENT)
            # Add value labels
            for j, (wt, mwt) in enumerate(zip(rule_wts, max_wts)):
                lbl = f"+{wt}" if wt > 0 else "0"
                ax.text(mwt+0.5, j, lbl, va='center', color=color if wt > 0 else SUBTEXT,
                        fontsize=7, fontweight='bold')

            # Add base score annotation
            ax.set_xlim(0, max(max_wts)+8)
            if algo == best:
                ax.set_facecolor('#1a1a00')
                for spine in ax.spines.values():
                    spine.set_color(YELLOW); spine.set_linewidth(2)

        self.fig_rules.suptitle(
            "Rule-by-Rule Scoring Breakdown  ▸  Each bar = points earned (gray = possible, colored = triggered)",
            color=TEXT, fontsize=9)
        self.fig_rules.tight_layout()
        self.canvas_rules.draw()

        # ── Feature values chart
        self.ax_features.clear()
        self.ax_features.set_facecolor('#0d0d1a')
        self.fig_features.patch.set_facecolor(BG)
        feat_labels = ['Avg Burst', 'Var Burst', 'Range Burst', 'Density', 'Pri Var', 'Uniq Pri', 'N', 'Quantum']
        feat_display= [feat_vals['avg_burst'], feat_vals['var_burst'], feat_vals['range_burst'],
                       feat_vals['density'], feat_vals['pri_var'], feat_vals['unique_pri'],
                       feat_vals['n'], feat_vals['quantum']]
        bars = self.ax_features.bar(feat_labels, feat_display,
                                    color=[BLUE, ORANGE, GREEN, PURPLE, PINK, TEAL, LIME, YELLOW],
                                    edgecolor=BG)
        for bar, val in zip(bars, feat_display):
            self.ax_features.text(bar.get_x()+bar.get_width()/2,
                                  bar.get_height()+0.05, f"{val:.1f}",
                                  ha='center', va='bottom', color=TEXT, fontsize=8)
        self.ax_features.set_title("Your Workload Feature Values (inputs to scoring engine)",
                                   color=LIME, fontsize=10, fontweight='bold')
        self.ax_features.tick_params(colors=TEXT, labelsize=9)
        self.ax_features.spines[:].set_color(ACCENT)
        self.fig_features.tight_layout()
        self.canvas_features.draw()

        # ── Detail cards
        for w in self.pred_cards_frame.winfo_children():
            w.destroy()
        tk.Label(self.pred_cards_frame,
                 text="📖  Why each algorithm scored this way (Rule-Based Breakdown):",
                 font=('Segoe UI', 11, 'bold'), bg=BG, fg=TEXT).pack(anchor='w', pady=(8, 4))

        for algo in algos:
            color   = ALGO_COLORS.get(algo, BLUE)
            is_best = (algo == best)
            card    = tk.Frame(self.pred_cards_frame, bg=CARD,
                               highlightbackground=color if is_best else ACCENT,
                               highlightthickness=2 if is_best else 1)
            card.pack(fill='x', pady=5)
            header = tk.Frame(card, bg=color if is_best else ACCENT)
            header.pack(fill='x')
            earned = scores[algo] - 50  # points earned beyond base
            tk.Label(header,
                     text=f"{'🏆 ' if is_best else ''}  {algo}  —  Score: {scores[algo]}  "
                          f"(Base 50 + {earned} earned){'  ← WINNER' if is_best else ''}",
                     font=('Segoe UI', 10, 'bold'),
                     bg=color if is_best else ACCENT,
                     fg=BG if is_best else TEXT,
                     pady=5, padx=10).pack(anchor='w')
            tk.Label(card, text=reasons[algo]['summary'],
                     font=('Segoe UI', 9, 'italic'), bg=CARD, fg=SUBTEXT,
                     padx=12, pady=4, wraplength=800, justify='left').pack(anchor='w')
            for detail in reasons[algo]['details']:
                row = tk.Frame(card, bg=CARD); row.pack(fill='x', padx=12, pady=1)
                is_pos = detail.startswith('✅')
                dot_col= color if is_pos else SUBTEXT
                tk.Label(row, text="●", font=('Segoe UI', 9), bg=CARD, fg=dot_col).pack(side='left')
                tk.Label(row, text=detail, font=('Segoe UI', 9), bg=CARD,
                         fg=TEXT if is_pos else SUBTEXT,
                         wraplength=780, justify='left').pack(side='left', padx=6)
            tk.Frame(card, bg=CARD, height=6).pack()

        self.notebook.select(4)
        self.status_var.set(f"🔮 Prediction done — {best} is recommended (Rule-Based)")
        self.current_algo.set(best)
        threading.Thread(target=self._sim_thread, args=(best,), daemon=True).start()

    # ── SIMULATION PLAYBACK ───────────────────
    def _play_sim(self):
        if not self.processes:
            messagebox.showwarning("No Processes", "Add processes first"); return
        algo = self.current_algo.get()
        if algo == 'Compare All': algo = 'FCFS'
        q   = self.quantum_var.get()
        tl  = self._get_timeline(algo, q)
        self._sim_timeline = tl
        self._sim_step     = 0
        self._sim_running  = True
        self._sim_paused   = False
        # Update sim metrics chart
        if tl:
            metrics, avg_ct, avg_tat, avg_wt, avg_rt = calculate_metrics(self.processes, tl)
            self.root.after(100, self._draw_sim_avg_chart, metrics, avg_ct, avg_tat, avg_wt, avg_rt)
        self.notebook.select(3)
        self._advance_sim()

    def _stop_sim(self):
        self._sim_running = False
        if self._sim_after:
            self.root.after_cancel(self._sim_after)

    def _step_once(self):
        if not self._sim_timeline:
            self._play_sim()
        else:
            self._sim_paused = True
            self._draw_sim_frame(self._sim_timeline, self._sim_step)
            self._sim_step += 1
            if self._sim_step > len(self._sim_timeline):
                self._sim_step = 0

    def _advance_sim(self):
        if not self._sim_running or self._sim_paused:
            return
        tl   = self._sim_timeline
        step = self._sim_step
        if step > len(tl):
            self._sim_running = False
            return
        self._draw_sim_frame(tl, step)
        self._sim_step += 1
        self._sim_after = self.root.after(self.sim_speed.get(), self._advance_sim)

    def _draw_sim_frame(self, tl, step):
        c  = self.sim_canvas
        c.delete('all')
        W  = c.winfo_width()  or 950
        H  = c.winfo_height() or 380

        if not tl: return

        pids      = list(dict.fromkeys(seg['pid'] for seg in tl))
        color_map = {pid: PROC_COLORS[i % len(PROC_COLORS)] for i, pid in enumerate(pids)}

        current_time = tl[step]['start'] if step < len(tl) else tl[-1]['end']

        # ── Time display
        c.create_text(W//2, 22, text=f"⏱  Time = {current_time}",
                      fill=YELLOW, font=('Segoe UI', 14, 'bold'))

        algo = self.current_algo.get()
        c.create_text(W//2, 42, text=f"Algorithm: {algo}",
                      fill=SUBTEXT, font=('Segoe UI', 9))

        # ─── WAITING QUEUE (left side, vertical)
        wq_x = 40
        wq_title_y = 75
        c.create_text(wq_x + 60, wq_title_y, text="⏳ Waiting\nQueue",
                      fill=ORANGE, font=('Segoe UI', 10, 'bold'), anchor='w')

        # Determine which processes are in waiting queue at this step
        # A process is "waiting" if it has arrived but not yet started its first CPU burst
        first_start = {}
        for seg in tl:
            if seg['pid'] not in first_start:
                first_start[seg['pid']] = seg['start']

        waiting_procs = []
        for p in self.processes:
            pid = p['pid']
            if (p['arrival'] <= current_time and
                    first_start.get(pid, 999999) > current_time):
                waiting_procs.append(p)

        # Also: processes that have been seen before but still have remaining work
        # and are not currently in CPU
        currently_in_cpu = tl[step]['pid'] if step < len(tl) else None
        partially_done   = set()
        for i, seg in enumerate(tl):
            if i < step and seg['pid'] != currently_in_cpu:
                # check if it has remaining segments
                remaining_segs = [s for j, s in enumerate(tl) if j > i and s['pid'] == seg['pid']]
                if remaining_segs and remaining_segs[0]['start'] <= current_time:
                    partially_done.add(seg['pid'])

        for pid in partially_done:
            proc = next((p for p in self.processes if p['pid'] == pid), None)
            if proc and not any(wp['pid'] == pid for wp in waiting_procs):
                waiting_procs.append(proc)

        box_w, box_h = 80, 32
        wq_start_y   = 100
        c.create_rectangle(wq_x, wq_start_y - 8, wq_x + box_w + 10, wq_start_y + len(waiting_procs)*45 + 10,
                           outline=ORANGE, fill='', width=1, dash=(4, 4))

        for i, wp in enumerate(waiting_procs[:6]):
            col = color_map.get(wp['pid'], PROC_COLORS[0])
            by  = wq_start_y + i*42
            c.create_rectangle(wq_x + 5, by, wq_x + box_w + 5, by + box_h,
                                fill=col, outline='white', width=1)
            c.create_text(wq_x + 5 + box_w//2, by + box_h//2,
                          text=f"{wp['pid']}\nb={wp['burst']}",
                          fill='black', font=('Segoe UI', 8, 'bold'), justify='center')

        if not waiting_procs:
            c.create_text(wq_x + 50, wq_start_y + 20, text="(empty)",
                          fill=SUBTEXT, font=('Segoe UI', 9, 'italic'))

        # ─── READY QUEUE (horizontal, middle left → CPU)
        rq_label_x = 200
        rq_label_y = H//2 - 75
        c.create_text(rq_label_x, rq_label_y, text="📥 Ready Queue",
                      fill=GREEN, font=('Segoe UI', 11, 'bold'))

        # Build ready queue: arrived, not finished, not currently in CPU
        finished_pids = set()
        for i in range(len(tl)):
            pid = tl[i]['pid']
            last_idx = max((j for j, s in enumerate(tl) if s['pid'] == pid), default=-1)
            if last_idx < step and last_idx >= 0:
                finished_pids.add(pid)

        queued = []
        seen_in_queue = set()
        for seg in tl[step:]:
            pid = seg['pid']
            if (pid not in finished_pids and
                    pid not in seen_in_queue and
                    pid != currently_in_cpu):
                proc = next((p for p in self.processes if p['pid'] == pid), None)
                if proc and proc['arrival'] <= current_time:
                    queued.append(seg)
                    seen_in_queue.add(pid)

        rq_box_w, rq_box_h = 72, 40
        rq_x = 195
        rq_y = H//2 - 42
        rq_border_w = (rq_box_w + 8) * min(len(queued[:5]), 5) + 20
        c.create_rectangle(rq_x - 5, rq_y - 8, rq_x + rq_border_w, rq_y + rq_box_h + 10,
                           outline=GREEN, fill='', width=1)
        for i, seg in enumerate(queued[:5]):
            col = color_map[seg['pid']]
            bx  = rq_x + i*(rq_box_w + 8)
            c.create_rectangle(bx, rq_y, bx+rq_box_w, rq_y+rq_box_h,
                                fill=col, outline='white', width=1)
            c.create_text(bx+rq_box_w//2, rq_y+rq_box_h//2,
                          text=seg['pid'], fill='black', font=('Segoe UI', 10, 'bold'))

        if not queued:
            c.create_text(rq_x + 80, rq_y + rq_box_h//2, text="(empty)",
                          fill=SUBTEXT, font=('Segoe UI', 9, 'italic'))

        # Arrow from ready queue to CPU
        arr_x1 = rq_x + rq_border_w
        cpu_x  = W//2 - 90
        arr_y  = H//2
        if queued:
            c.create_line(arr_x1 + 5, arr_y, cpu_x - 5, arr_y,
                          fill=GREEN, width=3, arrow=tk.LAST)

        # ── CPU box
        cpu_y, cpu_w, cpu_h = H//2 - 55, 180, 110
        c.create_rectangle(cpu_x, cpu_y, cpu_x+cpu_w, cpu_y+cpu_h,
                            fill=ACCENT, outline=BLUE, width=3)
        c.create_text(cpu_x+cpu_w//2, cpu_y+20, text="⚙ CPU", fill=BLUE,
                      font=('Segoe UI', 13, 'bold'))

        if step < len(tl):
            seg = tl[step]
            col = color_map[seg['pid']]
            c.create_rectangle(cpu_x+15, cpu_y+40, cpu_x+cpu_w-15, cpu_y+cpu_h-15,
                                fill=col, outline='white', width=2)
            c.create_text(cpu_x+cpu_w//2, cpu_y+75,
                          text=f"{seg['pid']}\n[{seg['start']}→{seg['end']}]",
                          fill='black', font=('Segoe UI', 10, 'bold'), justify='center')

            # Step-by-step decision explanation
            if self._step_mode.get():
                decision = self._explain_decision(seg, tl, step, current_time)
                self.step_label.config(text=f"🔍 Decision at t={current_time}: {decision}")
        else:
            c.create_text(cpu_x+cpu_w//2, cpu_y+cpu_h//2,
                          text="✅ DONE", fill=GREEN, font=('Segoe UI', 12, 'bold'))
            self.step_label.config(text="✅ Simulation complete.")

        # Arrow from CPU to completed
        c.create_line(cpu_x+cpu_w+5, H//2, cpu_x+cpu_w+50, H//2,
                      fill=BLUE, width=2, arrow=tk.LAST)
        c.create_text(cpu_x+cpu_w+60, H//2, text="→ Done", fill=BLUE,
                      font=('Segoe UI', 9), anchor='w')

        # ── Completed processes
        done_pids = []
        for pid in pids:
            last_idx = max((i for i, s in enumerate(tl) if s['pid'] == pid), default=-1)
            if last_idx < step and last_idx >= 0:
                done_pids.append(pid)

        c.create_text(W - 130, H//2 - 80, text="✅ Completed:",
                      fill=GREEN, font=('Segoe UI', 10, 'bold'))
        for i, pid in enumerate(done_pids[:4]):
            col = color_map[pid]
            bx  = W - 230 + i*58
            by  = H//2 - 62
            c.create_rectangle(bx, by, bx+50, by+30, fill=col, outline='white')
            c.create_text(bx+25, by+15, text=pid, fill='black',
                          font=('Segoe UI', 9, 'bold'))

        # ── Gantt progress bar at bottom
        max_t  = tl[-1]['end'] if tl else 1
        bar_y  = H - 25
        bar_x0 = 40; bar_x1 = W - 40
        bar_len= bar_x1 - bar_x0
        c.create_rectangle(bar_x0, bar_y-8, bar_x1, bar_y+8,
                            fill=ACCENT, outline=SUBTEXT)
        for seg in tl[:step+1]:
            sx = bar_x0 + (seg['start']/max_t)*bar_len
            ex = bar_x0 + (seg['end']/max_t)*bar_len
            c.create_rectangle(sx, bar_y-7, ex, bar_y+7,
                                fill=color_map[seg['pid']], outline='')
            if (ex-sx) > 14:
                c.create_text((sx+ex)/2, bar_y, text=seg['pid'],
                              fill='black', font=('Segoe UI', 7, 'bold'))
        cx = bar_x0 + (current_time/max_t)*bar_len
        c.create_line(cx, bar_y-12, cx, bar_y+12, fill=YELLOW, width=2)
        c.create_text(cx, bar_y - 15, text=f"t={current_time}",
                      fill=YELLOW, font=('Segoe UI', 7, 'bold'))

    def _explain_decision(self, seg, tl, step, current_time):
        algo = self.current_algo.get()
        pid  = seg['pid']
        proc = next((p for p in self.processes if p['pid'] == pid), None)
        if not proc: return f"Executing {pid}"

        if algo == 'FCFS':
            return f"FCFS: {pid} arrived at t={proc['arrival']}, it arrived earliest among ready processes. Executing until t={seg['end']}."
        elif algo == 'SJF':
            return f"SJF: {pid} has the shortest burst ({proc['burst']} units) among all ready processes. Executing until t={seg['end']}."
        elif algo == 'Round Robin':
            q = self.quantum_var.get()
            return f"Round Robin: {pid} gets its time slice (quantum={q}). It runs from t={seg['start']} to t={seg['end']}."
        elif algo == 'Priority':
            return f"Priority: {pid} has the highest priority ({proc['priority']}, lower=higher). Preempts lower-priority processes."
        elif algo == 'SMART-SJF':
            return f"SMART-SJF: Chose {pid}. Either it's shortest remaining, or it was starving beyond patience threshold."
        return f"Executing {pid} from t={seg['start']} to t={seg['end']}."

    def _draw_sim_avg_chart(self, metrics, avg_ct, avg_tat, avg_wt, avg_rt):
        """Draw average metrics chart below the simulation canvas."""
        self.ax_sim_metrics.clear()
        self.ax_sim_metrics.set_facecolor('#0d0d1a')
        self.fig_sim_metrics.patch.set_facecolor(BG)

        labels = ['Avg Response Time (RT)', 'Avg Completion Time (CT)',
                  'Avg Turnaround Time (TAT)', 'Avg Waiting Time (WT)']
        values = [avg_rt, avg_ct, avg_tat, avg_wt]
        colors = [PURPLE, BLUE, GREEN, ORANGE]

        bars = self.ax_sim_metrics.barh(labels, values, color=colors, edgecolor=BG, height=0.55)
        for bar, val in zip(bars, values):
            self.ax_sim_metrics.text(bar.get_width() + 0.1,
                                     bar.get_y() + bar.get_height()/2,
                                     f"{val:.2f}", va='center', color=TEXT,
                                     fontsize=10, fontweight='bold')

        self.ax_sim_metrics.set_title(
            f"Average Metrics — RT={avg_rt:.2f}  CT={avg_ct:.2f}  TAT={avg_tat:.2f}  WT={avg_wt:.2f}",
            color=TEAL, fontsize=10, fontweight='bold')
        self.ax_sim_metrics.tick_params(colors=TEXT, labelsize=9)
        self.ax_sim_metrics.spines[:].set_color(ACCENT)
        self.ax_sim_metrics.set_xlim(0, max(values) * 1.2 + 1)
        self.fig_sim_metrics.tight_layout()
        self.canvas_sim_metrics.draw()


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == '__main__':
    root = tk.Tk()
    app  = CPUSchedulerApp(root)
    root.mainloop()
