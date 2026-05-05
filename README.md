<div align="center">

```
╔═══════════════════════════════════════════════════════════════╗
║         CPU SCHEDULING VISUALIZER WITH PREDICTION MODE        ║
╚═══════════════════════════════════════════════════════════════╝
```

<img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Tkinter-GUI-FF6F00?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/Matplotlib-Charts-11557C?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge"/>

<br/>

> **A feature-rich, interactive CPU scheduling simulator with ML-based algorithm prediction, animated step-by-step execution, and an integrated OS chatbot.**

<br/>

*Marzana Tabassum · 2232327642 · CSE323, Section 2*  
*North South University · Spring 2026*

</div>

---

## 📌 Table of Contents

- [📸 Demo](#-demo)
- [✨ Overview](#-overview)
- [🚀 Features](#-features)
- [🧠 Algorithms](#-algorithms)
- [🤖 ML Prediction Engine](#-ml-prediction-engine)
- [🎬 Animated Simulator](#-animated-simulator)
- [📊 Analytics Dashboard](#-analytics-dashboard)
- [🛠️ Installation](#️-installation)
- [▶️ How to Run](#️-how-to-run)
- [📸 Screenshots](#-screenshots)
- [🏗️ Architecture](#️-architecture)
- [🔬 Challenges & Solutions](#-challenges--solutions)
- [📄 License](#-license)
  ## 📸 Demo

<div align="center">

[![CPU Scheduling Visualizer Demo](https://img.youtube.com/vi/TKIO9v6sjeE/maxresdefault.jpg)](https://www.youtube.com/watch?v=TKIO9v6sjeE)

*Click to watch the full demo on YouTube*

</div>

---

## ✨ Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Standard scheduling tools show a Gantt chart and stop.     │
│  This project goes further — it teaches, predicts, and      │
│  animates every scheduling decision in real time.           │
└─────────────────────────────────────────────────────────────┘
```

The **CPU Scheduling Visualizer with Prediction Mode** is not just a visualizer — it is a **learning and decision-support environment**. A student or system designer can:

- Input any real or hypothetical process set
- Run any of **5 scheduling algorithms** (including a custom hybrid)
- Compare all algorithms **side-by-side**
- Watch a **step-by-step animated execution**
- Ask the **built-in ML model** which algorithm would perform best
- Use the **OS Chatbot** to ask natural-language questions about scheduling concepts

---

## 🚀 Features

| Feature | Description |
|---|---|
| 🧮 **5 Scheduling Algorithms** | FCFS, SJF, Round Robin, Preemptive Priority, SMART-SJF |
| 🎨 **Gantt Chart** | Color-coded, scrollable, zoomable real-time chart |
| 🤖 **ML Prediction** | Random Forest trained on live simulation outputs |
| 🎬 **Animated Simulation** | Frame-by-frame CPU state, pause/resume/restart |
| 📊 **Analytics Dashboard** | Bar & pie charts for metrics comparison |
| 🔄 **Compare All Mode** | All 5 algorithms run on same input, side-by-side table |
| 💉 **Live Process Injection** | Add processes to a **running** simulation |
| ⚙️ **Auto-Tune** | Suggests optimal quantum size for Round Robin |
| ⚠️ **Starvation Detection** | Real-time warnings with configurable threshold |
| 💬 **OS Chatbot** | Natural-language Q&A about scheduling concepts |
| 🖱️ **Drag-to-Reorder** | Drag processes in the priority queue |

---

## 🧠 Algorithms

```
┌──────────────────────────────────────────────────────────────────┐
│  FCFS  ──  First Come First Served                               │
│  SJF   ──  Shortest Job First (Non-Preemptive)                   │
│  RR    ──  Round Robin                                           │
│  PP    ──  Preemptive Priority                                   │
│  ★ SMART-SJF  ──  Novel Hybrid (SJF + Aging Anti-Starvation)    │
└──────────────────────────────────────────────────────────────────┘
```

### ⭐ SMART-SJF — The Custom Algorithm

SMART-SJF is the project's **original contribution** — a hybrid scheduling policy designed from scratch. It solves the classic SJF starvation problem using an **aging mechanism**:

```
Effective Priority Score = Burst ÷ (1 + waiting_time × 0.1)
```

- The **longer** a process waits → the **lower** its score → the **sooner** it runs
- No process is ever ignored indefinitely
- Short jobs still get preference — fairness and efficiency coexist

**Adaptive parameters auto-tune to the workload:**
```
PATIENCE  = max(3, avg_burst × 1.5)   → starvation threshold
fair_q    = max(2, avg_burst × 0.6)   → CPU slice for starving processes
```

---

## 🤖 ML Prediction Engine

```
Input Workload
      │
      ▼
 8 Features Extracted
 (avg burst, variance, arrival density, priority spread, ...)
      │
      ▼
 Random Forest Classifier  (60 estimators, trained on 1200 samples)
      │
      ▼
 Best Algorithm Predicted  +  Rule-by-rule explanation shown in UI
```

- Training data is **generated from live simulation outputs** — not a static dataset
- Achieves **~78% accuracy** on held-out test split
- Runs in a **background daemon thread** so the GUI never freezes
- Returns a **full breakdown**: which rules fired, how many points each added, and why

---

## 🎬 Animated Simulator

The simulator converts a static schedule timeline into a **live, interactive animation**:

```
▶ Play → Pause → Resume → Step-by-Step → Restart
         ↕
   Speed Control (slow down / speed up)
```

At every frame, the canvas redraws:
- 🟦 **Waiting Queue** — processes not yet ready
- 🟩 **Ready Queue** — processes waiting for CPU
- 🟥 **CPU** — currently running process
- 📈 **Gantt Chart** — growing one column at a time

In **step-by-step mode**, each advance also generates a plain-language explanation of *why* the current process was chosen — turning the simulator into an **interactive teaching tool**.

---

## 📊 Analytics Dashboard

Embedded `matplotlib` charts rendered inside Tkinter:

- 📊 Bar charts — Waiting Time, Turnaround Time, CPU Utilization per algorithm
- 🥧 Pie charts — CPU share breakdown per process
- 📋 Compare All table — all 5 algorithms × all metrics, side by side

---

## 🛠️ Installation

### Prerequisites

```bash
Python 3.8+
```

### Install Dependencies

```bash
pip install scikit-learn matplotlib numpy
```

> Tkinter comes pre-installed with Python on most systems. If missing:
> ```bash
> sudo apt-get install python3-tk   # Linux
> ```

### Clone the Repository

```bash
git clone https://github.com/Marzana-tabassum/CPU-Scheduling-Visualizer.git
cd CPU-Scheduling-Visualizer
```

---

## ▶️ How to Run

```bash
python scheduler_v3_ultimate_(1).py
```

The GUI will launch. The ML model trains automatically in the background at startup.

### Quick Start Guide

```
1. Enter process details (PID, Arrival, Burst, Priority)
2. Click [Add] → process appears in the queue
3. Select an algorithm from the sidebar
4. Click [Run Simulation] → Gantt chart renders live
5. Click [Predict Best Algorithm] → ML recommends optimal choice
6. Click [Compare All] → see all 5 algorithms side by side
7. Click [Animation] → watch step-by-step execution
```

---

## 📸 Screenshots


<div align="center">

<img width="800" alt="CPU Scheduling Visualizer" src="https://github.com/user-attachments/assets/fa609897-abea-4637-9953-e3a0ce95cecb" />

<br/><br/>

<img width="800" alt="Gantt Chart & Metrics" src="https://github.com/user-attachments/assets/76444f3b-9a14-486c-af5c-e5cf53fc2db0" />

<br/><br/>

<img width="800" alt="ML Prediction Engine" src="https://github.com/user-attachments/assets/bfa7d49e-e441-405b-ab42-1a5dc1baf494" />

<br/><br/>

<img width="800" alt="Analytics Dashboard" src="https://github.com/user-attachments/assets/f1c90ed8-f328-4fa3-a4f7-27bda22c5310" />

</div>
---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INPUT LAYER                         │
│   Process Queue    │   Algorithm Selector   │   Settings        │
└────────────────────────────┬────────────────────────────────────┘
                             │ process list + config
┌────────────────────────────▼────────────────────────────────────┐
│                         CORE ENGINE                             │
│                                                                 │
│   Scheduler Engine              ML Prediction Engine            │
│   ├── FCFS                      ├── Generate 1200 samples       │
│   ├── SJF                       ├── Score all 5 algorithms      │
│   ├── Round Robin               ├── Train Random Forest         │
│   ├── Preemptive Priority       └── Predict best algorithm      │
│   └── SMART-SJF                                                 │
└────────────────────────────┬────────────────────────────────────┘
                             │ schedule + metrics + prediction
┌────────────────────────────▼────────────────────────────────────┐
│                       OUTPUT MODULES                            │
│  Gantt  │  Metrics  │  Animator  │  Compare All  │  Chatbot     │
└─────────────────────────────────────────────────────────────────┘
```

### Module Overview

| Module | Responsibility |
|---|---|
| **Process Manager** | Input, editing, drag-to-reorder, live injection |
| **Scheduler Engine** | Pure implementations of all 5 algorithms |
| **Gantt Renderer** | Color-coded charts with zoom & scroll |
| **Metrics Calculator** | WT, TAT, Response Time, Utilization, Throughput |
| **Animated Simulator** | Frame-by-frame animation engine |
| **ML Prediction Engine** | Random Forest trained on simulation outputs |
| **Analytics Dashboard** | Embedded matplotlib bar/pie charts |
| **Compare All Module** | Side-by-side metrics for all algorithms |
| **OS Chatbot** | Keyword-based scheduling Q&A engine |

---

## 🔬 Challenges & Solutions

### Challenge 1 — Designing SMART-SJF

> *SJF is optimal for average waiting time but starves long processes. Round Robin is fair but ignores burst length entirely.*

**Solution:** Built `smart_sjf()` with adaptive aging. A `last_served` dictionary tracks each process's wait time. When any process exceeds `PATIENCE`, it gets a fair CPU slice before normal SJF resumes. Parameters auto-scale to the workload — no manual tuning needed.

---

### Challenge 2 — ML Training Without Ground Truth Labels

> *What does "best algorithm" even mean? Average WT? Throughput? Fairness? Each metric tells a different story.*

**Solution:** Designed a `RULE_DEFINITIONS` dictionary that evaluates each algorithm against 8 workload features (burst variance, arrival density, priority spread, process count, etc.) and assigns points. This rule scorer generated labels for 1200 synthetic process sets, which trained the Random Forest. The model achieves ~78% accuracy and runs in a background thread.

---

### Challenge 3 — Animated Simulation Without Freezing the GUI

> *Tkinter's event model is incompatible with a naive animation loop — `sleep()` inside a loop freezes the entire window.*

**Solution:** The full timeline is computed once upfront. Each frame schedules the next using `root.after()` — delegating timing entirely to Tkinter's event loop. Pause cancels the next scheduled frame; resume reschedules from the same step counter. The GUI stays fully responsive throughout.

---

## 📄 License

This project was developed as an academic submission for **CSE323 — Operating Systems Lab**  
at **North South University, Spring 2026**.

---

<div align="center">

**Built with 💙 by Marzana Tabassum**  
*North South University · 2232327642*

```
"Not just a visualizer — a learning and decision-support environment."
```

</div>
