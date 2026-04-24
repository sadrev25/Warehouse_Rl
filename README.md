# Warehouse_RL
# PPO-Based Antagonistic Robot in Cooperative Warehouse Swarm

## Overview

This repository implements a **PPO-based adversarial agent** that disrupts cooperative warehouse robot swarms by manipulating inter-robot communication broadcasts — without any physical collision or detectable anomaly.

**Master's Thesis** — University of Stuttgart, 2026  
**Program** — Erasmus Mundus MIR (Marine & Maritime Intelligent Robotics)  
**Author** — Mukesh Sadasivam

---

## Research Question

> Can a single compromised robot reduce cooperative warehouse swarm throughput by manipulating the communication layer — without any physical interference?

**Answer: Yes — up to 94% throughput reduction with a single antagonist robot.**

---

## Key Results

| Experiment | Attack Type | Mean Reward | Best Episode | Throughput Drop |
|---|---|---|---|---|
| Baseline | No antagonist | — | 13/24 tasks, 237s | 0% |
| Exp 1 — Hotspot | load + priority at key nodes | 0.023 | reward=0.055 | 38% mean |
| Exp 2 — Every waypoint | load + priority everywhere | 0.023 | 1/24 tasks! | 94% best |
| Exp 3 — time_offset only | timing manipulation only | 0.008 | reward=0.019 | ~15% |
| Exp 4 — Combined (WIP) | all three actions | TBD | TBD | TBD |

---

## How the Attack WorksNormal cooperative robots:
Broadcast real state every 1.2s
are_next_crossings_available() checks:
1. Is another robot heading to same node?
2. Within 6s crossing buffer window?
3. Who has load/priority? → that robot passesAntagonist manipulation:
fake_priority    → claims high task priority
fake_has_load    → claims to carry load
fake_time_offset → claims earlier arrival timeEffect:
Other robots see antagonist as high priority
They yield unnecessarily at every crossing
Throughput drops significantly
No collision — completely undetectable!
