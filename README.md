# Mechatronics, Control & Robotics Portfolio
**Danial Heidar** — B.Sc. Student in Mechanical Engineering, University of Tehran

This repository organizes my coursework and personal programming projects in **Control Systems**, **Robot Kinematics**, and **AI/Search Algorithms**.

---

## 1. Rotary Inverted Pendulum — Dynamic Modeling & Control
* **Project Report:** [`Control_Project_Report.pdf`](./Control_Project_Report.pdf) *(Group project; my primary role was dynamic modeling, stability analysis, and controller design)*
* **Overview:**
  * Derived nonlinear Euler-Lagrange equations of motion and linearized the system around the unstable upright equilibrium.
  * Designed and evaluated **P, PI, PD, PID** controllers and a **Lead Compensator** in **MATLAB** using Root Locus, Bode, and Nyquist stability analyses.
  * Collaborated with teammates responsible for SolidWorks CAD modeling and Proteus/Arduino hardware simulation.

---

## 2. Humanoid Robot Parallel Wrist Kinematics (Surena V)
* **Contribution Commit:** [surenaV_core — commit `a5a7379`](https://github.com/danialheidar/surenaV_core/commit/a5a73798365c1ac2ada1f31bd9f1f97b267616d5)
* **Overview:**
  * Implemented an updated geometric/analytical inverse kinematics solution for the Surena V parallel wrist mechanism.
  * Added boundary handling for **out-of-workspace target ranges** and selection logic to pick the optimal solution among **multiple analytical branches**.

---

## 3. AI & Algorithmic Programming Projects (Python)
* **Constraint-Propagation Sudoku Solver** ([`./3-sodoku ai`](./3-sodoku%20ai)): Combines **Naked/Hidden Singles** inference over a `9×9×9` candidate matrix with **Minimum Remaining Values (MRV)** heuristic branching and iterative stack-based backtracking, wrapped in a **Pygame** GUI.
* **AlphaZero Practice Implementations:** Self-play agents integrating Monte Carlo Tree Search (MCTS) with neural networks:
  * **Backgammon** ([`./4-backgammon`](./4-backgammon)) — adapted to stochastic dice transitions.
  * **Fading Tic-Tac-Toe** ([`./2-fading_tictactoe`](./2-fading_tictactoe))
  * **Tic-Tac-Toe** ([`./1-TIC TAC TOE`](./1-TIC%20TAC%20TOE))
* **Neural Pong** ([`./0-neural pong`](./0-neural%20pong)): Real-time paddle control agent trained via neural networks.
