# Fading-Move Tic-Tac-Toe — AlphaZero Agent

An AlphaZero-style reinforcement learning agent (PUCT MCTS + a
policy/value residual CNN, trained via self-play) for **Fading-Move
Tic-Tac-Toe**: standard 3x3 Tic-Tac-Toe, except each player may only
have 3 active marks at once — placing a 4th mark fades that player's
oldest mark.

## Project layout

```
fading_tictactoe/
├── game.py        # Rules engine (no ML dependencies)
├── symmetry.py     # D4 board-symmetry augmentation utilities
├── network.py      # PyTorch policy/value residual CNN
├── mcts.py         # AlphaZero-style PUCT Monte Carlo Tree Search
├── self_play.py    # Self-play data generation + replay buffer
├── train.py        # Checkpointing + iterative-improvement training loop
├── evaluate.py      # Random / Network / depth-limited-minimax agents + match runner
├── play.py         # Human-vs-agent command-line interface
├── config.py       # All hyperparameters, in one dataclass
├── tests/
│   └── test_game.py
└── README.md
```

## Design notes

**State representation.** A flat 3x3 grid of {X, O, empty} loses the
information needed to predict fading, so the state is instead encoded
as a `(7, 3, 3)` tensor:

* Planes 0–2: player X's active marks, indexed by **recency rank**
  (plane 0 = X's newest mark, plane 2 = X's oldest — i.e. the one that
  fades on X's *next* move, if X currently has 3). A plane is all-zero
  if no mark currently holds that rank.
* Planes 3–5: the same, for player O.
* Plane 6: constant plane, 1.0 if X is to move, 0.0 if O is to move.

This is a bijection with the engine's own canonical state signature
(`(history_X, history_O, current_player)`), so it is Markovian and
loses no information relevant to fading, winning, or repetition.

**Repetition / cycle handling.** Because marks disappear, the same
board can recur. The engine hashes the full canonical state signature
after every move and declares a draw once a state has been seen
`config.repetition_draw_count` times (default 3); a hard ply cap
(`max_game_length`) is kept only as a defensive fallback that should
never actually trigger. Note that with at most 3 active marks per
side, at least 3 of the 9 squares are always empty, so "no legal
moves" can never occur as a terminal condition in this variant —
draws only arise from repetition.

**Symmetry augmentation.** Every plane in the state tensor is purely
spatial (which *square* holds "X's newest mark", etc.), so any of the
8 dihedral (rotation/reflection) transforms can be applied directly to
the last two axes of the tensor, and identically to the 9-way policy
vector reshaped to `3x3`, without corrupting move-age semantics — a
transform only changes *where* a mark of a given age is, never *which*
age-rank plane it lives in.

**MCTS value perspective.** Every search node stores a value from the
perspective of the player to move *at that node*. Since
`FadingTicTacToe.make_move` always switches `current_player` — even on
a game-ending move — the mover alternates strictly at every ply, so a
single sign flip per tree level during backup is always correct. See
the extensive comments in `mcts.py` for the one subtlety that is easy
to get backwards: PUCT selection must negate a child's average value
before comparing it against `c_puct * P * sqrt(N) / (1+n)`, because
the child's stored value is from the *opponent's* perspective, not the
parent's.

**Minimax reference agent.** Because the game can cycle indefinitely,
`evaluate.py`'s `MinimaxAgent` is a depth-limited, alpha-beta-pruned
search with a simple line-counting heuristic at the frontier — a
useful fixed-strength baseline, but explicitly **not** a perfect
solver. The AlphaZero agent itself never uses or depends on it.

## Installation

```bash
cd fading_tictactoe
pip install -r requirements.txt
```

Requires Python 3.9+, PyTorch, and NumPy.

## Running unit tests

```bash
python -m unittest tests.test_game -v
```

(No PyTorch dependency is needed to run these — they only exercise
`game.py`.)

## Training

```bash
python train.py --iterations 20
```

This runs the iterative-improvement loop: self-play with the current
best network → train a candidate on the replay buffer → evaluate
candidate vs. best → promote if it wins enough decisive games.
Checkpoints are written to `checkpoints/best.pt` (always the current
best) and `checkpoints/iter_XXXX.pt` (one per iteration).

To resume:

```bash
python train.py --resume checkpoints/best.pt --iterations 10
```

Useful overrides: `--games-per-iter`, `--simulations`, `--seed`,
`--checkpoint-dir`. All other hyperparameters live in `config.py`.

## Evaluating

```bash
python evaluate.py --checkpoint checkpoints/best.pt --opponent random --games 100
python evaluate.py --checkpoint checkpoints/best.pt --opponent minimax --minimax-depth 6 --games 40
python evaluate.py --checkpoint checkpoints/best.pt --opponent checkpoint \
    --opponent-checkpoint checkpoints/iter_0001.pt --games 40
```

Reports wins / losses / draws and win rate.

## Playing against the trained agent

```bash
python play.py --checkpoint checkpoints/best.pt --human-plays X --simulations 400
```

The board is redrawn after every move, and a line is printed whenever
a player's next move is about to fade one of their marks (and again
right after a fade happens).

## Suggested starting hyperparameters

**Laptop / CPU** (defaults in `config.py` are already tuned for this):

```python
num_res_blocks = 4
num_filters = 64
num_simulations = 200
num_self_play_games = 200   # per iteration
batch_size = 256
num_iterations = 20
eval_games = 40
```

A full run of ~20 iterations at these settings should take from tens
of minutes to a few hours on a modern laptop CPU, depending on core
count — this is a tiny 3x3 game, so it is far cheaper than typical
AlphaZero applications.

**With a CUDA GPU available** (`config.device = "cuda"`, auto-detected
by `train.py`/`play.py`/`evaluate.py`):

```python
num_res_blocks = 6
num_filters = 128
num_simulations = 400
num_self_play_games = 500
batch_size = 1024
num_iterations = 50
eval_games = 100
eval_num_simulations = 200
```

GPU throughput mostly helps by letting you afford far more self-play
games and simulations per move per unit time, not by needing a bigger
network — this game is small enough that even the CPU-sized network
above is already generous.

## Increasing playing strength

* **More simulations per move** (`num_simulations`, `eval_num_simulations`,
  and `--simulations` at play time) is the single most direct lever —
  deeper search finds tactics a shallow search misses, independent of
  network quality.
* **More self-play games per iteration** (`num_self_play_games`) and
  **more iterations** (`num_iterations`) give the network more, and
  more diverse, high-quality training signal.
* **Raise the acceptance bar** (`acceptance_win_rate`,
  `min_decisive_games`) so weaker candidates are rejected more often,
  at the cost of slower (but more reliable) progress.
* **Larger network** (`num_filters`, `num_res_blocks`) if you have the
  compute to back it — this game is small enough that the default
  network is unlikely to be a bottleneck until quite late in training.
* **Longer/annealed temperature schedule** (`temperature_moves`,
  `temperature_early`, `temperature_late`) trades early-game
  exploration for early-game exploitation; too little exploration can
  cause the agent to converge on a narrow, exploitable strategy.
* **Symmetry augmentation** (`use_symmetry_augmentation = True`,
  already the default) effectively multiplies your training data by 8
  at no extra self-play cost — leave it on unless you have a specific
  reason to disable it.
