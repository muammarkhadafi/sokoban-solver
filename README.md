# Sokoban Solver

Push-optimal Sokoban solver for make.fun's sokoban-solver-01-run-20261121.

## How it works

A level is a hallway with 1-4 rooms off it (3-4 boxes each). A box pushed out of a
room into the hallway can never come back, so every room is an independent problem:
the solver finds the rooms, solves each one on its own with A* over pushes, walks
the hallway between rooms, and the push counts add up to the level's push optimum.

Per room, the search is A* over pushes (a state is the boxes' squares plus the
player's reachable region, named by its lowest square) with:

- an admissible Hungarian heuristic (minimum box-goal matching over walk distances),
  so the first solved state popped is push-optimal for that room;
- dead-square pruning (a box that can never reach a goal is never pushed there);
- frozen 2x2 pruning (a block of walls and boxes with a box off its goal is dead);
- a greedy best-first retry if A* ever exhausts its expansion budget.

Determinism: every budget is a count of work (state expansions), never the clock.
No `Date`, no `Math.random`, no environment reads. Two runs write identical bytes.

## Run locally

```
MAKEFUN_INPUT_DIR=<kit>/practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Scores 1,000,000 on the practice set (every instance solved in exactly P* pushes)
in about 0.4 seconds, and 1,000,000 on 20 freshly generated sets (100 instances).
The house baseline plain-astar 1.0.0 scores 200,000 on the practice set.

## License

MIT (see LICENSE).
