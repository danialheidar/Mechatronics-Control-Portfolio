import time
def is_complete(sudoku):
    """Check if the board is fully filled (no zeros)."""
    for i in range(9):
        for j in range(9):
            if sudoku[i][j] == 0:
                return False
    return True


def build_table(sudoku):
    """
    Build a 3D 'forbidden' table.
    table[value][row][col] == 1 means that (value+1) cannot be placed in that cell.
    """
    table = [[[0] * 9 for _ in range(9)] for _ in range(9)]

    for r in range(9):
        for c in range(9):
            val = sudoku[r][c]
            if val == 0:
                continue

            v = val - 1  # zero‑based index for the value

            # A filled cell forbids every value
            for x in range(9):
                table[x][r][c] = 1

            # Same row, column and box cannot have the same value
            for k in range(9):
                table[v][k][c] = 1
                table[v][r][k] = 1

            br, bc = (r // 3) * 3, (c // 3) * 3
            for rr in range(br, br + 3):
                for cc in range(bc, bc + 3):
                    table[v][rr][cc] = 1

            # The cell itself is allowed to contain its own value
            table[v][r][c] = 0

    return table


def print_sudoku(sudoku):
    """Pretty‑print the board with 3x3 box separators."""
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("-" * 21)
        row_str = ""
        for j in range(9):
            if j % 3 == 0 and j != 0:
                row_str += " | "
            row_str += str(sudoku[i][j]) + " "
        print(row_str.strip())


def propagate(sudoku):
    """
    Apply simple logical rules (hidden/naked singles) as long as possible.
    Returns:
        'solved'          – board is completely filled.
        'contradiction'   – an empty cell has no valid candidates.
        'stuck'           – no progress made, board is not solved.
    """
    while True:
        if is_complete(sudoku):
            return 'solved'

        table = build_table(sudoku)

        # Check for contradiction: an empty cell with zero candidates
        for r in range(9):
            for c in range(9):
                if sudoku[r][c] == 0:
                    if all(table[v][r][c] == 1 for v in range(9)):
                        return 'contradiction'

        changed = False

        # Hidden single in a row
        for v in range(9):
            for r in range(9):
                possible = [c for c in range(9) if sudoku[r][c] == 0 and table[v][r][c] == 0]
                if len(possible) == 1:
                    c = possible[0]
                    sudoku[r][c] = v + 1
                    changed = True
                    break
            if changed:
                break
        if changed:
            continue

        # Hidden single in a column
        for v in range(9):
            for c in range(9):
                possible = [r for r in range(9) if sudoku[r][c] == 0 and table[v][r][c] == 0]
                if len(possible) == 1:
                    r = possible[0]
                    sudoku[r][c] = v + 1
                    changed = True
                    break
            if changed:
                break
        if changed:
            continue

        # Hidden single in a 3x3 box
        for v in range(9):
            for br in range(3):
                for bc in range(3):
                    possible = []
                    for rr in range(br * 3, br * 3 + 3):
                        for cc in range(bc * 3, bc * 3 + 3):
                            if sudoku[rr][cc] == 0 and table[v][rr][cc] == 0:
                                possible.append((rr, cc))
                    if len(possible) == 1:
                        rr, cc = possible[0]
                        sudoku[rr][cc] = v + 1
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break
        if changed:
            continue

        # Naked single in a cell
        for r in range(9):
            for c in range(9):
                if sudoku[r][c] != 0:
                    continue
                possible = [v for v in range(9) if table[v][r][c] == 0]
                if len(possible) == 1:
                    sudoku[r][c] = possible[0] + 1
                    changed = True
                    break
            if changed:
                break
        if changed:
            continue

        # No changes made – stuck
        return 'stuck'


def find_best_guess(sudoku):
    """
    Find the empty cell with the fewest possible candidates.
    Returns (row, col, candidates) or None if the board is complete.
    """
    table = build_table(sudoku)
    best_r, best_c = -1, -1
    best_candidates = None
    best_count = 10  # more than 9

    for r in range(9):
        for c in range(9):
            if sudoku[r][c] == 0:
                candidates = [v + 1 for v in range(9) if table[v][r][c] == 0]
                if len(candidates) < best_count:
                    best_count = len(candidates)
                    best_r, best_c = r, c
                    best_candidates = candidates
                    if best_count == 1:  # cannot get better than 1
                        return best_r, best_c, best_candidates

    if best_r == -1:
        return None  # board is complete
    return best_r, best_c, best_candidates

t1 = time.time()
# ------------------- Initial puzzle -------------------
sudoku = [
    [0, 0, 0, 2, 0, 0, 0, 0, 0],
    [8, 0, 9, 0, 0, 0, 1, 0, 0],
    [0, 2, 0, 0, 0, 0, 0, 0, 0],
    [6, 0, 3, 0, 0, 9, 0, 0, 0],
    [0, 7, 0, 6, 0, 0, 0, 5, 0],
    [0, 0, 0, 0, 4, 0, 9, 0, 3],
    [0, 4, 0, 0, 8, 0, 0, 3, 0],
    [0, 0, 6, 9, 0, 0, 0, 7, 0],
    [0, 9, 5, 0, 0, 1, 0, 0, 2]
]

# Stack for guesses: each element is (row, col, candidates, next_index, board_snapshot)
stack = []

while True:
    status = propagate(sudoku)

    if status == 'solved':
        print("✅ Puzzle solved successfully!")
        break

    if status == 'contradiction':
        print("⚠️ Contradiction reached – backtracking to the last guess...")
        backtrack_success = False
        while stack:
            row, col, candidates, idx, snapshot = stack.pop()
            # Restore board to the state before this guess
            for r in range(9):
                for c in range(9):
                    sudoku[r][c] = snapshot[r][c]

            # Try the next candidate
            while idx < len(candidates):
                guess = candidates[idx]
                sudoku[row][col] = guess
                # Push the new state with the next index for future backtracking
                stack.append((row, col, candidates, idx + 1, [row[:] for row in sudoku]))
                print(f"🔁 New guess at ({row+1},{col+1}) = {guess} (candidate {idx+1} of {len(candidates)})")
                backtrack_success = True
                break

            if backtrack_success:
                break
            # If all candidates for this guess are exhausted, pop the previous one

        if not backtrack_success:
            print("❌ No more guesses left – puzzle has no solution.")
            break

        # After finding a new guess, continue with propagation
        continue

    # If we are 'stuck', we need to make a new guess
    guess_info = find_best_guess(sudoku)
    if guess_info is None:
        # Should not happen if propagate said 'stuck' but board not complete
        print("The board is complete.")
        break

    row, col, candidates = guess_info
    if not candidates:
        print("❌ Empty cell with no candidates – contradiction in previous guess.")
        continue

    # Take the first candidate
    guess = candidates[0]
    snapshot = [row[:] for row in sudoku]   # save current board
    sudoku[row][col] = guess
    # Push with next_index = 1 (so we can try later candidates if needed)
    stack.append((row, col, candidates, 1, snapshot))
    print(f"🎯 Guessing at ({row+1},{col+1}) = {guess} (out of {len(candidates)} candidates)")

# Show final result
print("\nSolved board:")
print_sudoku(sudoku)
t2 = time.time()
print(t2 - t1)