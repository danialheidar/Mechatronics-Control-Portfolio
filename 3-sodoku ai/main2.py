def is_complete(sudoku):
    for i in range(9):
        for j in range(9):
            if sudoku[i][j] == 0:
                return False
    return True


def build_table(sudoku):
    # table[value][row][col] == 1 means value+1 cannot go in that cell
    table = [[[0] * 9 for _ in range(9)] for _ in range(9)]

    for r in range(9):
        for c in range(9):
            val = sudoku[r][c]
            if val == 0:
                continue

            v = val - 1

            # Already filled cells are impossible for every value
            for x in range(9):
                table[x][r][c] = 1

            # Same row, column, and box cannot contain the same value
            for k in range(9):
                table[v][k][c] = 1
                table[v][r][k] = 1

            br = (r // 3) * 3
            bc = (c // 3) * 3

            for rr in range(br, br + 3):
                for cc in range(bc, bc + 3):
                    table[v][rr][cc] = 1

            # The cell itself is allowed to contain its own value
            table[v][r][c] = 0

    return table


# ---------- backtracking helper ----------
def is_valid(sudoku, row, col, num):
    """Check if placing num at (row, col) is valid."""
    # Row check
    for c in range(9):
        if sudoku[row][c] == num:
            return False
    # Column check
    for r in range(9):
        if sudoku[r][col] == num:
            return False
    # Box check
    br, bc = (row // 3) * 3, (col // 3) * 3
    for rr in range(br, br + 3):
        for cc in range(bc, bc + 3):
            if sudoku[rr][cc] == num:
                return False
    return True


def solve_backtracking(sudoku):
    """Recursive DFS that modifies the board in place.
       Returns True if solved, False if no solution exists."""
    for r in range(9):
        for c in range(9):
            if sudoku[r][c] == 0:
                # Try numbers 1..9
                for num in range(1, 10):
                    if is_valid(sudoku, r, c, num):
                        sudoku[r][c] = num
                        if solve_backtracking(sudoku):
                            return True
                        sudoku[r][c] = 0  # backtrack
                return False  # no number works
    return True  # all cells filled


# ---------- pretty print function ----------
def print_sudoku(sudoku):
    """Print the Sudoku board with 3x3 box separators."""
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("-" * 21)  # 3*7 dashes? Actually we need 3 groups of 3 numbers, each separated by " | "
        row_str = ""
        for j in range(9):
            if j % 3 == 0 and j != 0:
                row_str += " | "
            row_str += str(sudoku[i][j]) + " "
        print(row_str.strip())


# ---------- main solver ----------
sudoku = [
    [8, 9, 1, 0, 0, 6, 0, 7, 0],
    [3, 4, 0, 0, 7, 0, 0, 0, 6],
    [7, 0, 6, 0, 4, 0, 0, 0, 0],
    [4, 7, 0, 1, 3, 0, 0, 0, 5],
    [1, 6, 0, 5, 0, 8, 0, 3, 0],
    [5, 3, 8, 7, 6, 4, 2, 1, 9],
    [0, 0, 0, 4, 0, 7, 5, 0, 0],
    [0, 8, 0, 0, 0, 0, 0, 4, 2],
    [0, 0, 4, 6, 0, 0, 0, 0, 0]
]

# ---- constraint propagation ----
while not is_complete(sudoku):
    table = build_table(sudoku)
    changed = False

    # Hidden single in a row
    for v in range(9):
        for r in range(9):
            possible = [
                c for c in range(9)
                if sudoku[r][c] == 0 and table[v][r][c] == 0
            ]
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
            possible = [
                r for r in range(9)
                if sudoku[r][c] == 0 and table[v][r][c] == 0
            ]
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

    # Stuck – backtracking will take over
    break

# ---- if still not complete, use backtracking ----
if not is_complete(sudoku):
    print("Logic stuck – using backtracking...")
    if solve_backtracking(sudoku):
        print("Solution found via backtracking.")
    else:
        print("No solution exists!")

# ---- print final board in the desired format ----
print("\nSolved puzzle:")
print_sudoku(sudoku)