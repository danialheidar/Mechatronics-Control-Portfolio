import pygame
import sys
import time

# ======================================================================
#  SUDOKU SOLVER (copied from your code)
# ======================================================================

def is_complete(sudoku):
    for i in range(9):
        for j in range(9):
            if sudoku[i][j] == 0:
                return False
    return True

def build_table(sudoku):
    table = [[[0] * 9 for _ in range(9)] for _ in range(9)]
    for r in range(9):
        for c in range(9):
            val = sudoku[r][c]
            if val == 0:
                continue
            v = val - 1
            for x in range(9):
                table[x][r][c] = 1
            for k in range(9):
                table[v][k][c] = 1
                table[v][r][k] = 1
            br, bc = (r // 3) * 3, (c // 3) * 3
            for rr in range(br, br + 3):
                for cc in range(bc, bc + 3):
                    table[v][rr][cc] = 1
            table[v][r][c] = 0
    return table

def propagate(sudoku):
    while True:
        if is_complete(sudoku):
            return 'solved'
        table = build_table(sudoku)
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

        # Hidden single in a box
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

        return 'stuck'

def find_best_guess(sudoku):
    table = build_table(sudoku)
    best_r, best_c = -1, -1
    best_candidates = None
    best_count = 10
    for r in range(9):
        for c in range(9):
            if sudoku[r][c] == 0:
                candidates = [v + 1 for v in range(9) if table[v][r][c] == 0]
                if len(candidates) < best_count:
                    best_count = len(candidates)
                    best_r, best_c = r, c
                    best_candidates = candidates
                    if best_count == 1:
                        return best_r, best_c, best_candidates
    if best_r == -1:
        return None
    return best_r, best_c, best_candidates

def solve_sudoku(sudoku):
    """Returns True if solved, False if unsolvable."""
    stack = []
    while True:
        status = propagate(sudoku)
        if status == 'solved':
            return True
        if status == 'contradiction':
            backtrack_success = False
            while stack:
                row, col, candidates, idx, snapshot = stack.pop()
                for r in range(9):
                    for c in range(9):
                        sudoku[r][c] = snapshot[r][c]
                while idx < len(candidates):
                    guess = candidates[idx]
                    sudoku[row][col] = guess
                    stack.append((row, col, candidates, idx + 1, [row[:] for row in sudoku]))
                    backtrack_success = True
                    break
                if backtrack_success:
                    break
            if not backtrack_success:
                return False
            continue
        # stuck – make a guess
        guess_info = find_best_guess(sudoku)
        if guess_info is None:
            return True   # should not happen
        row, col, candidates = guess_info
        if not candidates:
            return False
        guess = candidates[0]
        snapshot = [row[:] for row in sudoku]
        sudoku[row][col] = guess
        stack.append((row, col, candidates, 1, snapshot))

# ======================================================================
#  PYGAME GUI
# ======================================================================

# Constants
CELL_SIZE = 60
GRID_OFFSET = 30
GRID_SIZE = 9 * CELL_SIZE
WINDOW_WIDTH = GRID_SIZE + 2 * GRID_OFFSET
WINDOW_HEIGHT = GRID_SIZE + 2 * GRID_OFFSET + 80   # extra space for buttons

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
LIGHT_BLUE = (173, 216, 230)
RED = (255, 0, 0)
GREEN = (0, 200, 0)
BUTTON_COLOR = (100, 150, 200)
BUTTON_HOVER = (70, 120, 170)
BUTTON_TEXT = (255, 255, 255)

def draw_grid(screen, board, selected):
    # Draw background
    screen.fill(WHITE)

    # Draw cells
    font = pygame.font.Font(None, 48)
    for r in range(9):
        for c in range(9):
            x = GRID_OFFSET + c * CELL_SIZE
            y = GRID_OFFSET + r * CELL_SIZE
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            # Highlight selected cell
            if selected and selected == (r, c):
                pygame.draw.rect(screen, LIGHT_BLUE, rect)
            # Draw number
            val = board[r][c]
            if val != 0:
                text = font.render(str(val), True, BLACK)
                text_rect = text.get_rect(center=(x + CELL_SIZE//2, y + CELL_SIZE//2))
                screen.blit(text, text_rect)

    # Draw grid lines (thin for cells, thick for boxes)
    for i in range(10):
        line_width = 3 if i % 3 == 0 else 1
        # vertical lines
        pygame.draw.line(screen, BLACK,
                         (GRID_OFFSET + i * CELL_SIZE, GRID_OFFSET),
                         (GRID_OFFSET + i * CELL_SIZE, GRID_OFFSET + GRID_SIZE),
                         line_width)
        # horizontal lines
        pygame.draw.line(screen, BLACK,
                         (GRID_OFFSET, GRID_OFFSET + i * CELL_SIZE),
                         (GRID_OFFSET + GRID_SIZE, GRID_OFFSET + i * CELL_SIZE),
                         line_width)

def draw_buttons(screen, mouse_pos):
    # Solve button
    solve_rect = pygame.Rect(GRID_OFFSET, GRID_OFFSET + GRID_SIZE + 20, 120, 40)
    clear_rect = pygame.Rect(GRID_OFFSET + 150, GRID_OFFSET + GRID_SIZE + 20, 120, 40)

    # Hover effect
    color_solve = BUTTON_HOVER if solve_rect.collidepoint(mouse_pos) else BUTTON_COLOR
    color_clear = BUTTON_HOVER if clear_rect.collidepoint(mouse_pos) else BUTTON_COLOR

    pygame.draw.rect(screen, color_solve, solve_rect, border_radius=8)
    pygame.draw.rect(screen, color_clear, clear_rect, border_radius=8)

    font = pygame.font.Font(None, 32)
    text_solve = font.render("Solve", True, BUTTON_TEXT)
    text_clear = font.render("Clear", True, BUTTON_TEXT)
    screen.blit(text_solve, (solve_rect.x + 30, solve_rect.y + 8))
    screen.blit(text_clear, (clear_rect.x + 30, clear_rect.y + 8))

    return solve_rect, clear_rect

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Sudoku Solver")
    clock = pygame.time.Clock()

    board = [[0] * 9 for _ in range(9)]
    selected = None   # (row, col) or None
    status_message = "Click a cell, then press a number (1-9) or Backspace to clear."
    solve_rect = clear_rect = None

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    # Check if click on grid
                    x, y = event.pos
                    if (GRID_OFFSET <= x <= GRID_OFFSET + GRID_SIZE and
                        GRID_OFFSET <= y <= GRID_OFFSET + GRID_SIZE):
                        col = (x - GRID_OFFSET) // CELL_SIZE
                        row = (y - GRID_OFFSET) // CELL_SIZE
                        if 0 <= row < 9 and 0 <= col < 9:
                            selected = (row, col)
                            status_message = f"Selected cell ({row+1}, {col+1})"
                    elif solve_rect and solve_rect.collidepoint(event.pos):
                        # Solve
                        status_message = "Solving..."
                        solved = solve_sudoku(board)
                        if solved:
                            status_message = "Solved!"
                        else:
                            status_message = "No solution exists."
                        selected = None
                    elif clear_rect and clear_rect.collidepoint(event.pos):
                        # Clear board
                        board = [[0] * 9 for _ in range(9)]
                        selected = None
                        status_message = "Board cleared."
            elif event.type == pygame.KEYDOWN:
                if selected is not None:
                    r, c = selected
                    # Number keys (1-9)
                    if event.key in range(pygame.K_1, pygame.K_9 + 1):
                        num = event.key - pygame.K_1 + 1
                        board[r][c] = num
                        status_message = f"Set ({r+1},{c+1}) = {num}"
                    elif event.key in range(pygame.K_KP1, pygame.K_KP9 + 1):
                        num = event.key - pygame.K_KP1 + 1
                        board[r][c] = num
                        status_message = f"Set ({r+1},{c+1}) = {num}"
                    elif event.key == pygame.K_BACKSPACE or event.key == pygame.K_DELETE:
                        board[r][c] = 0
                        status_message = f"Cleared ({r+1},{c+1})"
                # Global shortcuts
                if event.key == pygame.K_s:          # 'S' key (lowercase constant)
                    status_message = "Solving..."
                    solved = solve_sudoku(board)
                    if solved:
                        status_message = "Solved!"
                    else:
                        status_message = "No solution exists."
                    selected = None
                if event.key == pygame.K_c:          # 'C' key
                    board = [[0] * 9 for _ in range(9)]
                    selected = None
                    status_message = "Board cleared."

        # Drawing
        draw_grid(screen, board, selected)
        solve_rect, clear_rect = draw_buttons(screen, mouse_pos)

        # Status message
        font_status = pygame.font.Font(None, 28)
        text_status = font_status.render(status_message, True, BLACK)
        status_y = GRID_OFFSET + GRID_SIZE + 70
        screen.blit(text_status, (GRID_OFFSET, status_y))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()