from game.action_space import ActionSpace
from game.backgammon import WHITE, BLACK, BAR


assert ActionSpace.encode(0, 1, WHITE) == 0
assert ActionSpace.encode(23, 1, BLACK) == 0

assert ActionSpace.encode(23, 6, WHITE) == 143
assert ActionSpace.encode(0, 6, BLACK) == 143

assert ActionSpace.encode(BAR, 1, WHITE) == 144
assert ActionSpace.encode(BAR, 1, BLACK) == 144

assert ActionSpace.decode(0, WHITE) == (0, 1)
assert ActionSpace.decode(0, BLACK) == (23, 1)

assert ActionSpace.decode(143, WHITE) == (23, 6)
assert ActionSpace.decode(143, BLACK) == (0, 6)

assert ActionSpace.decode(144, WHITE) == (BAR, 1)
assert ActionSpace.decode(144, BLACK) == (BAR, 1)

print("ActionSpace perspective test: OK")