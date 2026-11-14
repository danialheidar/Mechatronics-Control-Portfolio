from pong_env import PongEnv


env = PongEnv()

state = env.reset()

print("Initial state:")
print(state)

for i in range(2000):

    action = 0

    next_state, reward, done = env.step(action)

    print(
        "step =", i,
        "state =", next_state,
        "reward =", reward,
        "done =", done
    )

    if done:
        break