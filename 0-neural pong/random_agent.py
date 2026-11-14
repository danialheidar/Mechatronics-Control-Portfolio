from pong_env import PongEnv
import random


env = PongEnv()

num_episodes = 10

for episode in range(num_episodes):

    state = env.reset()

    total_reward = 0
    steps = 0

    while True:

        # انتخاب تصادفی Action
        action = random.randint(0, 2)

        # اجرای Action
        next_state, reward, done = env.step(action)

        total_reward += reward
        steps += 1

        state = next_state

        if done:
            break

    print(
        f"Episode {episode + 1}: "
        f"steps={steps}, "
        f"reward={total_reward}"
    )