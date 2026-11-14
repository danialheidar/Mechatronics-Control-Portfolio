from pong_env import PongEnv
from agent import Agent

from visualize import watch_game

env = PongEnv()
agent = Agent()


NUM_EPISODES = 1000


for episode in range(NUM_EPISODES):

    state = env.reset()

    total_reward = 0
    total_loss = 0

    steps = 0
    loss_count = 0

    while True:

        # -------------------------
        # انتخاب Action
        # -------------------------

        action = agent.choose_action(state)

        # -------------------------
        # اجرای Action
        # -------------------------

        next_state, reward, done = env.step(
            action
        )

        # -------------------------
        # ذخیره تجربه
        # -------------------------

        agent.remember(
            state,
            action,
            reward,
            next_state,
            done
        )

        # -------------------------
        # آموزش
        # -------------------------

        loss = agent.train_step()

        if loss is not None:
            total_loss += loss
            loss_count += 1

        state = next_state

        total_reward += reward

        steps += 1

        if done:
            break

    # -------------------------
    # کاهش exploration
    # -------------------------

    agent.update_epsilon()

    average_loss = (
        total_loss / loss_count
        if loss_count > 0
        else 0
    )

    if (episode + 1) % 100 == 0:

        print("\n==============================")
        print(f"Evaluation after episode {episode + 1}")
        print("==============================")

        watch_game(env, agent)

    print(
        f"Episode: {episode + 1:4d} | "
        f"Steps: {steps:4d} | "
        f"Reward: {total_reward:6.1f} | "
        f"Loss: {average_loss:.4f} | "
        f"Epsilon: {agent.epsilon:.3f}"
    )