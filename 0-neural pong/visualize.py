import pygame
import time


def watch_game(env, agent):

    pygame.init()

    screen = pygame.display.set_mode(
        (env.WIDTH, env.HEIGHT)
    )

    pygame.display.set_caption(
        "Neural Pong - Evaluation"
    )

    clock = pygame.time.Clock()

    state = env.reset()

    # در بازی نمایشی exploration نداریم
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    running = True

    while running and not env.done:

        # -------------------------
        # Event ها
        # -------------------------

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

        # -------------------------
        # Agent action
        # -------------------------

        action = agent.choose_action(state)

        next_state, reward, done = env.step(action)

        state = next_state

        # -------------------------
        # رسم
        # -------------------------

        screen.fill((0, 0, 0))

        # Paddle ما
        pygame.draw.rect(
            screen,
            (255, 255, 255),
            (
                30,
                int(
                    env.paddle_y
                    - env.PADDLE_HEIGHT / 2
                ),
                env.PADDLE_WIDTH,
                env.PADDLE_HEIGHT
            )
        )

        # Paddle حریف
        pygame.draw.rect(
            screen,
            (255, 255, 255),
            (
                env.WIDTH - 40,
                int(
                    env.opponent_paddle_y
                    - env.PADDLE_HEIGHT / 2
                ),
                env.PADDLE_WIDTH,
                env.PADDLE_HEIGHT
            )
        )

        # توپ
        pygame.draw.circle(
            screen,
            (255, 255, 255),
            (
                int(env.ball_x),
                int(env.ball_y)
            ),
            env.BALL_SIZE // 2
        )

        pygame.display.flip()

        clock.tick(60)

    # چند لحظه نتیجه را نگه دار
    time.sleep(1)

    # برگرداندن epsilon قبلی
    agent.epsilon = old_epsilon

    pygame.quit()