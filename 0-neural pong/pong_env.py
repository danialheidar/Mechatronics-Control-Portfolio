import random


class PongEnv:

    WIDTH = 800
    HEIGHT = 500

    PADDLE_WIDTH = 10
    PADDLE_HEIGHT = 100
    PADDLE_SPEED = 6

    BALL_SIZE = 10

    def __init__(self):
        self.reset()

    def reset(self):
        # موقعیت paddle
        self.paddle_y = self.HEIGHT // 2

        self.opponent_paddle_y = self.HEIGHT // 2

        # موقعیت توپ
        self.ball_x = self.WIDTH // 2
        self.ball_y = self.HEIGHT // 2

        # سرعت توپ
        self.ball_vx = random.choice([-5, 5])
        self.ball_vy = random.choice([-5, 5])

        self.done = False

        return self.get_state()

    def get_state(self):

        return [
            self.ball_x / self.WIDTH,
            self.ball_y / self.HEIGHT,

            self.ball_vx / 5,
            self.ball_vy / 5,

            self.paddle_y / self.HEIGHT,

            self.opponent_paddle_y / self.HEIGHT
        ]

    def move_opponent(self):

        if self.opponent_paddle_y < self.ball_y:
            self.opponent_paddle_y += 4

        elif self.opponent_paddle_y > self.ball_y:
            self.opponent_paddle_y -= 4

        self.opponent_paddle_y = max(
            self.PADDLE_HEIGHT // 2,
            min(
                self.HEIGHT - self.PADDLE_HEIGHT // 2,
                self.opponent_paddle_y
            )
        )

    def step(self, action):

        # -------------------------
        # حرکت paddle
        # -------------------------

        if action == 1:
            self.paddle_y -= self.PADDLE_SPEED

        elif action == 2:
            self.paddle_y += self.PADDLE_SPEED

        self.move_opponent()

        # محدود کردن paddle
        self.paddle_y = max(
            self.PADDLE_HEIGHT // 2,
            min(
                self.HEIGHT - self.PADDLE_HEIGHT // 2,
                self.paddle_y
            )
        )

        # -------------------------
        # حرکت توپ
        # -------------------------

        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # برخورد با بالا و پایین
        if self.ball_y <= 0 or self.ball_y >= self.HEIGHT:
            self.ball_vy *= -1

        reward = 0


        opponent_left = self.WIDTH - 40
        opponent_right = opponent_left + self.PADDLE_WIDTH

        opponent_top = (
            self.opponent_paddle_y
            - self.PADDLE_HEIGHT // 2
        )

        opponent_bottom = (
            self.opponent_paddle_y
            + self.PADDLE_HEIGHT // 2
        )

        if (
            self.ball_vx > 0
            and self.ball_x >= opponent_left
            and self.ball_x <= opponent_right
            and opponent_top <= self.ball_y <= opponent_bottom
        ):
            self.ball_vx *= -1

        # -------------------------
        # برخورد با paddle
        # -------------------------

        paddle_left = 30
        paddle_right = paddle_left + self.PADDLE_WIDTH

        paddle_top = (
            self.paddle_y - self.PADDLE_HEIGHT // 2
        )

        paddle_bottom = (
            self.paddle_y + self.PADDLE_HEIGHT // 2
        )

        if (
            self.ball_vx < 0
            and self.ball_x <= paddle_right
            and self.ball_x >= paddle_left
            and paddle_top <= self.ball_y <= paddle_bottom
        ):
            self.ball_vx *= -1

            reward = 1

        # -------------------------
        # توپ از سمت ما رد شد
        # -------------------------

        if self.ball_x < 0:
            reward = -10
            self.done = True

        # -------------------------
        # توپ از سمت حریف رد شد
        # -------------------------

        if self.ball_x > self.WIDTH:
            reward = 10

            self.ball_x = self.WIDTH // 2
            self.ball_y = self.HEIGHT // 2

            self.ball_vx = random.choice([-5, 5])
            self.ball_vy = random.choice([-5, 5])

        return self.get_state(), reward, self.done