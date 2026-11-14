import pygame
import random

pygame.init()

# -------------------------
# تنظیمات بازی
# -------------------------

WIDTH = 800
HEIGHT = 500

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Neural Pong")

clock = pygame.time.Clock()

# -------------------------
# Paddle ها
# -------------------------

PADDLE_WIDTH = 10
PADDLE_HEIGHT = 100
PADDLE_SPEED = 6

left_paddle = pygame.Rect(
    30,
    HEIGHT // 2 - PADDLE_HEIGHT // 2,
    PADDLE_WIDTH,
    PADDLE_HEIGHT
)

right_paddle = pygame.Rect(
    WIDTH - 40,
    HEIGHT // 2 - PADDLE_HEIGHT // 2,
    PADDLE_WIDTH,
    PADDLE_HEIGHT
)

# -------------------------
# توپ
# -------------------------

BALL_SIZE = 10

ball = pygame.Rect(
    WIDTH // 2 - BALL_SIZE // 2,
    HEIGHT // 2 - BALL_SIZE // 2,
    BALL_SIZE,
    BALL_SIZE
)

ball_speed_x = 5
ball_speed_y = 5

# -------------------------
# امتیاز
# -------------------------

left_score = 0
right_score = 0

font = pygame.font.Font(None, 50)


def reset_ball():
    global ball_speed_x, ball_speed_y

    ball.center = (WIDTH // 2, HEIGHT // 2)

    ball_speed_x = random.choice([-5, 5])
    ball_speed_y = random.choice([-5, 5])

def get_state():
    state = [
        ball.centerx / WIDTH,
        ball.centery / HEIGHT,

        ball_speed_x / 5,
        ball_speed_y / 5,

        left_paddle.centery / HEIGHT
    ]

    return state

def apply_action(action):
    if action == 1:
        left_paddle.y -= PADDLE_SPEED

    elif action == 2:
        left_paddle.y += PADDLE_SPEED

    # action == 0 یعنی هیچ کاری نکن

    # جلوگیری از خروج paddle
    left_paddle.y = max(
        0,
        min(HEIGHT - PADDLE_HEIGHT, left_paddle.y)
    )

def calculate_reward():

    # اگر توپ از سمت چپ خارج شده،
    # یعنی ما توپ را از دست داده‌ایم.
    if ball.left <= 0:
        return -10

    # اگر توپ به paddle چپ برخورد کرده،
    # یک پاداش کوچک می‌گیریم.
    if ball.colliderect(left_paddle) and ball_speed_x < 0:
        return 1

    return 0


# -------------------------
# حلقه اصلی بازی
# -------------------------

running = True

while running:

    # دریافت Event ها
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_SPACE:
            print(get_state())

    # -------------------------
    # کنترل Paddle ها
    # -------------------------

    keys = pygame.key.get_pressed()

    # Paddle چپ
    action = random.randint(0, 2)

    apply_action(action)

    # Paddle راست
    if keys[pygame.K_UP]:
        right_paddle.y -= PADDLE_SPEED

    if keys[pygame.K_DOWN]:
        right_paddle.y += PADDLE_SPEED

    # جلوگیری از خروج paddle از صفحه
    left_paddle.y = max(0, min(HEIGHT - PADDLE_HEIGHT, left_paddle.y))
    right_paddle.y = max(0, min(HEIGHT - PADDLE_HEIGHT, right_paddle.y))

    # -------------------------
    # حرکت توپ
    # -------------------------

    ball.x += ball_speed_x
    ball.y += ball_speed_y

    # برخورد توپ با بالا و پایین
    if ball.top <= 0 or ball.bottom >= HEIGHT:
        ball_speed_y *= -1

    # برخورد با paddle چپ
    if ball.colliderect(left_paddle) and ball_speed_x < 0:
        ball_speed_x *= -1

    # برخورد با paddle راست
    if ball.colliderect(right_paddle) and ball_speed_x > 0:
        ball_speed_x *= -1

    # -------------------------
    # امتیاز
    # -------------------------

    if ball.left <= 0:
        right_score += 1
        reset_ball()

    if ball.right >= WIDTH:
        left_score += 1
        reset_ball()

    # -------------------------
    # رسم بازی
    # -------------------------

    screen.fill((0, 0, 0))

    pygame.draw.rect(screen, (255, 255, 255), left_paddle)
    pygame.draw.rect(screen, (255, 255, 255), right_paddle)

    pygame.draw.ellipse(screen, (255, 255, 255), ball)

    # خط وسط
    pygame.draw.line(
        screen,
        (100, 100, 100),
        (WIDTH // 2, 0),
        (WIDTH // 2, HEIGHT),
        2
    )

    state = get_state()

    # امتیاز
    score_text = font.render(
        f"{left_score} : {right_score}",
        True,
        (255, 255, 255)
    )

    screen.blit(
        score_text,
        (WIDTH // 2 - score_text.get_width() // 2, 20)
    )

    pygame.display.flip()

    clock.tick(60)


pygame.quit()