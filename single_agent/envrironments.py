import pygame
import math

# --- Setup ---
# Map dimensions in meters
MAP_WIDTH_M = 50.0   # meters
MAP_HEIGHT_M = 50.0  # meters



# Screen size in pixels
SCREEN_WIDTH_PX = 800
SCREEN_HEIGHT_PX = 800
AREA_WIDTH = SCREEN_WIDTH_PX
AREA_HEIGHT = SCREEN_HEIGHT_PX
FPS = 60
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
ROBOT_RADIUS = 10

# Scale: pixels per meter
SCALE = SCREEN_WIDTH_PX / MAP_WIDTH_M  # Assuming square pixels and uniform scale

def world_to_screen(x_m, y_m):
    x_px = int(x_m * SCALE)
    y_px = int(SCREEN_HEIGHT_PX - y_m * SCALE)  # Flip y-axis for display
    return x_px, y_px

def screen_to_world(x_px, y_px):
    x_m = x_px / SCALE
    y_m = (SCREEN_HEIGHT_PX - y_px) / SCALE
    return x_m, y_m




from ma_exploration.obstacle_generator import generate_forest, generate_giant_shapes, generate_moving_obstacles, generate_mixed_scene


# Choose one:
# obstacles = generate_forest(AREA_WIDTH, AREA_HEIGHT, tree_density=0.01)
# obstacles = generate_giant_shapes(AREA_WIDTH, AREA_HEIGHT, num_shapes=10)
# obstacles = generate_moving_obstacles(AREA_WIDTH, AREA_HEIGHT, num_moving=5)
obstacles = generate_mixed_scene(MAP_WIDTH_M, MAP_HEIGHT_M)

# Drawing inside your Pygame loop
def draw_obstacles(screen, obstacles):
    for x, y, r in obstacles["static"]:
        pygame.draw.circle(screen, BLACK, (int(x), int(y)), r)
    for mob in obstacles["moving"]:
        pygame.draw.circle(screen, (150, 0, 0), (int(mob["pos"][0]), int(mob["pos"][1])), mob["radius"])

def update_moving_obstacles(obstacles, dt, map_bounds_m):
    x_min, y_min, x_max, y_max = map_bounds_m
    for mob in obstacles["moving"]:
        x, y = mob["pos"]
        vx, vy = mob["vel"]
        x += vx * dt
        y += vy * dt

        # Bounce off bounds in meters
        if x - mob["radius"] < x_min or x + mob["radius"] > x_max:
            vx *= -1
        if y - mob["radius"] < y_min or y + mob["radius"] > y_max:
            vy *= -1

        mob["pos"] = [x, y]
        mob["vel"] = [vx, vy]


# Initialize
pygame.init()
screen = pygame.display.set_mode((AREA_WIDTH, AREA_HEIGHT))
clock = pygame.time.Clock()


# --- Robot ---
class Robot:
    def __init__(self, x, y, theta):
        self.x, self.y, self.theta = x, y, theta
        self.v = 0
        self.w = 0

    def update(self, dt):
        self.x += self.v * math.cos(self.theta) * dt
        self.y += self.v * math.sin(self.theta) * dt
        self.theta += self.w * dt


    def draw(self, screen):
        pygame.draw.circle(screen, GREEN, (int(self.x), int(self.y)), ROBOT_RADIUS)
        dx = math.cos(self.theta) * 15
        dy = math.sin(self.theta) * 15
        pygame.draw.line(screen, BLACK, (self.x, self.y), (self.x + dx, self.y + dy), 2)

# --- Main Loop ---
running = True
robot = Robot(50, 50, 0)  # Dummy robot class

while running:
    dt = clock.tick(FPS) / 1000.0  # Delta time in seconds

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # --- Update ---
    robot.update(dt)  # Assuming your robot has an update function
    update_moving_obstacles(obstacles, dt, map_bounds_m=(0.0,0.0,MAP_WIDTH_M, MAP_HEIGHT_M))

    # --- Draw ---
    screen.fill((255, 255, 255))  # White background
    robot.draw(screen)  # Assuming it has a draw method

    for x_m, y_m, r_m in obstacles["static"]:
        x_px, y_px = world_to_screen(x_m, y_m)
        pygame.draw.circle(screen, BLACK, (x_px, y_px), int(r_m * SCALE))

    for mob in obstacles["moving"]:
        x_px, y_px = world_to_screen(*mob["pos"])
        pygame.draw.circle(screen, (255, 0, 0), (x_px, y_px), int(mob["radius"] * SCALE))



    pygame.display.flip()

pygame.quit()
