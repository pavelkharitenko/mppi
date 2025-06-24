import random, math

def generate_forest(area_width_m, area_height_m, tree_density_per_m2=0.01, tree_radius=0.2, min_obstacle_spacing=1.0, seed=4.0):
    """
    Generates non-overlapping trees as circular obstacles.
    Each tree is at least `min_obstacle_spacing` meters apart from others.
    Returns: {"static": [(x, y, radius)], "moving": []}
    """

    if seed is not None:
        random.seed(seed)
    num_trees = int(area_width_m * area_height_m * tree_density_per_m2)
    max_attempts = num_trees * 20
    trees = []

    spacing = max(min_obstacle_spacing, 2 * tree_radius)

    attempts = 0
    while len(trees) < num_trees and attempts < max_attempts:
        x = random.uniform(tree_radius, area_width_m - tree_radius)
        y = random.uniform(tree_radius, area_height_m - tree_radius)

        valid = True
        for tx, ty, _ in trees:
            dist = math.hypot(tx - x, ty - y)
            if dist < spacing:
                valid = False
                break

        if valid:
            trees.append((x, y, tree_radius))

        attempts += 1

    return {"static": trees, "moving": []}


def generate_giant_shapes(area_width_m, area_height_m, num_shapes, radius_range=(2.0, 8.0)):
    """
    Generates large circular obstacles.
    Returns: {"static": [(x, y, radius)], "moving": []}
    """
    shapes = [
        (
            random.uniform(0, area_width_m),
            random.uniform(0, area_height_m),
            random.uniform(*radius_range)
        )
        for _ in range(num_shapes)
    ]
    return {"static": shapes, "moving": []}


def generate_moving_obstacles(area_width_m, area_height_m, num_moving, radius=0.5, velocity_range=(0.2, 1.0)):
    """
    Generates moving circular obstacles with linear velocity in m/s.
    Returns: {"static": [], "moving": [{"pos": [x, y], "radius": r, "vel": [vx, vy]}]}
    """
    moving = []
    for _ in range(num_moving):
        x = random.uniform(0, area_width_m)
        y = random.uniform(0, area_height_m)
        vx = random.choice([-1, 1]) * random.uniform(*velocity_range)
        vy = random.choice([-1, 1]) * random.uniform(*velocity_range)
        moving.append((x,y,radius,vx,vy))
    return {"static": [], "moving": moving}


def generate_mixed_scene(area_width_m, area_height_m, tree_density=0.028, num_moving=5, tree_radius=1.2):
    """
    Combines forest and moving obstacles.
    Returns: {"static": [...], "moving": [...]}
    """
    forest = generate_forest(area_width_m, area_height_m, tree_density, tree_radius)
    moving = generate_moving_obstacles(area_width_m, area_height_m, num_moving)
    return {
        "static": forest["static"],
        "moving": moving["moving"]
    }
