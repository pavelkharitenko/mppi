import random

def generate_forest(area_width_m, area_height_m, tree_density_per_m2=0.01, tree_radius=0.2):
    """
    Generates trees as fixed-position circular obstacles.
    Returns: {"static": [(x, y, radius)], "moving": []}
    """
    num_trees = int(area_width_m * area_height_m * tree_density_per_m2)
    trees = [
        (
            random.uniform(0, area_width_m),
            random.uniform(0, area_height_m),
            tree_radius
        )
        for _ in range(num_trees)
    ]
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


def generate_moving_obstacles(area_width_m, area_height_m, num_moving, radius=1.0, velocity_range=(0.5, 2.0)):
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
        moving.append({
            "pos": [x, y],
            "radius": radius,
            "vel": [vx, vy]
        })
    return {"static": [], "moving": moving}


def generate_mixed_scene(area_width_m, area_height_m, tree_density=0.05, num_moving=5, tree_radius=0.3):
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
