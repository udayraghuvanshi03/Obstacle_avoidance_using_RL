import random
import math
import numpy as np

import pygame
from pygame.color import THECOLORS

import pymunk
from pymunk.vec2d import Vec2d
from pymunk.pygame_util import DrawOptions as draw
from typing import Optional
# PyGame init
width = 1000
height = 700
pygame.init()
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

screen.set_alpha(None)

show_sensors = True
draw_screen = True


class GameState:
    def __init__(self,random_env:bool):
        self.crashed = False

        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0., 0.)

        self.create_car(100, 100, 0.5)

        self.num_steps = 0

        static = [
            pymunk.Segment(
                self.space.static_body,
                (0, 1), (0, height), 1),
            pymunk.Segment(
                self.space.static_body,
                (1, height), (width, height), 1),
            pymunk.Segment(
                self.space.static_body,
                (width-1, height), (width-1, 1), 1),
            pymunk.Segment(
                self.space.static_body,
                (1, 1), (width, 1), 1)
        ]
        for s in static:
            s.friction = 1.
            s.group = 1
            s.collision_type = 1
            s.color = THECOLORS['red']
        self.space.add(*static)

        self.obstacles = []
        if random_env:
            self.obstacles.append(self.create_obstacle(random.randint(0, width - 100),random.randint(0, height-100),random.randint(50, 100), "circle"))
            self.obstacles.append(self.create_obstacle(random.randint(0, width - 100), random.randint(0, height - 100),
                                                       random.randint(50, 100), "circle"))
            self.obstacles.append(self.create_obstacle(random.randint(0, width - 100), random.randint(0, height - 100),
                                                       random.randint(50, 100), "circle"))
            self.obstacles.append(self.create_obstacle(random.randint(0, width - 100), random.randint(0, height - 100),
                                                       random.randint(50, 100), "Rectangle"))
            self.obstacles.append(self.create_obstacle(random.randint(0, width - 100), random.randint(0, height - 100),
                                                       random.randint(50, 100), "Rectangle"))
        else:
            self.obstacles.append(self.create_obstacle(200, 350, 50, "circle"))
            self.obstacles.append(self.create_obstacle(450, 200, 105, "circle"))
            self.obstacles.append(self.create_obstacle(600, 450, 75, "circle"))
            self.obstacles.append(self.create_obstacle(800, 105, 55, "Rectangle"))
            self.obstacles.append(self.create_obstacle(250, 550, 110, "Rectangle"))
            self.obstacles.append(self.create_obstacle(870, 550, 75, "Rectangle"))

        #Dynamic obstacle
        self.create_cat()

    def create_obstacle(self, x, y, r, shape_type):
        c_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        if shape_type == "circle":
            c_shape = pymunk.Circle(c_body, r)
        else:
            c_shape = pymunk.Poly.create_box(c_body, (r, r + 5))
        c_shape.elasticity = 1.0
        c_body.position = x, y
        c_shape.color = THECOLORS["blue"]
        self.space.add(c_body, c_shape)
        return c_body

    def create_cat(self):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.cat_body = pymunk.Body(1, inertia)
        self.cat_body.position = 50, height - 100
        self.cat_shape = pymunk.Circle(self.cat_body, 30)
        self.cat_shape.color = THECOLORS["orange"]
        self.cat_shape.elasticity = 1.0
        self.cat_shape.angle = 0.5
        direction = Vec2d(1, 0).rotated(self.cat_body.angle)
        self.space.add(self.cat_body, self.cat_shape)

    def create_car(self, x, y, r):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.car_body = pymunk.Body(1, inertia)
        self.car_body.position = x, y
        self.car_shape = pymunk.Circle(self.car_body, 25)
        self.car_shape.color = THECOLORS["green"]
        self.car_shape.elasticity = 1.0
        self.car_body.angle = r
        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.car_body.apply_impulse_at_local_point(driving_direction)
        self.space.add(self.car_body, self.car_shape)

    def frame_step(self, action):
        if action == 0:
            self.car_body.angle -= .2
        elif action == 1:
            self.car_body.angle += .2

        if self.num_steps % 100 == 0:
            self.move_obstacles()

        if self.num_steps % 5 == 0:
            self.move_cat()

        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.car_body.velocity = 100 * driving_direction

        screen.fill(THECOLORS["black"])
        draw(screen)
        for shape in self.space.shapes:
            if isinstance(shape, pymunk.Segment):
                body = shape.body
                start_pos = int(body.position.x + shape.a.x), int(height - (body.position.y + shape.a.y))
                end_pos = int(body.position.x + shape.b.x), int(height - (body.position.y + shape.b.y))
                pygame.draw.lines(screen, (255, 255, 255), False, [start_pos, end_pos], 2)

            elif isinstance(shape, pymunk.Circle):
                body = shape.body
                pos = int(body.position.x), int(height - body.position.y)
                if body.body_type == 0:
                    dynamic_body_index = self.space.bodies.index(body)
                    if dynamic_body_index == 0:
                        pygame.draw.circle(screen, (255, 0, 0), pos, int(shape.radius), 0)
                    else:
                        pygame.draw.circle(screen, (0, 255, 0), pos, int(shape.radius), 0)
                else:
                    pygame.draw.circle(screen, (0, 0, 255), pos, int(shape.radius), 0)
            else:
                body = shape.body
                vertices = [(int(body.position.x + vertex.x), int(height - (body.position.y + vertex.y))) for vertex in
                            shape.get_vertices()]

                pygame.draw.polygon(screen, (0, 0, 255), vertices)

        self.space.step(1./10)
        if draw_screen:
            pygame.display.flip()
        clock.tick()

        x, y = self.car_body.position
        readings = self.get_sonar_readings(x, y, self.car_body.angle)
        normalized_readings = [(x-20.0)/20.0 for x in readings]
        state = np.array([normalized_readings])

        if self.car_is_crashed(readings):
            self.crashed = True
            reward = -500
            self.recover_from_crash(driving_direction)
        else:
            reward = -5 + int(self.sum_readings(readings) / 10)
        self.num_steps += 1

        return reward, state

    def move_obstacles(self):
        for obstacle in self.obstacles:
            speed = random.randint(1, 5)
            direction = Vec2d(1, 0).rotated(self.car_body.angle + random.randint(-2, 2))
            obstacle.velocity = speed * direction

    def move_cat(self):
        speed = random.randint(20, 200)
        self.cat_body.angle -= random.randint(-1, 1)
        direction = Vec2d(1, 0).rotated(self.cat_body.angle)
        self.cat_body.velocity = speed * direction

    def car_is_crashed(self, readings):
        if readings[0] == 1 or readings[1] == 1 or readings[2] == 1:
            return True
        else:
            return False

    def recover_from_crash(self, driving_direction):
        while self.crashed:
            self.car_body.velocity = -100 * driving_direction
            self.crashed = False
            for i in range(10):
                self.car_body.angle += .2
                screen.fill(THECOLORS["grey7"])
                draw(screen)
                for shape in self.space.shapes:
                    if isinstance(shape, pymunk.Segment):
                        body = shape.body
                        start_pos = int(body.position.x + shape.a.x), int(height - (body.position.y + shape.a.y))
                        end_pos = int(body.position.x + shape.b.x), int(height - (body.position.y + shape.b.y))
                        pygame.draw.lines(screen, (255, 255, 255), False, [start_pos, end_pos], 2)

                    elif isinstance(shape, pymunk.Circle):
                        body = shape.body
                        pos = int(body.position.x), int(
                            height - body.position.y)
                        if body.body_type == 0:
                            dynamic_body_index = self.space.bodies.index(body)
                            if dynamic_body_index == 0:
                                pygame.draw.circle(screen, (255, 0, 0), pos, int(shape.radius), 0)
                            else:
                                pygame.draw.circle(screen, (0, 255, 0), pos, int(shape.radius), 0)
                        else:
                            pygame.draw.circle(screen, (0, 0, 255), pos, int(shape.radius), 0)
                    else:
                        body = shape.body
                        vertices = [(int(body.position.x + vertex.x), int(height - (body.position.y + vertex.y))) for
                                    vertex in
                                    shape.get_vertices()]

                        pygame.draw.polygon(screen, (0, 0, 255), vertices)

                self.space.step(1./10)
                if draw_screen:
                    pygame.display.flip()
                clock.tick()

    def sum_readings(self, readings):
        tot = 0
        for i in readings:
            tot += i
        return tot

    def get_sonar_readings(self, x, y, angle):
        readings = []
        # Make our arms.
        arm_left = self.make_sonar_arm(x, y)
        arm_middle = arm_left
        arm_right = arm_left

        readings.append(self.get_arm_distance(arm_left, x, y, angle, 0.75))
        readings.append(self.get_arm_distance(arm_middle, x, y, angle, 0))
        readings.append(self.get_arm_distance(arm_right, x, y, angle, -0.75))

        if show_sensors:
            pygame.display.update()

        return readings

    def get_arm_distance(self, arm, x, y, angle, offset):
        i = 0

        # Look at each point and see if we've hit something.
        for point in arm:
            i += 1
            rotated_p = self.get_rotated_point(
                x, y, point[0], point[1], angle + offset
            )

            if rotated_p[0] <= 0 or rotated_p[1] <= 0 \
                    or rotated_p[0] >= width or rotated_p[1] >= height:
                return i  # Sensor is off the screen.
            else:
                obs = screen.get_at(rotated_p)
                if self.get_track_or_not(obs) != 0:
                    return i

            if show_sensors:
                pygame.draw.circle(screen, (255, 255, 255), (rotated_p), 2)

        return i

    def make_sonar_arm(self, x, y):
        spread = 10
        distance = 20
        arm_points = []
        for i in range(1, 40):
            arm_points.append((distance + x + (spread * i), y))

        return arm_points

    def get_rotated_point(self, x_1, y_1, x_2, y_2, radians):
        x_change = (x_2 - x_1) * math.cos(radians) + \
            (y_2 - y_1) * math.sin(radians)
        y_change = (y_1 - y_2) * math.cos(radians) - \
            (x_1 - x_2) * math.sin(radians)
        new_x = x_change + x_1
        new_y = height - (y_change + y_1)
        return int(new_x), int(new_y)

    def get_track_or_not(self, reading):
        if reading == THECOLORS['black']:
            return 0
        else:
            return 1

if __name__ == "__main__":
    game_state = GameState(random_env=True)
    while True:
        game_state.frame_step((random.randint(0, 2)))
