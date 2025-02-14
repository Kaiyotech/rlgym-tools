import random
from typing import List, Union

import numpy as np
from numpy import random as rand
from rlgym.utils.state_setters import StateSetter
from rlgym.utils.state_setters import StateWrapper
from rlgym.utils.common_values import BOOST_LOCATIONS
import math


def _expand_cars(state_wrapper: StateWrapper, data: np.ndarray):
    rng = np.random.default_rng()
    num_cars_in_replay = (len(data) - 9) // 13
    num_cars_to_add = len(state_wrapper.cars) - num_cars_in_replay
    car_data = np.split(data[9:], num_cars_in_replay)
    item_positions = [data[:3]]
    for i, car in enumerate(car_data):
        item_positions.append(car_data[i][:3])
    # longest dimension of any hitbox is 131.49 (breakout) so check 150 distance for safety
    for i in range(num_cars_to_add):
        checks = 0
        closest_dist = 0
        x = 0
        y = 0
        z = 17
        while closest_dist <= 150:
            x = rng.uniform(-2800, 2800)
            y = rng.uniform(-3800, 3800)
            item_positions.sort(key=lambda p: np.linalg.norm(p - np.array([x, y, z])))
            closest_dist = np.linalg.norm(item_positions[0] - np.array([x, y, z]))
            checks += 1
            # having trouble finding space on the floor, put it in the air
            # this seems extremely rare. I ran 100_000 tests with 6 initial cars and ball at 17
            # and the most checks ever was 4
            if checks > 5:
                z = rng.uniform(140, 1800)
        item_positions.append(np.array([x, y, z]))
        data = np.append(data, [x, y, z, 0, rng.uniform(-np.pi, np.pi), 0, rng.uniform(0, 500), rng.uniform(0, 500),
                                0, 0, 0, 0, rng.uniform(0, 1)])

    return data


def _shrink_cars(state_wrapper: StateWrapper, data: np.ndarray):
    num_cars_in_replay = (len(data) - 9) // 13
    num_cars_to_remove = num_cars_in_replay - len(state_wrapper.cars)
    per_team_to_remove = num_cars_to_remove // 2
    leftover = num_cars_to_remove - (per_team_to_remove * 2)
    blue_end = (num_cars_in_replay // 2) - 1
    orange_end = (num_cars_in_replay - 1)
    to_delete = [*range(blue_end - per_team_to_remove + 1, blue_end + 1)]
    to_delete.extend(range(orange_end - per_team_to_remove + 1 - leftover, orange_end + 1))
    car_data = np.split(data[9:], num_cars_in_replay)
    car_data = np.delete(car_data, to_delete, axis=0)
    car_data = car_data.flatten()
    ball_data = data[:9]
    return np.concatenate((ball_data, car_data))


class ReplaySetter(StateSetter):
    def __init__(self, ndarray_or_file: Union[str, np.ndarray], random_boost=False, remove_defender_weight=0,
                 defender_front_goal_weight=0, vel_div_range=(2, 10), vel_div_weight=0, end_object_tracker=None,
                 zero_ball_weight=0, zero_car_weight=0, rotate_car_weight=0, backward_car_weight=0, special_loc_weight=0,
                 zero_boost_weight=0, dtap_dict=None, initial_state_dict=(False, False, False), expand_shrink_cars=False):
        """
        ReplayBasedSetter constructor

        :param ndarray_or_file: A file string or a numpy ndarray of states for a single game mode.
        :param expand_shrink_cars: an integer number of cars to expect, to expand or contract a setter which isn't the right size
        """
        super().__init__()

        self.initial_state_dict = initial_state_dict
        self.dtap_dict = dtap_dict
        self.zero_boost_weight = zero_boost_weight
        self.special_loc_weight = special_loc_weight
        self.backward_car_weight = backward_car_weight
        self.rotate_car_weight = rotate_car_weight
        self.zero_car_weight = zero_car_weight
        self.expand_shrink_cars = expand_shrink_cars
        if isinstance(ndarray_or_file, np.ndarray):
            self.states = ndarray_or_file
        elif isinstance(ndarray_or_file, str):
            self.states = np.load(ndarray_or_file)
        self.probabilities = self.generate_probabilities()
        self.random_boost = random_boost
        self.remove_defender_weight = remove_defender_weight
        self.defender_front_goal_weight = defender_front_goal_weight
        self.vel_div_weight = vel_div_weight
        assert vel_div_range[0] >= 1
        assert vel_div_range[0] < vel_div_range[1]
        self.vel_div_range = vel_div_range
        self.divisor = 1
        self.big_boosts = [BOOST_LOCATIONS[i] for i in [3, 4, 15, 18, 29, 30]]
        self.big_boosts = np.asarray(self.big_boosts)
        self.big_boosts[:, -1] = 18
        self.end_object_tracker = end_object_tracker
        self.zero_ball_weight = zero_ball_weight

    def generate_probabilities(self):
        """
        Generates probabilities for each state.
        :return: Numpy array of probabilities (summing to 1)
        """
        return np.ones(len(self.states)) / len(self.states)

    @classmethod
    def construct_from_replays(cls, paths_to_replays: List[str], frame_skip: int = 150):
        """
        Alternative constructor that constructs ReplayBasedSetter from replays given as paths.

        :param paths_to_replays: Paths to all the reapls
        :param frame_skip: Every frame_skip frame from the replay will be converted
        :return: Numpy array of frames
        """
        return cls(cls.convert_replays(paths_to_replays, frame_skip))

    @staticmethod
    def convert_replays(paths_to_each_replay: List[str], frame_skip: int = 150, verbose: int = 0, output_location=None):
        from rlgym_tools.replay_converter import convert_replay
        states = []
        for replay in paths_to_each_replay:
            replay_iterator = convert_replay(replay)
            remainder = random.randint(0, frame_skip - 1)  # Vary the delays slightly
            for i, value in enumerate(replay_iterator):
                if i % frame_skip == remainder:
                    game_state, _ = value

                    whole_state = []
                    ball = game_state.ball
                    ball_state = np.concatenate((ball.position, ball.linear_velocity, ball.angular_velocity))

                    whole_state.append(ball_state)
                    for player in game_state.players:
                        whole_state.append(np.concatenate((player.car_data.position,
                                                           player.car_data.euler_angles(),
                                                           player.car_data.linear_velocity,
                                                           player.car_data.angular_velocity,
                                                           np.asarray([player.boost_amount]))))

                    np_state = np.concatenate(whole_state)
                    states.append(np_state)
            if verbose > 0:
                print(replay, "done")

        states = np.asarray(states)
        if output_location is not None:
            np.save(output_location, states)
        return states

    def reset(self, state_wrapper: StateWrapper):
        """
        Modifies the StateWrapper to contain random values the ball and each car.

        :param state_wrapper: StateWrapper object to be modified with desired state values.
        """

        data = self.states[np.random.choice(len(self.states), p=self.probabilities)]
        self.divisor = 1
        if self.vel_div_weight > rand.uniform(0, 1):
            self.divisor = rand.uniform(*self.vel_div_range)
        if not self.expand_shrink_cars:
            assert len(data) == len(state_wrapper.cars) * 13 + 9, "Data given does not match current game mode"
        elif len(data) < len(state_wrapper.cars) * 13 + 9:
            data = _expand_cars(state_wrapper, data)
        else:
            data = _shrink_cars(state_wrapper, data)
        self._set_ball(state_wrapper, data)
        self._set_cars(state_wrapper, data)

        if self.dtap_dict is not None:
            self.dtap_dict["hit_towards_bb"] = self.initial_state_dict[0]
            self.dtap_dict["ball_hit_bb"] = self.initial_state_dict[1]
            self.dtap_dict["hit_towards_goal"] = self.initial_state_dict[2]

    def _set_cars(self, state_wrapper: StateWrapper, data: np.ndarray):
        """
        Sets the players according to the game state from replay

        :param state_wrapper: StateWrapper object to be modified with desired state values.
        :param data: Numpy array from the replay to get values from.
        """
        ball_pos = data[:3]
        data = np.split(data[9:], len(state_wrapper.cars))
        attack_team = -1
        mid = len(state_wrapper.cars) // 2
        if self.remove_defender_weight > 0 or self.defender_front_goal_weight > 0:
            close_dist = 1000000
            for i, car in enumerate(state_wrapper.cars):
                car_pos = data[i][:3]
                dist = ball_pos - car_pos
                new_dist = math.sqrt(dist[0] ** 2 + dist[1] ** 2 + dist[2] ** 2)
                if new_dist < close_dist:
                    close_dist = new_dist
                    if i < mid:
                        attack_team = 0
                    else:
                        attack_team = 1

        remove_defender = self.remove_defender_weight > rand.uniform(0, 1)
        defender_goal = self.defender_front_goal_weight > rand.uniform(0, 1)
        rand_boost = self.random_boost and rand.choice([True, False])
        zero_boost = self.zero_boost_weight > rand.uniform(0, 1)
        for i, car in enumerate(state_wrapper.cars):
            if rand_boost:
                boost = rand.uniform(0.35, 1.0)
                if rand.uniform(0, 1) > 0.95:
                    boost = boost / 10
            if zero_boost:
                boost = 0
            else:
                boost = data[i][12]

            if remove_defender:
                if attack_team == 0 and i >= mid:
                    car.set_pos(i * 100, -5100, rand.uniform(17, 300))
                elif attack_team == 1 and i < mid:
                    car.set_pos(i * 100, 5100, rand.uniform(17, 300))
                else:
                    car.set_pos(*data[i][:3])
            elif defender_goal:
                if attack_team == 0 and i >= mid:
                    car.set_pos(rand.uniform(-1300, 1300), rand.uniform(4000, 5100), 17)
                    defender_goal = False  # only move one defender to goal
                elif attack_team == 1 and i < mid:
                    car.set_pos(rand.uniform(-1300, 1300), rand.uniform(-5100, -4000), 17)
                    defender_goal = False  # only move one defender to goal
                else:
                    car.set_pos(*data[i][:3])
            else:
                car.set_pos(*data[i][:3])

            car.set_rot(*data[i][3:6])
            car.set_lin_vel(*data[i][6:9]/self.divisor)
            car.set_ang_vel(*data[i][9:12])
            if rand.uniform(0, 1) < self.zero_car_weight:
                car.set_lin_vel(0, 0, 0)
                car.set_ang_vel(0, 0, 0)
            if rand.uniform(0, 1) < self.backward_car_weight:
                car.set_rot(data[i][3], -data[i][4], data[i][5])
            elif rand.uniform(0, 1) < self.rotate_car_weight:
                car.set_rot(data[i][3], rand.uniform(-np.pi, np.pi), data[i][5])
            car.boost = boost

    def _set_ball(self, state_wrapper: StateWrapper, data: np.ndarray):
        """
        Sets the ball according to the game state from replay

        :param state_wrapper: StateWrapper object to be modified with desired state values.
        :param data: Numpy array from the replay to get values from.
        """
        if self.end_object_tracker is not None and self.end_object_tracker[0] != 0:
            state_wrapper.ball.set_pos(*self.big_boosts[self.end_object_tracker[0] - 1])
            state_wrapper.ball.set_lin_vel(0, 0, 0)
            state_wrapper.ball.set_ang_vel(0, 0, 0)
        else:
            state_wrapper.ball.set_pos(*data[:3])
            state_wrapper.ball.set_lin_vel(*data[3:6]/self.divisor)
            state_wrapper.ball.set_ang_vel(*data[6:9])

        if rand.uniform(0, 1) < self.zero_ball_weight:
            state_wrapper.ball.set_pos(z=90)
            state_wrapper.ball.set_lin_vel(0, 0, 0)
            state_wrapper.ball.set_ang_vel(0, 0, 0)

        elif rand.uniform(0, 1) < self.special_loc_weight:
            if rand.uniform(0, 1) < 0.5:
                state_wrapper.ball.set_pos(*self.big_boosts[rand.choice(len(self.big_boosts))])
            else:
                mult = rand.choice([-1, 1])
                mult2 = rand.choice([-1, 1])
                loc = [1000 * mult, mult2 * 4800, 90]
                state_wrapper.ball.set_pos(*loc)
            state_wrapper.ball.set_pos(z=90)