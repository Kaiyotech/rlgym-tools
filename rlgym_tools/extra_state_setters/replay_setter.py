import random
from typing import List, Union

import numpy as np
from numpy import random as rand
from rlgym.utils.state_setters import StateSetter
from rlgym.utils.state_setters import StateWrapper
import math


class ReplaySetter(StateSetter):
    def __init__(self, ndarray_or_file: Union[str, np.ndarray], random_boost=False, remove_defender_weight=0,
                 defender_front_goal_weight=0, vel_div_range=(2, 10), vel_div_weight=0):
        """
        ReplayBasedSetter constructor

        :param ndarray_or_file: A file string or a numpy ndarray of states for a single game mode.
        """
        super().__init__()

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
        assert len(data) == len(state_wrapper.cars) * 13 + 9, "Data given does not match current game mode"
        self.divisor = 1
        if self.vel_div_weight > rand.uniform(0, 1):
            self.divisor = rand.uniform(*self.vel_div_range)
        self._set_ball(state_wrapper, data)
        self._set_cars(state_wrapper, data)

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
                    if i < mid:
                        attack_team = 0
                    else:
                        attack_team = 1

        for i, car in enumerate(state_wrapper.cars):
            boost = data[i][12]
            if self.random_boost and rand.choice([True, False]):
                boost = rand.uniform(0.35, 1.0)
                if rand.uniform(0, 1) > 0.95:
                    boost = boost / 10
            if self.remove_defender_weight > rand.uniform(0, 1):
                if attack_team == 0 and i >= mid:
                    car.set_pos(i * 100, 0, rand.uniform(17, 300))
                elif attack_team == 1 and i < mid:
                    car.set_pos(i * 100, 0, rand.uniform(17, 300))
            if self.defender_front_goal_weight > rand.uniform(0, 1):
                if attack_team == 0 and i >= mid:
                    car.set_pos(rand.uniform(-1300, 1300), rand.uniform(4000, 5100), 17)
                elif attack_team == 1 and i < mid:
                    car.set_pos(rand.uniform(-1300, 1300), rand.uniform(-5100, -4000), 17)
            else:
                car.set_pos(*data[i][:3])
            car.set_rot(*data[i][3:6])
            car.set_lin_vel(*data[i][6:9]/self.divisor)
            car.set_ang_vel(*data[i][9:12])
            car.boost = boost

    def _set_ball(self, state_wrapper: StateWrapper, data: np.ndarray):
        """
        Sets the ball according to the game state from replay

        :param state_wrapper: StateWrapper object to be modified with desired state values.
        :param data: Numpy array from the replay to get values from.
        """
        state_wrapper.ball.set_pos(*data[:3])
        state_wrapper.ball.set_lin_vel(*data[3:6]/self.divisor)
        state_wrapper.ball.set_ang_vel(*data[6:9])
