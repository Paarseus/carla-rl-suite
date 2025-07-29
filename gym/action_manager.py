import cv2
import carla
from gymnasium import spaces
import math
import numpy as np

smoothing_factor = 0.1
enable_smoothing = False


class ActionSpace:
        def __init__(self):
                self.smoothing_factor = smoothing_factor
                self.enable_smoothing = enable_smoothing
                self.previous_action = np.array([0.0, 0.0], dtype=np.float32)
                print(f"[ACTION SPACE] Initialized with smoothing_factor={smoothing_factor}, smoothing={enable_smoothing}")
                
        def get_action_bounds(self):
                """
                Returns action bounds for creating Gymnasium space.
                Use this in main.py to create spaces.Box()
                """
                return {
                        'low': np.array([-1.0, -1.0], dtype=np.float32),
                        'high': np.array([1.0, 1.0], dtype=np.float32),
                        'shape': (2,),
                        'dtype': np.float32
                        }

        def process_action(self, action):
                """
                Convert RL action to CARLA VehicleControl.
                
                Args:
                action: numpy array [throttle, steer, brake]
                
                Returns:
                carla.VehicleControl object
                """

                if not isinstance(action, np.ndarray):
                        action = np.array(action, dtype=np.float32)

                if self.enable_smoothing:
                        action = self._apply_smoothing(action)
                steer = float(action[0])
                acceleration = float(action[1])
                
                if acceleration >= 0:
                        throttle = acceleration
                        brake = 0
                else:
                        throttle = 0
                        brake = abs(acceleration)

                control = carla.VehicleControl(
                        throttle=throttle,
                        steer=steer,
                        brake=brake,
                        hand_brake=False,
                        reverse=False,
                        manual_gear_shift=False
                        )
                print(f"[ACTION] T:{throttle:.3f}, S:{steer:.3f}, B:{brake:.3f}")
                return control

        def reset(self):
                self.previous_action = np.array([0.0, 0.0], dtype=np.float32)
                print("[ACTION SPACE] Reset - previous action cleared")

        def sample_action(self):
                """Sample a random valid action.  Accelartion, Steering"""
                sample_action = np.random.uniform(
                                        low=[-1.0, -1.0], 
                                        high=[1.0, 1.0]
                                        ).astype(np.float32)

                print(f"[SAMPLE ACTION]: steer = {sample_action[0]:.3f}, acceleration = {sample_action[1]:.3f}")
                return sample_action

        def _apply_smoothing(self, action):
                """Apply action smoothing to prevent jerky movements."""
                smoothed_action = (self.smoothing_factor * self.previous_action + 
                                (1.0 - self.smoothing_factor) * action)
                self.previous_action = smoothed_action.copy()
                return smoothed_action
        