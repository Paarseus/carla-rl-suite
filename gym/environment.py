import carla

import gymnasium as gym
from gymnasium import spaces

import cv2
import time
import numpy as np

from connection_manager import ClientConnection
from sensors.sensor_manager import SensorManager
from sensors.sensor_interface import SensorInterface
from traffic_manager import TrafficManager
from hero_manager import HeroManager
from action_manager import ActionSpace
from observation_manager2 import ObservationSpace
from reward_manager import RewardManager

from hero_config import *


class CarlaEnv(gym.Env):
        def __init__(self):
                super().__init__()

                #CONNECTION
                self.conn = ClientConnection(town="Town01")       
                self.client, self.world = self.conn.connect() 
                self.conn.cleanup_all_actors() 

                #TRAFFIC_MANAGER
                traffic_manager = TrafficManager(self.client, self.world)                
                
                #SENSOR
                self.sensor_interface = SensorInterface()
                self.sensor_manager = SensorManager()

                #REWARD
                self.reward_manager = RewardManager()
                # self.reward_manager.reset()
                self.episode_step = 0
                self.total_reward = 0

                #HERO_MANAGER
                self.hero_manager = HeroManager(self.client, self.world, self.sensor_interface, self.sensor_manager)
                # self.hero_manager.reset_hero(hero_config)
                # self.hero_manager.set_spectator_camera_view()

                #ACTION_MANAGER & OBSERVATION_MANAGER
                self.action_manager = ActionSpace()
                self.observation_manager = ObservationSpace(self.hero_manager, self.sensor_interface)
                # self.observation_manager.reset()
                action_bounds = self.action_manager.get_action_bounds()
                self.action_space = spaces.Box(
                        low=action_bounds['low'],
                        high=action_bounds['high'], 
                        shape=action_bounds['shape'],
                        dtype=action_bounds['dtype']
                )
                self.observation_space = spaces.Dict({
                'hero_state': spaces.Box(
                        low=np.array([
                        -1.0,  # steering [-1, 1] (CARLA control range)
                        0.0,   # throttle [0, 1] (CARLA control range)
                        0.0,   # brake [0, 1] (CARLA control range)
                        0.0,   # speed [0, 1] (normalized: speed / 100.0 km/h)
                        -1.0,  # position_x [-1, 1] (normalized: relative_pos / 1000.0m)
                        -1.0,  # position_y [-1, 1] (normalized: relative_pos / 1000.0m)
                        -1.0,  # yaw [-1, 1] (normalized: yaw_rad / π)
                        -1.0,  # velocity_x [-1, 1] (normalized: velocity / 30.0 m/s)
                        -1.0,  # velocity_y [-1, 1] (normalized: velocity / 30.0 m/s)
                        0.0    # collision_status [0, 1] (0.0 or 1.0)
                        ], dtype=np.float32),
                        high=np.array([
                        1.0,   # steering
                        1.0,   # throttle
                        1.0,   # brake  
                        1.0,   # speed (normalized)
                        1.0,   # position_x (normalized)
                        1.0,   # position_y (normalized)
                        1.0,   # yaw (normalized)
                        1.0,   # velocity_x (normalized)
                        1.0,   # velocity_y (normalized)
                        1.0    # collision_status
                        ], dtype=np.float32),
                        shape=(10,),
                        dtype=np.float32
                ),
                'navigation': spaces.Box(
                        low=np.array([0.0, -1.0], dtype=np.float32),   # [distance_from_center, angle_to_road]
                        high=np.array([1.0, 1.0], dtype=np.float32),   # distance: [0,1], angle: [-1,1]
                        shape=(2,),
                        dtype=np.float32
                ),
                'camera': spaces.Box(
                        low=0,
                        high=255,
                        shape=(300, 400, 3),  # height, width, channels (RGB)
                        dtype=np.uint8
                )
                })


        def reset(self, *, seed = None, options = None):
                if seed is not None:
                        np.random.seed(seed)
                
                self.hero_manager.reset_hero(hero_config)
                self.reward_manager.reset()
                self.observation_manager.reset()
                self.hero_manager.set_spectator_camera_view()

                self.episode_step = 0
                self.total_reward = 0

                #self.hero_manager.hero.set_autopilot(True)

                observation = self.observation_manager.obs()
                info = {}

                return observation, info         


        def step(self, action=None):                
                if action is None:
                        action = self.action_manager.sample_action()
                
                carla_control = self.action_manager.process_action(action)
                self.hero_manager.apply_ego_control(carla_control)
                
                self.world.tick()
                
                obs = self.observation_manager.obs()

                reward = self.reward_manager.calculate_reward(obs, action)

                terminated, truncated = self.reward_manager.get_termination_conditions(obs)

                self.episode_step += 1
                self.total_reward += reward

                self.previous_obs = obs


                info = {
                        'episode_step': self.episode_step,
                        'total_reward': self.total_reward
                }

                if terminated or truncated:
                        info = {
                                'episode_step': self.episode_step,
                                'total_reward': self.total_reward
                        }

                self.world.tick()


                return obs, float(reward), terminated, truncated, info


        def render(self):
                try:
                        self._camera_name = ['camera_front', 'camera_right', 'camera_left', 'camera_rear']
                        self.render_mode = "human"
                        sensor_data = self.hero_manager.get_sensor_data()
                        for name in self._camera_name:
                                if name in sensor_data:
                                        frame_id, image_data = sensor_data[name]
                                else:
                                        print(f"Invalid image data from {name}")


                                if self.render_mode == "human" and 'camera' in name :
                                        bgr_image = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
                                        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
                                        cv2.imshow(name, bgr_image)
                                        
                                key = cv2.waitKey(1) & 0xFF


                                if key == ord('q'):
                                        return
                                
                        return

                except Exception as e:
                        print(f"[ERROR] Gymnasium Render(): {e}")
                


        
        def close(self):
                print("[EXITING] Gymnasium: Closing CARLA environment...")

                try:
                        cv2.destroyAllWindows()
                
                        self.conn.cleanup_all_actors()

                        settings = self.world.get_settings()
                        settings.synchronous_mode = False
                        settings.no_rendering_mode = False
                        settings.fixed_delta_seconds = None
                        self.world.apply_settings(settings)

                        time.sleep(0.5) 

                        print("[EXIT SUCCESSFULL]")

                except Exception as e:
                        print(f"[ERROR] Gymnasium Close(): {e}")
                



if __name__ == "__main__":
        env=CarlaEnv()
        time.sleep(0.5) 
        env.reset()
        while(True):
                try:
                        env.step()
                        env.render()
                        time.sleep(0.01)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                                env.close()
                                break
                
                except Exception as e:
                        print(f"[ERROR] Gymnasium \"__main__\": {e}")

                except KeyboardInterrupt:
                        env.close()
                        break






