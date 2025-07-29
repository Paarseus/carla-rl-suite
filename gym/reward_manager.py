import carla
import numpy as np
import math


"""
==============================================================================
CARLA RL ENVIRONMENT - REWARD SYSTEM OVERVIEW
==============================================================================

ACTION SPACE: 2D Continuous [acceleration, steering]
- acceleration: -1.0 (full brake) to +1.0 (full throttle)  
- steering: -1.0 (full left) to +1.0 (full right)

OBSERVATION SPACE:
- hero_state: [steering, throttle, brake, speed, x, y, yaw, vel_x, vel_y, collision] (10 values)
- navigation: [distance_from_center, angle_to_road] (2 values)

REWARD COMPONENTS:
==============================================================================

1. SPEED REWARD (weight: 1.0)
   └── Target: 25 km/h | Range: -1.0 to +1.0w
   └── Penalizes: <5 km/h (too slow) or >50 km/h (too fast)

2. LANE KEEPING REWARD (weight: 2.0) 
   └── Target: Stay in lane center | Range: -2.0 to +1.0
   └── Heavy penalty (-2.0) for leaving lane (>2m from center)

3. COLLISION PENALTY (weight: -100.0)
   └── Immediate -50.0 penalty on collision detection
   └── Additional -20.0 penalty if episode ends in collision

4. PROGRESS REWARD (weight: 1.0)
   └── +0.1 reward per meter traveled (when speed > 2 km/h)
   └── Encourages forward movement and exploration

5. SMOOTHNESS REWARD (weight: 0.5)
   └── Penalizes jerky steering (>0.3 change) and acceleration (>0.5 change)
   └── Promotes human-like smooth driving behavior

6. ROAD ALIGNMENT REWARD (weight: 1.0)
   └── Target: Stay aligned with road direction | Range: -1.0 to +0.2
   └── Heavy penalty (-1.0) for being >45° off road direction

7. TERMINATION BONUSES
   └── +10.0 bonus for successful episodes (>100m distance)
   └── Episode ends on: collision, >4m off road, or stuck (<1 km/h)

TOTAL REWARD RANGE:
└── Best case: ~+15.0/step (perfect highway driving)
└── Worst case: ~-170.0/step (collision while doing everything wrong)
└── Typical good driving: +2.0 to +5.0/step

TRAINING OBJECTIVE: Learn smooth, safe highway driving with lane keeping
==============================================================================
"""


class RewardManager:
        def __init__(self):
                # Reward weights (tune these for different behaviors)
                self.speed_weight = 1.0
                self.lane_weight = 2.0  
                self.collision_weight = 100.0
                self.progress_weight = 1.0
                self.smoothness_weight = 0.5
                self.alignment_weight = 1.0
                # Target parameters
                self.target_speed = 25.0  # km/h - optimal driving speed
                self.max_speed = 50.0     # km/h - speed limit
                self.max_lane_deviation = 2.0  # meters - acceptable lane deviation
                # Previous values for smoothness calculation
                self.prev_steering = 0.0
                self.prev_acceleration = 0.0
                self.prev_position = None
                # Episode tracking
                self.episode_distance = 0.0
                self.collision_occurred = False
                print(f"[REWARD MANAGER] Initialized with target_speed={self.target_speed} km/h")


        def calculate_reward(self, observation, action, terminated=False):
                """
                Calculate comprehensive reward based on observation and action.

                Args:
                observation: Dict with 'hero_state' and 'navigation' 
                action: [acceleration, steering] from action space
                terminated: Whether episode ended

                Returns:
                float: Total reward for this step
                """
                # Extract data from observation
                hero_state = observation['hero_state']
                navigation = observation['navigation']
                
                # Parse hero_state: [steering, throttle, brake, speed, x, y, yaw, vel_x, vel_y, collision]
                current_steering = hero_state[0]
                current_throttle = hero_state[1] 
                current_brake = hero_state[2]
                current_speed = hero_state[3] * 100.0  # Convert normalized back to km/h
                position_x = hero_state[4]
                position_y = hero_state[5]
                current_position = np.array([position_x, position_y])
                collision_status = hero_state[9]
                
                # Parse navigation: [distance_from_center, angle_to_road]
                distance_from_center = navigation[0] * 5.0  # Convert normalized back to meters
                angle_to_road = navigation[1] * np.pi       # Convert normalized back to radians
                
                # Parse action for smoothness
                current_acceleration = action[0] if action is not None else 0.0
                action_steering = action[1] if action is not None else 0.0
                
                # Initialize reward components
                reward_components = {}
                
                # 1. SPEED REWARD - Encourage maintaining target speed
                speed_reward = self._calculate_speed_reward(current_speed)
                reward_components['speed'] = speed_reward
                
                # 2. LANE KEEPING REWARD - Stay in lane center
                lane_reward = self._calculate_lane_reward(distance_from_center)
                reward_components['lane'] = lane_reward
                
                # 3. COLLISION PENALTY - Heavily penalize crashes
                collision_reward = self._calculate_collision_reward(collision_status)
                reward_components['collision'] = collision_reward
                
                # 4. PROGRESS REWARD - Encourage forward movement
                progress_reward = self._calculate_progress_reward(current_position, current_speed)
                reward_components['progress'] = progress_reward
                
                # 5. SMOOTHNESS REWARD - Encourage smooth driving
                smoothness_reward = self._calculate_smoothness_reward(
                current_steering, current_acceleration, action_steering
                )
                reward_components['smoothness'] = smoothness_reward
                
                # 6. ROAD ALIGNMENT REWARD - Stay aligned with road direction
                alignment_reward = self._calculate_alignment_reward(angle_to_road)
                reward_components['alignment'] = alignment_reward
                
                # 7. TERMINATION PENALTIES
                termination_reward = self._calculate_termination_reward(terminated, collision_status)
                reward_components['termination'] = termination_reward
                
                # Calculate total reward
                total_reward = (
                        speed_reward * self.speed_weight +
                        lane_reward * self.lane_weight + 
                        collision_reward * self.collision_weight +
                        progress_reward * self.progress_weight +
                        smoothness_reward * self.smoothness_weight +
                        alignment_reward * self.alignment_weight +
                        termination_reward
                )
                
                # Update previous values for next step
                self.prev_steering = current_steering
                self.prev_acceleration = current_acceleration  
                self.prev_position = current_position.copy()
                
                # Debug logging (remove in production)
                if np.random.random() < 0.1:  # Log 10% of steps
                        self._log_reward_breakdown(reward_components, total_reward)
                
                return float(total_reward)

        def _calculate_speed_reward(self, speed):
                """Reward for maintaining optimal speed"""
                if speed < 5.0:  # Too slow
                        return -0.5
                elif speed > self.max_speed:  # Too fast 
                        return -1.0
                else:
                        # Gaussian reward around target speed
                        speed_diff = abs(speed - self.target_speed)
                        return math.exp(-0.1 * speed_diff**2)
                
        def _calculate_lane_reward(self, distance_from_center):
                """Reward for staying in lane center"""
                if distance_from_center > self.max_lane_deviation:
                        return -2.0  # Heavy penalty for leaving lane
                else:
                        # Linear reward: closer to center = higher reward
                        normalized_distance = distance_from_center / self.max_lane_deviation
                        return 1.0 - normalized_distance

        def _calculate_collision_reward(self, collision_status):
                """Heavy penalty for collisions"""
                if collision_status > 0.5:  # Collision detected
                        self.collision_occurred = True
                        return -50.0  # Immediate large penalty
                return 0.0
        
        def _calculate_progress_reward(self, current_position, speed):
                """Reward for making forward progress"""
                if self.prev_position is None:
                        self.prev_position = current_position.copy()
                        return 0.0
                
                # Calculate distance traveled
                distance_moved = np.linalg.norm(current_position - self.prev_position)
                self.episode_distance += distance_moved
                
                # Reward based on forward progress
                if speed > 2.0:  # Only reward if moving meaningfully
                        return distance_moved * 0.1  # Small positive reward for progress
                return 0.0
        
         
        def _calculate_smoothness_reward(self, current_steering, current_acceleration, action_steering):
                """Reward smooth driving (avoid jerky movements)"""
                steering_change = abs(current_steering - self.prev_steering)
                acceleration_change = abs(current_acceleration - self.prev_acceleration)
                
                # Penalize large sudden changes
                steering_penalty = -steering_change * 0.5 if steering_change > 0.3 else 0.0
                accel_penalty = -acceleration_change * 0.3 if acceleration_change > 0.5 else 0.0
                
                return steering_penalty + accel_penalty

        def _calculate_alignment_reward(self, angle_to_road):
                """Reward for staying aligned with road direction"""
                angle_penalty = abs(angle_to_road)
                if angle_penalty > np.pi/4:  # 45 degrees
                        return -1.0  # Large penalty for being way off
                else:
                        # Small reward for good alignment
                        return 0.2 * (1.0 - angle_penalty / (np.pi/4))


        def reset(self):
                """Reset for new episode"""
                self.prev_steering = 0.0
                self.prev_acceleration = 0.0
                self.prev_position = None
                self.episode_distance = 0.0
                self.collision_occurred = False
                print("[REWARD MANAGER] Reset for new episode")
                return
        

        def _log_reward_breakdown(self, components, total):
                """Debug logging of reward components"""
                print(f"[REWARD] \n | Total: {total:.3f} \n | " + 
                "\n | ".join([f"{k}: {v:.3f}" for k, v in components.items()]))
    

        def _calculate_termination_reward(self, terminated, collision_status):
                """Handle episode termination rewards/penalties"""
                if terminated:
                        if collision_status > 0.5:
                                return -20.0  # Additional penalty for ending in collision
                        elif self.episode_distance > 100.0:  # Completed significant distance
                                return 10.0   # Bonus for successful episode
                return 0.0

        def get_termination_conditions(self, observation):
                """
                Check if episode should terminate.
                
                Returns:
                terminated: bool - Episode ended due to success/failure
                truncated: bool - Episode ended due to time limit
                """
                hero_state = observation['hero_state']
                collision_status = hero_state[9]
                distance_from_center = observation['navigation'][0] * 5.0  # Convert to meters
                
                # Terminate on collision
                if collision_status > 0.5:
                        return True, False
                
                # Terminate if too far from road
                if distance_from_center > 4.0:  # 4 meters off road
                        return True, False
                
                # Terminate if stuck (very low speed for too long)
                # speed = hero_state[3] * 100.0  # Convert to km/h
                # if speed < 1.0 and self.episode_distance < 5.0:  # Barely moved in episode
                #         return True, False
                
                return False, False
