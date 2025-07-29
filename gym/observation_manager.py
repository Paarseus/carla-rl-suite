import cv2
import carla
from gymnasium import spaces
import math
import numpy as np



class ObservationSpace:
        def __init__(self, hero_manager, sensor_interface, encoding_dim=512, use_resnet=True):
                self.hero_manager = hero_manager
                self.sensor_interface = sensor_interface
                self.max_position_offset = 1000.0  # meters from start
                self.max_lane_deviation = 5.0  # meters from lane center
                self.start_position = None  # Track starting position

                self.control = self.hero_manager.hero.get_control()

                print("[OBSERVATION SPACE] Initialized")


        def hero_control(self):
                self.control = self.hero_manager.hero.get_control()
                steering = self.control.steer
                throttle = self.control.throttle
                brake = self.control.brake
                speed = self.get_speed()

                return np.array([steering, throttle, brake, speed], dtype=np.float32)


        def get_acceleration(self):
                acceleration = self.hero.get_acceleration()
                accel_x = acceleration.x
                accel_y = acceleration.y
                accel_z = acceleration.z
                accel_magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)

                return np.array([accel_magnitude], dtype=np.float32)


        def get_navigation(self):
                """Simple fix for lane distance calculation"""
                if self.hero is None:
                        return np.array([0.0, 0.0], dtype=np.float32)
                
                world = self.hero.get_world()
                map_obj = world.get_map()
                
                # Get current waypoint
                current_waypoint = map_obj.get_waypoint(
                        self.hero.get_location(),
                        project_to_road=True,
                        lane_type=carla.LaneType.Driving
                )
                
                if current_waypoint is None:
                        return np.array([0.0, 0.0], dtype=np.float32)
                
                # Get next waypoint
                next_waypoint = current_waypoint.next(1.0)[0]
                
                # Convert to simple coordinates
                car_x = self.hero.get_location().x
                car_y = self.hero.get_location().y
                
                wp1_x = current_waypoint.transform.location.x
                wp1_y = current_waypoint.transform.location.y
                
                wp2_x = next_waypoint.transform.location.x
                wp2_y = next_waypoint.transform.location.y
                
                # Simple distance to line calculation
                A = np.array([wp2_x - wp1_x, wp2_y - wp1_y])  # Line direction
                B = np.array([car_x - wp1_x, car_y - wp1_y])  # Car relative to line start
                
                # Distance = |cross product| / |line length|
                cross = abs(A[0] * B[1] - A[1] * B[0])
                line_length = np.sqrt(A[0]**2 + A[1]**2)
                
                if line_length < 0.01:  # Avoid division by zero
                        lateral_distance = np.sqrt(B[0]**2 + B[1]**2)
                else:
                        lateral_distance = cross / line_length
                
                # Keep your existing heading calculation
                hero_heading = self.hero.get_transform().get_forward_vector()
                hero_heading_vec = np.array([hero_heading.x, hero_heading.y])
                
                wp_heading = current_waypoint.transform.get_forward_vector()
                wp_heading_vec = np.array([wp_heading.x, wp_heading.y])
                
                dot_product = np.dot(hero_heading_vec, wp_heading_vec)
                cross_product = hero_heading_vec[0] * wp_heading_vec[1] - hero_heading_vec[1] * wp_heading_vec[0]
                angle_to_road = np.arctan2(cross_product, dot_product)
                
                return np.array([lateral_distance, angle_to_road / np.pi], dtype=np.float32)

        def get_speed(self):
                vel = self.hero.get_velocity()
                speed = 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
                return speed


        def get_camera_data(self):
                # Get raw camera data
                sensor_data = self.sensor_interface.get_data()
                camera_data = sensor_data.get('camera_front')
                
                if camera_data is None:
                        print("No image received from cameras")
                        raw_image = np.zeros((300, 400, 3), dtype=np.uint8)  # Updated size to match your config
                else:
                        frame_id, raw_image = camera_data
                
        
                return raw_image
                

        def get_camera_data(self):
                sensor_data = self.sensor_interface.get_data()
                camera_data = sensor_data.get('camera_front')
                if camera_data is None:
                        print("No image recieved from cameras")
                        return np.zeros((300, 400, 3), dtype=np.uint8)
                
                frame_id, image_data = camera_data
                return image_data





        def get_collision_status(self):
                """Get collision detection status from collision sensor"""
                try:
                        sensor_data = self.sensor_interface.get_data()
                        collision_data = sensor_data.get('collision')
                        
                        if collision_data is not None:
                                # Collision detected!
                                frame_id, collision_event = collision_data
                                print(f"[COLLISION DETECTED] Frame: {frame_id}")
                                return 1.0
                        else:
                                # No collision
                                return 0.0
                        
                except Exception as e:
                        print(f"[COLLISION ERROR] {e}")
                        return 0.0


        def _get_empty_obs(self):
                """Return empty observation when hero is None"""
                return {
                        'control': np.zeros(4, dtype=np.float32),
                        'collision': np.zeros(1, dtype=np.float32),
                        'acceleration': np.zeros(1, dtype=np.float32),
                        'camera': np.zeros((300, 400, 3), dtype=np.uint8)
                }

        def reset(self):
                """Reset the observation space state (call this on environment reset)"""
                self.start_position = None
                print("[OBSERVATION SPACE] Reset - start position cleared")    
                

        def obs(self):
                self.hero = self.hero_manager.hero
                if self.hero is None:
                        print("[ERROR] No hero recieved by observation space, sending empty obs")
                        return self._get_empty_obs() 
                
                # Get all observation components
                control_data = self.hero_control()          # [steering, throttle, brake, speed]
                acceleration_data = self.get_acceleration()
                navigation_data = self.get_navigation()
                collision_status = self.get_collision_status()
                camera_data = self.get_camera_data()
                
                observation = {
                        'control': control_data.astype(np.float32),
                        'collision': np.array([collision_status], dtype=np.float32),
                        'acceleration': acceleration_data.astype(np.float32),
                        'navigation': navigation_data.astype(np.float32),
                        'camera': camera_data.astype(np.uint8)  # camera_data is just raw image
                }

                        
                return observation

