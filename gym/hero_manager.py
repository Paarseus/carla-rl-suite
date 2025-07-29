import carla

import random
import time


class HeroManager:
        def __init__(self, client=None, world=None, sensor_interface=None, sensor_manager=None):
                if client is not None and world is not None:
                        self.client = client
                        self.world = world
                        self.hero = None
                        self.map = self.world.get_map()

                        print("[CONNECTION SUCCESS] HeroManager: Using provided CARLA client/world")
                if client is None or world is None:
                        raise RuntimeError("[CONNECTION FAILED] HeroManager: Failed to get CARLA client/world")
                
                if sensor_interface is not None and sensor_manager is not None:
                        self.sensor_interface = sensor_interface
                        self.sensor_manager = sensor_manager
                else:
                        raise RuntimeError("[SENSOR CONNECTION FAILED] HeroManager: Failed to get Sensor Interface/Sensor Manager")


        def reset_hero(self, hero_config):
                """This function resets / spawns the hero vehicle and its sensors"""

                # Part 1: destroy all sensors (if necessary)
                self.sensor_interface.destroy()

                self.world.tick()

                # Part 2: Spawn the ego vehicle
                user_spawn_points = hero_config["spawn_points"]
                if user_spawn_points:
                        spawn_points = []
                        for transform in user_spawn_points:

                                transform = [float(x) for x in transform.split(",")]
                                if len(transform) == 3:
                                        location = carla.Location(
                                                transform[0], transform[1], transform[2]
                                        )
                                        waypoint = self.map.get_waypoint(location)
                                        waypoint = waypoint.previous(random.uniform(0, 5))[0]
                                        transform = carla.Transform(
                                                location, waypoint.transform.rotation
                                        )
                                else:
                                        assert len(transform) == 6
                                        transform = carla.Transform(
                                                carla.Location(transform[0], transform[1], transform[2]),
                                                carla.Rotation(transform[4], transform[5], transform[3])
                                        )
                                spawn_points.append(transform)
                else:
                        spawn_points = self.map.get_spawn_points()

                self.hero_blueprints = self.world.get_blueprint_library().find(hero_config['blueprint'])
                self.hero_blueprints.set_attribute("role_name", "hero")

                # If already spawned, destroy it
                if self.hero is not None:
                        self.hero.destroy()
                        self.hero = None

                random.shuffle(spawn_points)
                for i in range(0,len(spawn_points)):
                        next_spawn_point = spawn_points[i % len(spawn_points)]
                        self.hero = self.world.try_spawn_actor(self.hero_blueprints, next_spawn_point)
                        if self.hero is not None:
                                print("[SPAWNED] Hero: ", self.hero)
                                break
                        else:
                                print("Could not spawn hero, changing spawn point")

                if self.hero is None:
                        print("We ran out of spawn points")
                        return

                self.world.tick()

                # Part 3: Spawn the new sensors
                for name, attributes in hero_config["sensors"].items():
                        sensor = self.sensor_manager.spawn(name, attributes, self.sensor_interface, self.hero)
                        print("[SPAWNED] Sensor: " , sensor)

                # Not needed anymore. This tick will happen when calling CarlaCore.tick()
                # self.world.tick()

                self.world.tick()

                return self.hero

        def set_spectator_camera_view(self):
                """This positions the spectator as a 3rd person view of the hero vehicle"""
                transform = self.hero.get_transform()

                # Get the camera position
                server_view_x = transform.location.x - 5 * transform.get_forward_vector().x
                server_view_y = transform.location.y - 5 * transform.get_forward_vector().y
                server_view_z = transform.location.z + 3

                # Get the camera orientation
                server_view_roll = transform.rotation.roll
                server_view_yaw = transform.rotation.yaw
                server_view_pitch = transform.rotation.pitch

                # Get the spectator and place it on the desired position
                self.spectator = self.world.get_spectator()
                self.spectator.set_transform(
                        carla.Transform(
                                carla.Location(x=server_view_x, y=server_view_y, z=server_view_z),
                                carla.Rotation(pitch=server_view_pitch,yaw=server_view_yaw,roll=server_view_roll),
                        )
                )

                self.world.tick()

        def apply_ego_control(self, control):
                """Applies the control calcualted at the experiment to the hero"""
                self.hero.apply_control(control)
                print("[ACTION APPLIED]")

        def get_sensor_data(self):
                """Returns the data sent by the different sensors at this tick"""
                sensor_data = self.sensor_interface.get_data()
                # print("---------")
                # world_frame = self.world.get_snapshot().frame
                # print("World frame: {}".format(world_frame))
                # for name, data in sensor_data.items():
                #     print("{}: {}".format(name, data[0]))
                return sensor_data
