"""
Handles CARLA server connection and basic world setup.
"""

import carla
import random
from typing import Optional, Tuple
from connection_config import *
import time


class ClientConnection:
    """Manages connection to CARLA server and basic world operations."""
    
    def __init__(self, town: str = "Town01"):
        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.town = town
        
    def connect(self) -> Tuple[Optional[carla.Client], Optional[carla.World]]:
        """
        Establish connection to CARLA server.
        
        Returns:
            Tuple of (client, world) or (None, None) if connection fails
        """
        try:
            self.client = carla.Client(HOST, PORT)
            self.client.set_timeout(TIMEOUT)
            self.world = self.client.get_world()
            self.world.set_weather(carla.WeatherParameters.WetSunset)
            
            print(f"[CONNECTION SUCCESS] Successfully connected to CARLA server")
            print(f"Client version: {self.client.get_client_version()}")
            print(f"Server version: {self.client.get_server_version()}")
            
            return self.client, self.world
            
        except Exception as e:
            print(f'[CONNECTION FAIL] Failed to connect to CARLA server: {e}')
            self._print_version_info()
            return None, None
    
    def _print_version_info(self):
        """Print version information for debugging."""
        if self.client:
            client_version = self.client.get_client_version()
            server_version = self.client.get_server_version()
            print(f"Client version: {client_version}")
            print(f"Server version: {server_version}")
            
            if client_version != server_version:
                print("[WARNING] Client and Server version mismatch!")
    
    def test_spawn_vehicle(self) -> bool:
        """
        Test vehicle spawning capability.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = blueprint_library.filter(CAR_NAME)[0]
            spawn_points = self.world.get_map().get_spawn_points()
            
            if not spawn_points:
                raise RuntimeError("No spawn points available in this map.")
            
            spawn_point = random.choice(spawn_points)
            vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            vehicle.set_autopilot(True)
            # Set spectator to follow vehicle
            spectator = self.world.get_spectator()
            transform = carla.Transform(
                spawn_point.location + carla.Location(x=-1, z=1.5),
                spawn_point.rotation
            )
            spectator.set_transform(transform)
            print("[SPAWN SUCCESS] Vehicle spawned successfully for testing.")
            return True
            
        except Exception as e:
            print(f"[SPAWN FAIL] Failed during vehicle spawn test: {e}")
            return False
    
    def cleanup_all_actors(self):
        """Clean up all vehicles and sensors in the world."""
        if not self.world:
            return
            
        print("[CLEANING UP] ConnectionManager: Cleaning up all vehicles and sensors...")
        actors = self.world.get_actors()
        
        for actor in actors:
            if (actor.type_id.startswith("vehicle.") or 
                actor.type_id.startswith("sensor.") or actor.type_id.startswith("walker.")):
                try:
                    actor.destroy()
                    print(f"[DESTROY SUCCESS] ConnectionManager: {actor.type_id} (ID: {actor.id})")
                except Exception as e:
                    print(f"[DESTROY FAIL] ConnectionManager: Failed to destroy actor {actor.id}: {e}")                



if __name__ == "__main__":
   conn = ClientConnection(town="Town01")       
   client, world = conn.connect() 
   time.sleep(1)
   conn.cleanup_all_actors()
   time.sleep(1)
   conn.test_spawn_vehicle() 
   time.sleep(1)
   conn.cleanup_all_actors()

