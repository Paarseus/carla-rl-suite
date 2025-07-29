#hero_config.py

hero_config = {
    "blueprint": "vehicle.mini.cooper",
    "spawn_points": [],
    "sensors": {
        # Front camera
        "camera_front": {
            "type": "sensor.camera.rgb",
            "transform": "0.7,0.0,1.3,0.0,0.0,0.0",
            "image_size_x": "400",
            "image_size_y": "300",
            "fov": "90"
        },
        
        # Rear camera
        "camera_rear": {
            "type": "sensor.camera.rgb",
            "transform": "-2.0,0.0,1.4,0.0,0.0,180.0",
            "image_size_x": "400",
            "image_size_y": "300",
            "fov": "90"
        },
        
        # Left camera
        "camera_left": {
            "type": "sensor.camera.rgb",
            "transform": "0.0,-1.5,1.4,0.0,0.0,-90.0",
            "image_size_x": "400",
            "image_size_y": "300",
            "fov": "90"
        },
        
        # # Right camera
        "camera_right": {
            "type": "sensor.camera.rgb",
            "transform": "0.0,1.5,1.4,0.0,0.0,90.0",
            "image_size_x": "400",
            "image_size_y": "300",
            "fov": "90"
        },

        # "lidar": {
        #     "type": "sensor.lidar.ray_cast",
        #     "transform": "0.0,0.0,2.0,0.0,0.0,0.0",  # 2m above vehicle center
        #     "channels": "64",              # Number of laser channels
        #     "range": "100.0",             # Maximum detection range in meters
        #     "points_per_second": "600000", # Points per second
        #     "rotation_frequency": "20.0",  # Rotation frequency in Hz
        #     "upper_fov": "15.0",          # Upper field of view in degrees
        #     "lower_fov": "-25.0",         # Lower field of view in degrees
        #     "horizontal_fov": "360.0",    # Horizontal field of view in degrees
        #     "atmosphere_attenuation_rate": "0.004",
        #     "dropoff_general_rate": "0.45",
        #     "dropoff_intensity_limit": "0.8",
        #     "dropoff_zero_intensity": "0.4"
        # },

        "collision": {
        "type": "sensor.other.collision",
        "transform": "0.0,0.0,0.0,0.0,0.0,0.0"  # Center of vehicle
        },
        
        # Top-down camera
        "camera_top": {
            "type": "sensor.camera.rgb",
            "transform": "0.0,0.0,10.0,0.0,-90.0,0.0",  # 5m up, pointing down
            "image_size_x": "400",
            "image_size_y": "300",
            "fov": "90"
        }
    }
}