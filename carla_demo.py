import carla
import numpy as np
import cv2
import time
from ultralytics import YOLO
import torch
import random

# Check PyTorch and CUDA
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Load YOLOv8 model
print("Loading YOLOv8 model...")
model = YOLO("runs/detect/kitti_yolov82/weights/best.pt")
print("Model loaded.")

# Connect to CARLA
print("Connecting to CARLA...")
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
try:
    world = client.get_world()
    print(f"Connected to CARLA world: {world.get_map().name}")
except Exception as e:
    print(f"Failed to connect: {e}")
    exit()

blueprint_library = world.get_blueprint_library()

# Spawn ego vehicle
print("Spawning ego vehicle...")
vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
spawn_points = world.get_map().get_spawn_points()
print(f"Total spawn points: {len(spawn_points)}")
ego_vehicle = None
for i, spawn_point in enumerate(spawn_points[:5]):
    try:
        ego_vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        print(f"Ego vehicle spawned at spawn point {i}: {ego_vehicle.get_location()}")
        break
    except RuntimeError as e:
        print(f"Spawn failed at point {i}: {e}")
if ego_vehicle is None:
    print("Failed to spawn ego vehicle after multiple attempts! Exiting...")
    exit()

# Traffic Manager setup
traffic_manager = client.get_trafficmanager()
traffic_manager.set_global_distance_to_leading_vehicle(1.0)

# List to track all actors
actors = []

# Spawn random background traffic
print("Spawning background traffic...")
for _ in range(50): 
    bp = random.choice(blueprint_library.filter('vehicle.*'))
    spawn_point = random.choice(spawn_points[1:]) 
    vehicle = world.try_spawn_actor(bp, spawn_point)
    if vehicle:
        actors.append(vehicle)
        vehicle.set_autopilot(True)
        traffic_manager.auto_lane_change(vehicle, True)
        print(f"Traffic vehicle spawned at {vehicle.get_location()}")

# Spawn obstacles (3 cars, 7 pedestrians)
print("Spawning obstacles...")
car_bp = random.choice(blueprint_library.filter('vehicle.*'))
ped_bp = random.choice(blueprint_library.filter('walker.pedestrian.*'))
for i in range(10):
    bp = car_bp if i < 3 else ped_bp 
    spawn_point = spawn_points[i + 1]
    actor = world.try_spawn_actor(bp, spawn_point)
    if actor:
        actors.append(actor)
        print(f"{'Car' if i < 3 else 'Pedestrian'} spawned at {actor.get_location()}")
        if 'walker' in actor.type_id:
            controller_bp = blueprint_library.find('controller.ai.walker')
            controller = world.spawn_actor(controller_bp, carla.Transform(), actor)
            controller.start()
            controller.go_to_location(world.get_random_location_from_navigation())
            actors.append(controller)

# Camera setup
rgb_camera_bp = blueprint_library.find('sensor.camera.rgb')
rgb_camera_bp.set_attribute('image_size_x', '800')
rgb_camera_bp.set_attribute('image_size_y', '600')
rgb_camera = world.spawn_actor(rgb_camera_bp, carla.Transform(carla.Location(x=1.5, z=2.0)), attach_to=ego_vehicle)
print("RGB camera spawned.")

# Sensor data
rgb_data = None
def process_rgb(image):
    global rgb_data
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    rgb_data = array.reshape((image.height, image.width, 4))[:, :, :3]
    print("RGB data received.")

rgb_camera.listen(lambda image: process_rgb(image))

# Autopilot for ego vehicle
ego_vehicle.set_autopilot(True)

# Warm-up ticks
print("Warming up simulation...")
for _ in range(10):
    world.tick()
    print("Warm-up tick")

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output/demo.avi', fourcc, 20.0, (800, 600))

# Simulation loop
print("Starting simulation loop...")
try:
    for i in range(600):
        world.tick()
        print(f"Tick {i+1}, Location: {ego_vehicle.get_location()}")
        if rgb_data is None:
            print("Waiting for RGB data...")
            continue
        results = model.predict(rgb_data, imgsz=800, save=False)
        annotated_frame = results[0].plot()
        cv2.imshow('Detection', annotated_frame)
        out.write(annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    print("Cleaning up...")
    rgb_camera.stop()
    ego_vehicle.destroy()
    rgb_camera.destroy()
    for actor in actors:
        actor.destroy()
    out.release()
    cv2.destroyAllWindows()
    print("Done.")