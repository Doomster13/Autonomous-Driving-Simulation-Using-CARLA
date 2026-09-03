import glob
import os
import sys
import time
import cv2
import random
import math
from collections import deque
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    print("Error: CARLA egg file not found. Please ensure CARLA is installed correctly.")
    pass

import carla

sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
    sys.version_info.major,
    sys.version_info.minor,
    'win-amd64'))[0])

#Constants
SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 10
EPISODES = 100_000

# DQN Agent settings
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
UPDATE_TARGET_EVERY = 5
MODEL_NAME = "HOC"
MIN_REWARD = -200
DISCOUNT = 0.99

# Epsilon
epsilon = 1
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 10


class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_height = IM_HEIGHT
    im_width = IM_WIDTH
    front_camera = None
    
    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.actor_list = []
        self.start_location = None
    
    def reset(self):
        self.cleanup()
        self.collision_hist = []

        transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, transform)
        self.actor_list.append(self.vehicle)

        self.start_location = self.vehicle.get_location()
        # Camera Sensor
        rgb_cam_bp = self.blueprint_library.find("sensor.camera.rgb")
        rgb_cam_bp.set_attribute("image_size_x", f"{self.im_width}")
        rgb_cam_bp.set_attribute("image_size_y", f"{self.im_height}")
        rgb_cam_bp.set_attribute("fov", "110")
        
        cam_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(rgb_cam_bp, cam_transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))
        
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        
        # Collision Sensor
        col_sensor_bp = self.blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(col_sensor_bp, carla.Transform(), attach_to=self.vehicle)
        self.actor_list.append(self.collision_sensor)
        self.collision_sensor.listen(lambda event: self.collision_data(event))
        
        while self.front_camera is None:
            time.sleep(0.01)
        
        self.episode_start = time.time()
        return self.front_camera

    def cleanup(self):
        for actor in self.actor_list:
            if actor.is_alive:
                actor.destroy()
        self.actor_list = []

    def collision_data(self, event):
        self.collision_hist.append(event)

    def process_img(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3] # Remove alpha channel
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3

    def step(self, action):
        if action == 0:  # Go Left
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1 * self.STEER_AMT))
        elif action == 1:  # Go Straight
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0))
        elif action == 2:  # Go Right
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1 * self.STEER_AMT))

        self.world.tick()

        done = False

        # PROGRESS-BASED REWARD LOGIC
        current_location = self.vehicle.get_location()
        distance_from_start = current_location.distance(self.start_location)

        # for avoiding circle motion in safe space
        reward = distance_from_start

        if len(self.collision_hist) > 0:
            done = True
            reward = -200  #penalty for collision

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        return self.front_camera, reward, done, None

class DQNAgent:
    def __init__(self):
        self.model = self.create_model()

        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        
        log_dir = f"logs/{MODEL_NAME}-{int(time.time())}"
        self.summary_writer = tf.compat.v1.summary.FileWriter(log_dir, graph=self.sess.graph)
        self.target_update_counter = 0

    def create_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (5, 5), strides=(2, 2), padding="same", input_shape=(IM_HEIGHT, IM_WIDTH, 3),
                                   activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),

            # 3 actions: left, straight, right
            tf.keras.layers.Dense(3, activation='linear')
        ])

        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch]) / 255.0
        current_qs_list = self.model.predict(current_states, verbose=0)

        new_current_states = np.array([transition[3] for transition in minibatch]) / 255.0
        future_qs_list = self.target_model.predict(new_current_states, verbose=0)

        X = []
        y = []

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)  # Append the original image (0-255)
            y.append(current_qs)

        self.model.train_on_batch(np.array(X), np.array(y))

    def update_target_model(self):
        self.target_update_counter += 1
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255.0, verbose=0)[0]

    def log_stats(self, step, **stats):
        summary = tf.compat.v1.Summary()
        for key, value in stats.items():
            summary.value.add(tag=key, simple_value=value)

        # Explicitly write the summary to the log file
        self.summary_writer.add_summary(summary, step)

if __name__ == '__main__':
    ep_rewards = [-200]

    random.seed(1)
    np.random.seed(1)
    tf.set_random_seed(1)

    # Create models folder
    if not os.path.isdir("models"):
        os.makedirs("models")
        
    # Initialize agent and environment
    agent = DQNAgent()
    env = CarEnv()

    
    try:
        # Main training loop
        for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit="episodes"):
            episode_reward = 0
            step = 1

            current_state = env.reset()
            done = False

            while not done:
                if np.random.random() > epsilon:
                    action = np.argmax(agent.get_qs(current_state))
                else:
                    action = np.random.randint(0, 3)

                # Step the environment
                new_state, reward, done, _ = env.step(action)
                episode_reward += reward

                # Add experience to memory
                agent.update_replay_memory((current_state, action, reward, new_state, done))

                # Only train every 4 steps for better performance
                if step % 4 == 0 and len(agent.replay_memory) >= MIN_REPLAY_MEMORY_SIZE:
                    agent.train()

                agent.update_target_model()

                current_state = new_state
                step += 1
            
            # Update target model after each episode
            agent.update_target_model()

            #log
            ep_rewards.append(episode_reward)
            if not episode % AGGREGATE_STATS_EVERY or episode == 1:
                average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
                min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
                max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
                
                stats_to_log = {
                    'reward_avg': average_reward,
                    'reward_min': min_reward,
                    'reward_max': max_reward,
                    'epsilon': epsilon
                }
                agent.log_stats(episode, **stats_to_log)
                
                if min_reward >= MIN_REWARD:
                    model_save_path = f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.h5'
                    agent.model.save(model_save_path)

            # Decay epsilon
            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)
        
        # Final model save
        final_model_path = f'models/{MODEL_NAME}_final_{int(time.time())}.h5'
        agent.model.save(final_model_path)
        print(f"Training complete. Final model saved to {final_model_path}")

    finally:
        print("Cleaning up CARLA actors...")
        env.cleanup()
        print("Cleanup complete.")