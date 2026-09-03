#  Start of Modified DOC.py 
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
# Use tf.keras directly for Functional API
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import pickle

#  CARLA Egg finding (Improved) 
try:
    script_dir = os.path.dirname(os.path.realpath(__file__))
    egg_pattern = os.path.join(script_dir, '../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))

    found_eggs = glob.glob(egg_pattern)
    if not found_eggs:
        egg_pattern_alt = os.path.join(script_dir, '../../carla/dist/carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))
        found_eggs = glob.glob(egg_pattern_alt)

    if found_eggs:
        sys.path.append(found_eggs[0])
        print(f"CARLA egg found and added to path: {found_eggs[0]}")
    else:
        raise FileNotFoundError(f"CARLA egg file not found matching pattern: {egg_pattern} or {egg_pattern_alt}")

except FileNotFoundError as e:
    print(e)
    sys.exit()
except Exception as e:
    print(f"An unexpected error occurred during CARLA egg loading: {e}")
    sys.exit()

import carla
#  End CARLA Egg finding 

# Constants
SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 30 # Increased episode time for goal-reaching
EPISODES = 100_000

# DQN Agent settings
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
UPDATE_TARGET_EVERY = 5 # Steps between target model updates
MODEL_NAME = "DOC_Goal" # Updated model name
MIN_REWARD = -200
DISCOUNT = 0.99

epsilon = 1
EPSILON_DECAY = 0.9999 # Slower decay Expected to be around 40000 ep
MIN_EPSILON = 0.01 # Higher min epsilon

AGGREGATE_STATS_EVERY = 10
SAVE_STATE_EVERY = AGGREGATE_STATS_EVERY * 5 # Save every 50 episodes

#  Save/Load Paths 
SAVE_DIR = "saved_states_goal" # Separate directory
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, f"{MODEL_NAME}_latest.h5")
REPLAY_MEMORY_SAVE_PATH = os.path.join(SAVE_DIR, f"{MODEL_NAME}_replay_memory.pkl")
METADATA_SAVE_PATH = os.path.join(SAVE_DIR, f"{MODEL_NAME}_metadata.pkl")
# 

#  Normalization constants for goal vector 
# Estimate max possible distance in your map or use a large enough value
MAX_DISTANCE = 300.0 # Adjust based on your map size/typical goal distances
MAX_ANGLE_DEGREES = 180.0

#  CarEnv Class with Goal Vector 
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
        self.collision_hist = []
        self.goal_location = None
        self.start_location = None
        self.prev_distance_to_goal = None

        self.goal_bonus = 500
        self.collision_penalty = -200
        self.closer_reward_scale = 2.0 # Increased scaling for getting closer
        self.goal_reach_threshold = 5.0 # Meters within goal

    def _get_goal_info(self):
        """ Calculates distance and relative angle to the goal. Returns normalized values. """
        if self.vehicle is None or self.goal_location is None:
            return np.array([1.0, 0.0], dtype=np.float32) # Return normalized "far away, straight ahead" default

        vehicle_location = self.vehicle.get_location()
        vehicle_transform = self.vehicle.get_transform()
        vehicle_yaw_rad = math.radians(vehicle_transform.rotation.yaw)

        distance = vehicle_location.distance(self.goal_location)

        # Calculate vector from vehicle to goal in world coordinates
        goal_vector_world = self.goal_location - vehicle_location
        # Calculate angle of the goal vector in world coordinate system (degrees)
        goal_angle_world_rad = math.atan2(goal_vector_world.y, goal_vector_world.x)

        # Calculate relative angle: angle_world - vehicle_yaw
        relative_angle_rad = goal_angle_world_rad - vehicle_yaw_rad
        # Normalize angle to be within [-pi, pi]
        relative_angle_rad = (relative_angle_rad + math.pi) % (2 * math.pi) - math.pi

        # Normalize distance and angle for network input
        normalized_distance = min(distance / MAX_DISTANCE, 1.0) # Clip at 1.0
        normalized_angle = relative_angle_rad / math.pi # Normalize angle to [-1, 1]

        return np.array([normalized_distance, normalized_angle], dtype=np.float32)

    def reset(self):
        self.cleanup()
        self.collision_hist = []
        self.front_camera = None
        self.goal_location = None
        self.start_location = None
        self.prev_distance_to_goal = None

        start_transform = random.choice(self.world.get_map().get_spawn_points())
        try:
            # Try spawning, wait a bit if occupied
            self.vehicle = self.world.try_spawn_actor(self.model_3, start_transform)
            retry_count = 0
            while self.vehicle is None and retry_count < 5:
                 print(f"Spawn point occupied, trying another...")
                 time.sleep(0.5)
                 start_transform = random.choice(self.world.get_map().get_spawn_points())
                 self.vehicle = self.world.try_spawn_actor(self.model_3, start_transform)
                 retry_count += 1
            if self.vehicle is None:
                 raise RuntimeError("Failed to spawn vehicle after multiple attempts.")

            self.actor_list.append(self.vehicle)
            self.start_location = start_transform.location
        except RuntimeError as e:
            print(f"Error spawning vehicle: {e}. Retrying reset...")
            time.sleep(1)
            return self.reset() # Retry resetting the entire environment

        possible_goals = self.world.get_map().get_spawn_points()
        goal_transform = random.choice(possible_goals)
        while goal_transform.location.distance(self.start_location) < 20: # Ensure goal is reasonably far
             goal_transform = random.choice(possible_goals)
        self.goal_location = goal_transform.location

        # Debug draw goal
        self.world.debug.draw_string(self.goal_location, 'GOAL', draw_shadow=False,color=carla.Color(r=0, g=255, b=0), life_time=SECONDS_PER_EPISODE+1, persistent_lines=True)

        self.prev_distance_to_goal = self.vehicle.get_location().distance(self.goal_location)

        # Camera Sensor setup (same as before)
        rgb_cam_bp = self.blueprint_library.find("sensor.camera.rgb")
        rgb_cam_bp.set_attribute("image_size_x", f"{self.im_width}")
        rgb_cam_bp.set_attribute("image_size_y", f"{self.im_height}")
        rgb_cam_bp.set_attribute("fov", "110")
        cam_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(rgb_cam_bp, cam_transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        # Collision Sensor setup (same as before)
        col_sensor_bp = self.blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(col_sensor_bp, carla.Transform(), attach_to=self.vehicle)
        self.actor_list.append(self.collision_sensor)
        self.collision_sensor.listen(lambda event: self.collision_data(event))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(0.5) # Allow sensors to initialize and first image to arrive

        start_wait_time = time.time()
        while self.front_camera is None:
            if time.time() - start_wait_time > 5:
                 print("Error: Camera feed timed out during reset.")
                 self.cleanup()
                 raise TimeoutError("CARLA camera sensor failed to provide data during reset.")
            time.sleep(0.01)

        self.episode_start = time.time()
        initial_goal_info = self._get_goal_info() # Get initial goal vector
        return (self.front_camera, initial_goal_info) # Return tuple state


    def cleanup(self):
        # Stop sensors if they exist and are listening
        if hasattr(self, 'sensor') and self.sensor is not None and getattr(self.sensor, 'is_listening', False):
             self.sensor.stop()
        if hasattr(self, 'collision_sensor') and self.collision_sensor is not None and getattr(self.collision_sensor, 'is_listening', False):
             self.collision_sensor.stop()

        # Destroy actors if they exist and are alive
        actors_to_destroy = [actor for actor in self.actor_list if actor is not None and actor.id is not None and actor.is_alive]
        if actors_to_destroy:
             self.client.apply_batch([carla.command.DestroyActor(x) for x in actors_to_destroy])

        self.actor_list = []
        self.vehicle = None # Explicitly clear references
        self.sensor = None
        self.collision_sensor = None
        # Clear debug drawings
        self.world.debug.draw_string(carla.Location(), '', life_time=0.1, persistent_lines=True)


    def collision_data(self, event):
        self.collision_hist.append(event)


    def process_img(self, image):
        if not hasattr(image, 'raw_data'): return
        i = np.array(image.raw_data)
        expected_elements = self.im_height * self.im_width * 4
        if i.shape[0] != expected_elements: return
        try:
            i2 = i.reshape((self.im_height, self.im_width, 4))
            i3 = i2[:, :, :3]
            if self.SHOW_CAM:
                cv2.imshow("", i3)
                cv2.waitKey(1)
            self.front_camera = i3
        except ValueError as e:
            print(f"Error reshaping image: {e}")
            self.front_camera = None

    def step(self, action):
        if self.vehicle is None or not self.vehicle.is_alive:
             print("Error: Vehicle is not valid in step.")
             dummy_img = np.zeros((self.im_height, self.im_width, 3), dtype=np.uint8)
             dummy_goal = np.array([1.0, 0.0], dtype=np.float32)
             return (dummy_img, dummy_goal), self.collision_penalty, True, None

        # Apply control (same as before)
        if action == 0: control = carla.VehicleControl(throttle=1.0, steer=-1 * self.STEER_AMT, brake=0.0)
        elif action == 1: control = carla.VehicleControl(throttle=1.0, steer=0, brake=0.0)
        elif action == 2: control = carla.VehicleControl(throttle=1.0, steer=1 * self.STEER_AMT, brake=0.0)
        else: control = carla.VehicleControl(throttle=0.0, brake=1.0)
        self.vehicle.apply_control(control)

        self.world.tick()

        #  Calculate Reward 
        reward = 0
        done = False

        if len(self.collision_hist) > 0:
            done = True
            reward = self.collision_penalty
            print("Collision") # Less verbose

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True
            reward -= 10 # Small penalty for timeout
            print("Timeout") # Less verbose

        current_goal_info = np.array([1.0, 0.0], dtype=np.float32) # Default invalid state
        if not done:
             try:
                 current_location = self.vehicle.get_location()
                 if self.goal_location is None:
                      print("Error: Goal location missing during step.")
                      done = True; reward = self.collision_penalty
                 else:
                    distance_to_goal = current_location.distance(self.goal_location)
                    current_goal_info = self._get_goal_info() # Get current goal vector

                    # Reward getting closer, penalize getting farther
                    if self.prev_distance_to_goal is not None:
                         # Use non-normalized distance for reward calculation
                         distance_delta = self.prev_distance_to_goal - distance_to_goal
                         if distance_delta > 0: # Got closer
                              reward += distance_delta * self.closer_reward_scale
                         else: # Got farther or stayed same
                              reward += distance_delta * self.closer_reward_scale * 0.5 # Smaller penalty for getting farther
                    self.prev_distance_to_goal = distance_to_goal

                    # Bonus for reaching goal
                    if distance_to_goal < self.goal_reach_threshold:
                         done = True
                         reward += self.goal_bonus
                         print("GOAL REACHED!")

             except Exception as e:
                  print(f"Error calculating goal distance/reward: {e}")
                  done = True; reward = self.collision_penalty


        # Get current camera state, handle None case
        current_cam_state = self.front_camera
        if current_cam_state is None:
            # If camera is None mid-episode, it's a problem
            print("Warning: front_camera is None in step. Using zero state.")
            current_cam_state = np.zeros((self.im_height, self.im_width, 3), dtype=np.uint8)
            # Optionally force done=True if sensor fails repeatedly


        # Return tuple state: (image, goal_vector)
        return (current_cam_state, current_goal_info), reward, done, None

#  DQNAgent Class with Functional Model 
class DQNAgent:
    def __init__(self, load_model_path=None, load_replay_path=None):
        self.vector_input_shape = (2,) # distance, angle
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # Load replay memory or initialize
        if load_replay_path and os.path.exists(load_replay_path):
             print(f"Loading replay memory from {load_replay_path}...")
             try:
                 with open(load_replay_path, 'rb') as f:
                      self.replay_memory = pickle.load(f)
                 print(f"Loaded {len(self.replay_memory)} transitions.")
             except (EOFError, pickle.UnpicklingError, FileNotFoundError) as e:
                  print(f"Error loading replay memory ({e}). Initializing new.")
                  self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        else:
             print("Initializing new replay memory.")
             self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # TF 1.x Session Setup
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        tf.keras.backend.set_session(self.sess)

        # Initialize TF global variables
        try:
            self.sess.run(tf.global_variables_initializer())
            print("Global variables initialized.")
        except Exception as e:
            print(f"Error initializing TF variables: {e}")
            # Consider exiting or handling this more gracefully

        # Load model weights AFTER initializing variables
        if load_model_path and os.path.exists(load_model_path):
             print(f"Loading model weights from {load_model_path}...")
             try:
                 self.model.load_weights(load_model_path)
                 self.target_model.set_weights(self.model.get_weights())
                 print("Model weights loaded.")
             except Exception as e:
                 print(f"Error loading model weights: {e}. Using newly initialized weights.")
        else:
             print("Using newly initialized model weights.")

        # Logging setup
        log_dir = f"logs/{MODEL_NAME}-{int(time.time())}"
        self.summary_writer = tf.summary.FileWriter(log_dir)
        self.target_update_counter = 0

    def create_model(self):
        # Image Input Branch (CNN)
        image_input = Input(shape=(IM_HEIGHT, IM_WIDTH, 3), name='image_input')
        x = Conv2D(32, (5, 5), strides=(2, 2), padding="same", activation='relu')(image_input)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(64, (3, 3), padding="same", activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(64, (3, 3), padding="same", activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x) # CNN feature vector

        # Goal Vector Input Branch
        vector_input = Input(shape=self.vector_input_shape, name='vector_input')
        # Optional: Process vector input through a small dense layer
        y = Dense(16, activation='relu')(vector_input)

        # Concatenate features
        combined = Concatenate()([x, y])

        # Final Dense layers for Q-values
        z = Dense(256, activation='relu')(combined) # Added another dense layer
        output = Dense(3, activation='linear', name='output')(z) # 3 actions

        # Create and compile model
        model = Model(inputs=[image_input, vector_input], outputs=output)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])
        print("Model Summary:")
        model.summary() # Print model structure
        return model

    def update_replay_memory(self, transition):
        # Transition is now ( (img_state, vec_state), action, reward, (new_img_state, new_vec_state), done )
        current_state_tuple, action, reward, new_state_tuple, done = transition

        # Validate states before appending
        valid = True
        if not isinstance(current_state_tuple, tuple) or len(current_state_tuple) != 2 or \
           not isinstance(current_state_tuple[0], np.ndarray) or not isinstance(current_state_tuple[1], np.ndarray):
            print("Warning: Invalid current state format.")
            valid = False
        if not isinstance(new_state_tuple, tuple) or len(new_state_tuple) != 2 or \
           not isinstance(new_state_tuple[0], np.ndarray) or not isinstance(new_state_tuple[1], np.ndarray):
             print("Warning: Invalid new state format.")
             valid = False

        if valid:
             # Ensure vector states have the correct shape
             if current_state_tuple[1].shape == self.vector_input_shape and new_state_tuple[1].shape == self.vector_input_shape:
                 self.replay_memory.append(transition)
             else:
                 print(f"Warning: Vector state shape mismatch. Expected {self.vector_input_shape}, Got current={current_state_tuple[1].shape}, new={new_state_tuple[1].shape}")
        else:
             print("Skipping adding transition to replay memory due to invalid format.")


    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        try: # Add try-except around training batch
            minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

            #  Prepare inputs for the multi-input model 
            current_images = np.array([transition[0][0] for transition in minibatch]) / 255.0
            current_vectors = np.array([transition[0][1] for transition in minibatch])
            # Note: Vector data is assumed to be pre-normalized/scaled by CarEnv._get_goal_info

            new_images = np.array([transition[3][0] for transition in minibatch]) / 255.0
            new_vectors = np.array([transition[3][1] for transition in minibatch])

            # Get current and future Q-values using the multi-input predict
            current_qs_list = self.model.predict_on_batch([current_images, current_vectors])
            future_qs_list = self.target_model.predict_on_batch([new_images, new_vectors])

            y = [] # Target Q-values

            for index, (current_state_tuple, action, reward, new_state_tuple, done) in enumerate(minibatch):
                if not done:
                    max_future_q = np.max(future_qs_list[index])
                    new_q = reward + DISCOUNT * max_future_q
                else:
                    new_q = reward

                current_qs = current_qs_list[index]
                current_qs[action] = new_q
                y.append(current_qs)

            # Train using the multi-input format
            self.model.train_on_batch([current_images, current_vectors], np.array(y))

        except Exception as e:
             print(f"Error during training step: {e}")
             # Log the error or potentially skip this training step


    def update_target_model(self):
        self.target_update_counter += 1
        if self.target_update_counter >= UPDATE_TARGET_EVERY: # Use >= for safety
            print("Updating target model")
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state_tuple):
        image_state, vector_state = state_tuple

        # Prepare inputs for prediction (batch dimension needed)
        image_state_norm = np.array(image_state, dtype=np.float32) / 255.0
        image_batch = image_state_norm.reshape(-1, *image_state_norm.shape) # Add batch dim

        # Vector state should already be normalized/scaled
        vector_batch = vector_state.reshape(-1, *vector_state.shape) # Add batch dim

        # Predict using multi-input format
        q_values = self.model.predict([image_batch, vector_batch], verbose=0)[0]
        return q_values


    def log_stats(self, step, **stats):
        try:
            summary = tf.Summary()
            for key, value in stats.items():
                summary.value.add(tag=key, simple_value=value)
            self.summary_writer.add_summary(summary, step)
            self.summary_writer.flush()
        except Exception as e:
            print(f"Error writing summary: {e}")

#  Main execution block (with modifications for tuple state) 
if __name__ == '__main__':
    ep_rewards = [-200]
    start_episode = 1

    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"Created save directory: {SAVE_DIR}")

    # Load metadata
    if os.path.exists(METADATA_SAVE_PATH):
        print(f"Loading metadata from {METADATA_SAVE_PATH}...")
        try:
            with open(METADATA_SAVE_PATH, 'rb') as f:
                metadata = pickle.load(f)
                start_episode = metadata['episode'] + 1
                epsilon = metadata['epsilon']
                ep_rewards = metadata.get('ep_rewards', [-200])
            print(f"Resuming from episode {start_episode}, epsilon: {epsilon:.4f}")
        except Exception as e:
            print(f"Error loading metadata: {e}. Starting from scratch.")
            start_episode = 1
            epsilon = 1; ep_rewards = [-200]
            # If metadata fails, potentially delete corrupted replay/model too?
            if os.path.exists(REPLAY_MEMORY_SAVE_PATH): os.remove(REPLAY_MEMORY_SAVE_PATH)
            if os.path.exists(MODEL_SAVE_PATH): os.remove(MODEL_SAVE_PATH)
    else:
        print("No metadata found. Starting from scratch.")

    # Initialize agent
    agent = DQNAgent(load_model_path=MODEL_SAVE_PATH if os.path.exists(MODEL_SAVE_PATH) else None,
                     load_replay_path=REPLAY_MEMORY_SAVE_PATH if os.path.exists(REPLAY_MEMORY_SAVE_PATH) else None)

    env = CarEnv()

    # Seeding
    random.seed(start_episode)
    np.random.seed(start_episode)
    tf.random.set_random_seed(start_episode)


    try:
        for episode in tqdm(range(start_episode, EPISODES + 1), initial=start_episode, total=EPISODES, ascii=True, unit="episodes"):
            episode_reward = 0
            step = 1

            try:
                current_state_tuple = env.reset() # Now returns (image, vector)
                # Basic validation after reset
                if not isinstance(current_state_tuple, tuple) or len(current_state_tuple) != 2 or \
                   current_state_tuple[0] is None or current_state_tuple[0].shape != (IM_HEIGHT, IM_WIDTH, 3) or \
                   current_state_tuple[1] is None or current_state_tuple[1].shape != agent.vector_input_shape:
                     print(f"Error: env.reset() returned invalid state format/content in episode {episode}. Skipping.")
                     time.sleep(1); continue
                current_state_img, current_state_vec = current_state_tuple

            except TimeoutError as e:
                print(f"Timeout during env.reset() in episode {episode}: {e}. Skipping episode.")
                time.sleep(5); continue
            except Exception as e:
                print(f"Unexpected error during env.reset() in episode {episode}: {e}. Skipping...")
                time.sleep(2); continue

            done = False
            # start_time = time.time() # Already have episode_start in CarEnv

            while not done:
                # Validate current state tuple before using
                if current_state_img is None or current_state_img.shape != (IM_HEIGHT, IM_WIDTH, 3) or \
                   current_state_vec is None or current_state_vec.shape != agent.vector_input_shape:
                     print(f"Error: Invalid current state tuple in loop ep {episode}, step {step}. Ending episode.")
                     break # Exit inner loop

                # Get action based on combined state
                if np.random.random() > epsilon:
                    action = np.argmax(agent.get_qs((current_state_img, current_state_vec)))
                else:
                    action = np.random.randint(0, 3)

                try:
                    # Step returns ( (img, vec), reward, done, _)
                    new_state_tuple, reward, done, _ = env.step(action)

                    # Validate new state tuple
                    if not isinstance(new_state_tuple, tuple) or len(new_state_tuple) != 2 or \
                       new_state_tuple[0] is None or new_state_tuple[0].shape != (IM_HEIGHT, IM_WIDTH, 3) or \
                       new_state_tuple[1] is None or new_state_tuple[1].shape != agent.vector_input_shape:
                        print(f"Error: env.step() returned invalid new_state format/content in ep {episode}, step {step}. Ending.")
                        done = True; reward = env.collision_penalty
                        # Use previous valid state if possible
                        new_state_img = current_state_img
                        new_state_vec = current_state_vec
                    else:
                        new_state_img, new_state_vec = new_state_tuple

                except Exception as e:
                    print(f"Error during env.step() in episode {episode}, step {step}: {e}. Ending episode.")
                    done = True; reward = env.collision_penalty
                    new_state_img = current_state_img # Use last known good state
                    new_state_vec = current_state_vec

                episode_reward += reward

                # Update replay memory with tuple states
                agent.update_replay_memory(((current_state_img, current_state_vec), action, reward, (new_state_img, new_state_vec), done))

                # Train
                if len(agent.replay_memory) >= MIN_REPLAY_MEMORY_SIZE and step % 4 == 0:
                    agent.train() # Train handles the tuple state internally

                # Update target model check (every step, internal counter)
                agent.update_target_model()

                # Update current state for next iteration
                current_state_img = new_state_img
                current_state_vec = new_state_vec
                step += 1

            #  End of Episode 

            # Log stats (same as before)
            ep_rewards.append(episode_reward)
            if not episode % AGGREGATE_STATS_EVERY or episode == 1:
                average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
                min_reward_agg = min(ep_rewards[-AGGREGATE_STATS_EVERY:]) # Use different name
                max_reward_agg = max(ep_rewards[-AGGREGATE_STATS_EVERY:]) # Use different name

                stats_to_log = {
                    'reward_avg': average_reward,
                    'reward_min': min_reward_agg,
                    'reward_max': max_reward_agg,
                    'epsilon': epsilon,
                    'replay_memory_size': len(agent.replay_memory),
                    'episode_reward': episode_reward # Log current episode reward too
                }
                agent.log_stats(episode, **stats_to_log)


            # Save state periodically (same as before)
            if not episode % SAVE_STATE_EVERY:
                try:
                    print(f"\nSaving state at episode {episode}...")
                    agent.model.save(MODEL_SAVE_PATH)
                    print(f"  Model saved to {MODEL_SAVE_PATH}")
                    with open(REPLAY_MEMORY_SAVE_PATH, 'wb') as f:
                        pickle.dump(agent.replay_memory, f)
                    print(f"  Replay memory saved ({len(agent.replay_memory)} transitions)")
                    metadata = {'episode': episode, 'epsilon': epsilon, 'ep_rewards': ep_rewards}
                    with open(METADATA_SAVE_PATH, 'wb') as f:
                        pickle.dump(metadata, f)
                    print(f"  Metadata saved (Episode: {episode}, Epsilon: {epsilon:.4f})\n")
                except Exception as e:
                    print(f"Error saving state at episode {episode}: {e}")

            # Decay epsilon (same as before)
            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)

    #  End of Training Loop 

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving final state...")
        try:
            current_episode_num = episode if 'episode' in locals() else start_episode -1 # Get last completed or starting ep
            agent.model.save(MODEL_SAVE_PATH)
            with open(REPLAY_MEMORY_SAVE_PATH, 'wb') as f: pickle.dump(agent.replay_memory, f)
            metadata = {'episode': current_episode_num, 'epsilon': epsilon, 'ep_rewards': ep_rewards}
            with open(METADATA_SAVE_PATH, 'wb') as f: pickle.dump(metadata, f)
            print("Latest model, replay memory, and metadata saved.")
        except Exception as e:
            print(f"Error saving state during interruption: {e}")

    finally:
        print("Cleaning up CARLA actors...")
        if 'env' in locals() and env is not None:
             try: env.cleanup(); print("Cleanup complete.")
             except Exception as e: print(f"Error during final cleanup: {e}")
        else: print("Environment object not found for cleanup.")

        if 'agent' in locals():
             if hasattr(agent, 'sess') and agent.sess is not None:
                  print("Closing TensorFlow session."); agent.sess.close()
             if hasattr(agent, 'summary_writer') and agent.summary_writer is not None:
                  agent.summary_writer.close()
