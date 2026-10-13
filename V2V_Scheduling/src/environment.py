import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.vehicle import Vehicle
from src.message import Message
from src.semantic_model import SemanticModel
from src.aoi_tracker import AoITracker
from src.channel_model import ChannelModel
from src.metrics import MetricsTracker


class V2VEnv(gym.Env):
    def __init__(self, config):
        super(V2VEnv, self).__init__()
        self.config = config
        self.num_vehicles = config['simulation']['num_vehicles']
        self.dt = config['simulation']['dt_seconds']
        self.episode_steps = config['simulation']['episode_steps']
        self.capacity = config['simulation']['transmission_resources']
        self.max_queue = config['simulation']['max_queue_size']

        self.semantic_model = SemanticModel(
            config['messages']['safety_priority'],
            config['messages']['traffic_priority'],
            config['messages']['routine_priority'],
        )
        self.channel = ChannelModel()
        self.aoi_tracker = AoITracker(self.num_vehicles)
        self.metrics = MetricsTracker(self.num_vehicles)

        self.action_space = spaces.MultiDiscrete([3] * self.num_vehicles)
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf,
            shape=(self.num_vehicles, 6), dtype=np.float32
        )

        self.vehicles = []
        self.current_time = 0.0
        self.step_count = 0
        self.global_state = None

    # ---------- Gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.current_time = 0.0
        self.step_count = 0
        self.metrics = MetricsTracker(self.num_vehicles)
        self.aoi_tracker = AoITracker(self.num_vehicles)

        self.vehicles = []
        for i in range(self.num_vehicles):
            pos = np.random.uniform(0, 1000)
            speed = np.random.uniform(10, 30)
            self.vehicles.append(Vehicle(i, pos, speed, self.max_queue))

        self.global_state = self._get_global_state()
        return self._get_observations(), {}

    def step(self, actions):
        self.current_time += self.dt
        self.step_count += 1

        for v in self.vehicles:
            v.update_position(self.dt)

        self._generate_messages()
        requested_transmissions = self._resolve_actions(actions)

        per_agent_rewards = np.zeros(self.num_vehicles, dtype=np.float32)
        delivered_per_agent = np.zeros(self.num_vehicles, dtype=np.float32)

        alpha = self.config['reward']['alpha_semantic']
        beta = self.config['reward']['beta_aoi']
        gamma = self.config['reward']['gamma_penalty']

        attempted = len(requested_transmissions)
        delivered = 0
        delivered_utility = 0.0

        for tx in requested_transmissions:
            sender_id = tx['sender_id']
            msg = tx['message']
            sender = self.vehicles[sender_id]

            receivers = [r for r in range(self.num_vehicles) if r != sender_id]
            if not receivers:
                continue
            receiver_id = max(
                receivers,
                key=lambda r: self.aoi_tracker.aoi_matrix[sender_id, r]
            )
            receiver = self.vehicles[receiver_id]

            aoi_before = float(self.aoi_tracker.aoi_matrix[sender_id, receiver_id])
            distance = abs(sender.position - receiver.position)
            success = self.channel.attempt_transmission(distance)

            if success:
                delivered += 1
                delivered_per_agent[sender_id] += 1
                delivered_utility += msg.importance

                self.aoi_tracker.update_aoi(
                    sender_id, receiver_id,
                    self.current_time, msg.creation_time
                )
                aoi_after = float(self.aoi_tracker.aoi_matrix[sender_id, receiver_id])
                improvement = max(0.0, aoi_before - aoi_after)

                per_agent_rewards[sender_id] += (
                    alpha * msg.importance + beta * improvement
                )

                try:
                    sender.message_queue.remove(msg)
                except ValueError:
                    pass
            else:
                per_agent_rewards[sender_id] -= gamma

        # Time advances for AoI
        self.aoi_tracker.step(self.dt)

        # Small shared AoI penalty
        avg_aoi = self.aoi_tracker.get_average_aoi()
        per_agent_rewards -= 0.01 * (avg_aoi / 10.0)

        # Fairness bonus: starved vehicles get a stronger positive nudge
        if self.metrics.delivery_counts is not None:
            recent = self.metrics.delivery_counts
        else:
            recent = np.zeros(self.num_vehicles, dtype=np.float32)
        per_agent_rewards += 1.0 / (1.0 + recent)

        # Semantic-weighted AoI (for evaluation, not reward)
        weighted_aoi = self.metrics.get_semantic_weighted_aoi(self.aoi_tracker.aoi_matrix)

        self.metrics.record_transmission(attempted, delivered, delivered_utility)
        self.metrics.record_aoi(avg_aoi)
        self.metrics.record_weighted_aoi(weighted_aoi)
        self.metrics.record_deliveries(delivered_per_agent)

        terminated = self.step_count >= self.episode_steps
        truncated = False

        info = {
            'avg_aoi': avg_aoi,
            'weighted_aoi': weighted_aoi,
            'pdr': self.metrics.get_pdr(),
            'semantic_utility': self.metrics.semantic_utility,
            'attempted': attempted,
            'delivered': delivered,
            'fairness': self.metrics.get_fairness(),
            'per_agent_rewards': per_agent_rewards.copy(),
        }

        obs = self._get_observations()
        self.global_state = self._get_global_state()

        return obs, per_agent_rewards, terminated, truncated, info

    # ---------- Internals ----------
    def _generate_messages(self):
        for v in self.vehicles:
            if np.random.rand() < self.config['messages']['arrival_prob']:
                class_type = np.random.choice(['safety', 'traffic', 'routine'])
                importance = self.semantic_model.get_importance(class_type)
                msg = Message(v.id, -1, class_type, importance, self.current_time)
                v.enqueue_message(msg)
                self.metrics.record_message(v.id, importance)

    def _resolve_actions(self, actions):
        requests = []
        for i, action in enumerate(actions):
            v = self.vehicles[i]
            if action == 0 or len(v.message_queue) == 0:
                continue

            if action == 1:
                msg = max(v.message_queue, key=lambda m: m.importance)
                score = 10.0 * msg.importance
            elif action == 2:
                msg = v.message_queue[-1]
                age = msg.get_age(self.current_time)
                score = 10.0 / (1.0 + age)
            else:
                continue

            requests.append({
                'sender_id': i,
                'message': msg,
                'priority': score,
            })

        requests.sort(key=lambda x: x['priority'], reverse=True)
        return requests[:self.capacity]

    def _get_observations(self):
        obs = np.zeros((self.num_vehicles, 6), dtype=np.float32)
        for i, v in enumerate(self.vehicles):
            q = v.message_queue
            self_aoi = float(np.mean(self.aoi_tracker.aoi_matrix[v.id, :]))
            obs[i] = [
                v.position / 1000.0,
                v.speed / 30.0,
                len(q) / self.max_queue,
                q[-1].get_age(self.current_time) / 20.0 if q else 0.0,
                max((m.importance for m in q), default=0.0),
                self_aoi / 20.0,
            ]
        return obs

    def _get_global_state(self):
        local_flat = self._get_observations().flatten()
        aoi_flat = self.aoi_tracker.aoi_matrix.flatten()
        return np.concatenate([local_flat, aoi_flat]).astype(np.float32)