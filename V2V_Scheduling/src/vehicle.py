from collections import deque
from src.message import Message


class Vehicle:
    def __init__(self, vehicle_id: int, initial_position: float,
                 speed: float, max_queue_size: int = 10):
        self.id = vehicle_id
        self.position = float(initial_position)
        self.speed = float(speed)
        self.message_queue = deque(maxlen=max_queue_size)

    def update_position(self, dt: float):
        self.position += self.speed * dt

    def enqueue_message(self, message: Message):
        self.message_queue.append(message)

    def get_state(self) -> dict:
        return {
            "id": self.id,
            "position": self.position,
            "speed": self.speed,
            "queue_size": len(self.message_queue),
        }