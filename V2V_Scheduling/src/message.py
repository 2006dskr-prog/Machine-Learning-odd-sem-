import uuid


class Message:
    def __init__(self, source_id: int, dest_id: int, class_type: str,
                 importance: float, creation_time: float):
        self.id = str(uuid.uuid4())[:8]
        self.source_id = source_id
        self.dest_id = dest_id
        self.class_type = class_type
        self.importance = float(importance)
        self.creation_time = float(creation_time)

    def get_age(self, current_time: float) -> float:
        return current_time - self.creation_time

    def __repr__(self):
        return (f"Msg({self.id}, src={self.source_id}, "
                f"class={self.class_type}, imp={self.importance:.2f})")