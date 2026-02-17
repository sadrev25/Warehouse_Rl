class Task:

    def __init__(
        self,
        task_number: int,
        task_type: str,
        priority: float,
        current_station: str,
        target_station: str,
    ):

        self.task_number = task_number
        self.task_type = task_type
        self.priority = priority
        self.current_station = current_station
        self.target_station = target_station
