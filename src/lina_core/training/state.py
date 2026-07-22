from enum import StrEnum


class TrainingState(StrEnum):
    CREATED = "created"
    VALIDATING = "validating"
    PREPARING = "preparing"
    RUNNING = "running"
    EVALUATING = "evaluating"
    SAVING = "saving"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
