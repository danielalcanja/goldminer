"""Background-job adapter for hosted Gold Miner runs."""

from .models import JobResult, JobSpec
from .service import JobService

__all__ = ["JobResult", "JobService", "JobSpec"]
