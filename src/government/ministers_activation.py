from enum import Enum

class MinisterType(str, Enum):
    EDUCATION = "education"
    TRAINING = "training"
    SECURITY = "security"
    DEVELOPMENT = "development"
    RESEARCH = "research"
    KNOWLEDGE = "knowledge"
    PRIVACY = "privacy"
    CREATIVITY = "creativity"
    ANALYSIS = "analysis"
    STRATEGY = "strategy"
    REASONING = "reasoning"
    COMMUNICATION = "communication"
    RESOURCES = "resources"
    FINANCE = "finance"

class MinistersActivationSystem:
    def __init__(self, brain_hub):
        self.brain_hub = brain_hub
        self.active_ministers = []

    def activate_all(self):
        pass

    async def delegate_task(self, minister_type, task):
        return {}

    def get_all_stats(self):
        return {}
