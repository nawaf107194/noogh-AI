# Placeholder for autonomous_client.py

class AutonomousClient:
    """Placeholder for the async AutonomousClient."""
    async def get_resources(self):
        return {
            'gpu_memory_percent': 42.0,
            'gpu_temperature': 65.0,
            'overall_status': 'normal'
        }
    
    async def decide_device(self, task_type, estimated_vram, priority):
        return {'device': 'cpu', 'reason': 'Placeholder decision'}

    async def is_ready_for_training(self, estimated_vram):
        return True

    async def prepare_training(self, model_name, estimated_vram_needed):
        return {
            'freed_vram': 2.0,
            'available_vram': 10.0,
            'paused_ministers': 5
        }

    async def complete_training(self, success):
        return {'progress': 100.0}

class SyncAutonomousClient:
    """
    Placeholder for a synchronous wrapper around the AutonomousClient.
    In this placeholder, the methods are just regular methods, not async.
    """
    def get_resources(self):
        return {
            'gpu_memory_percent': 42.0,
            'gpu_temperature': 65.0,
            'overall_status': 'normal',
            'warnings': []
        }
    
    def decide_device(self, task_type, estimated_vram, priority):
        return {'device': 'cpu', 'reason': 'Placeholder decision'}

    def is_ready_for_training(self, estimated_vram):
        return True

    def prepare_training(self, model_name, estimated_vram_needed):
        return {
            'freed_vram': 2.0,
            'available_vram': 10.0,
            'paused_ministers': 5
        }

    def complete_training(self, success):
        return {'progress': 100.0}
