from typing import Optional

class NooghGovernmentSystem:
    def __init__(self, verbose: bool = True):
        from .president import President
        self.president = President(verbose=verbose)

    async def process_request(self, user_input: str, context: Optional[dict] = None, priority: str = "medium"):
        """
        Process a user request through the government system.
        """
        return await self.president.process_request(user_input, context, priority)

    def get_status(self):
        """
        Get the status of the entire government system.
        """
        return self.president.get_cabinet_status()

    def print_status(self):
        """
        Print the status of the government system.
        """
        self.president.print_status()
