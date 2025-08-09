"""
Simple stub for piopiy module to prevent import errors.
In a real implementation, this would be the actual piopiy SDK.
"""


class Action:
    def __init__(self):
        self.actions = []

    def stream(self, url, params=None):
        """Add a stream action."""
        action = {
            "action": "stream",
            "url": url,
            "parameters": params or {}
        }
        self.actions.append(action)

    def PCMO(self):
        """Return the action list in PCMO format."""
        return {"actions": self.actions}