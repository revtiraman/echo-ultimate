from openenv.core.client import HTTPEnvClient
from models import EchoAction, EchoObservation


class EchoClient(HTTPEnvClient):
    """HTTP client for the ECHO calibration environment."""

    action_class = EchoAction
    observation_class = EchoObservation

    def step_with_response(self, response_text: str) -> EchoObservation:
        """Helper: submit a raw response string as an action."""
        action = EchoAction(response=response_text)
        return self.step(action)
