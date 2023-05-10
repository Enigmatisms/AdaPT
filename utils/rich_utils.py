"""
    Rich console progress bar
    from nerfstudio: (script/render.py)
    @author: NeRFStudio
"""

from rich.progress import ProgressColumn
from rich.text import Text

class ItersPerSecColumn(ProgressColumn):
    """Renders the iterations per second for a progress bar."""

    def __init__(self, suffix="it/s") -> None:
        super().__init__()
        self.suffix = suffix

    def render(self, task) -> Text:
        """Show data transfer speed."""
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text("?", style="progress.data.speed")
        return Text(f"{speed:.2f} {self.suffix}", style="progress.data.speed")
    