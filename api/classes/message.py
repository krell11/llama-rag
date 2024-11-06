from pydantic import BaseModel
from datetime import datetime


class Message(BaseModel):
    sender: str
    text: str
    timestamp: datetime
