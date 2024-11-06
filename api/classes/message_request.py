from pydantic import BaseModel
from typing import List, Optional


class ChatCreationRequest(BaseModel):
    message: str
    user_id: str
    image_path: Optional[str] = None


class SendMessageRequest(BaseModel):
    message: str
    chat_id: str
    user_id: str
    image_path: Optional[str] = None


class SendMessageResponse(BaseModel):
    message: str
    chat_id: str
