from pydantic import BaseModel
from typing import List
from message import Message


class Chat(BaseModel):
    chat_id: str
    title: str
    messages: List[Message]
    links: List[str]


class ChatCreationRequest(BaseModel):
    message: str
    user_id: str


class SendMessageRequest(BaseModel):
    text: str
    chat_id: str
    user_id: str


class SendMessageResponse(BaseModel):
    text: str
    links: List[str]
    chat_id: str
