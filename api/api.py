from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Any
from models.embedder import Embedder
from models.llama_basic import BaseLLM
from models.image_descriptioner import Describer
from datetime import datetime
import uuid
from bson.json_util import dumps
import json
from .classes.message_request import ChatCreationRequest, SendMessageRequest, SendMessageResponse
from redis_base.chat_history import ChatHistory
from redis_base.retriever import RedisClient

app = FastAPI()
origins = [""]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embedder = Embedder('distiluse-base-multilingual-cased-v1')
rcli = RedisClient(text_model=embedder)
llm = BaseLLM(model_name='model/model-q8_0.gguf')
chat_history = ChatHistory()
describer = Describer()


@app.post("/chats")
async def create_chat(chat_request: ChatCreationRequest) -> SendMessageResponse:
    chat_id = str(uuid.uuid4())
    if chat_request.image_path:
        search_results = rcli.search_query(k=3, user_query=chat_request.message)
        context_for_llm = rcli.get_context_from_ids(search_results)
        image_description = describer.process_image(chat_request.image_path)[0]
        answer = llm.answer(user_message=chat_request.message, context=context_for_llm+image_description)
    else:
        search_results = rcli.search_query(k=3, user_query=chat_request.message)
        context_for_llm = rcli.get_context_from_ids(search_results)
        answer = llm.answer(user_message=chat_request.message, context=context_for_llm)
    dialog = [{"sender": chat_request.user_id, "message": chat_request.message},
              {"sender": "bot", "message": answer}]
    chat_history.store_chat_data(user_id=chat_request.user_id, dialog_id=chat_id, messages=dialog)
    chat_history.summarize_dialog(user_id=chat_request.user_id, dialog_id=chat_id, llm=llm)
    return SendMessageResponse(message=answer, chat_id=chat_id)


@app.post("/message")
async def handle_message(request: SendMessageRequest) -> SendMessageResponse:
    chat_id = request.chat_id
    if request.image_path:
        search_results = rcli.search_query(k=3, user_query=request.message)
        context_for_llm = rcli.get_context_from_ids(search_results)
        image_description = describer.process_image(request.image_path)[0]
        dialog_summary = chat_history.get_chat_summary(dialog_id=chat_id, user_id=request.user_id)
        answer = llm.answer(user_message=request.message, context=context_for_llm+image_description,
                            history=dialog_summary)
    else:
        search_results = rcli.search_query(k=3, user_query=request.message)
        context_for_llm = rcli.get_context_from_ids(search_results)
        dialog_summary = chat_history.get_chat_summary(dialog_id=chat_id, user_id=request.user_id)
        answer = llm.answer(user_message=request.message, context=context_for_llm, history=dialog_summary)
    dialog = [{"sender": request.user_id, "message": request.message},
              {"sender": "bot", "message": answer}]
    chat_history.store_chat_data(user_id=request.user_id, dialog_id=chat_id, messages=dialog)
    chat_history.summarize_dialog(user_id=request.user_id, dialog_id=chat_id, llm=llm)
    return SendMessageResponse(message=answer, chat_id=chat_id)


@app.get("/user/{user_id}")
async def get_all_user_chats(user_id: str) -> Any:
    chats = chat_history.get_all_dialogs(user_id=user_id)
    return JSONResponse(content=dumps(chats))


@app.get("/chat/{chat_id}")
async def get_chat_by_id(user_id: str, chat_id: str) -> Any:
    chat = chat_history.get_chat_history(user_id=user_id, dialog_id=chat_id)
    messages = []
    for message in chat:
        messages.append({"user": message["user"], "text": message["text"]})
    return {
        "user_id": user_id,
        "chat_id": chat_id,
        "messages": messages
    }
