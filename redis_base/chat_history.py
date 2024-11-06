import redis
import time
from typing import List, Dict


class ChatHistory:
    def __init__(self, HOST='localhost', PORT=6379, PASSWORD=None):
        try:
            self.client = redis.Redis(host=HOST, port=PORT, password=PASSWORD,
                                      decode_responses=True)
        except redis.RedisError as e:
            print(f"Error connecting to Redis: {e}")

    def store_chat_data(self, user_id: str, dialog_id: str, messages: List[dict]):
        index_key = f"chat:{user_id}:{dialog_id}:index"

        current_index = self.client.get(index_key)
        if current_index is None:
            current_index = 0
        else:
            current_index = int(current_index)

        pipeline = self.client.pipeline()

        for message in messages:
            timestamp = time.time()
            redis_key = f"chat:{user_id}:{dialog_id}:{current_index % 6}"
            message["timestamp"] = timestamp
            pipeline.json().set(redis_key, "$", message)

            current_index += 1

        pipeline.set(index_key, current_index)

        pipeline.execute()
        pipeline.reset()

    def get_chat_history(self, user_id: str, dialog_id: str) -> List[Dict]:
        keys = [f"chat:{user_id}:{dialog_id}:{i}" for i in range(6)]
        history = self.client.json().mget(keys, "$")
        return [message for sublist in history if sublist for message in sublist]

    def clear_chat_history(self, user_id: str, dialog_id: str):
        keys = self.client.keys(f"chat:{user_id}:{dialog_id}:*")
        for key in keys:
            self.client.delete(key)

    def get_all_dialogs(self, user_id: str) -> List[str]:
        keys = self.client.keys(f"chat:{user_id}:*")
        dialog_ids = set(key.split(":")[2] for key in keys)
        return list(dialog_ids)

    def summarize_dialog(self, user_id: str, dialog_id: str, llm):
        chat_history = self.get_chat_history(user_id=user_id, dialog_id=dialog_id)
        context = ""
        for message in chat_history:
            if isinstance(message, dict):
                if message["sender"] == user_id:
                    context += f"Пользователь: {message['message']}\n"
                else:
                    context += f"Бот: {message['message']}\n"
            else:
                pass
        summary = llm.summarize(query=context)
        summary_key = f"chat:{user_id}:{dialog_id}:summary"
        self.client.json().set(summary_key, "$", summary)

    def get_chat_summary(self, dialog_id: str, user_id: str) -> str:
        summary_key = f"chat:{user_id}:{dialog_id}:summary"
        return self.client.json().get(summary_key)


if __name__ == "__main__":
    chat_history = ChatHistory()
    from models.llama_basic import BaseLLM
    model_name='C:/work/llama-rag/api/model/model-q8_0.gguf'
    model = BaseLLM(model_name=model_name)
    messages = [
        {"sender": "user", "message": "Привет!"},
        {"sender": "bot", "message": "Здравствуйте, чем могу помочь?"}
    ]
    chat_history.store_chat_data(user_id="12345", dialog_id="dialog_1", messages=messages)
    history = chat_history.get_chat_history(user_id="12345", dialog_id="dialog_1")
    chat_history.summarize_dialog(user_id="12345", dialog_id="dialog_1", llm=model)
    summary = chat_history.get_chat_summary(dialog_id="dialog_1", user_id="12345")
    dialogs = chat_history.get_all_dialogs(user_id="12345")
    print(f"Все диалоги пользователя: {dialogs}")
    print(summary)
