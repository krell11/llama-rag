from torch import cuda
from llama_cpp import Llama
from typing import Generator
import time


class BaseLLM:
    def __init__(self, model_name, system_prompt_template=None, creativity=0.3):
        self.device = f'cuda:{cuda.current_device()}' \
            if cuda.is_available() else 'cpu'
        print(self.device)
        self.model_name = model_name
        self.temperature = creativity

        self.model = Llama(
            model_path=model_name,
            n_batch=2048,
            n_gpu_layers=100,
            n_ctx=5000,
            temperature=self.temperature,
            verbose=True)

        if system_prompt_template is not None:
            self.prompt_template = system_prompt_template
        else:
            self.prompt_template = """
                Ты русскоязычный ассистент, и твоя задача — помогать пользователям решать вопросы, связанные с программным комплексом CML-Bench. 
                Всегда ориентируйся на предоставленные знания для ответа на вопросы.

                Важно:
                1. Отвечай только на вопросы, связанные с работой программного обеспечения CML-Bench.
                2. Если вопрос не касается CML-Bench или программного обеспечения, вежливо скажи, что не можешь помочь с этим запросом.
                3. Если ты не знаешь точного ответа на вопрос, не придумывай. Ответь, что не обладаешь нужной информацией.
                4. Старайся избегать ошибок при взятии данных из контекста, некоторые слова там повреждены. Постарайся понять суть вопроса и ответить правильно.
                5. Стремись к кратким, но точным и информативным ответам.

                Контекст для ответа:
                {context}

                Ответь на вопрос, используя информацию из контекста.
                """

    def stream_answer(self, user_message, context, history='') -> Generator[str, None, None]:
        response = self.model.create_chat_completion(
            messages=[{"role": "system",
                       "content": self.prompt_template.format(context=context) + history},
                      {
                      "role": "user",
                      "content": user_message}],
            temperature=self.temperature,
            stream=True,
        )

        for chunk in response:
            if "choices" in chunk and len(chunk["choices"]) > 0:
                if "delta" in chunk["choices"][0]:
                    delta_content = chunk["choices"][0]["delta"].get("content", "")
                    if delta_content:
                        yield f"data: {delta_content}\n\n"

    def summarize(self, query: dict):
        query = str(query)
        prompt_template = """
                        Сгенерируй краткое описание диалога, которое будет отображать наиболее важные моменты, чтобы при
                        дальнейшем общении пользовательские предпочтения были учтены и было понятно, что спрашивает и 
                        как отвечает другой участник диалога. Опирайся на следующий запрос:
                        {query}
                        """
        summarization = self.model.create_chat_completion(
            messages=[{"role": "system",
                       "content": prompt_template.format(query=query)}],
            temperature=0.3,
            max_tokens=500)

        return summarization['choices'][0]['message']['content']

    def answer(self, user_message, context, history='') -> str:
        response = self.model.create_chat_completion(
            messages=[{"role": "system",
                       "content": self.prompt_template.format(context=context) + history},
                      {
                      "role": "user",
                      "content": user_message}],
            temperature=self.temperature,
            max_tokens=2000,
        )

        return response['choices'][0]['message']['content']
