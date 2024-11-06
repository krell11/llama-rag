import torch.cuda
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import dotenv
import os
import torch


dotenv.load_dotenv('../.env')


class Describer:
    def __init__(self, device_map: str = "cuda"):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.float16, device_map=torch.cuda.current_device()
        )
        self.min_pixels = int(os.environ.get('MIN_PIXELS'))
        self.max_pixels = int(os.environ.get('MAX_PIXELS'))
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct",
                                                       min_pixels=self.min_pixels, max_pixels=self.max_pixels)

    def process_image(self, image_path: str):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                        "resized_height": 280,
                        "resized_width": 420,

                    },
                    {
                        "type": "text",
                        "text": "Опиши это изображение.Если на изображении интерфейс, то максимально точно передай его "
                                "особенности и постарайся описать так, чтобы при прочтении было понятно о чем речь,"
                                "цвета и вид иконки элементов можно опускать, если это не важно"},
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to("cuda")
        generated_ids = self.model.generate(**inputs, max_new_tokens=500)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)
        return output_text


if __name__ == "__main__":
    describer = Describer()
    answer = describer.process_image("C:/practice/SoHCCL6AGhw.jpg")[0]
    #torch.cuda.empty_cache()
    from models.llama_basic import BaseLLM
    # language = BaseLLM('model-q8_0.gguf')
    print(answer)
    # print(language.stream_answer(answer, 'просто опиши что написано и исправь ошибки если они есть'))
    # print(language.answer(answer, 'просто опиши что написано и исправь ошибки если они есть'))

