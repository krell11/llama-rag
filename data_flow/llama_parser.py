from llama_parse import LlamaParse
import os
import dotenv
import json


dotenv.load_dotenv('../.env')
api_key = os.environ.get('PARSER_KEY')
parsing_instruction = """
"""


class Parser:
    def __init__(self, key: str = api_key, language: str = 'ru', result_type='markdown'):
        self.parser = LlamaParse(
            api_key=key,
            result_type=result_type,
            parsing_instructions=parsing_instruction,
            verbose=True,
            language=language
        )

    def parse(self, path: str, output_file: str = 'parsing_result.json', images_folder: str = 'images'):
        result = self.parser.get_json_result(path)
        images = self.parser.get_images(result, images_folder)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        return result


if __name__ == "__main__":
    path = 'CML_Bench_Руководство_оператора_v17.docx'
    parser = Parser(api_key)
    data = parser.parse(path)
    new_data = {'pages': []}
    for page in data[0]['pages']:
        number = page['page']
        content = page['text']
        try:
            images_on_page = page['images']
        except KeyError:
            images_on_page = []
        new_data['pages'].append({'page': number, 'content': content, 'images': images_on_page})
