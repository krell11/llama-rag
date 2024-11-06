from llama_parser import Parser
from models.embedder import Embedder
from models.image_descriptioner import Describer
from redis_base.retriever import RedisClient
import json


parser = Parser()
embedder = Embedder('distiluse-base-multilingual-cased-v1')
# describer = Describer()
rcli = RedisClient(text_model=embedder)


def parsing(parser: Parser, embedder: Embedder, describer: Describer, rcli: RedisClient,
            images_folder: str, docs_folder: str):
    #data = parser.parse(document, images_folder, docs_folder)
    # data = json.load(open('parsing_result.json', 'r', encoding='utf-8'))
    # new_data = {'pages': []}
    # for page in data[0]['pages'][51:]:
    #     number = page['page']
    #     content = page['text']
    #     images_on_page = page['images']
    #     if len(images_on_page) != 0:
    #         for image_path in images_on_page:
    #             correct_path = image_path['path']
    #             image_description = describer.process_image(image_path=correct_path)
    #             content += image_description[0]
    #     new_data['pages'].append({'page': number, 'content': content, 'images': images_on_page})
    #     new_data_list = new_data['pages']
    #     rcli.store_new_data(new_data_list)
    #     new_data = {'pages': []}
    #     print(new_data)
    #     print(number)
        #начиная с data[0]['pages'][79:]
    rcli.create_vector_field()
    keys, text_embeddings = rcli.create_embeddings()
    rcli.store_embeddings(keys=keys, text_embeddings=text_embeddings)

if __name__ == "__main__":
    parsing(parser, embedder, None, rcli, 'images', 'docs')
    print("Parsing is done")
