# llama-rag


## Установка

1. **clone repo:**
   ```bash
   git clone https://github.com/krell11/llama-rag.git
   ```
2. **install requirements**
   ```
   pip install -r requirements.txt
   ```
3. **build docker image**
   ```
   docker-compose up
   ```
4. Скачайте файл модели по этой [ссылке](https://huggingface.co/IlyaGusev/saiga_llama3_8b_gguf/resolve/main/model-q8_0.gguf?download=true) и поместите его в папку /api/model.
5. run container
   ```
   docker start llama-rag-redis-1
   ```
6. Parse Json files
   ```
   python data_flow/store_data.py
   ```
7. start api
   ```
   cd api
   fastapi run api.py
   ```
8. visit [swagger UI](http://localhost:8000/)
