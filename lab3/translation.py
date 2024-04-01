from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline


class Item(BaseModel):
    text: str


def load_model():
    return pipeline("translation_ru_to_en", "Helsinki-NLP/opus-mt-ru-en")


model = load_model()
app = FastAPI()


@app.post('/translate')
async def translate(item: Item):
    return model(item.text)[0]['translation_text']