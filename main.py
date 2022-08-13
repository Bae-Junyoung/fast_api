from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import numpy as np
from pydantic import BaseModel
from datetime import datetime
from typing import Union
from starlette.responses import StreamingResponse
from models.skin_models import *
from router import image_url
from fastapi.staticfiles import StaticFiles
import io, os, sys, logging

app = FastAPI()

app.include_router(image_url.router)

app.mount("/static", StaticFiles(directory="static"), name="static")

check_is_skin = CheckIsSkin('./scope_model_true_false')


class SkinImage(BaseModel):
    url: Union[str, None] = None


class ImageWithScore(BaseModel):
    left_cheek: Union[SkinImage, None] = None
    left_cheek_score: Union[int, None] = None
    right_cheek: Union[SkinImage, None] = None
    right_cheek_score: Union[int, None] = None


class Evaluation(ImageWithScore):
    pass


class Pore(ImageWithScore):
    gender_rank: Union[int, None] = None
    age_gender_rank: Union[int, None] = None
    pore_cnt: Union[int, None] = None
    pore_size: Union[float, None] = None


class Texture(ImageWithScore):
    skin_elasticity: Union[float, None] = None


class Wrinkle(ImageWithScore):
    pass


class Trouble(ImageWithScore):
    pass


class Results(BaseModel):
    user_id: str
    gender: str
    age: int
    measure_date: Union[datetime, None] = None

    evaluation: Union[Evaluation, None] = None
    pore: Union[Pore, None] = None
    texture: Union[Texture, None] = None
    wrinkle: Union[Wrinkle, None] = None
    trouble: Union[Trouble, None] = None


def function_results(user_id: str, gender: str, age: int, left_img: Image, right_img: Image):
    results = Results(user_id=user_id, gender=gender, age=age)
    now_time = datetime.now()

    ## evaluation
    results.evaluation = Evaluation(left_cheek=SkinImage(), left_cheek_score=70, right_cheek=SkinImage(), right_cheek_score=80)

    ## pore
    results.pore = Pore(left_cheek=SkinImage(), left_cheek_score=65, right_cheek=SkinImage(), right_cheek_score=75,
                             gender_rank=80, age_gender_rank=85, pore_cnt=13, pore_size=12)

    ## texture
    left_texture = TextureModel(user_id=user_id, img=left_img, side='left', time=now_time)
    right_texture = TextureModel(user_id=user_id, img=right_img, side='right', time=now_time)
    results.texture = Texture(left_cheek=SkinImage(url=left_texture.get_img_url()), left_cheek_score=left_texture.get_score(),
                              right_cheek=SkinImage(url=right_texture.get_img_url()), right_cheek_score=right_texture.get_score(),
                              skin_elasticity=10)

    ## wrinkle
    results.wrinkle = Wrinkle(left_cheek=SkinImage(), left_cheek_score=70, right_cheek=SkinImage(), right_cheek_score=70)

    ## trouble
    results.trouble = Trouble(left_cheek=SkinImage(), left_cheek_score=85, right_cheek=SkinImage(), right_cheek_score=85)

    return results


@app.get('/')
def root_route(results: Results):
    return results


@app.post('/model')
async def model_test(user_id: str, gender: str, age: int, image1: UploadFile = File(...), image2: UploadFile = File(...)):
    if image1.content_type.startswith('image/') is False:
        raise HTTPException(status_code=400, detail=f'File \'{image1.filename}\' is not an image.')

    if image2.content_type.startswith('image/') is False:
        raise HTTPException(status_code=400, detail=f'File \'{image2.filename}\' is not an image.')

    try:
        left_img = await image1.read()
        right_img = await image2.read()
        left_pil_image = Image.open(BytesIO(left_img))
        right_pil_image = Image.open(BytesIO(right_img))

        if not check_is_skin.pred(left_pil_image):
            return {'result': 'please try again with new left-cheek image'}

        if not check_is_skin.pred(right_pil_image):
            return {'result': 'please try again with new right-cheek image'}

        results = function_results(user_id=user_id, gender=gender, age=age, left_img=left_pil_image, right_img=right_pil_image)

        return results

    except Exception as error:
        logging.exception(error)
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/check_images")
def images():
    print(os.getcwd())

    out = []
    for filename in os.listdir("static/images"):
        out.append({
            "name": filename.split(".")[0],
            "path": filename
        })
    return out