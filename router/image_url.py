from fastapi import APIRouter, Depends

from dependencies import get_token_header


router = APIRouter(
    prefix="/get_image",
    tags=["get_image"],
    # dependencies=[Depends(get_token_header)],
    # responses={404: {"description": "Not found"}},
)

import cv2
import io
from starlette.responses import StreamingResponse
import os


@router.get("/{image_name}")
def image_url(image_name: str):
    img = cv2.imread(os.getcwd() + "/static/images/" + image_name)

    res, im_png = cv2.imencode(".jpg", img,)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/jpg")