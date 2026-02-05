from fastapi import APIRouter, UploadFile, File
from ...rag.parse import parse_any
from ...rag.index import index

router = APIRouter(prefix="/docs", tags=["docs"])

@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    filetype, text = parse_any(file.filename, data)
    res = await index.add_document(file.filename, filetype, text, {"filename": file.filename, "filetype": filetype})
    return {"ok": True, **res}
