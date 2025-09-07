from fastapi import FastAPI, UploadFile, File
import uvicorn

app = FastAPI(title="FaceSense API")


@app.get("/")


def home():
    return {"message": "Welcome to FaceSense API"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
