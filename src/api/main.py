import uvicorn

from src.api.settings import Settings

if __name__ == "__main__":
    settings = Settings()
    uvicorn.run("src.api.app:app", host=settings.HOST, port=settings.PORT)
