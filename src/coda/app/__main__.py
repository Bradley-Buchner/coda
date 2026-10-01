import uvicorn

from coda.config import configure_logging, settings

if __name__ == "__main__":
    configure_logging()
    from .server import app
    uvicorn.run(app, host=settings.app.host, port=settings.app.port,
                log_config=None)
