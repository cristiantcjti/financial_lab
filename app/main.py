import os

import uvicorn

from application import ApplicationBuilder

app = ApplicationBuilder().get_app()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=5100,
        log_config=None,
        access_log=False,
    )
