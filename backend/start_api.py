"""Development launcher for the sole medical v1 application.

Apply Alembic migrations first. Training, datasets and inference run in separate
workers; this launcher does not import Torch or start background industry jobs.
"""
import uvicorn

if __name__ == '__main__':
    uvicorn.run('app.main:app', host='127.0.0.1', port=8000)
