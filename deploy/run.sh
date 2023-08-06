gunicorn main:app --workers ${NUM_WORKERS:-1} --worker-class uvicorn.workers.UvicornWorker \
                --bind 0.0.0.0:8000 --log-level debug --error-logfile log/server.log