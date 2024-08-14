FROM nvcr.io/nvidia/pytorch:24.07-py3
WORKDIR /app
COPY gpu-stress.py .
CMD ["python", "gpu-stress.py", "--device gpu"]
