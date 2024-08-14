FROM nvcr.io/nvidia/pytorch:24.07-py3
WORKDIR /app
COPY gpu-stress.py .
RUN nohup python gpu-stress.py > /dev/null 2>&1 &
CMD ["watch", "-d", "-n 0.5", "nvidia-smi"]