FROM nvcr.io/nvidia/tritonserver:22.09-py3

COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN pip3 install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

ENTRYPOINT [ "tritonserver" ]
