FROM python:3.11-slim

# 設定環境變數
ENV DEBIAN_FRONTEND=noninteractive
# 讓 Python 輸出直接印在終端機，方便 debug
ENV PYTHONUNBUFFERED=1 

WORKDIR /workspace

# 合併指令減少 Layer 數量，並加入常用的工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    ca-certificates \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 升級 pip，建議加入 --timeout 避免網路不穩直接斷掉
RUN pip install --no-cache-dir --upgrade pip --default-timeout=100

RUN pip install --no-cache-dir torch torchvision torchaudio

# 安裝其餘套件
# 建議將 requirements.txt 先 COPY 進去，否則這行會失敗
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 預設指令
CMD ["/bin/bash"]