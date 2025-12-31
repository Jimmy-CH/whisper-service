# 使用官方 Python 3.10 镜像（基于 Debian）
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖：ffmpeg（Whisper 必需）、gcc（用于某些包编译）
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        gcc \
        && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖（使用国内镜像加速可选）
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
# RUN pip install --no-cache-dir -r requirements.txt
# 复制应用代码
COPY whisper_service.py .
COPY models/ ./models/
# 暴露端口
EXPOSE 9090
# 启动命令
CMD ["uvicorn", "whisper_service:app", "--host", "0.0.0.0", "--port", "9090"]

