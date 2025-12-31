import whisper  # noqa

print("正在下载 base 模型...")
# 可改为 small/medium/large/base
model = whisper.load_model("large", download_root="./models")
print("✅ 下载完成！模型保存在 ./models 目录")
