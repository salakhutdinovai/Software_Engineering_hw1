import torch
from transformers import pipeline

print("=== 1. Обработка текста: Анализ тональности ===")
sentiment_analyzer = pipeline("sentiment-analysis", model="blanchefort/rubert-base-cased-sentiment")
text = "Мне очень понравилась эта работа, всё сделано качественно!"
result_text = sentiment_analyzer(text)
print(f"Текст: {text}\nРезультат: {result_text}\n")

import soundfile as sf

print("=== 2. Обработка аудио: Распознавание речи (Speech-to-Text) ===")
asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
try:
    # Читаем аудио напрямую через soundfile, чтобы обойтись без ffmpeg на Windows
    audio_data, samplerate = sf.read("test_audio.wav")

    # Если аудио стерео (2 канала), делаем его моно (1 канал)
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]

    result_audio = asr({"sampling_rate": samplerate, "raw": audio_data})
    print(f"Распознанный текст: {result_audio['text']}\n")
except Exception as e:
    print(f"Ошибка при обработке аудио: {e}\n")

print("=== 3. Обработка изображений: Классификация объектов ===")
image_classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
try:
    result_image = image_classifier("test_image.jpg")
    print(f"Объекты на фото: {result_image[:3]}\n")
except Exception as e:
    print(f"Ошибка при обработке фото: {e}\n")

print("=== 4. Обработка видео: Классификация действий ===")
video_classifier = pipeline("video-classification", model="MCG-NJU/videomae-base-finetuned-kinetics")
try:
    result_video = video_classifier("test_video.mp4")
    print(f"Действие на видео: {result_video[:2]}\n")
except Exception as e:
    print(f"Ошибка при обработке видео: {e}\n")