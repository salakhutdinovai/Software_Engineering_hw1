# ML Models & Local LLM Demo

Репозиторий содержит реализацию 5 задач машинного обучения с использованием готовых моделей (Hugging Face / PyTorch).

## Выполненные задачи
1. **Текст:** Анализ тональности текста (модель: `blanchefort/rubert-base-cased-sentiment`).
2. **Аудио:** Распознавание речи / Speech-to-Text (модель: `openai/whisper-tiny`).
3. **Изображения:** Классификация объектов на фото (модель: `google/vit-base-patch16-224`).
4. **Видео:** Классификация действий на видео (модель: `MCG-NJU/videomae-base-finetuned-kinetics`).
5. **Локальная LLM:** Генерация текста (модель: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`).

## Запуск проекта
1. Клонируйте репозиторий.
2. Установите зависимости: `pip install -r requirements.txt`.
3. Добавьте в корень проекта тестовые файлы: `test_image.jpg`, `test_audio.wav`, `test_video.mp4`.
4. Запустите скрипты:
   - `python models_demo.py` (для базовых моделей)
   - `python local_llm.py` (для локальной LLM)