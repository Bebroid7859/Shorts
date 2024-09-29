# main.py

import os
import cv2
import torch
import numpy as np
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import speech_recognition as sr
from moviepy.editor import VideoFileClip
from transformers import pipeline

# Загрузка модели для анализа тональности текста
sentiment_pipeline = pipeline("sentiment-analysis", model="blanchefort/rubert-base-cased-sentiment")

# Загрузка YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Загрузка классов объектов
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Распознаватель речи
recognizer = sr.Recognizer()

# Функция для анализа объектов в кадре
def detect_objects(frame):
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    objects = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = scores[class_id]
            if confidence > 0.5:
                object_name = classes[class_id]
                objects.append(object_name)
    return objects

# Функция для распознавания речи из аудио
def transcribe_audio(audio_clip):
    try:
        with sr.AudioFile(audio_clip) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio, language="ru-RU")
    except sr.UnknownValueError:
        return "Не удалось распознать речь"
    except sr.RequestError as e:
        return f"Ошибка сервиса: {e}"

# Функция для анализа текста с помощью Transformers (тональность)
def analyze_text_with_transformers(transcript):
    sentences = transcript.split(". ")
    important_sentences = []
    
    for sentence in sentences:
        result = sentiment_pipeline(sentence)[0]
        sentiment = result['label']
        score = result['score']
        
        if sentiment in ['positive', 'negative'] and score > 0.85:
            important_sentences.append((sentence, sentiment, score))
    
    return important_sentences

# Функция для преобразования видеоклипа в вертикальный формат
def convert_to_vertical(clip, target_width=1080, target_height=1920):
    """
    Преобразует видеоклип в вертикальный формат с заданными размерами.
    Обрезает по центру, если соотношение сторон не совпадает.

    :param clip: исходный VideoFileClip
    :param target_width: желаемая ширина вертикального видео
    :param target_height: желаемая высота вертикального видео
    :return: преобразованный VideoFileClip
    """
    # Соотношение сторон для вертикального видео
    target_ratio = target_width / target_height
    # Текущие размеры видео
    width, height = clip.size
    current_ratio = width / height

    if current_ratio > target_ratio:
        # Шире, чем вертикальное соотношение, обрезаем по ширине
        new_width = int(target_ratio * height)
        x1 = (width - new_width) // 2
        clip = clip.crop(x1=x1, y1=0, width=new_width, height=height)
    else:
        # Выше, чем вертикальное соотношение, обрезаем по высоте
        new_height = int(width / target_ratio)
        y1 = (height - new_height) // 2
        clip = clip.crop(x1=0, y1=y1, width=width, height=new_height)

    # Масштабируем видео до целевых размеров
    clip = clip.resize(newsize=(target_width, target_height))
    return clip

# Основной процесс анализа видео
def process_video(video_path):
    print(f"Начало обработки видео: {video_path}")
    
    # Извлечение аудио из видео
    video_clip = VideoFileClip(video_path)
    audio_clip_path = "temp_audio.wav"
    video_clip.audio.write_audiofile(audio_clip_path)
    video_clip.close()

    # Распознавание речи из аудиодорожки
    audio_transcript = transcribe_audio(audio_clip_path)
    print("Текст из аудио:", audio_transcript)

    # Анализ текста с помощью Transformers
    important_sentences_transformers = analyze_text_with_transformers(audio_transcript)

    # Печать ключевых предложений
    print("Ключевые предложения (Transformers):", important_sentences_transformers)

    # Открытие видео для анализа кадров
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps == 0:
        print("Не удалось определить частоту кадров (FPS).")
        return []

    duration = frame_count / fps
    print(f"Видео загружено: {frame_count} кадров, {fps} FPS, {duration} секунд.")

    key_moments = []
    frame_index = 0
    skip_frames = 15  # Обработка каждого 15-го кадра для ускорения, регулируется

    last_moment_end = None
    current_moment_start = None
    current_moment_score = 0

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        if frame_index % skip_frames == 0:
            print(f"Обрабатывается кадр: {frame_index} из {frame_count}")
            objects = detect_objects(frame)
            importance_score = len(objects)

            if importance_score > 3:
                current_time = frame_index / fps
                current_moment_score += importance_score

                if current_moment_start is None:
                    current_moment_start = current_time

                last_moment_end = current_time
            elif current_moment_start is not None and last_moment_end is not None:
                if (last_moment_end - current_moment_start) > 5:
                    key_moments.append((current_moment_start, last_moment_end, current_moment_score))
                current_moment_start = None
                last_moment_end = None
                current_moment_score = 0

        frame_index += 1

    video.release()

    # Если ключевых моментов нет, добавляем момент с начала видео
    if not key_moments:
        print("Не найдено ключевых моментов, добавляется начало видео.")
        key_moments.append((0, min(10, duration), 0))  # добавляем момент от начала до десятой секунды

    print(f"Найдено {len(key_moments)} ключевых моментов.")

    # фильтрация ключевых моментов
    # Пример: фильтрация ключевых моментов, если они перекрываются
    filtered_key_moments = []
    for moment in key_moments:
        add = True
        for filt_moment in filtered_key_moments:
            # Проверка перекрытия более чем на 50%
            overlap = max(0, min(moment[1], filt_moment[1]) - max(moment[0], filt_moment[0]))
            duration_overlap = overlap / (min(moment[1], filt_moment[1]) - max(moment[0], filt_moment[0])) if max(moment[0], filt_moment[0]) < min(moment[1], filt_moment[1]) else 0
            if duration_overlap > 0.5:
                add = False
                break
        if add:
            filtered_key_moments.append(moment)

    print(f"После фильтрации осталось {len(filtered_key_moments)} ключевых моментов.")

    # Дополнительно можно сортировать или выбирать лучшие моменты на основе оценки
    # Например, сортировка по score и выбор топ-10
    filtered_key_moments = sorted(filtered_key_moments, key=lambda x: x[2], reverse=True)[:10]

    print(f"Отобрано {len(filtered_key_moments)} ключевых моментов после сортировки.")

    # Удаление оценок значимости для дальнейшей обработки
    key_moments_final = [(start, end) for start, end, score in filtered_key_moments]

    # Очищаем временный аудиофайл
    if os.path.exists(audio_clip_path):
        os.remove(audio_clip_path)

    return key_moments_final

# Функция для нарезки видео на клипы и преобразования в вертикальный формат
def cut_clips(video_path, key_moments, output_dir, output_size=(1080, 1920)):
    """
    Нарезает видео на клипы по ключевым моментам и сохраняет их в вертикальном формате.

    :param video_path: путь к исходному видео
    :param key_moments: список ключевых моментов (start_time, end_time)
    :param output_dir: папка для сохранения клипов
    :param output_size: кортеж (ширина, высота) для вертикального формата
    :return: список имен сохранённых клипов
    """
    os.makedirs(output_dir, exist_ok=True)
    clips = []
    for i, (start_time, end_time) in enumerate(key_moments):
        output_filename = f"clip_{i+1}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        # Вырезаем субклип
        subclip = VideoFileClip(video_path).subclip(start_time, end_time)
        # Преобразуем в вертикальный формат
        vertical_clip = convert_to_vertical(subclip, target_width=output_size[0], target_height=output_size[1])
        # Сохраняем клип
        vertical_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        print(f"Сохранён вертикальный клип: {output_filename} ({start_time:.2f} - {end_time:.2f}) секунд.")
        clips.append(output_filename)
    return clips

# Анализ видео и нарезка клипов
def create_video_clips(video_path, output_dir):
    """
    Основная функция для создания клипов из видео.

    :param video_path: путь к загруженному видео
    :param output_dir: папка для сохранения клипов
    :return: список имен сохранённых клипов
    """
    print("Анализ видео для выделения ключевых моментов...")
    key_moments = process_video(video_path)
    
    print(f"Найдено {len(key_moments)} ключевых сцен.")
    if key_moments:
        print("Нарезаем видео на вертикальные клипы...")
        clips = cut_clips(video_path, key_moments, output_dir, output_size=(1080, 1920))  # Задаем желаемый размер вертикального видео
        print("Процесс завершен.")
    else:
        print("Ключевые моменты не найдены.")
        clips = []
    
    return clips

if __name__ == "__main__":
    video_path = "G:/II2/ИИ/videos/video.mp4"
    if not os.path.exists(video_path):
        print(f"Видео файл не найден по пути: {video_path}")
    else:
        create_video_clips(video_path, "clips")
