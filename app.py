# app.py

import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from main import create_video_clips
import uuid
import zipfile

app = Flask(__name__)

# Настройки папок
UPLOAD_FOLDER = 'uploads'
CLIPS_FOLDER = 'clips'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CLIPS_FOLDER'] = CLIPS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 * 1024  # 10 гигабайт

# Разрешённые типы файлов для загрузки
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi'}

# папки существуют
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CLIPS_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'Нет файла в запросе'}), 400

    file = request.files['video']

    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400

    if file and allowed_file(file.filename):
        # Сохранение файла с уникальным именем
        unique_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_id + "_" + filename)
        file.save(upload_path)

        # Создание папки для клипов этого видео
        clips_dir = os.path.join(app.config['CLIPS_FOLDER'], unique_id)
        os.makedirs(clips_dir, exist_ok=True)

        try:
            # Обработка видео и получение списка клипов
            clips = create_video_clips(upload_path, clips_dir)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

        if not clips:
            return jsonify({'error': 'Не удалось сгенерировать клипы.'}), 500

        # Создание ссылок на клипы
        clips_urls = [f"/clips/{unique_id}/{clip}" for clip in clips]

        # Создание ссылки для скачивания всех клипов
        download_all_url = f"/download_all/{unique_id}"

        return jsonify({
            'unique_id': unique_id,
            'clips': clips_urls,
            'download_all': download_all_url
        }), 200
    else:
        return jsonify({'error': 'Недопустимый формат файла.'}), 400

@app.route('/clips/<unique_id>/<clip_name>')
def download_clip(unique_id, clip_name):
    clips_dir = os.path.join(app.config['CLIPS_FOLDER'], unique_id)
    return send_from_directory(clips_dir, clip_name, as_attachment=True)

@app.route('/download_all/<unique_id>')
def download_all_clips(unique_id):
    clips_dir = os.path.join(app.config['CLIPS_FOLDER'], unique_id)
    zip_filename = f"{unique_id}_clips.zip"
    zip_path = os.path.join(CLIPS_FOLDER, zip_filename)

    # Создание ZIP-архива
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, dirs, files in os.walk(clips_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, clips_dir)
                zipf.write(file_path, arcname)

    return send_from_directory(CLIPS_FOLDER, zip_filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
