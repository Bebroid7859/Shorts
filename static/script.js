// static/script.js

document.addEventListener('DOMContentLoaded', function() {
    const videoUpload = document.getElementById('video-upload');
    const uploadButton = document.getElementById('upload-button');
    const fileNameDisplay = document.getElementById('file-name');
    const resultSection = document.querySelector('.result-section');
    const clipsContainer = document.querySelector('.clips-container');
    const downloadAllButton = document.getElementById('download-all');
    const videoUploadForm = document.getElementById('video-upload-form');

    // Показать имя выбранного файла
    videoUpload.addEventListener('change', function() {
        if (videoUpload.files.length > 0) {
            let fileName = videoUpload.files[0].name;
            fileNameDisplay.textContent = fileName;
        } else {
            fileNameDisplay.textContent = 'Файл не выбран';
        }
    });

    // Обработчик загрузки видео
    videoUploadForm.addEventListener('submit', function(e) {
        e.preventDefault();

        if (videoUpload.files.length === 0) {
            alert('Пожалуйста, выберите видео для загрузки.');
            return;
        }

        const formData = new FormData();
        formData.append('video', videoUpload.files[0]);

        // Отображаем загрузку
        uploadButton.disabled = true;
        uploadButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Обработка...';

        fetch('/upload', {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
            uploadButton.disabled = false;
            uploadButton.innerHTML = '<i class="fas fa-magic"></i> Сгенерировать клипы';

            if (data.error) {
                alert(data.error);
            } else {
                // Показываем секцию результата и добавляем клипы
                resultSection.style.display = 'block';
                clipsContainer.innerHTML = '';

                data.clips.forEach((clipUrl, index) => {
                    const clipElement = document.createElement('div');
                    clipElement.classList.add('clip');

                    clipElement.innerHTML = `
                        <video src="${clipUrl}" controls></video>
                        <div class="clip-info">
                            <p>Клип ${index + 1}</p>
                            <a href="${clipUrl}" class="btn btn-secondary" download>
                                <i class="fas fa-download"></i> Скачать
                            </a>
                        </div>
                    `;

                    clipsContainer.appendChild(clipElement);
                });

                // Обработчик для кнопки "Скачать все клипы"
                downloadAllButton.onclick = function() {
                    window.location.href = data.download_all;
                };
            }
        })
        .catch(error => {
            uploadButton.disabled = false;
            uploadButton.innerHTML = '<i class="fas fa-magic"></i> Сгенерировать клипы';
            console.error('Error:', error);
            alert('Произошла ошибка при обработке видео.');
        });
    });
});
