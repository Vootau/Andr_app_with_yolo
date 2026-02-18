from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.uix.spinner import Spinner
from kivy.uix.camera import Camera
from kivy.uix.textinput import TextInput
from kivy.uix.checkbox import CheckBox
from kivy.metrics import dp
from kivymd.app import MDApp
from kivymd.theming import ThemeManager
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.textfield import MDTextField
from kivymd.uix.selectioncontrol import MDCheckbox
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.popup import Popup
from pathlib import Path

import cv2
import os
from ultralytics import YOLO
import datetime

# Модель YOLO
yolo_model = YOLO('yolov8n.pt')

# Глобальные переменные
SAVE_PATH = "./snapshot"

def detect_objects(frame):
    results = yolo_model.predict(frame)
    return results

def draw_boxes(frame, results):
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = yolo_model.names[class_id]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{class_name} ({confidence*100:.2f}%)"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return frame

class MainLayout(MDBoxLayout):
    auto_save_mode = False

    def __init__(self, **kwargs):
        super(MainLayout, self).__init__(orientation='vertical', **kwargs)

        # Виджет для вывода видео
        self.video_display = Image(size_hint=(1, 0.8), allow_stretch=True, keep_ratio=False)
        self.add_widget(self.video_display)

        # Панель управления
        control_panel = MDBoxLayout(size_hint=(1, 0.2))

        # Настройка кнопок
        self.start_button = MDRaisedButton(text="Старт", pos_hint={'center_x': .5})
        self.start_button.bind(on_release=self.start_video)
        control_panel.add_widget(self.start_button)

        self.stop_button = MDRaisedButton(text="Стоп", pos_hint={'center_x': .5})
        self.stop_button.bind(on_release=self.stop_video)
        control_panel.add_widget(self.stop_button)

        self.screenshot_button = MDRaisedButton(text="Скриншот", pos_hint={'center_x': .5})
        self.screenshot_button.bind(on_release=self.save_snapshot)
        control_panel.add_widget(self.screenshot_button)

        # Кнопка для выбора пути сохранения
        select_path_btn = MDRaisedButton(text="Выбрать путь сохранения", pos_hint={'center_x': .5})
        select_path_btn.bind(on_release=self.show_file_chooser)
        control_panel.add_widget(select_path_btn)

        # Выбор камеры
        switch_button = MDRaisedButton(text="Switch Camera", on_release=self.switch_camera)
        control_panel.add_widget(switch_button)

        # Авто-сохранение
        checkbox = MDCheckbox(group="auto-save")
        checkbox.bind(active=lambda _, val: self.toggle_auto_save(val))
        check_box_label = MDTextField(text="Авто-сохранение", readonly=True)
        control_panel.add_widget(checkbox)
        control_panel.add_widget(check_box_label)

        # Поле для списка объектов
        self.object_list = MDTextField(readonly=True, hint_text="Список объектов", multiline=True, size_hint=(1, 0.2))
        self.add_widget(self.object_list)

        # Добавляем панель управления
        self.add_widget(control_panel)

        # Камеры и событие обновления
        self.capture = None
        self.event = None
        self.current_frame = None

    def start_video(self, instance):
        if self.capture is not None:
            self.capture.release()
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            print("Не удалось открыть камеру.")
            return
        self.event = Clock.schedule_interval(self.update_frame, 1.0/30.0)

    def update_frame(self, dt):
        if self.capture is None:
            return
        ret, frame = self.capture.read()
        if not ret:
            return

        results = detect_objects(frame)
        frame = draw_boxes(frame, results)

        # Переводим кадр в нужный формат
        frame = cv2.flip(frame, 0)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.current_frame = frame.copy()

        # Сохраняем кадр, если активирован автоматический режим
        if self.auto_save_mode and len(results) > 0:
            filename = f"{SAVE_PATH}/snapshot_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            bgr_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR)
            bgr_frame = cv2.flip(bgr_frame, 0)
            cv2.imwrite(filename, bgr_frame)
            print(f"Кадр с объектами сохранён как {filename}")

        # Формируем текст с перечнем объектов
        objects_detected = []
        for result in results:
            for box in result.boxes:
                class_name = yolo_model.names[int(box.cls)]
                objects_detected.append(class_name)
        unique_objects = set(objects_detected)
        detected_str = ', '.join(unique_objects)
        self.object_list.text = detected_str

        # Отображаем кадр
        buf = frame.tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        self.video_display.texture = texture


    def stop_video(self, instance):
        if self.event is not None:
            self.event.cancel()
            self.event = None
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def switch_camera(self, instance):
        """
        Переключается циклично между доступными камерами.
        """
        if not hasattr(self, 'current_cam'):
            self.current_cam = 0

        available_cams = []
        for i in range(2):  # проверяем доступные камеры до индекса 2 делаем для телефона, больше 2-х не бывает.
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cams.append(i)
                cap.release()

        if len(available_cams) > 1:
            current_idx = self.current_cam % len(available_cams)
            next_idx = (current_idx + 1) % len(available_cams)

            new_cam_idx = available_cams[next_idx]
            self.capture.release()  # освобождаем предыдущую камеру перед переключением
            self.capture = cv2.VideoCapture(new_cam_idx)
            self.current_cam = new_cam_idx
            print(f"Переключение на камеру с индексом {new_cam_idx}.")
        else:
            print("Нет доступной второй камеры.")


    def toggle_auto_save(self, value):
        self.auto_save_mode = value
        if value:
            print("Режим автоматического сохранения включен.")
        else:
            print("Режим автоматического сохранения отключен.")

    def save_snapshot(self, instance):
        if self.current_frame is not None:
            filename = f"{SAVE_PATH}/snapshot_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            bgr_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR)
            bgr_frame = cv2.flip(bgr_frame, 0)
            cv2.imwrite(filename, bgr_frame)
            print(f"Снимок сохранен как {filename}")
        else:
            print("Нет кадра для сохранения")

    def show_file_chooser(self, _):
        """Показывает окно выбора каталога."""
        file_chooser = FileChooserListView(path=str(Path.home()), filters=[''], dirselect=True)
        popup = Popup(title="Выберите папку для сохранения:", content=file_chooser, size_hint=(0.9, 0.9))

        # Подтверждение выбора пути
        confirm_btn = Button(text="Подтвердить", size_hint=(0.3, 0.1))
        confirm_btn.bind(on_release=lambda _: self.set_save_path(file_chooser.path))
        popup.content.add_widget(confirm_btn)

        # Закрытие окна без подтверждения
        close_btn = Button(text="Закрыть", size_hint=(0.3, 0.1))
        close_btn.bind(on_release=popup.dismiss)
        popup.content.add_widget(close_btn)

        popup.open()

    def set_save_path(self, path):
        """Устанавливает путь для сохранения снимков."""
        global SAVE_PATH
        SAVE_PATH = str(Path(path))  # Преобразование строки в объект Path и получение абсолютного пути
        print(f"Установлен новый путь сохранения: {SAVE_PATH}")

class CameraApp(MDApp):
    def build(self):
        return MainLayout()

    def on_start(self):
        if not os.path.exists("snapshot"):
            os.makedirs("snapshot")
        layout = self.root  # root — ссылка на MainLayout
        layout.start_button.md_bg_color = self.theme_cls.primary_color
        layout.stop_button.md_bg_color = self.theme_cls.accent_color



if __name__ == '__main__':
    CameraApp().run()