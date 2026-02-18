from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
import cv2
from kivy.graphics.texture import Texture

class TestCamera(App):
    def build(self):
        self.img = Image()
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            print("Не удалось открыть камеру")
        Clock.schedule_interval(self.update, 1.0/30.0)
        return self.img

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            # Конвертируем в RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buf = frame.tobytes()
            # Создаём текстуру
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
            texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
            texture.flip_vertical()
            self.img.texture = texture

if __name__ == '__main__':
    TestCamera().run()
