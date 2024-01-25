import torch
import numpy as np
import cv2
from time import time
from datetime import datetime

class WasteDetector:

    def __init__(self, capture_index, model_name):
        """
        hangi kamerayı kullancağımız, hangi modeli kullanacağımız ekran kartı mı yoksa işlemci mi kullanacağız
        ve bazı değişkenlere atama yapıyoruz
        """
        self.capture_index = capture_index
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

    def get_video_capture(self):
        """
        kameradan görüntü alıyoruz
        """
        if(self.capture_index==0):
            cap = cv2.VideoCapture(self.capture_index, cv2.CAP_DSHOW) #CAP_DSHOW, kameranın Windows'ta hızlı açılmasını sağlıyor
        else:
            cap = cv2.VideoCapture(self.capture_index)

        return cap

    def load_model(self, model_name):
        """
        Pytorch hub'dan Yolov5 modelini indiriyoruz ve bunu modüle geri döndürüyoruz
        """
        if model_name:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        """
        kameradan aldığı görüntüyü modele sokarak ondan tahmin oranı alıyoruz
        """
        self.model.to(self.device)


        #modelin özgüven eşiğini ayarlıyoruz
        self.model.conf = 0.30

        #modelin IoU eşiğini ayarlıyoruz
        self.model.iou = 0.50

        #modele görüntü bazlı çıkarım yaptırıyoruz
        frame = [frame]
        results = self.model(frame)

        #çıkarım sonucu görüntüde bulunduğu tahmin edilen objelerin bilgilerini (etiket, koordinatlar, özgüven skorları) alıyoruz
        labels, cord, confidence_scores = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1], results.xyxyn[0][:, 4]

        #alınan bilgileri döndürüyoruz
        return labels, cord, confidence_scores

    def class_to_label(self, x):
        """
        sınıf id'lerini etikete dönüştürüyoruz.
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        aranan objenin hangi konumlar içinde ne kadar özgüven skoru ile yer aldığını kutular şeklinde ekrana yazdırıyoruz.
        """
        frame_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        labels, cord, confidence_scores = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        # her sınıf için farklı renk
        class_colors = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189), (140, 86, 75), (227, 119, 194)]

        for i in range(n):
            row = cord[i]

            class_index = int(labels[i])
            class_label = self.class_to_label(class_index)

            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                    row[3] * y_shape)

            conf = confidence_scores[i]

            # sınıf için farklı renk eşleştirmesi
            bgr = class_colors[class_index]

            # sınırlayıcı kutu
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)

            # sınırlayıcı kutuyu açıklayan text
            label_text = f"{class_label} {conf:.2f}"
            cv2.putText(frame, label_text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)

            # transparan sınırlayıcı kutu
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), bgr, cv2.FILLED)
            alpha = 0.3
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            print(f"Görüntü Tarihi: {frame_time}, Obje Sınıfı: {class_label}, Koordinatlar (x1,y1)(x2,y2): ({x1},{y1})({x2},{y2}), Özgüven Skoru: {conf:.2f}")

        return frame

    def __call__(self):

        """
        kamerayı açarak aranan nesnenin nerede olduğunu hangi nesne olduğunu ve % kaç olasılıkla onun olduğunu yazıyoruz.
        """
        cap = self.get_video_capture()
        try:
            print("Video kaynağının başlatılması kontrol ediliyor...")
            assert cap.isOpened()
            print("Video yakalama işlemi başlatıldı.")
        except:
            raise ValueError("HATA: Video yakalama başlatılamadı!")

        try:
            while True:

                ret, frame = cap.read()
                assert ret

                frame = cv2.resize(frame, (1000, 700))

                start_time = time()

                results = self.score_frame(frame)
                frame = self.plot_boxes(results, frame)

                # FPS için zamanı bitir
                end_time = time()
                fps = 1 / np.round(end_time - start_time, 2)
                cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

                #Görüntüyü bastır
                cv2.imshow('YOLOv5 Detection', frame)

                if cv2.waitKey(5) & 0xFF == ord('q'):
                    print("q tuşuna basıldı.")
                    break

            print("Kaynak kapatılıyor...")
            cap.release()
            cv2.destroyAllWindows()
            print("Kaynak kapatıldı.")
        except:
            print("Kaynak kapatılıyor...")
            cap.release()
            cv2.destroyAllWindows()
            print("Kaynak kapatıldı.")
            raise ValueError("HATA: Video yürütme esnasında sorun yaşandı.")


if __name__ == "__main__":

    # YOLOv5 model ağırlığı
    my_model_name = "best.pt"

    # Görüntü kaynağı olarak kamera seçiliyor
    my_capture_index = 0
    #my_capture_index = "susisesi.mp4"

    # sınıftan yeni bir obje oluşturarak instance başlatıyoruz.
    detector = WasteDetector(capture_index=my_capture_index, model_name=my_model_name)

    # instance'ı çalıştırıyoruz
    detector()