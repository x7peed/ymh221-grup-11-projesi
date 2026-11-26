import cv2, os, time, math, urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import ssl
import certifi

# mediapipe modullerini uzun uzun yazmamak icin kisalttik
mp_vision = vision
mp_core = python.BaseOptions

model_dosyasi = "face_landmarker.task"

# model dosyasi klasorde yoksa hata vermesin diye bastan indiriyoruz
if not os.path.exists(model_dosyasi):
    print("model dosyasi yok, netten indiriliyor...")
    
    # SSL context with certifi certificates (more up-to-date than system)
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    # Custom opener with SSL context
    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=ssl_context)
    )
    urllib.request.install_opener(opener)
    
    try:
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", 
            model_dosyasi
        )
        print("tamamdir indi.")
    except Exception as e:
        print(f"İndirme başarısız: {e}")
        print("Manuel indirme için: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
        raise

# mp video modu ayarlari, yuzu cabuk bulsun diye confidence 0.5 biraktim
opts = mp_vision.FaceLandmarkerOptions(
    base_options=mp_core(model_asset_path=model_dosyasi),
    running_mode=mp_vision.RunningMode.VIDEO,
    num_faces=4, 
    min_face_detection_confidence=0.5
)
model = mp_vision.FaceLandmarker.create_from_options(opts)
yuz_cizgileri = mp_vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION

# organ verileri: (nokta indeksleri, x offset, y offset, renk)
# if-else ameleliginden kurtulmak icin her seyi buraya paketledim
organlar = {
    "SOL KAS": ([293,300,334,296], -140, -55, (255,191,0)),
    "SAG KAS": ([63,70,105,66], 60, -55, (255,191,0)),
    "SOL GOZ": ([263,362,386,374], -150, -15, (255,0,255)),
    "SAG GOZ": ([33,133,159,145], 70, -15, (255,0,255)),
    "BURUN": ([1,2,98,327], 85, 75, (0,255,255)),
    "AGIZ": ([0,13,14,17], -150, 85, (0,165,255)),
    "CENE": ([152], 65, 115, (0,255,0))
}

def etiket_ciz(kare, x, y, yazi, offset_x, offset_y, renk):
    # hedefin ustune ufak bi radar noktasi ve cizgi atiyoruz
    cv2.circle(kare, (x, y), 4, renk, -1)
    cv2.line(kare, (x, y), (x+offset_x, y+offset_y), renk, 2)
    
    # yazinin arkasina siyah bi arka plan lazim yoksa okunmuyor
    (text_width, text_height), _ = cv2.getTextSize(yazi, cv2.FONT_HERSHEY_PLAIN, 1, 1)
    cv2.rectangle(kare, (x+offset_x-5, y+offset_y-text_height-5), (x+offset_x+text_width+5, y+offset_y+5), (0,0,0), -1)
    cv2.putText(kare, yazi, (x+offset_x, y+offset_y), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)

cap = cv2.VideoCapture(0)
eski_zaman = 0
etiketleri_goster = True

while True:
    ret, kare = cap.read()
    if not ret: break

    # aynalama yapmazsak kafa karisiyor o yuzden flip attik
    kare = cv2.flip(kare, 1) 
    h, w, _ = kare.shape
    
    # opencv bgr calisir ama mediapipe rgb ister, cevirmemiz lazim
    rgb = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
    mp_kare = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    # kronik mp sorunu: ayni ms gelirse cokuyor. try-except ile yamadik geciyoruz
    zaman_damgasi = int(time.time() * 1000)
    try:
        sonuc = model.detect_for_video(mp_kare, zaman_damgasi)
    except:
        continue 
        
    if sonuc.face_landmarks:
        for i, yuz_verisi in enumerate(sonuc.face_landmarks):
            # yuzdeki o havali orumcek agi (mesh) cizgileri
            for cizgi in yuz_cizgileri:
                x1, y1 = int(yuz_verisi[cizgi.start].x * w), int(yuz_verisi[cizgi.start].y * h)
                x2, y2 = int(yuz_verisi[cizgi.end].x * w), int(yuz_verisi[cizgi.end].y * h)
                cv2.line(kare, (x1, y1), (x2, y2), (100,100,100), 1)
            
            # ekranda birden fazla kisi olursa karismasin diye kafaya numara caktik
            cv2.putText(kare, f"Kisi {i+1}", (int(yuz_verisi[10].x*w), int(yuz_verisi[10].y*h)-20), 
                        cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 2)

            if etiketleri_goster:
                for isim, veri in organlar.items():
                    noktalar, ox, oy, rnk = veri
                    
                    # matematiksel ortalama alip organin tam merkezini buluyoruz
                    ort_x = int(sum([yuz_verisi[idx].x for idx in noktalar]) / len(noktalar) * w)
                    ort_y = int(sum([yuz_verisi[idx].y for idx in noktalar]) / len(noktalar) * h)
                    etiket_ciz(kare, ort_x, ort_y, isim, ox, oy, rnk)
                    
                # kameraya yaklasinca bozulmasin diye dudak acikligini goze oranladim (derinlik hilesi)
                dudak_mesafesi = math.hypot(yuz_verisi[13].x - yuz_verisi[14].x, yuz_verisi[13].y - yuz_verisi[14].y)
                goz_mesafesi = math.hypot(yuz_verisi[33].x - yuz_verisi[263].x, yuz_verisi[33].y - yuz_verisi[263].y)
                
                if goz_mesafesi > 0 and (dudak_mesafesi / goz_mesafesi) > 0.12:
                    dx = int(((yuz_verisi[13].x + yuz_verisi[14].x) / 2) * w)
                    dy = int(((yuz_verisi[13].y + yuz_verisi[14].y) / 2) * h)
                    etiket_ciz(kare, dx, dy, "DIL", -50, 80, (0,0,255))

    # klasik fps hesabi
    yeni_zaman = time.time()
    fps = int(1 / (yeni_zaman - eski_zaman)) if (yeni_zaman - eski_zaman) > 0 else 0
    eski_zaman = yeni_zaman
    
    cv2.putText(kare, f"FPS: {fps} (Gizle: L, Cikis: Q)", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.2, (0,255,0), 2)
    cv2.imshow("Kamera", kare)
    
    tus = cv2.waitKey(1) & 0xFF
    if tus == ord('q'): break
    if tus == ord('l'): etiketleri_goster = not etiketleri_goster

# isi bitince kamerayi salalim yoksa isigi yanik kaliyor
cap.release()
cv2.destroyAllWindows()