"""
Videonun analizi, raporlaması ve silinmesi bu modülde yapılır.
Nesne tanıma bu modülde gerçekleşir, çizgiler ve sayıcılar bu modülde çizilir.
Bu modül default olarak 10 fps ve 1138x640 çözünürlük değerindedir.

Bu modülü çalıştırmak için aşağıdaki kod girilir:

.. code-block:: python

    python analyzer.py --input girdi/girdivideo.MP4 --output cikti/ciktivideo.avi --yolo yolo-coco


Çizgi koordinatları paint  yardımı ile bulunur
(x1,y1) , (x2,y2) olacak şekilde aşağıdaki kod ile belirlenir ve Sayıcı tanımlanır.
Bu koordinatlar 1138X640 Çözünürlüklü görüntü içindir.\n
Sort algoritmasından tracker bağlantısı ve sayıcıların tanımlanması da burada gerçekleşir.

.. code-block:: python

    tracker = Sort()
    memory = {}

    # Çizgi çekilecek kordinatları paint  yardımı ile bulup  aşağıya yazıyoruz.
    # bu koordinatlar 1138X640 Çözünürlüklü görüntü içindir.
    line = [(500, 430), (810, 420)]  # çizgi 1  sarı,  coordinates from (x1,y1) to (x2,y2)
    line2 = [(365, 375), (365, 259)]  # çizgi 2 kırmızı
    line3 = [(155, 300), (300, 260)]  # çizgi 3 mavi
    line4 = [(688, 220), (860, 190)]  # çizgi 4 turkuaz


    counter = 0  # sarı çizginin sayıcıları
    carCounter = 0
    truckCounter = 0
    otherCounter = 0

    counter2 = 0  # kırmızı çizginin sayıcıları
    carCounter2 = 0
    truckCounter2 = 0
    otherCounter2 = 0

    counter3 = 0  # mavi çizginin sayıcıları
    carCounter3 = 0
    truckCounter3 = 0
    otherCounter3 = 0

    counter4 = 0  # turkuaz çizginin sayıcıları
    carCounter4 = 0
    truckCounter4 = 0
    otherCounter4 = 0

    counter5 = 0  # yol 1  sayıcısı

    counter6 = 0  # yol 2 sayıcısı


input değerlerin ayrıştırıldığı yer

.. code-block:: python

    if __name__ == '__main__':
        # construct the argument parse and parse the arguments
        # input parametrelerinin ayrıştırıldığı yer
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--input", required=True,  # input edilecek videonun adresi
                        help="path to input video")
        ap.add_argument("-o", "--output", required=True, #output edilecek videonun adresi ve adı.uzantısı, uzantısı avi
                        help="path to output video")
        ap.add_argument("-y", "--yolo", required=True,  # yolonun ağırlık, etiket ve cfg dosyalarının adresi
                        help="base path to YOLO directory")
        ap.add_argument("-c", "--confidence", type=float, default=0.47,  # tanımadaki eminlik değeri default'u 0.5'ti
                        help="minimum probability to filter weak detections")
        ap.add_argument("-t", "--threshold", type=float, default=0.3,
                        # non-maxima suppression için eşik değeri default'u 0.5'ti
                        help="threshold when applyong non-maxima suppression")
        args = vars(ap.parse_args())


Modülün işlem süresini tespit için kronometre başlatılır

.. code-block:: python

    startTime = time.time()

Kesişim fonksiyonu tanımlanır

.. code-block:: python

    # Return true if line segments AB and CD intersect
    def intersect(A, B, C, D):
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


Bu program CPU'da çalışacak şekilde tasarlanmıştır. Eğer GPU'da çalıştıralacak olursa Bu kod bloğundaki
" index [ln[i - 1] " kısmı " index [ln[i[0] - 1] " şekline getirilmelidir.
Ayrıca kod bloğunda nesnelerin etiketlerini ve ağırlıklarını yüklüyoruz, nesne işaretlemenin rengini belirliyoruz.

.. code-block:: python

    # load the COCO class labels our YOLO model was trained on
    # nesnelerin etiket isimlerinin olduğu yer
    labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

    # derive the paths to the YOLO weights and model configuration
    # YOLO modelinin weights ve cfg dosyasının olduğu yer
    weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
    configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

    # load our YOLO object detector trained on COCO dataset (80 classes)
    # and determine only the *output* layer names that we need from YOLO
    # YOLO neural network nesne tanıma için yükleniyor
    print("[INFO] YOLO diskten yükleniyor...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    # ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # [ln[i[0] - 1] Sadece GPU'da kullanılır yoksa index hatası olur.
    # default'da GPU kullanılıyordu ve index 0'dan başlıyordu
    # ancak bu projede CPU kullanılmakta ve index 0'dan başlamıyor bu da index hatasına neden oluyor
    # hata olmaması için index i'den başlıyor.

Videonun input edildiği yer

.. code-block:: python

    # initialize the video stream, pointer to output video file, and
    # frame dimensions
    vs = cv2.VideoCapture(args["input"])  # input video bilgileri opencv tarafından alınıyor
    writer = None
    (W, H) = (None, None)

    frameIndex = 0

    # try to determine the total number of frames in the video file
    # videodaki toplam frame sayısı
    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \\
            else cv2.CAP_PROP_FRAME_COUNT
        total = int(vs.get(prop))
        print("[INFO] Toplam frame sayısı : {} ".format(total))


    # an error occurred while trying to determine the total
    # number of frames in the video file
    except:
        print("[INFO] could not determine # of frames in video")
        print("[INFO] no approx. completion time can be provided")
        total = -1


Yol tayini için nesnelerin id'si tutulacak listeleri hazırlıyoruz

.. code-block:: python

    # Yol tayini için listeler tanımlıyoruz.
    countedByLine1 = list()
    countedByLine2 = list()



Görüntünün çekilip, open cv ile işlendikten sonra YOLO tarafından nesnelerin tespit edildiği;
confidence, class ID ve koordinatlarının belirlediği yer

.. code-block:: python

    # loop over frames from the video file stream
    # videonun çekildiği video döngüsü
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()
        # frame = cv2.resize(frame, (1138, 640))

        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break

        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > args["confidence"]:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

Tespit edilen nesnelerin noktalara işaretlenerek; çizgilerle kesişiminin belirlendiği,
nesnelerin yol takibinin yapıldığı ve sayıcıların artırıldığı yer

.. code-block:: python

        dets = []
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                dets.append([x, y, x + w, y + h, confidences[i]])

        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        dets = np.asarray(dets)
        tracks = tracker.update(dets)

        boxes = []
        indexIDs = []
        c = []
        previous = memory.copy()
        memory = {}

        for track in tracks:
            boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track[4]))
            memory[indexIDs[-1]] = boxes[-1]

        if len(boxes) > 0:
            i = int(0)
            for box in boxes:
                # extract the bounding box coordinates
                (x, y) = (int(box[0]), int(box[1]))
                (w, h) = (int(box[2]), int(box[3]))

                # draw a bounding box rectangle and label on the image
                # color = [int(c) for c in COLORS[classIDs[i]]]
                # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
                #cv2.rectangle(frame, (x, y), (w, h), color, 2) # tespit edilen nesnenin çevresine kare çizme fonksiyonu

                if indexIDs[i] in previous:
                    previous_box = previous[indexIDs[i]]
                    (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                    (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                    p0 = (int(x + (w - x) / 2), int(y + (h - y) / 2))
                    p1 = (int(x2 + (w2 - x2) / 2), int(y2 + (h2 - y2) / 2))
                    cv2.line(frame, p0, p1, color, 5)  # default 3 tespit edilen nesnenin üzerine nokta koyulması

                    # Sarı çizgi
                    if intersect(p0, p1, line[0], line[1]):  # nokta ile çizgi kesişiyorsa
                        if (classIDs[i] == 1 or classIDs[i] == 2 or classIDs[i] == 3 or
                                classIDs[i] == 5 or classIDs[i] == 7):  #sadece bicycle,car, motorbike,bus,truck sayılır
                            counter += 1
                            if classIDs[i] == 2:  # classIDs[i] = otomobil ise carCounter artar
                                carCounter += 1
                            elif classIDs[i] == 7:  # classIDs[i] == kamyom ise truckCounter artar
                                truckCounter += 1
                            else:
                                otherCounter += 1  #classIDs[i] == bisiklet, motorsiklet ve otobüs ise otherCounter += 1
                            print("counter 1 = ", counter, LABELS[classIDs[i]])
                            countedByLine1.append(indexIDs[i])  # yol tayin için çizgiye değeni liste 1'e ekle

                    # Mavi çizgi
                    if intersect(p0, p1, line3[0], line3[1]):
                        if (classIDs[i] == 1 or classIDs[i] == 2 or classIDs[i] == 3 or
                                classIDs[i] == 5 or classIDs[i] == 7):  #sadece bicycle,car, motorbike,bus,truck sayılır
                            counter3 += 1
                            if classIDs[i] == 2:  # classIDs[i] = otomobil ise carCounter3 artar
                                carCounter3 += 1
                            elif classIDs[i] == 7:  # classIDs[i] = kamyon ise truckCounter3 artar
                                truckCounter3 += 1
                            else:
                                otherCounter3 += 1  #classIDs[i] = bisiklet,motorsiklet ve otobüs ise otherCounter3 += 1
                            print("counter 3 = ", counter3, LABELS[classIDs[i]])
                            countedByLine2.append(indexIDs[i])  # yol tayin için çizgiye değeni liste 2'ye ekle

                    # Kırmızı çizgi
                    if intersect(p0, p1, line2[0], line2[1]):
                        if (classIDs[i] == 1 or classIDs[i] == 2 or classIDs[i] == 3 or
                                classIDs[i] == 5 or classIDs[i] == 7):  #sadece bicycle,car, motorbike,bus,truck sayılır
                            counter2 += 1
                            if classIDs[i] == 2:  # classIDs[i] = otomobil ise carCounter2 artar
                                carCounter2 += 1
                            elif classIDs[i] == 7:  # classIDs[i] = kamyon ise truckCounter2 artar
                                truckCounter2 += 1
                            else:
                                otherCounter2 += 1  #classIDs[i] = bisiklet,motorsiklet ve otobüs ise otherCounter2 += 1
                            print("counter 2 = ", counter2, LABELS[classIDs[i]])
                            idBul2 = indexIDs[i]
                            if (idBul2 in countedByLine2):  # yol tayin için çizgiye değenin adını liste 2'de ara
                                counter6 += 1  # Listede varsa bir artır
                                print("counter 6 = ", counter6, LABELS[classIDs[i]])  # Yol 2 consola yazdır

                    # Turkuaz çizgi
                    if intersect(p0, p1, line4[0], line4[1]):
                        if (classIDs[i] == 1 or classIDs[i] == 2 or classIDs[i] == 3 or
                                classIDs[i] == 5 or classIDs[i] == 7):  #sadece bicycle,car, motorbike,bus,truck sayılır
                            counter4 += 1
                            if classIDs[i] == 2:  # classIDs[i] = otomobil ise carCounter4 artar
                                carCounter4 += 1
                            elif classIDs[i] == 7:  # classIDs[i] = kamyon ise truckCounter4 artar
                                truckCounter4 += 1
                            else:
                                otherCounter4 += 1  #classIDs[i] = bisiklet,motorsiklet ve otobüs ise otherCounter4 += 1
                            print("counter 4 = ", counter4, LABELS[classIDs[i]])
                            idBul = indexIDs[i]
                            if (idBul in countedByLine1):  # yol tayin için çizgiye değenin adını liste 1'de ara
                                counter5 += 1  # listede varsa bir artır
                                print("counter 5 = ", counter5, LABELS[classIDs[i]]) #yol 1 bilgisini consola yazdır

                # text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                # tespit edilen nesnenin üzerinde sınıfı ve eminlik değeri yazar
                text = "{} {}".format(indexIDs[i], LABELS[classIDs[i]])
                # tespit edilen nesnenin üzerinde id ve sınıfı yazar
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                i += 1

Çizgi ve sayıcıların görüntü üzerine yazıldığı yer.

.. code-block:: python

        # draw line  # çizgilerin ekranda gösterildiği yer
        cv2.line(frame, line[0], line[1], (0, 255, 255), 10) # Blue Green Red, Çizgi Kalınlığı
        cv2.line(frame, line2[0], line2[1], (0, 0, 255), 10)
        cv2.line(frame, line3[0], line3[1], (255, 0, 0), 10)
        cv2.line(frame, line4[0], line4[1], (255, 255, 0), 10)

        # draw counter   #  sayıcıların ekranda gösterildiği yer
        cv2.putText(frame, str(counter), (830, 420), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2) #sarı (850,150)
        cv2.putText(frame, str(counter2), (380, 230), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2) #kırmızı (350,150)
        cv2.putText(frame, str(counter3), (310, 245), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0), 2) #mavi (100, 150)
        cv2.putText(frame, str(counter4), (860, 170), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 0), 2) #turkuaz (600,150)
        cv2.putText(frame, str(counter5), (910, 135),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)  # beyaz (600,150) yön1
        cv2.putText(frame, str(counter6), (365, 189),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)  # beyaz (600,150) yön2

İşlenmiş videoyu kayıt ettiğimiz yer.

.. code-block:: python

        # saves image file, fotoğraf olarak kayıt etmek istersek
        # cv2.imwrite("output/frame-{}.png".format(frameIndex), frame)

        # check if the video writer is None
        if writer is None:
            # initialize our video writer
            # cv2.imshow("Kamera", frame)
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 10, # default 30 fps idi, input video fps
                                     (frame.shape[1], frame.shape[0]), True)

            # some information on processing single frame
            # bir frame'nin işlem süresi hakkında bilgilendirmeler
            if total > 0:
                elap = (end - start)
                print("[INFO] Tek frame için işlen süresi {:.4f} saniye".format(elap))
                print("[INFO] Tahmini toplam işlem süresi: {:.4f}".format(elap * total))

        # write the output frame to disk
        writer.write(frame)

Videoyu analiz esnasında ekranda göstermek istenirse aşağıdaki kodun yorumdan çıkarılması gerekir.

.. code-block:: python

    # cv2.imshow("Kamera", frame)
    # if cv2.waitKey(1) == 27:
    #   break

    # cv2.destroyAllWindows()

Videonun analiz tarihini kayıt edebilmek için sistemden zaman bilgisini alıp formatlıyoruz.

.. code-block:: python

    an = datetime.datetime.now()  # şu an, videonun analiz edildiği tarihi otomatik olarak listenin başına yazmak için.
    videoTime = time.ctime(os.path.getctime(args["input"]))
    videoTime = datetime.datetime.strptime(videoTime, "%a %b %d %H:%M:%S %Y")

Sonuçların txt dosyasına yazdırıldığı kısım

.. code-block:: python

        # sonuçların yazdırıldığı kısım
    toText = open("AnalizSonucu.txt", "a") #sonuçları yeni satır olarak eklemek için append modunda txt dosyası açıyoruz
    toText.write("Analiz Tarihi = {0} | ".format(str(an)))

    toText.write("Sarı Çizgi Toplam Araç = {0} - ".format(str(counter)))  # Sarı çizgiden geçen toplam araç sayısı
    toText.write("Otomobil = {0} - ".format(str(carCounter)))  # Sarı çizgiden geçen otomobil sayısı
    toText.write("Kamyon = {0} - ".format(str(truckCounter)))  # sarı çizgiden geçen kamyon sayısı
    toText.write("Diğer Araçlar = {0} | ".format(str(otherCounter)))  # sarı çizgiden geçen diğer araçların sayısı

    toText.write("Mavi Çizgi Toplam Araç = {0} - ".format(str(counter2)))
    toText.write("Otomobil = {0} - ".format(str(carCounter2)))
    toText.write("Kamyon = {0} - ".format(str(truckCounter2)))
    toText.write("Diğer Araçlar {0} | ".format(str(otherCounter2)))

    toText.write("Kırmızı Çizgi Toplam Araç {0} - ".format(str(counter3)))
    toText.write("Otomobil = {0} - ".format(str(carCounter3)))
    toText.write("Kamyon = {0} - ".format(str(truckCounter3)))
    toText.write("Diğer Araçlar {0} | ".format(str(otherCounter3)))

    toText.write("Turkuaz Çizgi Toplam Araç {0} - ".format(str(counter4)))
    toText.write("Otomobil = {0} - ".format(str(carCounter4)))
    toText.write("Kamyon = {0} - ".format(str(truckCounter4)))
    toText.write("Diğer Araçlar {0} | ".format(str(otherCounter4)))

    toText.write("Yol 1 = {0} | ".format(str(counter5)))
    toText.write("Yol 2 = {0} | ".format(str(counter6)))
    toText.write("Video Tarihi {0}".format(str(videoTime)))
    toText.write('\\n')
    toText.close()

İşlemin bitirildiği ve İşlem süresinin Console ekranına yazdırlıdığı yer.

.. code-block:: python

    print("[INFO] İşlem bitiriliyor...")
    writer.release()
    vs.release()
    endTime = time.time()

    print(f'Toplam işlem süresi: {endTime - startTime}')  # toplam süre consola yazdırılıyor.


HATA NOTU! \n "Eğer Belirtilen modül bulunamadı!" hatası verirse çözüm ilgili modülü yeniden yüklemektir.
Örnek hatada NumPy yüklü olmasına rağmen erişim prblemi vardı, NumPy "1.21.5" yeniden yüklenince hata giderildi.

HATA:

.. code-block:: python

    Please note and check the following:

    * The Python version is: Python3.7 from "C:/Users/aliozdemir/Trafik_Analiz/Kayittan/venvs/Scripts/python.exe"
    * The NumPy version is: "1.21.5"

    and make sure that they are the versions you expect.
    Please carefully study the documentation linked above for further help.

    Original error was: DLL load failed: Belirtilen modül bulunamadı.

ÇÖZÜM:

.. code-block:: python

     > pip install --upgrade --force-reinstall numpy==1.21.5


"""
import argparse
import imutils
import time
import cv2
import os
from sort import *
import datetime

# Kodu çalıştırmak için aşağıdaki kodu, proje klasöründe bulunan venv ile çalışan terminale yapıştırınız.
# python analyzer.py --input girdi/DJI_0096.MP4 --output cikti/DJI_0096_analiz.avi --yolo yolo-coco
# bu proje 1138X640 çözünürlükte görüntü için uyarlanmıştır


tracker = Sort()
memory = {}

# Çizgi çekilecek kordinatları paint  yardımı ile bulup  aşağıya yazıyoruz.
# bu koordinatlar 1138X640 Çözünürlüklü görüntü içindir.
line = [(500, 430), (810, 420)]  # çizgi 1  sarı,  coordinates from (x1,y1) to (x2,y2)
line2 = [(365, 375), (365, 259)]  # çizgi 2 kırmızı
line3 = [(155, 300), (300, 260)]  # çizgi 3 mavi
line4 = [(688, 220), (860, 190)]  # çizgi 4 turkuaz


counter = 0  # sarı çizginin sayıcıları
carCounter = 0
truckCounter = 0
otherCounter = 0

counter2 = 0  # kırmızı çizginin sayıcıları
carCounter2 = 0
truckCounter2 = 0
otherCounter2 = 0

counter3 = 0  # mavi çizginin sayıcıları
carCounter3 = 0
truckCounter3 = 0
otherCounter3 = 0

counter4 = 0  # turkuaz çizginin sayıcıları
carCounter4 = 0
truckCounter4 = 0
otherCounter4 = 0

counter5 = 0  # yol 1  sayıcısı

counter6 = 0  # yol 2 sayıcısı

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    # input parametrelerinin ayrıştırıldığı yer
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,  # input edilecek videonun adresi
                    help="path to input video")
    ap.add_argument("-o", "--output", required=True, #output edilecek videonun adresi ve adı.uzantısı, uzantısı avi
                    help="path to output video")
    ap.add_argument("-y", "--yolo", required=True,  #yolonun ağırlık,etiket ve cfg dosyalarının adresi
                    help="base path to YOLO directory")
    ap.add_argument("-c", "--confidence", type=float, default=0.47,  # tanımadaki eminlik değeri default'u 0.5'ti
                    help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=0.3,
                    # non-maxima suppression için eşik değeri default'u 0.5'ti
                    help="threshold when applyong non-maxima suppression")
    args = vars(ap.parse_args())


    startTime = time.time()


    # Return true if line segments AB and CD intersect
    def intersect(A, B, C, D):
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


    # load the COCO class labels our YOLO model was trained on
    # nesnelerin etiket isimlerinin olduğu yer
    labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

    # derive the paths to the YOLO weights and model configuration
    # YOLO modelinin weights ve cfg dosyasının olduğu yer
    weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
    configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

    # load our YOLO object detector trained on COCO dataset (80 classes)
    # and determine only the *output* layer names that we need from YOLO
    # YOLO neural network nesne tanıma için yükleniyor
    print("[INFO] YOLO diskten yükleniyor...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    # ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # [ln[i[0] - 1] Sadece GPU'da kullanılır yoksa index hatası olur.
    # default'da GPU kullanılıyordu ve index 0'dan başlıyordu
    # ancak bu projede CPU kullanılmakta ve index 0'dan başlamıyor bu da index hatasına neden oluyor
    # hata olmaması için index i'den başlıyor.


    # initialize the video stream, pointer to output video file, and
    # frame dimensions
    vs = cv2.VideoCapture(args["input"])  # input video bilgileri opencv tarafından alınıyor
    writer = None
    (W, H) = (None, None)

    frameIndex = 0

    # try to determine the total number of frames in the video file
    # videodaki toplam frame sayısı
    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
            else cv2.CAP_PROP_FRAME_COUNT
        total = int(vs.get(prop))
        print("[INFO] Toplam frame sayısı : {} ".format(total))


    # an error occurred while trying to determine the total
    # number of frames in the video file
    except:
        print("[INFO] could not determine # of frames in video")
        print("[INFO] no approx. completion time can be provided")
        total = -1

    # Yol tayini için listeler tanımlıyoruz.
    countedByLine1 = list()
    countedByLine2 = list()

    # loop over frames from the video file stream
    # videonun çekildiği video döngüsü
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()
        # frame = cv2.resize(frame, (1138, 640))

        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break

        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > args["confidence"]:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

        dets = []
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                dets.append([x, y, x + w, y + h, confidences[i]])

        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        dets = np.asarray(dets)
        tracks = tracker.update(dets)

        boxes = []
        indexIDs = []
        c = []
        previous = memory.copy()
        memory = {}

        for track in tracks:
            boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track[4]))
            memory[indexIDs[-1]] = boxes[-1]

        if len(boxes) > 0:
            i = int(0)
            for box in boxes:
                # extract the bounding box coordinates
                (x, y) = (int(box[0]), int(box[1]))
                (w, h) = (int(box[2]), int(box[3]))

                # draw a bounding box rectangle and label on the image
                # color = [int(c) for c in COLORS[classIDs[i]]]
                # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
                # cv2.rectangle(frame, (x, y), (w, h), color, 2) #tespit edilen nesnenin çevresine kare çizme fonksiyonu

                if indexIDs[i] in previous:
                    previous_box = previous[indexIDs[i]]
                    (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                    (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                    p0 = (int(x + (w - x) / 2), int(y + (h - y) / 2))
                    p1 = (int(x2 + (w2 - x2) / 2), int(y2 + (h2 - y2) / 2))
                    cv2.line(frame, p0, p1, color, 5)  # default 3 tespit edilen nesnenin üzerine nokta koyulması

                    # Sarı çizgi
                    if intersect(p0, p1, line[0], line[1]):  # nokta ile çizgi kesişiyorsa
                        if (classIDs[i] == 1 or classIDs[i] == 2 or classIDs[i] == 3 or
                                classIDs[i] == 5 or classIDs[i] == 7):  #sadece bicycle,car, motorbike,bus,truck sayılır
                            counter += 1
                            if classIDs[i] == 2:  # classIDs[i] = otomobil ise carCounter artar
                                carCounter += 1
                            elif classIDs[i] == 7:  # classIDs[i] == kamyom ise truckCounter artar
                                truckCounter += 1
                            else:
                                otherCounter += 1  #classIDs[i] = bisiklet, motorsiklet ve otobüs ise otherCounter += 1
                            print("counter 1 = ", counter, LABELS[classIDs[i]])
                            countedByLine1.append(indexIDs[i])  # yol tayin için çizgiye değeni liste 1'e ekle

                    # Mavi çizgi
                    if intersect(p0, p1, line3[0], line3[1]):
                        if (classIDs[i] == 1 or classIDs[i] == 2 or classIDs[i] == 3 or
                                classIDs[i] == 5 or classIDs[i] == 7): #sadece bicycle,car, motorbike,bus,truck sayılır
                            counter3 += 1
                            if classIDs[i] == 2:  # classIDs[i] = otomobil ise carCounter3 artar
                                carCounter3 += 1
                            elif classIDs[i] == 7:  # classIDs[i] = kamyon ise truckCounter3 artar
                                truckCounter3 += 1
                            else:
                                otherCounter3 += 1  #classIDs[i] = bisiklet,motorsiklet,otobüs ise otherCounter3 += 1
                            print("counter 3 = ", counter3, LABELS[classIDs[i]])
                            countedByLine2.append(indexIDs[i])  # yol tayin için çizgiye değeni liste 2'ye ekle

                    # Kırmızı çizgi
                    if intersect(p0, p1, line2[0], line2[1]):
                        if (classIDs[i] == 1 or classIDs[i] == 2 or classIDs[i] == 3 or
                                classIDs[i] == 5 or classIDs[i] == 7):  #sadece bicycle,car,motorbike,bus,truck sayılır
                            counter2 += 1
                            if classIDs[i] == 2:  # classIDs[i] = otomobil ise carCounter2 artar
                                carCounter2 += 1
                            elif classIDs[i] == 7:  # classIDs[i] = kamyon ise truckCounter2 artar
                                truckCounter2 += 1
                            else:
                                otherCounter2 += 1  #classIDs[i] = bisiklet,motorsiklet,otobüs ise otherCounter2 += 1
                            print("counter 2 = ", counter2, LABELS[classIDs[i]])
                            idBul2 = indexIDs[i]
                            if (idBul2 in countedByLine2):  # yol tayin için çizgiye değenin adını liste 2'de ara
                                counter6 += 1  # Listede varsa bir artır
                                print("counter 6 = ", counter6, LABELS[classIDs[i]])  # Yol 2 consola yazdır

                    # Turkuaz çizgi
                    if intersect(p0, p1, line4[0], line4[1]):
                        if (classIDs[i] == 1 or classIDs[i] == 2 or classIDs[i] == 3 or
                                classIDs[i] == 5 or classIDs[i] == 7): #sadece bicycle,car,motorbike,bus,truck sayılır
                            counter4 += 1
                            if classIDs[i] == 2:  # classIDs[i] = otomobil ise carCounter4 artar
                                carCounter4 += 1
                            elif classIDs[i] == 7:  # classIDs[i] = kamyon ise truckCounter4 artar
                                truckCounter4 += 1
                            else:
                                otherCounter4 += 1  # classIDs[i] = bisiklet,motorsiklet,otobüs ise otherCounter4 += 1
                            print("counter 4 = ", counter4, LABELS[classIDs[i]])
                            idBul = indexIDs[i]
                            if (idBul in countedByLine1):  # yol tayin için çizgiye değenin adını liste 1'de ara
                                counter5 += 1  # listede varsa bir artır
                                print("counter 5 = ", counter5, LABELS[classIDs[i]]) #yol 1 bilgisini consola yazdır

                # text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                # tespit edilen nesnenin üzerinde sınıfı ve eminlik değeri yazar
                text = "{} {}".format(indexIDs[i], LABELS[classIDs[i]])
                # tespit edilen nesnenin üzerinde id ve sınıfı yazar
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                i += 1

        # draw line  # çizgilerin ekranda gösterildiği yer
        cv2.line(frame, line[0], line[1], (0, 255, 255), 10) # Blue Green Red, Çizgi Kalınlığı
        cv2.line(frame, line2[0], line2[1], (0, 0, 255), 10)
        cv2.line(frame, line3[0], line3[1], (255, 0, 0), 10)
        cv2.line(frame, line4[0], line4[1], (255, 255, 0), 10)

        # draw counter   #  sayıcıların ekranda gösterildiği yer
        cv2.putText(frame, str(counter), (830, 420), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2) #sarı (850,150)
        cv2.putText(frame, str(counter2), (380, 230), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2) #kırmızı (350,150)
        cv2.putText(frame, str(counter3), (310, 245), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0), 2) #mavi (100, 150)
        cv2.putText(frame, str(counter4), (860, 170), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 0), 2) #turkuaz (600,150)
        cv2.putText(frame, str(counter5), (910, 135),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)  # beyaz (600,150) yön1
        cv2.putText(frame, str(counter6), (365, 189),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)  # beyaz (600,150) yön2


        # saves image file, fotoğraf olarak kayıt etmek istersek
        # cv2.imwrite("output/frame-{}.png".format(frameIndex), frame)

        # check if the video writer is None
        if writer is None:
            # initialize our video writer
            # cv2.imshow("Kamera", frame)
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 10, # default 30 fps idi, input video fps
                                     (frame.shape[1], frame.shape[0]), True)

            # some information on processing single frame
            # bir frame'nin işlem süresi hakkında bilgilendirmeler
            if total > 0:
                elap = (end - start)
                print("[INFO] Tek frame için işlen süresi {:.4f} saniye".format(elap))
                print("[INFO] Tahmini toplam işlem süresi: {:.4f}".format(elap * total))

        # write the output frame to disk
        writer.write(frame)

    # cv2.imshow("Kamera", frame)
    # if cv2.waitKey(1) == 27:
    #   break

    # cv2.destroyAllWindows()
    an = datetime.datetime.now()  # şu an, videonun analiz edildiği tarihi otomatik olarak listenin başına yazmak için.
    videoTime = time.ctime(os.path.getctime(args["input"]))
    videoTime = datetime.datetime.strptime(videoTime, "%a %b %d %H:%M:%S %Y")

    # sonuçların yazdırıldığı kısım
    toText = open("AnalizSonucu.txt", "a")  #sonuçları yeni satır olarak eklemek için append modunda txt dosyası açıyoruz
    toText.write("Analiz Tarihi = {0} | ".format(str(an)))

    toText.write("Sarı Çizgi Toplam Araç = {0} - ".format(str(counter)))  # Sarı çizgiden geçen toplam araç sayısı
    toText.write("Otomobil = {0} - ".format(str(carCounter)))  # Sarı çizgiden geçen otomobil sayısı
    toText.write("Kamyon = {0} - ".format(str(truckCounter)))  # sarı çizgiden geçen kamyon sayısı
    toText.write("Diğer Araçlar = {0} | ".format(str(otherCounter)))  # sarı çizgiden geçen diğer araçların sayısı

    toText.write("Mavi Çizgi Toplam Araç = {0} - ".format(str(counter2)))
    toText.write("Otomobil = {0} - ".format(str(carCounter2)))
    toText.write("Kamyon = {0} - ".format(str(truckCounter2)))
    toText.write("Diğer Araçlar {0} | ".format(str(otherCounter2)))

    toText.write("Kırmızı Çizgi Toplam Araç {0} - ".format(str(counter3)))
    toText.write("Otomobil = {0} - ".format(str(carCounter3)))
    toText.write("Kamyon = {0} - ".format(str(truckCounter3)))
    toText.write("Diğer Araçlar {0} | ".format(str(otherCounter3)))

    toText.write("Turkuaz Çizgi Toplam Araç {0} - ".format(str(counter4)))
    toText.write("Otomobil = {0} - ".format(str(carCounter4)))
    toText.write("Kamyon = {0} - ".format(str(truckCounter4)))
    toText.write("Diğer Araçlar {0} | ".format(str(otherCounter4)))

    toText.write("Yol 1 = {0} | ".format(str(counter5)))
    toText.write("Yol 2 = {0} | ".format(str(counter6)))
    toText.write("Video Tarihi {0}".format(str(videoTime)))
    toText.write('\n')
    toText.close()

    print("[INFO] İşlem bitiriliyor...")
    writer.release()
    vs.release()
    endTime = time.time()
    #  _     _
    #  \\ Λ //
    #  ||(|)||
    #  \\\|///
    #   \\|//
    #    \|/ 
    #     |  /
    #   \ | /
    #    \|/
    #     |
    #     |
    #     |
    #     |
    #     |
    #     |
    
    print(f'Toplam işlem süresi: {endTime - startTime}')  # toplam süre consola yazdırılıyor.