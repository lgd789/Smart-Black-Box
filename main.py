import cv2
import sys
import math
import timeit
import pymysql
import threading
import numpy as np

from matplotlib.path import Path
from ultralytics import YOLO
from ocr import OCRModel
from PIL import ImageFont, ImageDraw, Image

class Car:
    def __init__ (self, id, bbox, car_img, segment=None, lane=None, plate_num=None):
        self.id = id
        self.bbox = bbox
        self.segment = segment
        self.x, self.y = int(bbox[0]-(bbox[2]/2)), int(bbox[1]-(bbox[3]/2))
        self.w, self.h = int(bbox[2]), int(bbox[3])
        self.benchmark_x = self.x + (self.w//2)
        self.benchmark_y = self.y + self.h
        self.x_position = None
        self.y_position = None
        self.lane = lane
        self.plate_id = None
        self.car_img = car_img

    def setBenchmarks(self, benchmark_x, benchmark_y):
        self.benchmark_x = benchmark_x
        self.benchmark_y = benchmark_y

    def setXPosition(self, x_position):
        self.x_position = x_position

    def setYPosition(self, y_position):
        self.y_position = y_position

    def setLane(self, lane):
        self.lane = lane
      
    def setPlateId(self, plate_id):
        self.plate_id = plate_id

    def setImg(self, car_img):
        self.car_img = car_img

    def getId(self):
        return self.id
    
    def getBenchmarks(self):
        return self.benchmark_x, self.benchmark_y 
    
    def getXPosition(self):
        return self.x_position

    def getYPosition(self):
        return self.y_position

    def getLane(self):
        return self.lane     

    def getPlateId(self):
        return self.plate_id
    
    def getBbox(self):
        return self.bbox
    
    def identifyViolate(self):
        # masking 이미지 생성 
        img_lane = np.zeros(frame.shape, dtype=np.uint8)
        img_car = np.zeros(frame.shape, dtype=np.uint8)

        line_right = self.lane
        line_left = line_right-1

        lines[line_right].drawOnBlackboard(img_lane)
        lines[line_left].drawOnBlackboard(img_lane)

        self.drawOnBlackboard(img_car)

        overlap = cv2.bitwise_and(img_car, img_lane)
        mask_img = img_car + img_lane
        mask_img[np.where((overlap == (0, 255, 0)).all(axis=2))] = (255, 0, 255)


        result_classify = model_cls(mask_img, device=0)
        for result in result_classify:
            cls = result.probs.top1
            conf = result.probs.top1conf
            
            #_위반13차량 식별
            if cls == 2 and conf > 0.8:
                
                #_위반13량 저장
                if self.id not in violateCars.keys():
                    violateCars[self.id] = ViolateCar(1, False, 'violation')
                else:
                    if violateCars[self.id].getType() == 'danger':
                        violateCars[self.id].setType('violation')

                # 번호판 인식 차량 ->_위반13
                if not violateCars[self.id].getReport() and violateCars[self.id].getType() == 'violation':
                    if self.plate_id in plates.keys() and plates[self.plate_id].getConf() > 0.9:
                        violateSetDB(plates[self.plate_id].getPlatenum(), 1)
                        violateCars[self.id].setReport(True)

    def drawOnWhiteboard(self, usr_car_lane):
        global img_white

        segment_width = img_white_w // 5
        if self.lane is not None:
            if usr_car_lane-2 <= self.lane <= usr_car_lane+2:
                x = int(abs((usr_car_lane-2)-self.lane) * segment_width) + int(segment_width * self.x_position)
                y = int((img_white_h-100) * self.y_position)

                # 화이트보드 차량사진 넣기
                car_region = (0, 0, img_car_w, img_car_h)
                car_point, region_point = carRectangle((x, y), car_region)
                x1, y1, x2, y2 = car_point
                region_x1, region_y1, region_x2, region_y2 = region_point
                img_white[y1:y2, x1:x2] = self.car_img[region_y1:region_y2, region_x1:region_x2]
                
                if self.segment is not None and self.id in violateCars:
                    cv2.fillPoly(frame, [self.segment], (0, 0, 255))
                if self.plate_id is not None:
                    # cv2.putText(img_white, plates[self.plate_id].getPlatenum(), (x1, y1-20), 0, 1, (255, 0, 0), 2)
                    img_white = korPutText(img_white, x1, y2, plates[self.plate_id].getPlatenum(),  (255, 0, 0))

    def drawOnBlackboard(self, img):
        if self.segment is not None:
            cv2.fillPoly(img, [self.segment], (0, 255, 0))

class ViolateCar:
    def __init__(self, cnt, report, type, plateId = None):
        self.cnt = cnt
        self.report = report
        self.plateId = plateId
        self.type = type

    def setViolateCnt(self, cnt):
        self.cnt = cnt

    def setReport(self, report):
        self.report = report
    
    def setType(self, type):
        self.type = type

    def getViolateCnt(self):
        return self.cnt
    
    def getReport(self):
        return self.report
    
    def getType(self):
        return self.type
    
    def getPlateId(self):
        return self.plateId

    def addViolateCnt(self):
        self.cnt = self.cnt + 1


class Plate:
    def __init__(self, id, bbox, plate_num=None, conf=0.0, car_id=None):
        self.id = id
        self.bbox = bbox
        self.x, self.y = int(bbox[0]-(bbox[2]/2)), int(bbox[1]-(bbox[3]/2))
        self.w, self.h = int(bbox[2]), int(bbox[3])
        self.plate_num = plate_num
        self.conf = conf
        self.car_id = car_id
        
    def setCarId(self, car_id):
        self.car_id = car_id

    def setBbox(self, bbox):
        self.bbox = bbox
        self.x, self.y = int(bbox[0]-(bbox[2]/2)), int(bbox[1]-(bbox[3]/2))
        self.w, self.h = int(bbox[2]), int(bbox[3])

    def getCarId(self):
        return self.car_id 
    
    def getPlatenum(self):
        return self.plate_num  
    
    def getConf(self):
        return self.conf
    
    def ocr(self):
        if self.conf < 0.9:
            plate_num, conf = model_ocr.ocr(frame, self.bbox)

            if self.conf < conf:
                self.plate_num = plate_num
                self.conf = conf

                print(self.id, self.plate_num)

    def identifyPlateForCar(self):
        if self.car_id is None:
            for car_id, car in cars.items():
                if overLap(car.getBbox(), self.bbox):
                    self.car_id = car_id
                    car.setPlateId(self.id)      

class Person:
    def __init__(self, id, bbox):
        self.id = id
        self.bbox = bbox
        self.x, self.y = int(bbox[0]-(bbox[2]/2)), int(bbox[1]-(bbox[3]/2))
        self.w, self.h = int(bbox[2]), int(bbox[3])
        self.benchmark_x = self.x + (self.w//2)
        self.benchmark_y = self.y + self.h
 
    def setBenchmarks(self, benchmark_x, benchmark_y):
        self.benchmark_x = benchmark_x
        self.benchmark_y = benchmark_y

    def getBenchmarks(self):
        return self.benchmark_x, self.benchmark_y
    
    def getBbox(self):
        return self.bbox
    
class Line:
    def __init__(self, type, min_x, max_x, segment = None, graph = None, line_change_possible = None, spot=None):
        self.type = type
        self.graph = graph
        self.segment = segment
        self.line_change_possible = line_change_possible
        self.spot = spot
        self.min_x = min_x
        self.max_x = max_x

    def setSpot(self, spot):
        self.spot = spot

    def getType(self):
        return self.type

    def getGraph(self):
        return self.graph

    def getSegment(self):
        return self.segment

    def getLineChangePossible(self):
        return self.line_change_possible    
        
    def getSpot(self):
        return self.spot
    
    def getMinX(self):
        return self.min_x

    def getMaxX(self):
        return self.max_x
    
    def drawOnWhiteboard(self, index, usr_car_lane, left_color, right_color):
        segment_width = img_white_w // 5
        if usr_car_lane-4 < index < usr_car_lane+3:
            x = int(abs((usr_car_lane-3)-index) * segment_width)
            y1, y2 = 0, img_white_h
            # color = line_color[self.type]
            color = (127,127,127)
            if index==(usr_car_lane-1):
                color = left_color
            elif index==(usr_car_lane):
                color = right_color

            if self.type < 3:
                drawDottedLine(img_white, (x, y1), (x, y2), color, 10)
            else:
                cv2.line(img_white, (x, y1), (x, y2), color, 10)
    def drawOnBlackboard(self, img):
        if self.type < 3:
            color = (255, 255, 255)
        elif self.type < 6:
            color = (0, 0, 255)

        for x in range(self.min_x, self.max_x):
            y_curve=int(np.polyval(self.graph, x))
            
            cv2.circle(img,
                        center=(x,y_curve),
                        radius=3,
                        color=color,
                        thickness=-1)
            
def korPutText(img, x, y, text, color):
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    font=ImageFont.truetype("font/NanumGothicBold.ttf", 20)
    org=(x, y-60)

    draw.text(org, text, font=font, fill=color)
    img = np.array(img)
    return img
    
def carRectangle(point, car_region):
    # 차량사각형 좌표 계산

    car_x, car_y = point
    region_x1, region_y1, region_x2, region_y2 = car_region

    x1 = int(car_x - img_car_w / 2)
    y1 = int(car_y - img_car_h)
    x2 = int(car_x + img_car_w / 2)
    y2 = int(car_y)

    # 이미지 영역 넘어가는 좌표값 설정
    if x1 < 0:
        region_x1 = abs(x1)
        x1 = 0
    if x2 > img_white_w:
        region_x2 = region_x2 - (x2 - img_white_w)
        x2 = img_white_w
    if y1 < 0:
        region_y1 = abs(y1)
        y1 = 0
    if y2 > img_white_h:
        region_y2 = region_y2 - (y2 - img_white_h)
        y2 = img_white_h

    return (x1, y1, x2, y2), (region_x1, region_y1, region_x2, region_y2)

def contactPoint(point,graph):
    a, b, r = benchmark_x, benchmark_y, math.sqrt(((point[0] - benchmark_x)**2 + (point[1] - benchmark_y)**2))
    cv2.circle(frame, (int(benchmark_x), int(benchmark_y)), int(r), (255, 255, 255), 3)

    # 직선의 기울기와 y절편
    m = graph[0]
    c = graph[1]

    # 이차방정식 계수 계산 
    A = m**2 + 1
    B = 2 * (m*c - m*b - a)
    C = a**2 + b**2 - r**2 + c**2 - 2*b*c

    # 판별식(Discriminant)
    D = B**2 - 4*A*C

    if D < 0:
        print("No tangent points exist.")
    elif D == 0: # 한 점에서 만나는 경우 (직선이 원에 내접하는 경우)
        x1 =int( -B / (2*A))
        y1 = int(m*x1 + c )
        if y1 > center_y:
            return [x1, y1]
    else: # 두 점에서 만나는 경우 
        x1 = int((-B + math.sqrt(D)) / (2*A))
        y1= int(m*x1 + c)
        
        x2= int((-B - math.sqrt(D)) / (2*A))
        y2= int(m*x2 + c) 
        print('y1 y2', y1, y2)
        if y1 < y2:
            return [x2,y2]
        else:
            return [x1, y1]

def overLap(bbox1, bbox2):
    # bbox1
    x, y = int(bbox1[0]-(bbox1[2]/2)), int(bbox1[1]-(bbox1[3]/2))
    w, h = int(bbox1[2]), int(bbox1[3])
    rect1 = x, y, (x+w), (y+h)
    # bbox2
    x, y = int(bbox2[0]-(bbox2[2]/2)), int(bbox2[1]-(bbox2[3]/2))
    w, h = int(bbox2[2]), int(bbox2[3])
    rect2 = x, y, (x+w), (y+h)
    
    x1 = max(rect1[0], rect2[0])
    y1 = max(rect1[1], rect2[1])
    x2 = min(rect1[2], rect2[2])
    y2 = min(rect1[3], rect2[3])
    
    if x1 < x2 and y1 < y2:
        intersection_area = (x2 - x1) * (y2 - y1)
        bbox1_area = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
        bbox2_area = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
        overlap_ratio = intersection_area / min(bbox1_area, bbox2_area)

        if overlap_ratio > 0.9:
            return True
        else:
            return False
    else:
        return False
    
def objectTrack(model, **kwargs):
    result_detects = model.track(**kwargs)
    result_carDetects = model_car.track(source=frame, persist=True, tracker='bytetrack.yaml', conf=0.5, device=0, show_conf=True, classes=[2, 5, 7])
    
    track_ids = []
    track_car_ids = []

    track_clss = []
    track_car_clss = []
    
    bboxes = []
    track_car_bboxes = []
    
    track_car_segments = []

    confidences= []
    track_car_confidences = [] 

    if result_detects[0].boxes.id is not None:
        track_ids = result_detects[0].boxes.id.int().cpu().tolist()  
        track_clss = result_detects[0].boxes.cls.int().cpu().tolist()
        bboxes = result_detects[0].boxes.xywh.cpu().tolist()
        confidences = result_detects[0].boxes.conf.cpu().numpy().astype(int)

    if result_carDetects[0].boxes.id is not None:
        track_car_ids = result_carDetects[0].boxes.id.int().cpu().tolist()  
        track_car_clss = result_carDetects[0].boxes.cls.int().cpu().tolist()
        track_car_bboxes = result_carDetects[0].boxes.xywh.cpu().tolist()
        track_car_segments = result_carDetects[0].masks.xy
        track_car_confidences = result_carDetects[0].boxes.conf.cpu().numpy().astype(int)

    # Visualize the results on the frame
    # Plot the tracks
    plate_track_ids = []

    for track_id, track_cls, bbox, segment in zip(track_car_ids, track_car_clss, track_car_bboxes, track_car_segments):
        segment = segment.astype(int)

        car_img = img_car_gray
        if track_id in violateCars.keys():
            car_img = img_car_yellow

        cars[track_id] = Car(track_id, bbox, car_img, segment)
        
        for plate_id, plate in plates.items():
            if plate.getCarId() == track_id: # 이미 번호판 정보를 알고있는 자동차 일 때
                # 차 객체에 번호판 ID 등록 해서 서로 연결
                cars[track_id].setPlateId(plate_id)


    for track_id, track_cls, bbox in zip(track_ids, track_clss, bboxes):
        if track_cls == 1: # 1 == plate 
            plate_track_ids.append(track_id)            
            if track_id not in plates.keys(): # 처음 잡힌 번호판 일 때
                plates[track_id] = Plate(track_id, bbox)
            else:
                plates[track_id].setBbox(bbox)
            plates[track_id].ocr() # OCR 
            
        elif track_cls == 2: # 2 == person
            persons[track_id] = Person(track_id, bbox)

    # 차량과 해당 번호판 매칭
    for plate_track_id in plate_track_ids:
        plates[plate_track_id].identifyPlateForCar()
        
        # db_위반13량 여부 확인
        if plates[plate_track_id].getCarId() not in violateCars.keys():
            if plates[plate_track_id].getConf() > 0 and violateDB(plates[plate_track_id].getPlatenum()): 
                violateCars[plates[plate_track_id].getCarId()] = ViolateCar(1, False, 'danger')

        #_위반13-> 번호판인식
        if plates[plate_track_id].getCarId() in violateCars.keys():
            if violateCars[plates[plate_track_id].getCarId()].getType() == 'violation':
                if plates[plate_track_id].getConf() > 0.9 and not violateCars[plates[plate_track_id].getCarId()].getReport():
                    # db 저장
                    violateSetDB(plates[plate_track_id].getPlatenum(), 1)
                    # 한번만 저장 
                    violateCars[plates[plate_track_id].getCarId()].setReport(True)
               

def instanceTrack(model, **kwargs):
    result_segs = model.track(**kwargs)
    if result_segs[0].boxes.id is not None:
        track_ids = result_segs[0].boxes.id.int().cpu().tolist()
        track_clss = result_segs[0].boxes.cls.int().cpu().tolist()
        segments = result_segs[0].masks.xy
        confidences = result_segs[0].boxes.conf.cpu().numpy().astype(int)
        
        # white_dotted 0, yellow_dotted 1,  blue_dotted 2
        # white_solid 3, yello_solid 4, blue_solid 5
        # stop_line 6, cross 7
        graphs = []
        cps = [] 
        for track_cls, segment in zip(track_clss, segments):
            if track_cls < 6:
                segment = segment.astype(int)
                # Fill poly function applied
                # X와 y로 분리
                x = segment[:, 0]
                y = segment[:, 1]
                min_x = min(x)
                max_x = max(x)
                p = np.polyfit(x, y, deg=1)
                graphs.append(list(np.append(p, [track_cls, min_x, max_x, segment])))
    
                cp = contactPoint((img_w//2, img_h//10*9), p)
                if cp is not None:
                    cv2.circle(frame, (cp[0], cp[1]), 1, (0, 255, 255), 15)
                    cps.append(cp)
                    
        # 차선 정렬            
        # solutions = [3, 1, 2], graphs = ['A', 'B', 'C']  ---> [(1, 'B'), (2, 'C'), (3,'A')]
        for i, (_, graph) in enumerate(sorted(zip(cps, graphs), key=lambda x: x[0])):
            w, c, track_cls, min_x, max_x, segment = graph
            if track_cls < 3:
                lines[i] = Line(track_cls, min_x, max_x, segment, (w, c), True, i)
            else:
                lines[i] = Line(track_cls, min_x, max_x, segment, (w, c), False, i)

def identifyLanes(car):
    car_getBenchmark_x, car_getBenchmark_y = car.getBenchmarks()
    cv2.circle(frame, car.getBenchmarks(), 1, (0, 0, 255), 15)
    pre_cp = None

    for i, line in lines.items():
        if line.getGraph() is not None:
            cp = contactPoint((car_getBenchmark_x, car_getBenchmark_y), line.getGraph())
            cv2.circle(frame, cp, 1, (255, 0, 0), 15)
            if cp is not None:
                if car_getBenchmark_x < cp[0]:
                    if pre_cp is not None:
                        car.setLane(i)
                        d1 = math.sqrt((car_getBenchmark_x - pre_cp[0])**2 + (car_getBenchmark_y - pre_cp[1])**2)
                        d2 = math.sqrt((car_getBenchmark_x - cp[0])**2 + (car_getBenchmark_y - cp[1])**2)

                        car.setXPosition(d1/(d1+d2))
                        car.setYPosition(ratioY((car_getBenchmark_x, car_getBenchmark_y)))

                        car.identifyViolate()

                        break
                    break
                # elif car_getBenchmark_x == cp[0] and line.getLineChangePossible() == False:
                #     if line.getMinX() <= car_getBenchmark_x <= line.getMaxX():
                #         if car.getId() not in violateCars.keys():
                #             violateCars[car.getId()] = ViolateCar(cnt=1)
                #         else:
                #             violateCars[car.getId()].addViolateCnt()
            pre_cp = cp    
     

def drawDottedLine(img, pt1, pt2, color, thickness=1, gap_length=50):
    # 선의 길이, 방향 벡터
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    vx = (pt2[0] - pt1[0]) / dist
    vy = (pt2[1] - pt1[1]) / dist

    # 시작점 초기화
    x_curr = pt2[0]
    y_curr = pt2[1]

    while int(dist) > 0:
        # 끝점을 계산.
        x_end = x_curr - gap_length * vx
        y_end = y_curr - gap_length * vy

        cv2.line(img, (int(x_curr), int(y_curr)), (int(x_end), int(y_end)), color, thickness)

        # 간격만큼 움직임
        x_curr -= 2 * gap_length * vx
        y_curr -= 2 * gap_length * vy

        # 남은 거리 계산
        dist -= 2 * gap_length

def equationOfPoint(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    # 기울기 계산
    if y2 != y1 and x2 != x1:
        m = (y2 - y1) / (x2 - x1)

        # y 절편 계산
        c = y1 - m * x1

        return (m, c)
    else:
        return (0, 0)
    
def ratioY(point):
    p = equationOfPoint(point, (benchmark_x, benchmark_y))    
    if p[0] < 0:  # 기울기 우상향일 때
        x_values = 0
        y_values = p[0] * x_values + p[1]

        if y_values > img_h: # y절편이 이미지크기를 벗어나면 x값을 구함
            y_values = img_h
            x_values = (y_values - p[1]) // p[0]
            
    elif p[0] > 0: # 기울기가 좌상향일 때
        x_values = img_w
        y_values = p[0] * x_values + p[1]
 
        if y_values > img_h: # y절편이 이미지크기를 벗어나면 x값을 구함
            y_values = img_h 
            x_values = (y_values - p[1]) // p[0]
    else: # 기울기가 0일때
        x_values = img_w//2
        y_values = img_h
    
    last_point = int(x_values), int(y_values)

    # cv2.line(frame, (center_x, center_y), last_point, (0, 255, 0), 2)
    r = math.sqrt(((point[0] - benchmark_x)**2 + (point[1] - benchmark_y)**2))
    r2 = math.sqrt(((last_point[0] - benchmark_x)**2 + (last_point[1] - benchmark_y)**2))

    return r/r2

# 원근법
def perspectiveTransform(img):
    x = img_white_w
    y = img_white_h

    # 왼쪽 위, 오른쪽 위, 왼쪽 아래, 오른쪽 아래
    pts1 = np.float32([[0, 0], [x, 0], [0, y], [x, y]])
    pts2 = np.float32([[x*0.4,y*0.4], [x*0.6,y*0.4], [0,y*0.8], [x,y*0.8]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    
    # 이미지에 원근법 적용
    cv2.warpPerspective(img, M, (x,y), img_white, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, (255,255,255))

def polygonOverLap(point, polygon):
    path = Path(polygon)

    return path.contains_point(point) # True/False  

def violateDB(platenum):
    sql = "SELECT * FROM %s WHERE platenum = '%s'" % (table, platenum)
    sql = "SELECT COUNT(platenum) FROM %s WHERE platenum = '%s'" % (table, platenum)
    cur.execute(sql)
    cur.rowcount
    result = cur.fetchone()[0]

    if result != 0:
        return True
    else:
        return False

def violateSetDB(platenum, cnt):
    sql = "INSERT INTO %s (platenum, date, latitude, longitude) values ('%s', current_timestamp(), NULL, NULL)" % (table, platenum)
    cur.execute(sql)
    conn.commit()  

# 1초 마다 속도 계산
def setSpeed():
    global speed
    speed = speeds.pop(0)

# 속도값 전체 읽어오기 
speed = 0
with open('speed\\주행5_speed.txt', 'r') as file:
    lines = file.readlines()
    speeds = [int(line.strip()) for line in lines]


# db연결
conn = pymysql.connect(host='localhost', user='root', password='', db='capston', charset='utf8')
cur = conn.cursor()
table = 'violation'

# ocr 모델
model_ocr = OCRModel()

# car, plate, person 인식 모델
model = YOLO('weight\\best_car_detect.pt')
model.to('cuda')

# 차량 세그멘테이션 모델
model_car = YOLO('weight\\yolov8n-seg.pt')
model_car.to('cuda')

# 차선 세그멘테이션 모델
model_seg = YOLO('weight\\best_segment_1102.pt')
model_seg.to('cuda')

#_위반13분류 모델
model_cls = YOLO('weight\\best_classify.pt')
model_cls.to('cuda')

# video 파일 열기
video_path = ""
cap = cv2.VideoCapture(video_path)
# frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# num_frames = len(speeds)

# # 원하는 프레임 수로 분할하기 위해 간격 계산
# interval = frame_count // num_frames

frame_rate = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = 570
duration_ms = int(duration * 1000)

cnt = 0
# threading.Timer(1,getSpeed).start() 

target_frame = int(frame_rate * duration)  # 재생할 목표 프레임 계산
current_frame = 0

# 이미지 사이즈
img_w, img_h= 1280, 700
img_white_w, img_white_h = 800, 800
center_x, center_y = img_w//2, img_h//2
benchmark_x, benchmark_y = img_w//2, img_h//3
img_car_w = 50
img_car_h = 60

# 일반 이미지 열기
img_car_path = "img\\car_gray.png"

img_car_gray = cv2.imread(img_car_path)
img_car_gray = cv2.resize(img_car_gray, (img_car_w, img_car_h))

#_위반13량 이미지 열기
img_car_path = "img\\car_red.png"

img_car_yellow = cv2.imread(img_car_path)
img_car_yellow = cv2.resize(img_car_yellow, (img_car_w, img_car_h))

# 내 차량 이미지 열기
img_car_path = "img\\car_white.png"

img_usr_car = cv2.imread(img_car_path)
img_usr_car = cv2.resize(img_usr_car, (img_car_w, img_car_h))

# 차로 안전 이미지 열기
icon_lane_off = cv2.imread('img\\차로안전_off.png')
icon_lane_h, icon_lane_w = 30, 45
icon_lane_off = cv2.resize(icon_lane_off, (icon_lane_w, icon_lane_h))

icon_lane_normal = cv2.imread('img\\차로안전_normal.png')
icon_lane_normal = cv2.resize(icon_lane_normal, (icon_lane_w, icon_lane_h))

icon_lane_left_orange = cv2.imread('img\\차로안전_left_orange.png')
icon_lane_left_orange = cv2.resize(icon_lane_left_orange, (icon_lane_w, icon_lane_h))

icon_lane_left_red = cv2.imread('img\\차로안전_left_red.png')
icon_lane_left_red = cv2.resize(icon_lane_left_red, (icon_lane_w, icon_lane_h))

icon_lane_right_orange = cv2.imread('img\\차로안전_right_orange.png')
icon_lane_right_orange = cv2.resize(icon_lane_right_orange, (icon_lane_w, icon_lane_h))

icon_lane_right_red = cv2.imread('img\\차로안전_right_red.png')
icon_lane_right_red = cv2.resize(icon_lane_right_red, (icon_lane_w, icon_lane_h))


# 정차 시 이미지 열기 
speed = 0
show_icon_wait_signal = 'off'
icon_wait_signal_off = cv2.imread('img\\wait_signal_off.png')
icon_wait_signal_h, icon_wait_signal_w = 30, 39
icon_wait_signal_off = cv2.resize(icon_wait_signal_off, (icon_wait_signal_w, icon_wait_signal_h))
icon_wait_signal_go = cv2.imread('img\\wait_signal_go.png') # 앞 차량 출발 이미지 열기
icon_wait_signal_go = cv2.resize(icon_wait_signal_go, (icon_wait_signal_w, icon_wait_signal_h))
icon_wait_signal_back = cv2.imread('img\\wait_signal_back.png') # 앞 차량_위반13이미지 열기
icon_wait_signal_back = cv2.resize(icon_wait_signal_back, (icon_wait_signal_w, icon_wait_signal_h))

# 차간 거리 이미지 열기
icon_gap_normal = cv2.imread('img\\gap_normal.png')
icon_gap_h, icon_gap_w = 30, 45
icon_gap_normal = cv2.resize(icon_gap_normal, (icon_gap_w, icon_gap_h))

icon_gap_danger = cv2.imread('img\\gap_danger.png')
icon_gap_danger = cv2.resize(icon_gap_danger, (icon_gap_w, icon_gap_h))

icon_gap_off = cv2.imread('img\\gap_off.png')
icon_gap_off = cv2.resize(icon_gap_off, (icon_gap_w, icon_gap_h))
# 결과 저장 경로
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out_frame = cv2.VideoWriter("output\\output_frame_demo_주행5.avi", fourcc, 30.0, (img_w, img_h))

fourcc = cv2.VideoWriter_fourcc(*'IYUV')
out_white = cv2.VideoWriter("output\\output_white_demo_주행5.avi", fourcc, 30.0, (img_white_w, img_white_h-100))

# 번호판,_위반13차량 최초 1회 초기화
plates = {}
violateCars = {}

# 차선 색깔 지정
line_color = {0:(127, 127, 127), 1:(0, 255, 255), 2:(255, 0, 0), 3:(127, 127, 127), 4:(0, 255, 255), 5: (255, 0, 0)}

# 정차 시 앞 차량 y 값
pre_y = None
cnt = 0
# video가 열렸을 때
# for i in range(0, frame_count, interval):
#     cap.set(cv2.CAP_PROP_POS_FRAMES, i)
# while cap.isOpened():

while current_frame < target_frame and current_frame < total_frames:
    # 1 frame씩 읽기
    success, frame = cap.read()

    # 차 , 보행자, 차선 매 번 초기화 
    cars = {}
    persons = {}
    lines = {0: Line(3, 0, img_w), 1:Line(3,  0, img_w)}

    # 사용자 차로 색 매 번 초기화
    left_color, right_color = (255,0,0), (255,0,0)
    
    # 사용자 차량 위치 등록
    box = img_w//2-50, img_h, 0, 0

    cars['usr_car'] = Car('usr_car', box, img_usr_car, lane = 1)
    cars['usr_car'].setXPosition(0.5)
    cars['usr_car'].setYPosition(1)

    if success:
        #fps 계산
        start_t = timeit.default_timer()

        # 이미지 크기 조정
        frame = cv2.resize(frame, (img_w, img_h))

        # 흑백 이미지 생성 (분석용)
        img = np.zeros(frame.shape, dtype=np.uint8)

        # 흰색 이미지 생성 (시각화)
        img_white = np.full((img_white_w, img_white_h, 3), 255, dtype=np.uint8)

        # 차, 번호판, 보행자 트래킹
        objectTrack(model, source=frame, persist=True, tracker='bytetrack.yaml', conf=0.5, device=0, show_conf=True)
        
        # 차선 트래킹
        instanceTrack(model_seg, source=frame, persist=True, tracker='bytetrack.yaml', conf=0.5, device=0, show_conf=True)
      
        # 모든 차량 차로 식별
        for car in cars.values():
            identifyLanes(car)

    
         # ----- 편의 기능 --------------------------------------------
        usr_lane = cars['usr_car'].getLane()
        show_icon_lane = False
        if speed > 0:
            show_icon_wait_signal = 'off'
        show_icon_gap = 'off'
        show_icon_gap = 'off'
        
        # 차간 거리 위험도 평가 and 앞 차량 출발 및_위반13알림
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        usr_lane_graph = lines[usr_lane].getGraph()
        if usr_lane_graph is not None:
            a = 550
            b = 650
            c = 700
            p1 = (((a - usr_lane_graph[1]) // usr_lane_graph[0]), a)
            p2 = (((b - usr_lane_graph[1]) // usr_lane_graph[0]), b)
            p3 = (((c - usr_lane_graph[1]) // usr_lane_graph[0]), c)
            
            usr_lane_graph = lines[usr_lane-1].getGraph()
            if usr_lane_graph is not None:
                p4 = (((a - usr_lane_graph[1]) // usr_lane_graph[0]), a)
                p5 = (((b - usr_lane_graph[1]) // usr_lane_graph[0]), b)
                p6 = (((c - usr_lane_graph[1]) // usr_lane_graph[0]), c)

                if 200 < (p3[0]-p6[0]) < 850:
                    show_icon_lane = True
                    show_icon_gap = 'normal'
                    a = 0.3
                    b = 1.0 - a
                    frame_ori = frame.copy()
                    cv2.fillPoly(frame, np.int32([(p2, p5, p4, p1)]), (0,255,0))  # 사용자 차로 색칠
                    cv2.fillPoly(frame, np.int32([(p3, p6, p5, p2)]), (127,255,0)) # 차간거리 위험 영역 색칠
                    frame = cv2.addWeighted(frame_ori, a, frame, b, 0) # 이미지 합성 (투명도)
                    
                    for key, car in cars.items():
                        if key == 'usr_car':
                            continue
                        lane = car.getLane()
                        if lane == usr_lane: # 사용자 차량과 같은 차로 일 때
                            point = car.getBenchmarks()
                            if speed <= 0 and polygonOverLap(point, (p3, p6, p4, p1)): # 정차 및 신호대기 중일 때
                                if pre_y is not None:
                                    dist = pre_y - point[1]
                                    if dist > 15:
                                        show_icon_wait_signal = 'go'   # 앞 차량 출발 알림 켜기
                                        pre_y = None
                                    elif dist < -15:
                                        # show_icon_wait_signal = 'back'   # 앞 차량_위반13알림 켜기
                                        pre_y = None
                                else:
                                    pre_y = point[1]
                            # 주행 중 차간 거리가 가까워 충돌 위험이 있을 때
                            elif polygonOverLap(point, (p3, p6, p4, p1)):
                                show_icon_gap = 'danger'


        # 차로 이탈 위험도 평가
        x_position = cars['usr_car'].getXPosition()
        if x_position < 0.2:
            left_color = (0,0,255)
        elif x_position < 0.3:
            left_color = (0,127,255)
        elif x_position > 0.8:
            right_color = (0,0,255)
        elif x_position > 0.7:
            right_color = (0,127,255)

        # 화이트 보드에 시각화
        usr_lane = cars['usr_car'].getLane()
        if usr_lane is not None:
            for index, line in lines.items():
                line.drawOnWhiteboard(index, usr_lane, left_color, right_color)
            
            for _, car in cars.items():
                car.drawOnWhiteboard(usr_lane)

        # 원근법 적용 ( 화이트 보드 )
        perspectiveTransform(img_white)
        
        
            
        # threading.Timer(1,setSpeed).start() 
        # 속도 표시
        setSpeed()

        if speed >= 100:
            cv2.putText(img_white, str(speed), (180, 280), 0, 1, (0, 0, 0), 2, cv2.LINE_AA)
        elif speed >= 10:
            cv2.putText(img_white, str(speed), (190, 280), 0, 1, (0, 0, 0), 2, cv2.LINE_AA)
        elif speed >= 0:
            cv2.putText(img_white, str(speed), (200, 280), 0, 1, (0, 0, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(img_white, '0', (200, 280), 0, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img_white, 'km/h', (198, 295), 0, 0.3,(0, 0, 0), 1, cv2.LINE_AA)

        # 차로 안전 아이콘 출력
        if show_icon_lane:       
            img_white[702:icon_lane_h+702, 378:icon_lane_w+378] = icon_lane_normal
        else:
            img_white[702:icon_lane_h+702, 378:icon_lane_w+378] = icon_lane_off

        if left_color == (0,127,255):
            img_white[702:icon_lane_h+702, 378:icon_lane_w+378] = icon_lane_left_orange
        elif left_color == (0,0,255):
            img_white[702:icon_lane_h+702, 378:icon_lane_w+378] = icon_lane_left_red  
        elif right_color == (0,127,255):
            img_white[702:icon_lane_h+702, 378:icon_lane_w+378] = icon_lane_right_orange
        elif right_color == (0,0,255):
            img_white[702:icon_lane_h+702, 378:icon_lane_w+378] = icon_lane_right_red

        
        if show_icon_wait_signal == 'off':
            img_white[700:icon_wait_signal_h+700, 427:icon_wait_signal_w+427] = icon_wait_signal_off
        elif show_icon_wait_signal == 'go': # 앞 차량 출발 아이콘 출력
            img_white[700:icon_wait_signal_h+700, 427:icon_wait_signal_w+427] = icon_wait_signal_go    
        elif show_icon_wait_signal == 'back': # 앞 차량_위반13아이콘 출력
            img_white[700:icon_wait_signal_h+700, 427:icon_wait_signal_w+427] = icon_wait_signal_back

        # 차간 거리 유지 아이콘 출력
        if show_icon_gap == 'off':
            img_white[700:icon_gap_h+700, 331:icon_gap_w+331] = icon_gap_off 
        elif show_icon_gap == 'normal':
            img_white[700:icon_gap_h+700, 331:icon_gap_w+331] = icon_gap_normal 
        elif show_icon_gap == 'danger':
            img_white[700:icon_gap_h+700, 331:icon_gap_w+331] = icon_gap_danger    

        # --------- 편의 기능 종료 ---------------------

        # for line in lines.values():
        #     if line.getGraph() is not None:
        #         w, c = line.getGraph()
                
        #         for x in range(0, 1281):
        #             y = w*x + c

        #             cv2.circle(frame, (int(x), int(y)), 3, (0,0,255), 2)

        
        img_white = img_white[100:, :]

        terminate_t = timeit.default_timer()
        FPS = int(1./(terminate_t - start_t ))
        
        # cv2.putText(frame, str(FPS), (0, 100), 0,1 ,(255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('ori', frame)
        cv2.imshow('white', img_white)

        out_frame.write(frame)
        out_white.write(img_white)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
        current_frame += 1
        if current_frame == target_frame:
            cv2.waitKey(duration_ms % frame_rate)
        cnt += 1
        print('cnt', cnt)

cap.release()
out_frame.release()
out_white.release()
cv2.destroyAllWindows()  
