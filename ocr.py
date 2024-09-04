import torch
import cv2
import sys
import torch.nn.functional as F

sys.path.append('deepOCR')
from torchvision import transforms
from PIL import Image
from model import Model
from utils import CTCLabelConverter




class Options:
    def __init__(self):
        self.Transformation = 'TPS'
        self.FeatureExtraction = 'ResNet'
        self.SequenceModeling = 'BiLSTM'
        self.Prediction = 'CTC'
        self.num_fiducial = 20
        self.imgH = 32
        self.imgW = 100
        self.input_channel = 1
        self.output_channel = 512
        self.hidden_size = 256
        self.num_class = 67
        self.num_fiducial = 20

class OCRModel:
    def __init__(self) -> None:
        # 모델 및 가중치 경로
        model_path = "weight\\best_OCR.pth"
        # 추론 수행
        self.batch_max_length = 25
        self.character = "0123456789가강거경고구기광나남너노누다더도두대라러로루마머모무바배버보부북서사산소수아어영오우울인원자저전조주제천충하허호"
       
        # 이미지 전처리 및 모델 초기화
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = Model(Options())
        self.model = torch.nn.DataParallel(self.model).to(self.device)

        # 가중치 로드
        self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
        self.model.eval()
        self.converter = CTCLabelConverter(self.character)

        self.text = torch.LongTensor(1, self.batch_max_length + 1).fill_(0).to(self.device)


    def ocr(self, img, bbox):
        img_h, img_w = img.shape[:2]
        
        x, y, w, h = bbox
        x1 = max(int(x-w/2)-10, 0)
        x2 = min(int(x+w/2)+10, img_w)
        y1 = max(int(y-h/2)-10, 0)
        y2 = min(int(y+h/2)+10, img_h)

        bbox_region = img[y1:y2, x1:x2]
        bbox_region = cv2.cvtColor(bbox_region, cv2.COLOR_BGR2GRAY) 
        bbox_region = cv2.resize(bbox_region, dsize=(300,100))
        # cv2.imshow('region', bbox_region)
        # 이미지를 불러와 전처리
        
        transform = transforms.Compose([
            transforms.ToTensor(),  
        ])

        image = Image.fromarray(bbox_region)
        image = transform(image)
        image = image.unsqueeze(0)  # 배치 차원 추가

        with torch.no_grad():
            pred = self.model(image, self.text)

            pred_size = torch.IntTensor([pred.size(1)])
            _, pred_index = pred.max(2)

            # 글자 집합을 정의
            pred_str = self.converter.decode(pred_index, pred_size)

            pred_prob = F.softmax(pred, dim=2)
            pred_max_prob, _ = pred_prob.max(dim=2)
            confidence_score = torch.prod(pred_max_prob[0])

            return pred_str[0], confidence_score.item()
