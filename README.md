<div align="center">
<h2>[2023] Smart Black Box </h2>
</div>

## 목차
  - [개요](#개요) 
  - [프로젝트 구현 결과](#프로젝트-구현-결과)
  - [설계 내용](#설계-내용)
<br><br>

## 개요
- 프로젝트 이름: Smart Black Box 
- 프로젝트 기간: 2023.09 ~ 2023.11
- 개발 엔진 및 언어:
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![MySQL](https://img.shields.io/badge/MySQL-4479A1?style=for-the-badge&logo=mysql&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-00FFFF?style=for-the-badge&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)

<br><br>

## 프로젝트 구현 결과
|![input](https://github.com/jh990714/Smart-Blackbox/assets/144774186/b2a409c2-cfe5-4884-920e-b508239a5bd9)|![output](https://github.com/jh990714/Smart-Blackbox/assets/144774186/600b3892-6354-4084-80ae-4a71149bbe11)|
|:---:|:---:|
|Input|Output|

https://github.com/jh990714/Smart-Black-Box/assets/144774186/6f53c517-bf79-4939-b7e9-9d6efb856f70

https://github.com/jh990714/Smart-Black-Box/assets/144774186/ccea0649-4325-4953-a5f2-859762e0acfe

https://github.com/jh990714/Smart-Black-Box/assets/144774186/ed8e2b90-e627-417b-afc8-c0fd24321167



<br>

### ° 아이콘 설명
|![내 차량](https://github.com/jh990714/Smart-Blackbox/assets/144774186/04c1bd3f-0e95-4d19-8ac8-4b49df91389b)|![다른차량](https://github.com/jh990714/Smart-Blackbox/assets/144774186/061fcedc-b17a-48b2-84a1-13cafcf0bd95)|![위반 차량](https://github.com/jh990714/Smart-Blackbox/assets/144774186/a5bdb7aa-3978-4f62-b6b3-25264bf96982)|
|:---:|:---:|:---:|
|내 차량|다른 차량|위반 차량|
|![차로 이탈 방지(정상)](https://github.com/jh990714/Smart-Blackbox/assets/144774186/fccf3b82-6d0a-417d-9cc8-5157ef88731a)|![차로 이탈 방지(위험)](https://github.com/jh990714/Smart-Blackbox/assets/144774186/8b984cbd-6921-43c4-b80e-cd47ca631713)|![차로 이탈 방지(경고)](https://github.com/jh990714/Smart-Blackbox/assets/144774186/a28ce339-6f3d-47df-8ff1-0de1b46a58d2)|
|차로 이탈 방지(정상)|차로 이탈 방지(위험)|차로 이탈 방지(경고)|
|![앞 차량 출발 신호](https://github.com/jh990714/Smart-Blackbox/assets/144774186/257beb55-24c0-436a-ac33-65c270b38d2d)|![앞 차량 후진 주의](https://github.com/jh990714/Smart-Blackbox/assets/144774186/2f80e5e6-a23c-4d01-8a92-9b7a30fd3a6c)|![차간 간격 알림(위험)](https://github.com/jh990714/Smart-Blackbox/assets/144774186/a4c20433-475d-4858-a89b-de5f5023a0ad)|
|앞 차량 출발 신호|앞 차량 후진 주의|차간 간격 알림(위험)|

<br><br>

## 설계 내용
### ° Object Detection, Instance Segmentation, OCR

|![image](https://github.com/jh990714/Smart-Black-Box/assets/144774186/a8f0eb15-41a1-43ed-b605-d081191c3264)|![image](https://github.com/jh990714/Smart-Black-Box/assets/144774186/36b445e3-d1c3-41da-a7f4-b39081578c88)|![image](https://github.com/jh990714/Smart-Black-Box/assets/144774186/925845f5-579a-4a6d-a48f-4095bf226a30)|
|:---:|:---:|:---:|
|![image](https://github.com/jh990714/Smart-Black-Box/assets/144774186/f779375c-29e2-4f3c-8aa4-dcddf75d56f7)|![image](https://github.com/jh990714/Smart-Black-Box/assets/144774186/92c8f39c-0989-4245-b626-828c53bc424f)|![image](https://github.com/jh990714/Smart-Black-Box/assets/144774186/8cc1eb6b-9a99-4946-8cdb-77eff8ad034b)|
|차, 번호판, 보행자<br>Object Detection|차선, 정지선, 횡단보도<br>Instance Segmentation|번호판 OCR|

- Object Dectection과 Instance Segmentation은 YOLOv8를 사용하여 학습
- 차, 번호판, 보행자를 학습하여 Object Dectection 모델 생성
- 차선, 정지선, 횡단보도를 학습하여 Instance Segmentation 모델 생성
- CLOVA AI에서 제공하는 deep-text-recognition-benchmark 오픈소스 프로젝트를 이용해 OCR모델 학습 후 생성
<br>

### ° 난폭 운전 차량 Classification
마스킹 이미지

|![일반 차량](https://github.com/jh990714/Smart-Black-Box/assets/144774186/7ed42c35-ec67-4714-93bd-a165cf3e0b30)|![위험 차량](https://github.com/jh990714/Smart-Black-Box/assets/144774186/7a44c91c-97aa-4cb1-bdac-9558d430bf68)|![위반 차량](https://github.com/jh990714/Smart-Black-Box/assets/144774186/9d6657f3-1906-4f2a-999b-ed13bd00e18a)
|:---:|:---:|:---:|
|일반 차량|위험 차량|위반 차량|

판별과정

|![판별_원본](https://github.com/jh990714/Smart-Black-Box/assets/144774186/d6b985a1-9b4f-4fd2-8b71-86db60600127)|![판별_마스킹](https://github.com/jh990714/Smart-Black-Box/assets/144774186/57a0f836-89d1-4d16-a011-d4055302b56c)|![판별_결과](https://github.com/jh990714/Smart-Black-Box/assets/144774186/eee6e136-c5a5-4cbf-89c5-7f1c7a696472)|
|:---:|:---:|:---:|
|원본 이미지|마스킹 이미지|판별 결과|

- 차량과 차선 마스킹 이미지로 분류모델 생성 ( 일반 차량, 위험 차량, 위반 차량 )
- 마스킹 이미지 사용하여 불필요한 배경 정보를 제거 ( 다른 객체들에 대한 영향 최소화 )
- 분류모델을 통해 난폭운전 차량 판별
<br>

### ° 차로 판단 Algorithm

원의 방정식과 직선의 방정식을 연립하여 원과 차선의 접점을 계산
|![차로판단](https://github.com/jh990714/Smart-Black-Box/assets/144774186/f2e8969d-be49-41a5-ad2c-8cf3a84405f1)|
|---|

- 차선의 순서를 판단 ( 첫 번째 차선, 두 번째 차선, ... )
- 사용자 차량의 차로( 파란선 )뿐만 아니라 다른 차량의 차로도 판단
- 차로가 판단됐다면 해당 차로에 어느 위치에 존재하는지 판단
