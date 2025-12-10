# Ad Risk Guardian
고려대학교 지능정보 SW 아카데미 7기 1조

---

## 프로젝트 소개 
Ad Risk Guardian은 온라인 플랫폼 환경에서 딥페이크 영상과 과장광고 문장을 실시간으로 탐지하여 사용자를 보호하는 크롬 확장 프로그램입니다. 광고 영상에서 프레임을 추출하고 음성을 텍스트로 변환한 뒤, 각각 독립적으로 분석하여 위험도를 평가합니다. 영상 탐지에는 MobileNetV3와 LSTM을 결합한 딥페이크 탐지 모델을, 텍스트 판별에는 KcELECTRA 모델을 사용했습니다. 두 분석 결과를 종합하여 과장광고를 식별하고 사용자에게 경고 알림을 제공합니다.

## 데이터셋
과장광고 판별 모델에 사용된 데이터셋은 건강기능식품 관련 법령 및 식품의약품안전처 가이드라인을 기준으로 YouTube 아카이브와 메타 광고 라이브러리에서 yt-dlp 및 FFmpeg 라이브러리를 활용해 수집했습니다. Clova Speech API를 사용해 과대광고-정상광고 텍스트를 확보한 후, GPT API에 페르소나를 적용하여 데이터를 증강함으로써 총 2399개 학습 및 검증 데이터(학습:검증 = 8:2)를 구축하였습니다.

딥페이크 탐지 모델에 사용된 데이터셋은 AI HUB 딥페이크 변조 영상(KoDF) 데이터셋과 Kaggle FaceForensics++ 데이터셋을 7:3 비율로 혼합하여 사용하였습니다. 성별 균형(남:여 = 1:1)을 맞추고 이미지 크기를 256x256으로 통일하여 총 1500개의 학습 및 검증 데이터(학습:검증 = 8:2)를 구성하였습니다.

- 🎬 Video dataset Link: https://drive.google.com/drive/folders/1rnOvxRjQ3p4_vX_QYE1YVgDgwjtzFIyz?usp=drive_link 
    - original_data
    - processed_data


## 주요 기능
- 딥페이크 탐지: 영상 탐지 모델(MobileNetV3 + LSTM)을 활용하여 광고 영상 내 조작 여부를 탐지합니다.
- 과장광고 판별: 음성-텍스트 변환(STT) 및 KcELECTRA 모델을 통해 과장 문장을 문맥 기반으로 분류합니다.
- 통합 신뢰도 평가: 영상과 텍스트 분석 결과를 late fusion으로 종합하여 광고에 대해 예측합니다.
- 실시간 경고 제공: 탐지된 위험 광고에 대해 사용자에게 주의 알림을 제공합니다.

---

## 기술 스택 
### Frontend
  
<img src="https://img.shields.io/badge/html5-E34F26?style=for-the-badge&logo=html5&logoColor=white"> <img src="https://img.shields.io/badge/css-1572B6?style=for-the-badge&logo=css3&logoColor=white">  <img src="https://img.shields.io/badge/javascript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black">

### Browser Extension

<img src="https://img.shields.io/badge/Chrome Extension(Manifest V3)-1572B6?style=for-the-badge"> 

### Backend
<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white"> <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=FastAPI&logoColor=white"> <img src="https://img.shields.io/badge/Celery-37814A?style=for-the-badge&logo=Celery&logoColor=white">  <img src="https://img.shields.io/badge/Redis-FF4438?style=for-the-badge&logo=Redis&logoColor=white">

### AI / Machine Learning

<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white">  <img src="https://img.shields.io/badge/FaceNet-1572B6?style=for-the-badge"> <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=HuggingFace&logoColor=white"> <img src="https://img.shields.io/badge/Transformers-1572B6?style=for-the-badge"> <img src="https://img.shields.io/badge/Whisper-1572B6?style=for-the-badge"> 

### Computer Vision / Image Processing
<img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=OpenCV&logoColor=white"> <img src="https://img.shields.io/badge/Pillow-1572B6?style=for-the-badge"> 

### Media Processing
<img src="https://img.shields.io/badge/FFmpeg-007808?style=for-the-badge&logo=FFmpeg&logoColor=white"> <img src="https://img.shields.io/badge/yt--dlp-3399FF?style=for-the-badge" alt="yt-dlp">

---

## 프로젝트 아키텍쳐
<img width="3934" height="2073" alt="아키텍쳐" src="https://github.com/user-attachments/assets/5f0f2e7d-65d9-41ef-98b0-d780476881b9" />

---

## 프로젝트 구조
```
Ad_Risk_Guardian/
├── 📁 pipeline
│   └── 📁 systemization # 딥페이크 및 과장광고 탐지 통합 시스템
│   ├── 📁 video
│   │   ├── 📁 __pycache__
│   │   ├── 📁 env
│   │   ├── 📁 experiments # 시행착오
│   │   ├── collect.py # 데이터수집
│   │   ├── eda.py # EDA
│   │   ├── preprocessing.py # 전처리
│   │   ├── modeling.py # 모델링
│   │   └── requirements.txt
│   └── 📁 text
│   │   ├── 📁 data # 데이터셋
│   │   ├──  📁 experiments # 시행착오
│   │   ├──  collect.ipynb # 데이터수집
│   │   ├──  eda.ipynb # EDA
│   │   ├──  preprocessing.ipynb # 전처리
│   │   ├──  modeling.ipynb # 모델링, 검증, 시각화
│   │   ├──  requirements.txt
├─ 📁 backend # Fast API
│   ├── 📁__pycache__
│   ├── 📁 app # main.py, tasks.py, models.py, utils.py
│   ├── 📁 temp # frames, audio 저장
│   ├── 📁 uploads # 유튜브 영상 다운로드
│   ├── 📁 weights # 모델 가중치
│   ├── celery_app.py # celery 정의
│   ├── requirements.txt
│   ├── run_all.py # 백엔드 실행
├─ 📁 extensions # Chrome Extension
│   ├── 📁 css # popup / overlay UI 스타일 파일
│   ├── 📁 icons # 확장 프로그램 아이콘
│   ├── 📁 js # background, content, popup 스크립트
│   ├── manifest.json # Chrome Extension 설정 파일
│   ├── popup.html # 팝업 UI
├─ .DS_store 
├─ .gitattributes 
├─ README.md
```

### 참고 사항
#### text 
*  pipeline/text/preprocessing.ipynb (Preview ERROR) ★ 로컬 환경에서는 정상적으로 실행됨
*  pipeline/text/experiments/learning_rate(3e_5),weight_decay(0_1),드롭아웃.ipynb ★ 로컬 환경에서는 정상적으로 실행됨
*  pipeline/text/experiments/learning_rate(5e_4).ipynb ★ 로컬 환경에서는 정상적으로 실행됨
*  pipeline/text/experiments/learning_rate(5e_4)_드롭아웃.ipynb★ 로컬 환경에서는 정상적으로 실행됨
*  가중치 다운로드 링크   https://drive.google.com/drive/folders/1h_Lur4wJhcikdbL8YuFGBozlp4koktQG?usp=sharing

#### 구축 환경
- video - spyder
- text - Google Colab
- extensions/backend - VScode

---

## 실행 방법

### 1. Docker Desktop 설치
- https://www.docker.com/ 접속
- Docker Desktop Installer 실행 후 설치
- 설치 완료 후 PC 재부팅
- Docker Desktop 실행

### 2. VSCode 환경 설정
1. Python 3.11 설치  
2. 가상환경 생성 및 활성화

```bash
cd backend
python -m venv venv
.\venv\Scripts\activate     # Windows
```

3. 패키지 설치
```
pip install --upgrade pip
pip install -r requirements.txt
```

4. 백엔드 실행
```
python run_all.py
```

### 3. 크롬 익스텐션
- chrome://extensions/ 접속
- 개발자 모드 클릭
- 압축 해제된 확장 프로그램 로드 클릭
- extensions 폴더 클릭
- 프로그램 실행

### 4. 광고 영상 시청
- sample 광고 영상 링크
    - https://www.youtube.com/watch?v=FvDr924FgPQ
    - https://www.youtube.com/watch?v=voEeBgSMb1Q

---

## 👥 Team Members

<table>
  <tr>
    <td align="center">
      <img src="https://avatars.githubusercontent.com/Howard7898" width="100"/><br/>
      <b>등덕호</b>
    </td>
    <td align="center">
      <img src="https://avatars.githubusercontent.com/planethhp" width="100"/><br/>
      <b>박현호</b>
    </td>
    <td align="center">
      <img src="https://avatars.githubusercontent.com/suubinahn" width="100"/><br/>
      <b>안수빈</b>
    </td>
    <td align="center">
      <img src="https://avatars.githubusercontent.com/daeu3414-boop" width="100"/><br/>
      <b>이다은</b>
    </td>
    <td align="center">
      <img src="https://avatars.githubusercontent.com/ImChaeRim" width="100"/><br/>
      <b>임채림</b>
    </td>
    <td align="center">
      <img src="https://avatars.githubusercontent.com/hje0103" width="100"/><br/>
      <b>한주은</b>
    </td>
  </tr>
</table>


---

## 참고문헌
1.
