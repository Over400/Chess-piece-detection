# opencv_chess

체스 기물 YOLOv8 detection + 듀얼 웹캠 로봇 체스 파이프라인

## Setup
pip install ultralytics torch pyyaml

## Usage
python model_alpha_v2.py prepare --source huggingface
python model_alpha_v2.py train --epochs 50
python model_alpha_v2.py predict --source board.jpg