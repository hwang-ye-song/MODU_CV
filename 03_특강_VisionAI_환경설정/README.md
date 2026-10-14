# 🛠️ Vision AI 특강 정리: 환경 설정과 CLI 연결

> **내 PC는 명령만 내리고, 무거운 연산은 Colab GPU가 한다.**
> 이 특강의 핵심은 모델이 아니라, 그 연결을 가능하게 하는 **환경 설정**과 **CLI 연결**입니다.

실습 원본: [yblee110/vision-ai-notebook](https://github.com/yblee110/vision-ai-notebook)

![환경 3층 구조](images/env_layers.png)

## 📁 들어 있는 것

| 파일 | 내용 |
|---|---|
| [`Vision_AI_특강_정리.ipynb`](Vision_AI_특강_정리.ipynb) | 정리 노트북 |
| [`Vision_AI_특강_정리.pdf`](Vision_AI_특강_정리.pdf) | 같은 내용의 PDF (A4) |
| [`특강_정리본.md`](특강_정리본.md) | 같은 내용의 마크다운 정리본 |
| `images/` | 환경 구조도 |
| [`tools/content.md`](tools/content.md) | 원고. `python tools/build_all.py`로 노트북·md·PDF를 다시 만든다 |

## 🗂 목차

0. 내 소감: 막혔던 점, 깨달은 점, 앞으로 쓸 방법
1. 한눈에 보기: 환경 3층 구조 + ⚡ 핵심 명령 요약
2. 왜 환경을 이렇게 나누나 (내 PC, Codespaces, Colab GPU)
3. 1층: Codespaces 자동 세팅 (`devcontainer.json`, `Dockerfile`, `setup.sh`, `doctor.py`)
4. 실습에서 한 것 (수업 순서대로)
5. 2층: 인증 (Hugging Face, Google OAuth2)
6. 3층: Colab 런타임 연결 (`new`, `upload`, `exec`, `restart-kernel`, `download`, `stop`)
7. 연결 자동화: 셸 스크립트와 에이전트 스킬
8. GPU 고르기와 비용
9. 환경 트러블슈팅
10. 복습 체크리스트

## 🧪 실습에서 한 것

1. 수업 저장소 Fork → 내 Fork에서 Codespace 생성
2. 00번 노트북: `hf download`로 모델 받기, 지문 확인, 데이터 나누기
3. `colab --auth oauth2 sessions`로 Google 로그인
4. `colab new -s 이름 --gpu T4`로 GPU 세션 생성, PyTorch 설치
5. 01번 GPU 추론: 원본 모델은 컵을 face powder로 예측
6. 02번 파인튜닝: 5개 클래스로 다시 학습
7. 에이전트 스킬 소개, 세션 종료

## ⚡ 핵심 명령 요약

```bash
# 1층: 환경 점검 (Codespaces 터미널)
source .venv/bin/activate
python scripts/doctor.py

# 2층: Google 로그인
colab --auth oauth2 sessions

# 3층: Colab GPU 연결
colab new -s hf-vision-gpu --gpu T4
colab upload -s hf-vision-gpu hf_colab_gpu/requirements-torch.txt content/requirements-torch.txt
colab exec -s hf-vision-gpu --timeout 1800 <<'PY'
import subprocess, sys
subprocess.run([sys.executable,'-m','pip','install','-r','/content/requirements-torch.txt'], check=True)
PY
colab restart-kernel -s hf-vision-gpu
colab exec -s hf-vision-gpu -f hf_colab_gpu/notebooks/01_gpu_inference.ipynb --timeout 1800
colab stop -s hf-vision-gpu
colab sessions
```

## 📚 출처
- 실습 코드와 환경 설정: [yblee110/vision-ai-notebook](https://github.com/yblee110/vision-ai-notebook)
- Colab CLI: [googlecolab/google-colab-cli](https://github.com/googlecolab/google-colab-cli)
- 모델: [facebook/deit-tiny-patch16-224](https://huggingface.co/facebook/deit-tiny-patch16-224)
- 데이터: [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
