# going_modular 폴더 안에 utils.py 파일을 새로 만들고 아래 코드를 파일에 저장
"""
PyTorch 모델 학습과 저장에 필요한 다양한 유틸리티 함수들을 포함
"""
import torch
from pathlib import Path

def save_model(model: torch.nn.Module,
               target_dir: str,     # 저장할 폴더 경로
               model_name: str):    # 저장할 파일 이름 (.pth 또는 .pt)
  """PyTorch 모델을 특정 디렉토리에 저장

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. 
    (.pth 또는 .pt 확장자를 포함해야 함)

  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)

  # Create model save path 
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"        # 파일 이름이 .pth 또는 .pt로 끝나는지 검사. 아니면 에러 발생
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")   #저장 위치 출력
  torch.save(obj=model.state_dict(),        # 모델의 가중치(파라미터)만 저장
             f=model_save_path)         # 해당 데이터를 파일로 저장
