"""
이미지 분류 데이터를 위한 PyTorch DataLoader를 생성하는 기능을 포함하고 있음
"""
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()        # CPU 코어 개수 

def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS
):
  """훈련용과 테스트용 DataLoader를 생성

  훈련/테스트 폴더 경로를 받아서 Dataset → DataLoader로 변환함
  
  Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    transform: 이미지 전처리
    batch_size: 한 번에 몇 장씩 학습할지
    num_workers: 데이터를 병렬로 불러올 프로세스 수
    
  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             transform=some_transform,
                             batch_size=32,
                             num_workers=4)
  """
  # Use ImageFolder to create dataset(s)- 폴더 안 이미지를 PyTorch가 이해할 수 있는 Dataset 형태로 변환
  train_data = datasets.ImageFolder(train_dir, transform=transform)     # ImageFolder : 폴더에 라벨 붙여줌
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  # Get class names
  class_names = train_data.classes

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False, # don't need to shuffle test data
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names
