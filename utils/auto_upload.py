"""
자동 모델 업로드 유틸리티
학습 완료 후 훈련된 모델을 GitHub 저장소에 자동 업로드
"""

import os
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
import yaml
import torch


class ModelUploader:
    def __init__(self, config):
        """
        ModelUploader 초기화
        
        Args:
            config: YAML 설정 딕셔너리 (auto_upload 섹션 포함)
        """
        self.config = config
        self.auto_upload_config = config.get('auto_upload', {})
        self.enabled = self.auto_upload_config.get('enabled', False)
        self.repository = self.auto_upload_config.get('repository', '')
        self.include_config = self.auto_upload_config.get('include_config', True)
        self.include_training_log = self.auto_upload_config.get('include_training_log', True)
        
        if not self.enabled:
            print("Warning: Auto-upload is disabled in config")
            return
            
        if not self.repository:
            print("Error: Repository URL not specified in config")
            self.enabled = False
            return
    
    def upload_models(self, model_name=None):
        """
        훈련된 모델들을 GitHub에 업로드
        
        Args:
            model_name: 저장된 모델 이름 (옵션)
        """
        if not self.enabled:
            print("Model upload skipped (disabled)")
            return False
            
        try:
            print("Starting automatic model upload...")
            
            # 타임스탬프 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 모델명 + 타임스탬프로 폴더명 생성
            clean_model_name = (model_name or "model").replace("_final", "")
            model_dir = f"{clean_model_name}_{timestamp}"
            
            # 1. 모델 저장소 클론 또는 업데이트
            self._prepare_repository()
            
            # 2. 새 모델 디렉토리 생성
            upload_path = Path("vastai_trained_model") / model_dir
            upload_path.mkdir(parents=True, exist_ok=True)
            print(f"Created model directory: {model_dir}")
            
            # 3. 모델 파일 복사
            self._copy_models(upload_path)
            
            # 4. 메타데이터 생성
            self._create_metadata(upload_path, model_name, timestamp)
            
            # 5. 설정 파일 복사 (옵션)
            if self.include_config:
                self._copy_config_files(upload_path)
            
            # 6. Git에 추가 및 커밋
            self._commit_and_push(model_dir, timestamp)
            
            print(f"Model upload completed!")
            print(f"Upload location: {self.repository}/tree/main/{model_dir}")
            return True
            
        except Exception as e:
            print(f"Model upload failed: {str(e)}")
            return False
    
    def _prepare_repository(self):
        """모델 저장소 클론 또는 업데이트"""
        print("Preparing model repository...")
        
        if Path("vastai_trained_model").exists():
            # 기존 저장소 업데이트
            os.chdir("vastai_trained_model")
            subprocess.run(["git", "pull", "origin", "main"], 
                         capture_output=True, text=True)
            os.chdir("..")
        else:
            # 새로 클론
            result = subprocess.run(["git", "clone", self.repository], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"Failed to clone repository: {result.stderr}")
    
    def _copy_models(self, upload_path):
        """saved_model 폴더의 모든 모델 파일 복사"""
        print("Copying model files...")
        
        saved_model_path = Path("saved_model")
        if not saved_model_path.exists():
            print("Warning: No saved_model directory found")
            return
        
        model_files_copied = 0
        for model_file in saved_model_path.glob("*.pt"):
            dest_file = upload_path / model_file.name
            shutil.copy2(model_file, dest_file)
            model_files_copied += 1
            print(f"   Copied: {model_file.name}")
        
        print(f"Total model files copied: {model_files_copied}")
    
    def _create_metadata(self, upload_path, model_name, timestamp):
        """훈련 메타데이터 파일 생성"""
        print("Creating metadata...")
        
        # GPU 정보 가져오기
        gpu_info = "Unknown"
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=name", 
                                   "--format=csv,noheader,nounits"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gpu_info = result.stdout.strip().split('\n')[0]
        except:
            pass
        
        # Git 커밋 해시 가져오기
        commit_hash = "Unknown"
        try:
            result = subprocess.run(["git", "rev-parse", "HEAD"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                commit_hash = result.stdout.strip()[:8]
        except:
            pass
        
        # 메타데이터 생성
        metadata = f"""Training Information
====================
Training completed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Host: {os.environ.get('HOSTNAME', 'unknown')}
GPU: {gpu_info}
Repository: https://github.com/Joowonoil/channel-estimation
Commit: {commit_hash}
Model name: {model_name or 'Unknown'}
Upload timestamp: {timestamp}

Configuration:
- Learning rate: {self.config.get('training', {}).get('lr', 'Unknown')}
- Max iterations: {self.config.get('training', {}).get('num_iter', 'Unknown')}
- Batch size: {self.config.get('dataset', {}).get('batch_size', 'Unknown')}
- Model architecture: v4 (LoRA compatible)
- WandB project: {self.config.get('training', {}).get('wandb_proj', 'Unknown')}
"""
        
        metadata_file = upload_path / "training_info.txt"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            f.write(metadata)
    
    def _copy_config_files(self, upload_path):
        """설정 파일들 복사"""
        print("Copying configuration files...")
        
        config_path = Path("config")
        if config_path.exists():
            config_dest = upload_path / "config"
            config_dest.mkdir(exist_ok=True)
            
            for config_file in config_path.glob("*.yaml"):
                dest_file = config_dest / config_file.name
                shutil.copy2(config_file, dest_file)
                print(f"   Copied config: {config_file.name}")
    
    def _commit_and_push(self, model_dir, timestamp):
        """Git에 커밋하고 푸시"""
        print("Uploading to GitHub...")
        
        os.chdir("vastai_trained_model")
        
        try:
            # Git 설정 확인 및 설정
            subprocess.run(["git", "config", "user.email", "vastai@auto-upload.com"], 
                          capture_output=True)
            subprocess.run(["git", "config", "user.name", "Vast AI Auto Upload"], 
                          capture_output=True)
            
            # Git 추가 및 커밋
            subprocess.run(["git", "add", "."], capture_output=True)
            
            commit_message = f"""Add trained models from {timestamp}

- Training completed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- Model directory: {model_dir}
- Auto-uploaded from Vast AI training

Automated upload from channel-estimation training
"""
            
            result = subprocess.run(["git", "commit", "-m", commit_message], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                # 푸시
                push_result = subprocess.run(["git", "push", "origin", "main"], 
                                           capture_output=True, text=True)
                if push_result.returncode != 0:
                    raise Exception(f"Failed to push: {push_result.stderr}")
            else:
                print("Info: No changes to commit")
                
        finally:
            os.chdir("..")


def auto_upload_models(config, model_name=None):
    """
    편의 함수: 모델 자동 업로드
    
    Args:
        config: YAML 설정 딕셔너리
        model_name: 모델 이름 (옵션)
    
    Returns:
        bool: 업로드 성공 여부
    """
    uploader = ModelUploader(config)
    return uploader.upload_models(model_name)