"""
Firebase工具模組：將GameDataCollector與Firebase數據庫整合
提供從遊戲數據到Firebase的無縫轉換功能
"""

from schemas import (
    GameDataCollector, GameType, SessionData, GamePerformanceData
)
from firebase_data_manager import FirebaseDataManager
from typing import Dict, Any, Optional, Union
from datetime import datetime
import os


class FirebaseGameDataCollector(GameDataCollector):
    """整合Firebase的遊戲數據收集器"""
    
    def __init__(self, firebase_credentials: Optional[str] = None):
        """初始化收集器並連接Firebase"""
        super().__init__()
        self.firebase = FirebaseDataManager(firebase_credentials)
        print("Firebase數據收集器已初始化")
    
    def save_session_data(self, session_id: str, filepath: Optional[str] = None) -> None:
        """將會話數據同時保存到Firebase和本地文件（如需要）"""
        if session_id not in self.session_data:
            raise ValueError(f"會話ID不存在: {session_id}")
            
        session = self.session_data[session_id]
        
        # 保存到Firebase
        self.firebase.save_session(session)
        
        # 如果指定了本地文件路徑，也保存到文件
        if filepath:
            super().save_session_data(session_id, filepath)
    
    def record_game_performance(self, session_id: str, round_id: str, 
                              performance_data: Dict[str, Any]) -> GamePerformanceData:
        """記錄遊戲表現數據並同步到Firebase"""
        # 使用父類方法生成標準化的表現數據
        game_data = super().record_game_performance(session_id, round_id, performance_data)
        
        # 保存到Firebase
        self.firebase.save_game_performance(game_data)
        
        return game_data
    
    def end_session(self, session_id: str) -> SessionData:
        """結束會話並同步到Firebase"""
        # 使用父類方法結束會話
        session = super().end_session(session_id)
        
        # 保存到Firebase
        self.firebase.save_session(session)
        
        return session
    
    def sync_local_data_to_firebase(self, data_dir: str) -> int:
        """將本地數據文件同步到Firebase"""
        import json
        import glob
        
        # 找到所有JSON文件
        file_pattern = os.path.join(data_dir, "*.json")
        json_files = glob.glob(file_pattern)
        synced_count = 0
        
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    session_dict = json.load(f)
                
                # 檢查是否是會話數據
                if all(key in session_dict for key in ['user_id', 'session_id', 'game_type']):
                    # 重建會話數據
                    session = SessionData(
                        user_id=session_dict['user_id'],
                        session_id=session_dict['session_id'],
                        game_type=GameType(session_dict['game_type']),
                        start_time=datetime.fromisoformat(session_dict['start_time']),
                        end_time=datetime.fromisoformat(session_dict['end_time']) if session_dict.get('end_time') else None,
                        user_skill=session_dict.get('user_skill', 0.5),
                        initial_difficulty=session_dict.get('initial_difficulty', 2),
                        difficulty_history=session_dict.get('difficulty_history', []),
                        performance_history=session_dict.get('performance_history', []),
                        satisfaction_history=session_dict.get('satisfaction_history', []),
                        cognitive_assessments=session_dict.get('cognitive_assessments', [])
                    )
                    
                    # 保存到Firebase
                    self.firebase.save_session(session)
                    synced_count += 1
                    
            except Exception as e:
                print(f"同步文件 {file_path} 時出錯: {e}")
        
        return synced_count


def sync_all_data():
    """從本地同步所有數據到Firebase（實用工具）"""
    collector = FirebaseGameDataCollector()
    data_dir = "data"
    
    # 確保目錄存在
    if not os.path.exists(data_dir):
        print(f"數據目錄 {data_dir} 不存在")
        return
    
    # 同步數據
    count = collector.sync_local_data_to_firebase(data_dir)
    print(f"成功同步 {count} 個會話至Firebase")


def migrate_to_firebase():
    """將本地數據遷移到Firebase（一次性使用）"""
    # 此函數旨在遷移已有的遊戲數據
    
    # 初始化Firebase管理器
    firebase = FirebaseDataManager()
    
    # 創建收集器並同步數據
    collector = FirebaseGameDataCollector()
    
    # 同步會話數據
    count = collector.sync_local_data_to_firebase("data")
    print(f"成功遷移 {count} 個會話至Firebase")
    
    # 可添加更多遷移邏輯，如用戶數據等


if __name__ == "__main__":
    # 運行同步工具
    sync_all_data()