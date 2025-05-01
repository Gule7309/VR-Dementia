import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import numpy as np
from schemas.data_models import (
    GameType, UserProfile, GamePerformanceData, GamePerformanceMetrics,
    InteractionData, DifficultyAdjustmentData, SessionData, 
    MultimodalPerceptionData, CognitiveAssessmentData
)


def convert_timestamp(timestamp_str: str) -> datetime:
    """將ISO格式時間字符串轉換為datetime對象"""
    return datetime.fromisoformat(timestamp_str)


def normalize_value(value: float, min_val: float, max_val: float) -> float:
    """將數值標準化到0-1範圍"""
    if max_val == min_val:
        return 0.5
    return (value - min_val) / (max_val - min_val)


class GameDataCollector:
    """遊戲數據收集器：從遊戲系統收集原始數據並轉換為標準格式"""
    
    def __init__(self):
        self.session_data = {}
        
    def start_session(self, user_id: str, game_type: str) -> str:
        """開始一個新的遊戲會話"""
        session_id = f"{user_id}_{game_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.session_data[session_id] = SessionData(
            user_id=user_id,
            session_id=session_id,
            game_type=GameType(game_type),
            start_time=datetime.now()
        )
        return session_id
        
    def end_session(self, session_id: str) -> SessionData:
        """結束遊戲會話並返回會話數據"""
        if session_id in self.session_data:
            self.session_data[session_id].end_time = datetime.now()
            return self.session_data[session_id]
        raise ValueError(f"會話ID不存在: {session_id}")
        
    def record_game_performance(self, session_id: str, round_id: str, 
                              performance_data: Dict[str, Any]) -> GamePerformanceData:
        """記錄單輪遊戲表現數據"""
        if session_id not in self.session_data:
            raise ValueError(f"會話ID不存在: {session_id}")
            
        session = self.session_data[session_id]
        
        # 創建標準化的表現指標對象
        metrics = GamePerformanceMetrics(
            accuracy=performance_data.get('accuracy', 0.0),
            completion_time=performance_data.get('completion_time', 0.0),
            correct_responses=performance_data.get('correct_responses', 0),
            incorrect_responses=performance_data.get('incorrect_responses', 0),
            missed_responses=performance_data.get('missed_responses', 0),
            average_response_time=performance_data.get('average_response_time', 0.0),
            frustration_indicators=performance_data.get('frustration_indicators', 0.0),
            engagement_level=performance_data.get('engagement_level', 0.5),
            satisfaction_level=performance_data.get('satisfaction_level', 0.5)
        )
        
        # 記錄性能歷史
        session.performance_history.append(metrics.accuracy)
        session.satisfaction_history.append(metrics.satisfaction_level)
        
        # 創建並返回標準化的遊戲表現數據
        return GamePerformanceData(
            user_id=session.user_id,
            game_id=f"{session.game_type.value}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            round_id=round_id,
            game_type=session.game_type,
            timestamp=datetime.now(),
            difficulty_level=session.difficulty_history[-1] if session.difficulty_history else session.initial_difficulty,
            performance_metrics=metrics,
            session_id=session_id
        )
        
    def record_difficulty_adjustment(self, session_id: str, difficulty_level: int, 
                                  difficulty_params: Dict[str, Any]) -> None:
        """記錄難度調整數據"""
        if session_id not in self.session_data:
            raise ValueError(f"會話ID不存在: {session_id}")
            
        session = self.session_data[session_id]
        
        # 記錄難度歷史
        session.difficulty_history.append(difficulty_level)
        
    def save_session_data(self, session_id: str, filepath: str) -> None:
        """將會話數據保存到JSON文件"""
        if session_id not in self.session_data:
            raise ValueError(f"會話ID不存在: {session_id}")
            
        session = self.session_data[session_id]
        
        # 將數據轉換為可序列化的字典
        session_dict = {
            "user_id": session.user_id,
            "session_id": session.session_id,
            "game_type": session.game_type.value,
            "start_time": session.start_time.isoformat(),
            "end_time": session.end_time.isoformat() if session.end_time else None,
            "user_skill": session.user_skill,
            "initial_difficulty": session.initial_difficulty,
            "difficulty_history": session.difficulty_history,
            "performance_history": session.performance_history,
            "satisfaction_history": session.satisfaction_history,
            "cognitive_assessments": session.cognitive_assessments,
            "timestamp": datetime.now().isoformat()
        }
        
        # 保存到文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(session_dict, f, indent=2, ensure_ascii=False)


class ModelInputConverter:
    """模型輸入轉換器：將收集的數據轉換為模型所需的輸入格式"""
    
    @staticmethod
    def prepare_behavior_analysis_input(game_data: GamePerformanceData) -> Dict[str, np.ndarray]:
        """準備用戶行為分析模型的輸入數據"""
        metrics = game_data.performance_metrics
        
        # 提取響應時間數據
        response_times = np.array([metrics.average_response_time] * 10).reshape(1, -1)
        
        # 提取錯誤模式數據
        error_rate = metrics.incorrect_responses / (metrics.correct_responses + metrics.incorrect_responses + 0.001)
        error_patterns = np.array([error_rate] * 10).reshape(1, -1)
        
        # 創建交互序列數據
        # 實際情況中，這可能是從game_data的更多字段中提取的
        interaction_sequences = np.zeros((1, 10, 10))
        
        return {
            'response_times': response_times,
            'error_patterns': error_patterns,
            'interaction_sequences': interaction_sequences
        }
    
    @staticmethod
    def prepare_difficulty_adjustment_input(performance_data: GamePerformanceMetrics, 
                                          current_difficulty: int) -> np.ndarray:
        """準備難度調整模型的輸入數據"""
        # 提取狀態特徵
        state_features = [
            performance_data.accuracy,
            normalize_value(performance_data.completion_time, 0, 120),  # 假設最長時間為120秒
            performance_data.frustration_indicators,
            performance_data.engagement_level,
            performance_data.satisfaction_level,
            current_difficulty / 4.0,  # 標準化到0-1（假設難度範圍是0-4）
            performance_data.attempts_count / 5.0  # 標準化嘗試次數
        ]
        
        return np.array(state_features, dtype=np.float32)
    
    @staticmethod
    def prepare_cognitive_assessment_input(game_data_list: List[GamePerformanceData]) -> Dict[str, np.ndarray]:
        """準備認知功能評估模型的輸入數據"""
        # 提取各遊戲類型的表現指標
        game_metrics = {}
        for game_data in game_data_list:
            game_type = game_data.game_type.value
            if game_type not in game_metrics:
                game_metrics[game_type] = []
            
            metrics = game_data.performance_metrics
            game_metrics[game_type].append({
                'accuracy': metrics.accuracy,
                'response_time': metrics.average_response_time,
                'completion_rate': metrics.completion_percentage
            })
        
        # 計算每種遊戲類型的平均表現
        domain_features = {}
        for game_type, metrics_list in game_metrics.items():
            avg_accuracy = np.mean([m['accuracy'] for m in metrics_list])
            avg_response_time = np.mean([m['response_time'] for m in metrics_list])
            avg_completion_rate = np.mean([m['completion_rate'] for m in metrics_list])
            
            domain_features[game_type] = np.array([
                avg_accuracy, 
                normalize_value(avg_response_time, 0, 10),  # 假設最長響應時間為10秒
                avg_completion_rate
            ])
        
        return domain_features
    
    @staticmethod
    def prepare_multimodal_input(perception_data: MultimodalPerceptionData) -> Dict[str, np.ndarray]:
        """準備多模態感知模型的輸入數據"""
        # 這裡會依據實際遊戲實現來提取特徵
        # 先創建空白的數據結構
        visual_features = np.zeros((1, 128))
        audio_features = np.zeros((1, 64))
        
        # 如果有視覺數據，提取特徵
        if perception_data.visual_data:
            # 實際情況下會從perception_data.visual_data中提取具體特徵
            pass
            
        # 如果有聽覺數據，提取特徵
        if perception_data.auditory_data:
            # 實際情況下會從perception_data.auditory_data中提取具體特徵
            pass
        
        return {
            'visual_features': visual_features,
            'audio_features': audio_features
        }
    
    @staticmethod
    def prepare_report_generation_input(assessment_data: CognitiveAssessmentData, 
                                      game_history: List[GamePerformanceData]) -> Dict[str, Any]:
        """準備報告生成模型的輸入數據"""
        # 提取認知領域評分
        domain_scores = {}
        for domain, score_data in assessment_data.cognitive_domains.items():
            domain_scores[domain] = score_data.score
        
        # 提取遊戲表現歷史
        performance_history = {}
        for game_data in game_history:
            game_type = game_data.game_type.value
            if game_type not in performance_history:
                performance_history[game_type] = []
            
            performance_history[game_type].append({
                'timestamp': game_data.timestamp.isoformat(),
                'accuracy': game_data.performance_metrics.accuracy,
                'difficulty': game_data.difficulty_level
            })
        
        return {
            'user_id': assessment_data.user_id,
            'assessment_date': assessment_data.assessment_date.isoformat(),
            'domain_scores': domain_scores,
            'overall_score': assessment_data.overall_score,
            'performance_history': performance_history,
            'clinical_notes': assessment_data.clinical_notes
        }


def load_session_data(filepath: str) -> SessionData:
    """從JSON文件加載會話數據"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 轉換基本字段
    session = SessionData(
        user_id=data['user_id'],
        session_id=data['session_id'],
        game_type=GameType(data['game_type']),
        start_time=convert_timestamp(data['start_time']),
        end_time=convert_timestamp(data['end_time']) if data['end_time'] else None,
        user_skill=data['user_skill'],
        initial_difficulty=data['initial_difficulty']
    )
    
    # 添加歷史數據
    session.difficulty_history = data['difficulty_history']
    session.performance_history = data['performance_history']
    session.satisfaction_history = data['satisfaction_history']
    session.cognitive_assessments = data['cognitive_assessments']
    
    return session


def batch_process_game_data(data_files: List[str], output_file: str) -> None:
    """批量處理遊戲數據文件並將處理結果保存到輸出文件"""
    processed_data = []
    
    for file_path in data_files:
        try:
            session_data = load_session_data(file_path)
            
            # 處理會話數據，例如計算聚合指標
            avg_performance = sum(session_data.performance_history) / len(session_data.performance_history) if session_data.performance_history else 0
            avg_satisfaction = sum(session_data.satisfaction_history) / len(session_data.satisfaction_history) if session_data.satisfaction_history else 0
            
            processed_data.append({
                'user_id': session_data.user_id,
                'game_type': session_data.game_type.value,
                'session_duration': session_data.duration_minutes,
                'avg_performance': avg_performance,
                'avg_satisfaction': avg_satisfaction,
                'max_difficulty': max(session_data.difficulty_history) if session_data.difficulty_history else session_data.initial_difficulty,
                'final_difficulty': session_data.difficulty_history[-1] if session_data.difficulty_history else session_data.initial_difficulty
            })
        except Exception as e:
            print(f"處理文件 {file_path} 時出錯: {e}")
    
    # 保存處理結果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False) 