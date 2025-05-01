"""
VR樂園「憶」壽延年 數據模型與轉換工具
提供標準化的數據結構和轉換功能，用於遊戲數據收集和模型訓練
"""

# 匯出主要的數據模型
from schemas.data_models import (
    # 枚舉類型
    GameType, CognitiveStatus, ActionType, ErrorType, StimulusType, DifficultyAction,
    
    # 基礎數據模型
    Position3D, TimeInterval,
    
    # 用戶模型
    UserProfile,
    
    # 遊戲表現數據
    ErrorDetail, GamePerformanceMetrics, CalculationGameMetrics,
    MemoryGameMetrics, SpatialGameMetrics, GameSpecificMetrics,
    GamePerformanceData,
    
    # 互動數據
    UserAction, GazeData, MovementPattern, InteractionData,
    
    # 難度調整數據
    DifficultyParameters, DifficultyAdjustmentEvent, DifficultyAdjustmentData,
    
    # 認知評估數據
    DomainScore, CognitiveAssessmentData,
    
    # 多模態感知數據
    Stimulus, StimulusResponse, ModalityData, MultimodalPerceptionData,
    
    # 生理及情緒數據
    PhysiologicalMeasurement, EmotionalTrigger, EmotionalIndicator,
    PhysiologicalEmotionalData,
    
    # 長期進展數據
    ProgressPoint, GameProgressData, SkillTransfer, LongTermProgressData,
    
    # 報告生成數據
    AssessmentReport,
    
    # 會話數據
    SessionData
)

# 匯出數據轉換工具
from schemas.data_conversion import (
    # 工具函數
    convert_timestamp, normalize_value,
    
    # 數據收集類
    GameDataCollector,
    
    # 模型輸入轉換
    ModelInputConverter,
    
    # 數據加載和處理函數
    load_session_data, batch_process_game_data
)

__all__ = [
    # 枚舉類型
    'GameType', 'CognitiveStatus', 'ActionType', 'ErrorType', 'StimulusType', 'DifficultyAction',
    
    # 基礎數據模型
    'Position3D', 'TimeInterval',
    
    # 用戶模型
    'UserProfile',
    
    # 遊戲表現數據
    'ErrorDetail', 'GamePerformanceMetrics', 'CalculationGameMetrics',
    'MemoryGameMetrics', 'SpatialGameMetrics', 'GameSpecificMetrics', 
    'GamePerformanceData',
    
    # 互動數據
    'UserAction', 'GazeData', 'MovementPattern', 'InteractionData',
    
    # 難度調整數據
    'DifficultyParameters', 'DifficultyAdjustmentEvent', 'DifficultyAdjustmentData',
    
    # 認知評估數據
    'DomainScore', 'CognitiveAssessmentData',
    
    # 多模態感知數據
    'Stimulus', 'StimulusResponse', 'ModalityData', 'MultimodalPerceptionData',
    
    # 生理及情緒數據
    'PhysiologicalMeasurement', 'EmotionalTrigger', 'EmotionalIndicator',
    'PhysiologicalEmotionalData',
    
    # 長期進展數據
    'ProgressPoint', 'GameProgressData', 'SkillTransfer', 'LongTermProgressData',
    
    # 報告生成數據
    'AssessmentReport',
    
    # 會話數據
    'SessionData',
    
    # 數據轉換工具
    'convert_timestamp', 'normalize_value', 'GameDataCollector', 'ModelInputConverter',
    'load_session_data', 'batch_process_game_data'
] 