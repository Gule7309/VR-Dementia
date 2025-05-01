from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Any
from datetime import datetime
from enum import Enum


# 枚舉類型定義
class GameType(str, Enum):
    ATTENTION_CALCULATION = "attention_calculation"
    SHORT_TERM_MEMORY = "short_term_memory"
    TIME_RECOGNITION = "time_recognition"
    DRAWING = "drawing"
    REPEAT_LANGUAGE = "repeat_language"
    NAMING = "naming"
    OBJECT_MATCHING = "object_matching"
    SPATIAL_CONCEPT = "spatial_concept"


class CognitiveStatus(str, Enum):
    NORMAL = "normal"
    MCI = "mci"  # 輕度認知障礙
    MILD_DEMENTIA = "mild_dementia"
    MODERATE_DEMENTIA = "moderate_dementia"
    SEVERE_DEMENTIA = "severe_dementia"


class ActionType(str, Enum):
    SELECT = "select"
    DRAG = "drag"
    CLICK = "click"
    GAZE = "gaze"
    SPEAK = "speak"


class ErrorType(str, Enum):
    CALCULATION_ERROR = "calculation_error"
    MEMORY_ERROR = "memory_error"
    ATTENTION_ERROR = "attention_error"
    PERCEPTION_ERROR = "perception_error"
    SPATIAL_ERROR = "spatial_error"
    LANGUAGE_ERROR = "language_error"


class StimulusType(str, Enum):
    VISUAL_OBJECT = "visual_object"
    TEXT = "text"
    SYMBOL = "symbol"
    AUDIO = "audio"
    SPEECH = "speech"
    HAPTIC = "haptic"


class DifficultyAction(str, Enum):
    INCREASE = "increase"
    DECREASE = "decrease"
    MAINTAIN = "maintain"


# 基礎數據模型
@dataclass
class Position3D:
    x: float
    y: float
    z: float


@dataclass
class TimeInterval:
    start_time: datetime
    end_time: datetime
    
    @property
    def duration_seconds(self) -> float:
        return (self.end_time - self.start_time).total_seconds()


# 用戶模型
@dataclass
class UserProfile:
    user_id: str
    age: int
    gender: str
    education_level: str
    cognitive_status: CognitiveStatus = CognitiveStatus.NORMAL
    medical_conditions: List[str] = field(default_factory=list)
    cognitive_abilities: Dict[str, float] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


# 遊戲表現數據模型
@dataclass
class ErrorDetail:
    error_type: ErrorType
    severity: float  # 0-1
    context: str
    timestamp: float


@dataclass
class GamePerformanceMetrics:
    accuracy: float  # 0-1
    completion_time: float  # 秒
    errors: List[ErrorDetail] = field(default_factory=list)
    correct_responses: int = 0
    incorrect_responses: int = 0
    missed_responses: int = 0
    average_response_time: float = 0.0
    response_time_variance: float = 0.0
    completion_percentage: float = 1.0  # 0-1
    attempts_count: int = 1
    frustration_indicators: float = 0.0  # 0-1
    engagement_level: float = 0.5  # 0-1
    satisfaction_level: float = 0.5  # 0-1


@dataclass
class CalculationGameMetrics:
    operation_accuracy: Dict[str, float]  # 各種計算操作的準確率
    calculation_speed: float  # 每秒操作數
    error_distribution: Dict[str, int]  # 不同類型錯誤的數量


@dataclass
class MemoryGameMetrics:
    recall_accuracy: float
    sequence_length_max: int
    position_accuracy: float
    recall_pattern: str


@dataclass
class SpatialGameMetrics:
    navigation_accuracy: float
    spatial_orientation: float
    landmark_recognition: float
    path_reconstruction: float


@dataclass
class GameSpecificMetrics:
    calculation_metrics: Optional[CalculationGameMetrics] = None
    memory_metrics: Optional[MemoryGameMetrics] = None
    spatial_metrics: Optional[SpatialGameMetrics] = None


@dataclass
class GamePerformanceData:
    user_id: str
    game_id: str
    round_id: str
    game_type: GameType
    timestamp: datetime
    difficulty_level: int
    performance_metrics: GamePerformanceMetrics
    game_specific_metrics: Optional[GameSpecificMetrics] = None
    session_id: Optional[str] = None


# 互動數據模型
@dataclass
class UserAction:
    action_id: str
    action_type: ActionType
    target_object: str
    start_time: float
    end_time: float
    duration: float
    position: Optional[Position3D] = None
    is_correct: bool = False
    reaction_time: float = 0.0


@dataclass
class GazeData:
    timestamp: float
    target_object: str
    duration: float
    direction: Position3D


@dataclass
class MovementPattern:
    path_efficiency: float  # 0-1
    hesitation_count: int
    correction_movements: int


@dataclass
class InteractionData:
    user_id: str
    session_id: str
    timestamp: datetime
    game_type: GameType
    actions: List[UserAction] = field(default_factory=list)
    gaze_data: List[GazeData] = field(default_factory=list)
    movement_patterns: Optional[MovementPattern] = None


# 難度調整模型
@dataclass
class DifficultyParameters:
    time_limit: Optional[int] = None  # 秒
    distraction_level: Optional[int] = None  # 0-4
    calculation_complexity: Optional[int] = None  # 1-5
    sequence_length: Optional[int] = None  # 記憶序列長度
    display_time: Optional[int] = None  # 顯示時間（秒）
    interference_level: Optional[int] = None  # 干擾程度
    complexity: Optional[int] = None  # 圖形複雜度
    guidance_level: Optional[int] = None  # 引導程度
    sentence_length: Optional[int] = None  # 詞語數量
    speech_speed: Optional[float] = None  # 語速
    background_noise: Optional[float] = None  # 背景噪音級別
    image_clarity: Optional[float] = None  # 圖像清晰度
    object_familiarity: Optional[float] = None  # 物體熟悉度
    similarity_level: Optional[float] = None  # 物體相似度
    distractor_count: Optional[int] = None  # 干擾物數量
    arrangement_complexity: Optional[int] = None  # 排列複雜度
    environment_complexity: Optional[int] = None  # 環境複雜度
    landmark_visibility: Optional[float] = None  # 地標可見度
    path_complexity: Optional[int] = None  # 路徑複雜度


@dataclass
class DifficultyAdjustmentEvent:
    event_id: str
    timestamp: float
    adaptation_type: str
    trigger: Dict[str, float]  # 觸發調整的指標及其值
    adjustment: Dict[str, Any]  # 調整的參數及其值
    user_response: Dict[str, Any]  # 用戶對調整的反應


@dataclass
class DifficultyAdjustmentData:
    user_id: str
    session_id: str
    game_type: GameType
    timestamp: datetime
    current_difficulty: int
    previous_difficulty: int
    performance_data: GamePerformanceMetrics
    difficulty_params: DifficultyParameters
    adjustment_events: List[DifficultyAdjustmentEvent] = field(default_factory=list)
    adaptation_effectiveness: Dict[str, float] = field(default_factory=dict)


# 認知評估模型
@dataclass
class DomainScore:
    score: float  # 0-1
    subscores: Dict[str, float]
    clinical_interpretation: Optional[str] = None


@dataclass
class CognitiveAssessmentData:
    user_id: str
    assessment_date: datetime
    cognitive_domains: Dict[str, DomainScore]
    overall_score: float
    clinical_notes: Optional[str] = None
    previous_assessment_id: Optional[str] = None
    change_from_previous: Optional[Dict[str, float]] = None


# 多模態感知數據
@dataclass
class Stimulus:
    stimulus_id: str
    type: StimulusType
    presentation_time: float
    duration: Optional[float] = None
    position: Optional[Position3D] = None
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StimulusResponse:
    stimulus_id: str
    response_time: float
    accuracy: bool
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModalityData:
    stimuli: List[Stimulus]
    responses: List[StimulusResponse]


@dataclass
class MultimodalPerceptionData:
    user_id: str
    session_id: str
    game_id: str
    timestamp: datetime
    visual_data: Optional[ModalityData] = None
    auditory_data: Optional[ModalityData] = None
    haptic_data: Optional[ModalityData] = None
    cross_modal_interactions: Dict[str, Any] = field(default_factory=dict)


# 生理及情緒數據
@dataclass
class PhysiologicalMeasurement:
    timestamp: float
    value: float


@dataclass
class EmotionalTrigger:
    timestamp: float
    game_event: str
    intensity: float


@dataclass
class EmotionalIndicator:
    level: float  # 0-1
    triggers: List[EmotionalTrigger] = field(default_factory=list)
    pattern: Optional[str] = None
    peak_moments: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PhysiologicalEmotionalData:
    user_id: str
    session_id: str
    timestamp: datetime
    sampling_rate: int  # Hz
    heart_rate: List[PhysiologicalMeasurement] = field(default_factory=list)
    heart_rate_variability: Optional[float] = None
    respiration_rate: Optional[float] = None
    skin_conductance: List[PhysiologicalMeasurement] = field(default_factory=list)
    pupil_dilation: List[Dict[str, float]] = field(default_factory=list)
    frustration: Optional[EmotionalIndicator] = None
    engagement: Optional[EmotionalIndicator] = None
    satisfaction: Optional[EmotionalIndicator] = None
    cognitive_load: Dict[str, float] = field(default_factory=dict)


# 長期進展與學習數據
@dataclass
class ProgressPoint:
    timestamp: datetime
    performance_index: float
    skill_level: Optional[float] = None
    difficulty_level: Optional[int] = None


@dataclass
class GameProgressData:
    game_type: GameType
    progress: List[ProgressPoint] = field(default_factory=list)
    plateau_periods: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SkillTransfer:
    primary_game: GameType
    secondary_game: GameType
    correlation_strength: float  # -1 到 1
    lag_sessions: int  # 改善出現的延遲


@dataclass
class LongTermProgressData:
    user_id: str
    training_program_id: str
    period: TimeInterval
    session_frequency: float  # 每週次數
    session_duration_average: float  # 分鐘
    engagement_metrics: Dict[str, float] = field(default_factory=dict)
    overall_progress: List[ProgressPoint] = field(default_factory=list)
    game_specific_progress: Dict[GameType, GameProgressData] = field(default_factory=dict)
    skill_transfer: List[SkillTransfer] = field(default_factory=list)


# 報告生成模型數據
@dataclass
class AssessmentReport:
    user_id: str
    report_id: str
    generation_date: datetime
    period_covered: TimeInterval
    summary: str
    domain_assessments: Dict[str, Dict[str, Any]]
    recommendations: List[str]
    trends: Dict[str, List[float]]
    charts: Dict[str, str]  # 圖表名稱和檔案路徑
    clinical_significance: Optional[str] = None
    next_assessment_date: Optional[datetime] = None


# 會話數據
@dataclass
class SessionData:
    user_id: str
    session_id: str
    game_type: GameType
    start_time: datetime
    end_time: Optional[datetime] = None
    user_skill: float = 0.5  # 0-1
    initial_difficulty: int = 2
    difficulty_history: List[int] = field(default_factory=list)
    performance_history: List[float] = field(default_factory=list)
    satisfaction_history: List[float] = field(default_factory=list)
    cognitive_assessments: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def duration_minutes(self) -> Optional[float]:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() / 60
        return None 