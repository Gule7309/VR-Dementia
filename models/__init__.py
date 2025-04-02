import logging
import os
import tensorflow as tf

from .behavior_analysis_model import get_game_analyzer
from .difficulty_adjustment_model import create_difficulty_manager
from .multimodal_perception_model import load_pretrained_perception_model
from .cognitive_assessment_model import create_cognitive_assessment_service
from .report_generation_model import create_report_generation_service


def setup_logging(log_level=logging.INFO):
    """設置日誌記錄"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ai_models.log')
        ]
    )


class ModelManager:
    """AI模型管理器：統一管理和使用所有AI模型"""
    
    def __init__(self, config=None):
        """
        初始化模型管理器
        
        參數:
        - config: 配置字典，包含模型路徑等設置
        """
        self.logger = logging.getLogger("model_manager")
        self.config = config or {}
        
        # 模型路徑配置
        model_paths = self.config.get('model_paths', {})
        
        # 初始化各模型
        self.analyzers = {}  # 遊戲特定的行為分析器
        self.difficulty_managers = {}  # 遊戲特定的難度管理器
        
        # 載入多模態感知模型
        self.logger.info("載入多模態感知模型...")
        self.perception_model = load_pretrained_perception_model(
            model_paths.get('perception_models')
        )
        
        # 載入認知評估服務
        self.logger.info("載入認知評估服務...")
        self.assessment_service = create_cognitive_assessment_service(
            model_paths.get('assessment_model')
        )
        
        # 載入報告生成服務
        self.logger.info("載入報告生成服務...")
        self.report_service = create_report_generation_service(
            model_paths.get('report_model')
        )
        
        self.logger.info("所有模型載入完成")
    
    def get_game_analyzer(self, game_type):
        """獲取特定遊戲的行為分析器"""
        
        if game_type not in self.analyzers:
            self.logger.info(f"為遊戲類型 '{game_type}' 創建新的行為分析器")
            model_path = self.config.get('model_paths', {}).get(
                'analyzers', {}).get(game_type)
            self.analyzers[game_type] = get_game_analyzer(game_type, model_path)
        
        return self.analyzers[game_type]
    
    def get_difficulty_manager(self, game_type):
        """獲取特定遊戲的難度管理器"""
        
        if game_type not in self.difficulty_managers:
            self.logger.info(f"為遊戲類型 '{game_type}' 創建新的難度管理器")
            model_path = self.config.get('model_paths', {}).get(
                'difficulty_managers', {}).get(game_type)
            self.difficulty_managers[game_type] = create_difficulty_manager(
                game_type, model_path=model_path)
        
        return self.difficulty_managers[game_type]
    
    def analyze_game_performance(self, game_data):
        """分析遊戲表現"""
        
        game_type = game_data.get('game_type')
        
        if not game_type:
            self.logger.error("無法分析：遊戲類型未提供")
            return None
        
        # 獲取該遊戲的分析器
        analyzer = self.get_game_analyzer(game_type)
        
        # 分析行為
        analysis_results = analyzer.analyze(game_data)
        
        return analysis_results
    
    def adjust_game_difficulty(self, game_type, performance_data, deterministic=False):
        """調整遊戲難度"""
        
        # 獲取難度管理器
        manager = self.get_difficulty_manager(game_type)
        
        # 更新難度
        new_difficulty, params = manager.update_difficulty(
            performance_data, deterministic=deterministic)
        
        return {
            'difficulty_level': new_difficulty,
            'difficulty_params': params
        }
    
    def process_multimodal_input(self, input_type, data):
        """處理多模態輸入"""
        
        if input_type == 'speech_recognition':
            result = self.perception_model.recognize_speech(data)
        elif input_type == 'object_matching':
            result = self.perception_model.match_objects(
                data['reference_image'], data['candidate_images'])
        elif input_type == 'drawing_assessment':
            result = self.perception_model.assess_drawing(
                data['reference_image'], data['user_drawing'])
        elif input_type == 'time_recognition':
            result = self.perception_model.recognize_time(
                image=data.get('image'), audio=data.get('audio'))
        else:
            self.logger.error(f"不支持的輸入類型: {input_type}")
            result = None
        
        return result
    
    def assess_cognitive_function(self, user_id, game_data):
        """評估認知功能"""
        
        # 使用認知評估服務
        assessment = self.assessment_service.assess_game_performance(user_id, game_data)
        
        return assessment
    
    def get_comprehensive_assessment(self, user_id, time_range=None):
        """獲取綜合認知評估"""
        
        # 使用認知評估服務
        assessment = self.assessment_service.get_comprehensive_assessment(
            user_id, time_range)
        
        return assessment
    
    def generate_assessment_report(self, user_id, user_name, game_data_list):
        """生成評估報告"""
        
        # 使用報告生成服務
        report = self.report_service.generate_report(user_id, user_name, game_data_list)
        
        return report


def create_model_manager(config=None):
    """創建模型管理器"""
    setup_logging()
    
    # 默認配置
    default_config = {
        'model_paths': {
            'perception_models': {},
            'assessment_model': None,
            'report_model': None,
            'analyzers': {},
            'difficulty_managers': {}
        }
    }
    
    # 合併配置
    if config:
        for key, value in config.items():
            if key in default_config and isinstance(default_config[key], dict):
                default_config[key].update(value)
            else:
                default_config[key] = value
    
    # 創建管理器
    manager = ModelManager(default_config)
    
    return manager 