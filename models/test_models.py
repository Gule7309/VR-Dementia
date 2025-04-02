import numpy as np
import tensorflow as tf
import logging
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io
import base64
import sys
import os

# 添加當前目錄到路徑，以便導入模型
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 導入模型管理器
from models import create_model_manager


def setup_logging():
    """設置日誌記錄"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )


def generate_mock_game_data(game_type, performance_level='medium', user_id='test_user'):
    """生成模擬遊戲數據"""
    
    # 根據表現水平設置基準準確率
    if performance_level == 'high':
        base_accuracy = 0.85
    elif performance_level == 'medium':
        base_accuracy = 0.65
    else:  # low
        base_accuracy = 0.45
    
    # 添加一些隨機變化
    accuracy = min(1.0, max(0.1, base_accuracy + np.random.normal(0, 0.1)))
    response_time = max(0.5, 5.0 - accuracy * 3 + np.random.normal(0, 1))
    completion_rate = min(1.0, max(0.2, accuracy * 0.8 + np.random.normal(0, 0.1)))
    
    # 基本遊戲數據
    game_data = {
        'user_id': user_id,
        'game_type': game_type,
        'timestamp': datetime.now().isoformat(),
        'duration': int(response_time * 10),  # 遊戲時長（秒）
        'performance_metrics': {
            'accuracy': accuracy,
            'response_time': response_time,
            'completion_rate': completion_rate,
            'error_rate': 1.0 - accuracy,
            'attempts': np.random.randint(1, 4)
        }
    }
    
    # 根據遊戲類型添加特定數據
    if game_type == 'attention_calculation':
        game_data['calculation_data'] = {
            'errors': int((1.0 - accuracy) * 10),
            'total_problems': 10,
            'problems_per_minute': 60 / response_time,
            'attention_drops': np.random.randint(0, 5)
        }
    elif game_type == 'short_term_memory':
        game_data['memory_data'] = {
            'memory_span': np.random.randint(3, 8),
            'retention_rate': accuracy,
            'sequence_accuracy': accuracy * 0.9
        }
    elif game_type == 'drawing':
        game_data['drawing_data'] = {
            'construction_score': accuracy,
            'memory_component': accuracy * 0.8,
            'detail_attention': accuracy * 0.9
        }
    
    return game_data


def generate_mock_performance_data(accuracy=0.7, completion_time=30):
    """生成模擬表現數據（用於難度調整）"""
    
    return {
        'accuracy': accuracy,
        'completion_time': completion_time,
        'attempts': np.random.randint(1, 4),
        'frustration_indicators': max(0, min(1, (1.0 - accuracy) * 0.8 + np.random.normal(0, 0.1))),
        'engagement_level': max(0, min(1, accuracy * 0.7 + np.random.normal(0, 0.1))),
        'timestamp': datetime.now().isoformat()
    }


def generate_mock_multimodal_data(input_type):
    """生成模擬多模態數據"""
    
    if input_type == 'speech_recognition':
        # 模擬音頻頻譜圖（實際應用中應使用真實音頻數據）
        return np.random.rand(1, 128, 128, 1)
    
    elif input_type == 'object_matching':
        # 模擬參考圖像和候選圖像
        reference_image = np.random.rand(224, 224, 3)
        candidate_images = [np.random.rand(224, 224, 3) for _ in range(5)]
        return {
            'reference_image': reference_image,
            'candidate_images': candidate_images
        }
    
    elif input_type == 'drawing_assessment':
        # 模擬參考圖形和用戶繪製圖形
        reference_image = np.random.rand(224, 224, 3)
        user_drawing = np.random.rand(224, 224, 3)
        return {
            'reference_image': reference_image,
            'user_drawing': user_drawing
        }
    
    elif input_type == 'time_recognition':
        # 模擬圖像和音頻數據
        image = np.random.rand(224, 224, 3)
        audio = np.random.rand(1, 128, 128, 1)
        return {
            'image': image,
            'audio': audio
        }
    
    return None


def test_behavior_analysis(model_manager):
    """測試行為分析模型"""
    logging.info("測試行為分析模型...")
    
    # 生成模擬數據
    game_types = ['attention_calculation', 'short_term_memory', 'drawing']
    performance_levels = ['low', 'medium', 'high']
    
    for game_type in game_types:
        for level in performance_levels:
            game_data = generate_mock_game_data(game_type, level)
            
            # 分析行為
            analysis_results = model_manager.analyze_game_performance(game_data)
            
            # 打印結果
            logging.info(f"遊戲類型: {game_type}, 表現水平: {level}")
            logging.info(f"  認知指標: {analysis_results.get('cognitive_indicators', 'N/A')}")
            logging.info(f"  異常分數: {analysis_results.get('anomaly_score', 'N/A')}")
            
            # 如果有解釋，打印它們
            if 'interpretations' in analysis_results:
                logging.info(f"  解釋: {analysis_results['interpretations']}")
            
            if 'suggestions' in analysis_results:
                logging.info(f"  建議: {analysis_results['suggestions']}")
    
    logging.info("行為分析模型測試完成")


def test_difficulty_adjustment(model_manager):
    """測試難度調整模型"""
    logging.info("測試難度調整模型...")
    
    # 測試不同遊戲類型的難度調整
    game_types = ['attention_calculation', 'short_term_memory', 'object_matching']
    
    for game_type in game_types:
        logging.info(f"遊戲類型: {game_type}")
        
        # 模擬不同表現水平的調整
        performance_levels = [
            (0.3, 50),  # 低表現：準確率低，完成時間長
            (0.6, 35),  # 中等表現
            (0.9, 20)   # 高表現：準確率高，完成時間短
        ]
        
        for accuracy, completion_time in performance_levels:
            performance_data = generate_mock_performance_data(accuracy, completion_time)
            
            # 調整難度
            adjustment = model_manager.adjust_game_difficulty(
                game_type, performance_data, deterministic=True)
            
            # 打印結果
            logging.info(f"  表現 (準確率={accuracy:.2f}, 時間={completion_time}s)")
            logging.info(f"  調整後難度: {adjustment['difficulty_level']}")
            logging.info(f"  難度參數: {adjustment['difficulty_params']}")
    
    # 測試連續調整
    logging.info("測試連續難度調整...")
    game_type = 'attention_calculation'
    
    # 從低表現開始，逐漸提高
    accuracies = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    completion_times = [50, 45, 40, 35, 30, 25, 20]
    
    for accuracy, completion_time in zip(accuracies, completion_times):
        performance_data = generate_mock_performance_data(accuracy, completion_time)
        
        # 調整難度
        adjustment = model_manager.adjust_game_difficulty(
            game_type, performance_data, deterministic=False)
        
        # 打印結果
        logging.info(f"  表現 (準確率={accuracy:.2f}, 時間={completion_time}s)")
        logging.info(f"  調整後難度: {adjustment['difficulty_level']}")
    
    logging.info("難度調整模型測試完成")


def test_multimodal_perception(model_manager):
    """測試多模態感知模型"""
    logging.info("測試多模態感知模型...")
    
    # 測試各種感知任務
    input_types = [
        'speech_recognition', 
        'object_matching',
        'drawing_assessment', 
        'time_recognition'
    ]
    
    for input_type in input_types:
        logging.info(f"輸入類型: {input_type}")
        
        # 生成模擬數據
        mock_data = generate_mock_multimodal_data(input_type)
        
        # 處理輸入
        result = model_manager.process_multimodal_input(input_type, mock_data)
        
        # 打印結果（模擬環境下，結果可能為None）
        logging.info(f"  處理結果: {result}")
    
    logging.info("多模態感知模型測試完成")


def test_cognitive_assessment(model_manager):
    """測試認知功能評估模型"""
    logging.info("測試認知功能評估模型...")
    
    user_id = 'test_user_123'
    
    # 生成不同類型遊戲的模擬數據
    game_types = [
        'attention_calculation', 
        'time_recognition',
        'short_term_memory', 
        'drawing',
        'repeat_language', 
        'naming',
        'object_matching', 
        'spatial_concept'
    ]
    
    for game_type in game_types:
        # 生成中等表現的數據
        game_data = generate_mock_game_data(game_type, 'medium', user_id)
        
        # 評估認知功能
        assessment = model_manager.assess_cognitive_function(user_id, game_data)
        
        # 打印結果
        logging.info(f"遊戲類型: {game_type}")
        logging.info(f"  域評分: {assessment.get('domain_scores', 'N/A')}")
        logging.info(f"  整體評分: {assessment.get('overall_score', 'N/A')}")
        logging.info(f"  解釋: {assessment.get('interpretations', 'N/A')}")
    
    # 測試綜合評估
    logging.info("測試綜合認知評估...")
    comprehensive = model_manager.get_comprehensive_assessment(user_id)
    
    if comprehensive:
        logging.info(f"平均域評分: {comprehensive.get('avg_domain_scores', 'N/A')}")
        logging.info(f"平均整體評分: {comprehensive.get('avg_overall_score', 'N/A')}")
        logging.info(f"趨勢: {comprehensive.get('trends', 'N/A')}")
        logging.info(f"解釋: {comprehensive.get('interpretations', 'N/A')}")
    else:
        logging.info("無綜合評估結果")
    
    logging.info("認知評估模型測試完成")


def test_report_generation(model_manager):
    """測試報告生成模型"""
    logging.info("測試報告生成模型...")
    
    user_id = 'test_user_456'
    user_name = '測試用戶'
    
    # 生成多個遊戲數據，模擬一段時間的使用
    game_data_list = []
    game_types = ['attention_calculation', 'short_term_memory', 'drawing']
    performance_levels = ['low', 'medium', 'medium', 'high']
    
    # 生成過去幾天的數據
    for i, level in enumerate(performance_levels):
        days_ago = len(performance_levels) - i - 1
        timestamp = (datetime.now() - timedelta(days=days_ago)).isoformat()
        
        for game_type in game_types:
            game_data = generate_mock_game_data(game_type, level, user_id)
            game_data['timestamp'] = timestamp
            
            # 添加認知評估
            cognitive_assessment = {
                'domain_scores': {
                    'attention': 0.4 + i * 0.15,
                    'memory': 0.5 + i * 0.1,
                    'language': 0.6,
                    'visuospatial': 0.55 + i * 0.12,
                    'executive': 0.45 + i * 0.14
                },
                'overall_score': 0.5 + i * 0.1
            }
            game_data['cognitive_assessment'] = cognitive_assessment
            
            game_data_list.append(game_data)
    
    # 生成報告
    report = model_manager.generate_assessment_report(user_id, user_name, game_data_list)
    
    # 打印報告內容
    logging.info(f"用戶: {report.get('user_name')}")
    logging.info(f"時間戳: {report.get('timestamp')}")
    logging.info(f"摘要: {report.get('summary')}")
    logging.info(f"整體評估: {report.get('overall_assessment')}")
    logging.info(f"認知域評估: {report.get('domain_assessments')}")
    logging.info(f"建議: {report.get('recommendations')}")
    
    # 檢查是否生成了圖表
    if 'charts' in report and report['charts']:
        logging.info(f"生成了 {len(report['charts'])} 個圖表")
        for chart_name, chart_data in report['charts'].items():
            logging.info(f"  圖表: {chart_name} (數據長度: {len(chart_data) if chart_data else 0})")
    
    logging.info("報告生成模型測試完成")


def main():
    """主函數：執行所有測試"""
    setup_logging()
    
    logging.info("啟動模型測試...")
    
    # 創建模型管理器
    model_manager = create_model_manager()
    
    # 執行測試
    test_behavior_analysis(model_manager)
    test_difficulty_adjustment(model_manager)
    test_multimodal_perception(model_manager)
    test_cognitive_assessment(model_manager)
    test_report_generation(model_manager)
    
    logging.info("所有測試完成")


if __name__ == '__main__':
    main() 