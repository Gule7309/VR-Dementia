import numpy as np
import tensorflow as tf
import logging
import json
from datetime import datetime, timedelta
import random
import os
import sys

# 設置日誌記錄
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def generate_user_profile(user_id, age_range=(60, 90), gender=None, education_level=None):
    """生成模擬用戶資料"""
    
    if gender is None:
        gender = random.choice(['男', '女'])
    
    if education_level is None:
        education_level = random.choice(['小學', '中學', '高中', '大專', '大學', '研究所'])
    
    # 設定認知程度 (1-10, 10表示沒有任何認知障礙)
    cognitive_level = max(1, min(10, np.random.normal(7, 2)))
    
    return {
        'user_id': user_id,
        'age': random.randint(*age_range),
        'gender': gender,
        'education_level': education_level,
        'cognitive_level': cognitive_level,
        'medical_history': {
            'has_dementia': cognitive_level < 5,
            'dementia_stage': 'none' if cognitive_level >= 5 else random.choice(['mild', 'moderate', 'severe']),
            'has_hypertension': random.random() < 0.4,
            'has_diabetes': random.random() < 0.3,
        },
        'registration_date': (datetime.now() - timedelta(days=random.randint(0, 365))).isoformat()
    }

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
            'attempts': np.random.randint(1, 4),
            'engagement_score': min(1.0, max(0.1, base_accuracy * 0.8 + np.random.normal(0, 0.15))),
            'frustration_level': min(1.0, max(0.0, (1.0 - base_accuracy) * 0.7 + np.random.normal(0, 0.1)))
        }
    }
    
    # 根據遊戲類型添加特定數據
    if game_type == 'attention_calculation':
        calculation_errors = []
        for _ in range(int((1.0 - accuracy) * 10)):
            error_type = random.choice(['carry', 'operation', 'number', 'random'])
            calculation_errors.append(error_type)
        
        game_data['calculation_data'] = {
            'errors': int((1.0 - accuracy) * 10),
            'total_problems': 10,
            'problems_per_minute': 60 / response_time,
            'attention_drops': np.random.randint(0, 5)
        }
        game_data['calculation_errors'] = calculation_errors
    
    return game_data

def generate_game_data_batch(user_profiles, num_samples_per_user=10, time_span_days=30):
    """為多個用戶生成模擬遊戲數據"""
    game_types = [
        'attention_calculation', 
        'time_recognition', 
        'short_term_memory', 
        'drawing', 
        'repeat_language'
    ]
    
    all_game_data = []
    
    for user in user_profiles:
        user_id = user['user_id']
        cognitive_level = user['cognitive_level']
        
        # 根據認知水平調整表現水平的概率
        high_prob = min(0.9, max(0.1, cognitive_level / 10))
        low_prob = min(0.9, max(0.1, (10 - cognitive_level) / 10))
        medium_prob = 1.0 - high_prob - low_prob
        
        # 設定表現水平分布
        performance_dist = {
            'high': high_prob,
            'medium': medium_prob,
            'low': low_prob
        }
        
        # 為該用戶生成多個遊戲數據
        for _ in range(num_samples_per_user):
            game_type = random.choice(game_types)
            
            # 隨機選擇表現水平
            performance_level = random.choices(
                ['high', 'medium', 'low'], 
                weights=[performance_dist['high'], performance_dist['medium'], performance_dist['low']]
            )[0]
            
            # 隨機時間點（過去30天內）
            random_days = random.uniform(0, time_span_days)
            timestamp = (datetime.now() - timedelta(days=random_days)).isoformat()
            
            # 生成遊戲數據
            game_data = generate_mock_game_data(
                game_type, 
                performance_level, 
                user_id
            )
            
            # 覆蓋時間戳
            game_data['timestamp'] = timestamp
            
            all_game_data.append(game_data)
    
    return all_game_data

def main():
    """主函數：生成模擬數據"""
    logging.info("開始生成模擬數據...")
    
    # 生成模擬用戶
    num_users = 5  # 較小的數據集用於測試
    user_profiles = []
    
    logging.info(f"生成{num_users}個模擬用戶資料...")
    for i in range(num_users):
        user_id = f"user_{i:03d}"
        user_profile = generate_user_profile(user_id)
        user_profiles.append(user_profile)
        logging.info(f"用戶 {user_id}: 年齡 {user_profile['age']}, 認知水平 {user_profile['cognitive_level']:.2f}")
    
    # 生成遊戲數據
    logging.info("生成模擬遊戲數據...")
    game_data_batch = generate_game_data_batch(user_profiles, num_samples_per_user=3)
    logging.info(f"共生成{len(game_data_batch)}筆遊戲數據")
    
    # 顯示一些遊戲數據示例
    logging.info("遊戲數據示例:")
    for i in range(min(3, len(game_data_batch))):
        game = game_data_batch[i]
        logging.info(f"遊戲類型: {game['game_type']}, 用戶: {game['user_id']}")
        logging.info(f"  準確率: {game['performance_metrics']['accuracy']:.2f}")
        logging.info(f"  反應時間: {game['performance_metrics']['response_time']:.2f}")
    
    # 保存模擬數據
    os.makedirs('simulated_data', exist_ok=True)
    
    with open('simulated_data/user_profiles.json', 'w', encoding='utf-8') as f:
        json.dump(user_profiles, f, ensure_ascii=False, indent=2)
    
    with open('simulated_data/game_data.json', 'w', encoding='utf-8') as f:
        json.dump(game_data_batch, f, ensure_ascii=False, indent=2)
    
    logging.info("模擬數據已保存到 simulated_data/ 目錄")

if __name__ == "__main__":
    main() 