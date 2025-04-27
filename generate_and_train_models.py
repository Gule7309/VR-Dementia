import numpy as np
import tensorflow as tf
import logging
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import sys
import random
from tqdm import tqdm

# 添加當前目錄到路徑，以便導入模型
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 導入模型管理器
from models import create_model_manager
from models.behavior_analysis_model import BehaviorAnalysisModel, create_behavior_analysis_model
from models.difficulty_adjustment_model import DifficultyAdjustmentModel, AdaptiveDifficultyManager
from models.cognitive_assessment_model import CognitiveAssessmentModel
from models.multimodal_perception_model import MultimodalPerceptionModel


def setup_logging():
    """設置日誌記錄"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('model_training.log')
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


def generate_game_data_batch(user_profiles, num_samples_per_user=10, time_span_days=30):
    """為多個用戶生成模擬遊戲數據"""
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
        
    elif game_type == 'short_term_memory':
        # 生成原始序列和用戶回答
        sequence_length = random.randint(3, 8)
        original_sequence = [random.randint(1, 9) for _ in range(sequence_length)]
        
        # 根據準確率決定用戶答對的概率
        correct_prob = accuracy
        user_response = []
        
        for num in original_sequence:
            if random.random() < correct_prob:
                user_response.append(num)  # 正確記憶
            else:
                # 錯誤記憶：隨機數字、遺漏或重複
                error_type = random.choice(['random', 'skip', 'repeat'])
                if error_type == 'random':
                    user_response.append(random.randint(1, 9))
                elif error_type == 'repeat' and user_response:
                    user_response.append(user_response[-1])
                # skip不添加任何數字
        
        game_data['memory_data'] = {
            'memory_span': sequence_length,
            'retention_rate': accuracy,
            'sequence_accuracy': len(set(original_sequence) & set(user_response)) / len(original_sequence),
            'original_sequence': original_sequence,
            'user_response': user_response
        }
        
    elif game_type == 'drawing':
        # 模擬繪畫任務的各項指標
        construction_score = base_accuracy + np.random.normal(0, 0.1)
        
        game_data['drawing_data'] = {
            'construction_score': min(1.0, max(0.1, construction_score)),
            'memory_component': min(1.0, max(0.1, construction_score * 0.8 + np.random.normal(0, 0.1))),
            'detail_attention': min(1.0, max(0.1, construction_score * 0.9 + np.random.normal(0, 0.1))),
            'spatial_arrangement': min(1.0, max(0.1, construction_score * 0.85 + np.random.normal(0, 0.1))),
            'drawing_time': max(10, (1 - construction_score) * 100 + np.random.normal(0, 10))
        }
        
    elif game_type == 'repeat_language':
        sentence_length = random.randint(5, 15)
        correct_words = int(sentence_length * accuracy)
        
        game_data['language_data'] = {
            'sentence_length': sentence_length,
            'correct_words': correct_words,
            'pronunciation_score': min(1.0, max(0.1, base_accuracy + np.random.normal(0, 0.1))),
            'memory_score': min(1.0, max(0.1, base_accuracy * 0.9 + np.random.normal(0, 0.1))),
            'response_delay': max(0.5, (1 - base_accuracy) * 5 + np.random.normal(0, 1))
        }
    
    elif game_type == 'naming':
        total_objects = random.randint(10, 20)
        correctly_named = int(total_objects * accuracy)
        
        game_data['naming_data'] = {
            'total_objects': total_objects,
            'correctly_named': correctly_named,
            'semantic_errors': int((total_objects - correctly_named) * 0.5),
            'phonological_errors': int((total_objects - correctly_named) * 0.3),
            'no_response_count': int((total_objects - correctly_named) * 0.2),
            'average_response_time': max(1.0, (1 - base_accuracy) * 10 + np.random.normal(0, 2))
        }
    
    elif game_type == 'object_matching':
        total_pairs = random.randint(5, 15)
        correctly_matched = int(total_pairs * accuracy)
        
        game_data['matching_data'] = {
            'total_pairs': total_pairs,
            'correctly_matched': correctly_matched,
            'visual_similarity_score': min(1.0, max(0.1, base_accuracy + np.random.normal(0, 0.1))),
            'conceptual_similarity_score': min(1.0, max(0.1, base_accuracy * 0.9 + np.random.normal(0, 0.1))),
            'average_decision_time': max(1.0, (1 - base_accuracy) * 8 + np.random.normal(0, 1.5))
        }
    
    elif game_type == 'spatial_concept':
        # 模擬空間概念任務
        total_trials = random.randint(5, 15)
        correct_trials = int(total_trials * accuracy)
        
        # 模擬導航路徑
        path_length = random.randint(10, 30)
        navigation_path = []
        
        # 設定一個目標點
        target_point = (random.randint(-10, 10), random.randint(-10, 10))
        
        # 生成路徑點
        current_point = (0, 0)  # 起點
        for _ in range(path_length):
            # 根據表現水平決定朝向目標移動的概率
            if random.random() < accuracy:
                # 更接近目標
                dx = np.sign(target_point[0] - current_point[0])
                dy = np.sign(target_point[1] - current_point[1])
            else:
                # 隨機方向
                dx = random.choice([-1, 0, 1])
                dy = random.choice([-1, 0, 1])
            
            current_point = (current_point[0] + dx, current_point[1] + dy)
            navigation_path.append(current_point)
        
        game_data['spatial_data'] = {
            'total_trials': total_trials,
            'correct_trials': correct_trials,
            'navigation_efficiency': min(1.0, max(0.1, base_accuracy + np.random.normal(0, 0.1))),
            'spatial_memory_score': min(1.0, max(0.1, base_accuracy * 0.85 + np.random.normal(0, 0.1))),
            'orientation_errors': int((total_trials - correct_trials) * 0.7),
            'navigation_path': navigation_path,
            'target_point': target_point
        }
    
    return game_data


def prepare_training_data_for_behavior_model(game_data_batch):
    """準備行為分析模型的訓練數據"""
    X = []
    y = []
    
    for game_data in game_data_batch:
        # 從遊戲數據提取特徵
        metrics = game_data['performance_metrics']
        
        # 構建輸入特徵
        features = {
            'response_times': np.array([metrics['response_time']] * 10).reshape(1, -1),
            'error_patterns': np.array([metrics['error_rate']] * 10).reshape(1, -1),
            'interaction_sequences': np.zeros((1, 10, 10))  # 假設序列特徵
        }
        
        # 構建標籤（這裡根據準確率和反應時間生成模擬的認知指標）
        accuracy = metrics['accuracy']
        response_time = metrics['response_time']
        
        # 模擬8個認知指標，例如：注意力、記憶力、計算能力等
        cognitive_indicators = np.zeros(8)
        cognitive_indicators[0] = min(1.0, max(0.0, accuracy * 0.8 + np.random.normal(0, 0.1)))  # 注意力
        cognitive_indicators[1] = min(1.0, max(0.0, accuracy * 0.7 + np.random.normal(0, 0.1)))  # 短期記憶
        cognitive_indicators[2] = min(1.0, max(0.0, accuracy * 0.9 + np.random.normal(0, 0.1)))  # 計算能力
        cognitive_indicators[3] = min(1.0, max(0.0, accuracy * 0.75 + np.random.normal(0, 0.1)))  # 空間感知
        cognitive_indicators[4] = min(1.0, max(0.0, accuracy * 0.85 + np.random.normal(0, 0.1)))  # 語言能力
        cognitive_indicators[5] = min(1.0, max(0.0, accuracy * 0.6 + np.random.normal(0, 0.1)))  # 執行功能
        cognitive_indicators[6] = min(1.0, max(0.0, accuracy * 0.65 + np.random.normal(0, 0.1)))  # 判斷能力
        cognitive_indicators[7] = min(1.0, max(0.0, accuracy * 0.7 + np.random.normal(0, 0.1)))  # 反應速度
        
        # 異常檢測標籤 (低準確率高反應時間視為異常)
        anomaly_label = 1.0 if (accuracy < 0.4 and response_time > 4.0) else 0.0
        
        # 添加到訓練數據
        X.append(features)
        y.append({
            'cognitive_indicators': cognitive_indicators,
            'anomaly_label': np.array([anomaly_label])
        })
    
    return X, y


def prepare_training_data_for_difficulty_model(game_data_batch):
    """準備難度調整模型的訓練數據"""
    states = []
    actions = []
    rewards = []
    
    for game_data in game_data_batch:
        metrics = game_data['performance_metrics']
        
        # 構建狀態特徵（用戶表現數據）
        state = np.array([
            metrics['accuracy'],
            metrics['response_time'],
            metrics['completion_rate'],
            metrics['attempts'],
            metrics.get('engagement_score', 0.5),
            metrics.get('frustration_level', 0.5),
            0.5  # 當前難度（歸一化到0-1之間）
        ])
        
        # 模擬動作（難度調整）: 0=降低難度，1=保持難度，2=提高難度
        if metrics['accuracy'] > 0.8 and metrics['completion_rate'] > 0.7:
            action = 2  # 提高難度
        elif metrics['accuracy'] < 0.4 or metrics['frustration_level'] > 0.7:
            action = 0  # 降低難度
        else:
            action = 1  # 保持難度
            
        # 模擬獎勵（基於用戶滿意度）
        engagement = metrics.get('engagement_score', 0.5)
        frustration = metrics.get('frustration_level', 0.5)
        reward = engagement * 0.7 - frustration * 0.3
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
    
    return np.array(states), np.array(actions), np.array(rewards)


def create_and_train_behavior_model(game_data_batch, epochs=10):
    """創建並訓練行為分析模型"""
    logging.info("準備行為分析模型訓練數據...")
    X, y = prepare_training_data_for_behavior_model(game_data_batch)
    
    logging.info("創建行為分析模型...")
    model = create_behavior_analysis_model(input_dims={'response_times': 10, 'error_patterns': 10, 'interaction_sequences': (10, 10)})
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss={
            'cognitive_indicators': 'binary_crossentropy',
            'anomaly_score': 'binary_crossentropy'
        },
        metrics=['accuracy']
    )
    
    logging.info(f"開始訓練行為分析模型（{epochs}個迭代）...")
    
    # 由於我們的數據格式比較特殊，這裡使用自定義訓練循環
    for epoch in range(epochs):
        epoch_loss = 0
        for i in tqdm(range(len(X)), desc=f"Epoch {epoch+1}/{epochs}"):
            # 手動執行一步訓練
            with tf.GradientTape() as tape:
                predictions = model(X[i], training=True)
                
                # 計算損失
                cognitive_loss = tf.keras.losses.binary_crossentropy(
                    y[i]['cognitive_indicators'], predictions['cognitive_indicators'])
                anomaly_loss = tf.keras.losses.binary_crossentropy(
                    y[i]['anomaly_label'], predictions['anomaly_score'])
                
                total_loss = cognitive_loss * 0.7 + anomaly_loss * 0.3
                
            # 計算梯度並更新模型
            gradients = tape.gradient(total_loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            epoch_loss += total_loss.numpy()
        
        avg_loss = epoch_loss / len(X)
        logging.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    
    logging.info("行為分析模型訓練完成")
    
    # 保存模型
    save_dir = 'saved_models'
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'behavior_analysis_model')
    model.save(model_path)
    logging.info(f"行為分析模型已保存到 {model_path}")
    
    return model


def create_and_train_difficulty_model(game_data_batch, epochs=10):
    """創建並訓練難度調整模型"""
    logging.info("準備難度調整模型訓練數據...")
    states, actions, rewards = prepare_training_data_for_difficulty_model(game_data_batch)
    
    logging.info("創建難度調整模型...")
    model = DifficultyAdjustmentModel(state_dim=7, action_dim=3)
    
    logging.info(f"開始訓練難度調整模型（{epochs}個迭代）...")
    
    # 強化學習訓練循環
    for epoch in range(epochs):
        epoch_loss = 0
        
        # 遍歷每個樣本
        for i in tqdm(range(len(states)), desc=f"Epoch {epoch+1}/{epochs}"):
            state = states[i]
            action = actions[i]
            reward = rewards[i]
            
            # 存儲這個轉換
            model.store_transition(state, action, reward)
            
            # 每積累5個樣本就訓練一次
            if (i + 1) % 5 == 0 or i == len(states) - 1:
                training_result = model.train()
                if training_result:
                    epoch_loss += training_result['loss']
        
        avg_loss = epoch_loss / (len(states) // 5 + 1)
        logging.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    
    logging.info("難度調整模型訓練完成")
    
    # 保存模型
    save_dir = 'saved_models'
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'difficulty_adjustment_model')
    model.save(model_path)
    logging.info(f"難度調整模型已保存到 {model_path}")
    
    return model


def main():
    """主函數：生成模擬數據並訓練模型"""
    setup_logging()
    logging.info("開始生成模擬數據並訓練模型...")
    
    # 生成模擬用戶
    num_users = 50
    user_profiles = []
    
    logging.info(f"生成{num_users}個模擬用戶資料...")
    for i in range(num_users):
        user_id = f"user_{i:03d}"
        user_profile = generate_user_profile(user_id)
        user_profiles.append(user_profile)
    
    # 生成遊戲數據
    logging.info("生成模擬遊戲數據...")
    game_data_batch = generate_game_data_batch(user_profiles, num_samples_per_user=20)
    logging.info(f"共生成{len(game_data_batch)}筆遊戲數據")
    
    # 保存模擬數據
    os.makedirs('simulated_data', exist_ok=True)
    
    with open('simulated_data/user_profiles.json', 'w', encoding='utf-8') as f:
        json.dump(user_profiles, f, ensure_ascii=False, indent=2)
    
    with open('simulated_data/game_data.json', 'w', encoding='utf-8') as f:
        json.dump(game_data_batch, f, ensure_ascii=False, indent=2)
    
    logging.info("模擬數據已保存到 simulated_data/ 目錄")
    
    # 訓練行為分析模型
    behavior_model = create_and_train_behavior_model(game_data_batch, epochs=5)
    
    # 訓練難度調整模型
    difficulty_model = create_and_train_difficulty_model(game_data_batch, epochs=5)
    
    # 使用訓練好的模型進行預測
    logging.info("使用訓練好的模型進行預測...")
    
    # 創建模型管理器（使用訓練好的模型）
    model_config = {
        'model_paths': {
            'analyzers': {
                'attention_calculation': 'saved_models/behavior_analysis_model',
                'short_term_memory': 'saved_models/behavior_analysis_model'
            },
            'difficulty_managers': {
                'attention_calculation': 'saved_models/difficulty_adjustment_model',
                'short_term_memory': 'saved_models/difficulty_adjustment_model'
            }
        }
    }
    
    # 創建模型管理器
    model_manager = create_model_manager(model_config)
    
    # 測試模型
    test_game_data = generate_mock_game_data('attention_calculation', 'medium')
    analysis_results = model_manager.analyze_game_performance(test_game_data)
    
    logging.info("行為分析結果:")
    if 'cognitive_indicators' in analysis_results:
        for i, score in enumerate(analysis_results['cognitive_indicators']):
            logging.info(f"  認知指標 {i+1}: {score:.2f}")
    
    if 'anomaly_score' in analysis_results:
        logging.info(f"  異常分數: {analysis_results['anomaly_score']:.2f}")
    
    # 測試難度調整
    performance_data = {
        'accuracy': 0.7,
        'completion_time': 30,
        'attempts': 2,
        'frustration_indicators': 0.3,
        'engagement_level': 0.8
    }
    
    adjustment = model_manager.adjust_game_difficulty('attention_calculation', performance_data)
    logging.info(f"難度調整結果: 級別 {adjustment['difficulty_level']}")
    logging.info(f"難度參數: {adjustment['difficulty_params']}")
    
    logging.info("模擬數據生成和模型訓練完成")


if __name__ == "__main__":
    main() 