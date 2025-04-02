import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
import logging
import random
from tensorflow.keras.optimizers import Adam


class DifficultyAdjustmentModel(Model):
    """基於強化學習的動態難度調整模型"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DifficultyAdjustmentModel, self).__init__()
        
        # 狀態編碼器 - 處理用戶表現數據
        self.state_encoder = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation='relu', input_shape=(state_dim,)),
            layers.BatchNormalization(),
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dropout(0.3)
        ])
        
        # 策略網絡 - 決定難度調整動作
        self.policy_network = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dense(action_dim, activation='softmax')  # 輸出動作概率分布
        ])
        
        # 價值網絡 - 估計狀態價值
        self.value_network = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dense(1)  # 估計狀態價值
        ])
        
        # 操作記錄
        self.action_history = []
        self.reward_history = []
        self.state_history = []
        
        # 優化器
        self.optimizer = Adam(learning_rate=0.001)
        
    def call(self, state, training=False):
        state_features = self.state_encoder(state, training=training)
        action_probs = self.policy_network(state_features, training=training)
        state_value = self.value_network(state_features, training=training)
        
        return action_probs, state_value
    
    def get_action(self, state, deterministic=False):
        """選擇難度調整動作"""
        action_probs, _ = self(tf.convert_to_tensor([state], dtype=tf.float32))
        action_probs = action_probs.numpy()[0]
        
        if deterministic:
            # 確定性策略：選擇概率最高的動作
            action = np.argmax(action_probs)
        else:
            # 隨機策略：根據概率分布抽樣
            action = np.random.choice(len(action_probs), p=action_probs)
            
        return action
    
    def store_transition(self, state, action, reward):
        """存儲轉換（用於訓練）"""
        self.state_history.append(state)
        self.action_history.append(action)
        self.reward_history.append(reward)
    
    def train(self, gamma=0.99):
        """使用策略梯度方法訓練模型"""
        if len(self.state_history) < 10:
            logging.warning("數據不足，無法進行有效訓練")
            return
        
        states = tf.convert_to_tensor(self.state_history, dtype=tf.float32)
        actions = tf.convert_to_tensor(self.action_history, dtype=tf.int32)
        rewards = tf.convert_to_tensor(self.reward_history, dtype=tf.float32)
        
        # 計算每個時間步的折扣回報
        returns = []
        discounted_sum = 0
        for r in rewards[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        
        # 標準化回報以減少方差
        returns = (returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + 1e-8)
        
        with tf.GradientTape() as tape:
            action_probs, values = self(states)
            
            # 將動作轉換為one-hot編碼
            action_masks = tf.one_hot(actions, depth=action_probs.shape[1])
            
            # 選定動作的概率
            selected_action_probs = tf.reduce_sum(action_probs * action_masks, axis=1)
            
            # 計算優勢函數：使用returns作為估計的Q值，values作為基線
            advantages = returns - tf.squeeze(values)
            
            # 策略損失
            policy_loss = -tf.reduce_mean(tf.math.log(selected_action_probs + 1e-8) * advantages)
            
            # 價值損失
            value_loss = tf.reduce_mean(tf.square(returns - tf.squeeze(values)))
            
            # 總損失
            loss = policy_loss + 0.5 * value_loss
        
        # 計算梯度並應用
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # 清空歷史
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        
        return {
            'loss': loss.numpy(),
            'policy_loss': policy_loss.numpy(),
            'value_loss': value_loss.numpy()
        }


class AdaptiveDifficultyManager:
    """適應性難度管理器：應用強化學習模型智能調整遊戲難度"""
    
    def __init__(self, game_type, difficulty_levels, state_features, model_path=None):
        """
        初始化難度管理器
        
        參數:
        - game_type: 遊戲類型
        - difficulty_levels: 可用的難度級別數
        - state_features: 狀態特徵的維度（用戶表現數據維度）
        - model_path: 預訓練模型路徑（如果有）
        """
        self.game_type = game_type
        self.difficulty_levels = difficulty_levels
        self.state_features = state_features
        self.logger = logging.getLogger(f"difficulty_manager_{game_type}")
        
        # 初始化或載入難度調整模型
        if model_path:
            try:
                self.model = tf.keras.models.load_model(model_path)
                self.logger.info(f"成功載入難度調整模型: {model_path}")
            except Exception as e:
                self.logger.error(f"載入模型時出錯: {e}")
                self.model = DifficultyAdjustmentModel(state_features, difficulty_levels)
        else:
            self.model = DifficultyAdjustmentModel(state_features, difficulty_levels)
        
        # 使用者滿意度估計器
        self.satisfaction_estimator = self._create_satisfaction_estimator()
        
        # 當前難度級別
        self.current_difficulty = self._get_initial_difficulty()
        
        # 遊戲特定難度參數
        self.difficulty_params = self._get_game_specific_params()
        
        # 用戶表現歷史
        self.performance_history = []
        
    def _create_satisfaction_estimator(self):
        """創建使用者滿意度估計器"""
        return tf.keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(self.state_features + 1,)),  # +1表示當前難度
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # 輸出滿意度估計 (0-1)
        ])
        
    def _get_initial_difficulty(self):
        """獲取初始難度設置"""
        # 默認從中等難度開始
        return self.difficulty_levels // 2
        
    def _get_game_specific_params(self):
        """獲取特定遊戲的難度參數配置"""
        
        # 各遊戲類型的特定參數配置
        game_params = {
            'attention_calculation': {
                'time_limit': [60, 50, 40, 30, 20],  # 秒
                'distraction_level': [0, 1, 2, 3, 4],  # 干擾程度
                'calculation_complexity': [1, 2, 3, 4, 5]  # 計算複雜度
            },
            'time_recognition': {
                'ambiguity_level': [0, 1, 2, 3, 4],  # 場景/聲音的模糊程度
                'options_count': [2, 3, 4, 5, 6]  # 選項數量
            },
            'short_term_memory': {
                'sequence_length': [3, 4, 5, 6, 7],  # 記憶序列長度
                'display_time': [5, 4, 3, 2, 1],  # 顯示時間（秒）
                'interference_level': [0, 1, 2, 3, 4]  # 干擾程度
            },
            'drawing': {
                'complexity': [1, 2, 3, 4, 5],  # 圖形複雜度
                'guidance_level': [4, 3, 2, 1, 0]  # 引導程度（4=高引導，0=無引導）
            },
            'repeat_language': {
                'sentence_length': [3, 5, 7, 9, 11],  # 詞語數量
                'speech_speed': [0.8, 0.9, 1.0, 1.1, 1.2],  # 語速
                'background_noise': [0, 0.1, 0.2, 0.3, 0.4]  # 背景噪音級別
            },
            'naming': {
                'image_clarity': [1.0, 0.9, 0.8, 0.7, 0.6],  # 圖像清晰度
                'object_familiarity': [1.0, 0.8, 0.6, 0.4, 0.2],  # 物體熟悉度
                'time_limit': [10, 8, 6, 4, 3]  # 回答時間限制（秒）
            },
            'object_matching': {
                'similarity_level': [0.9, 0.8, 0.7, 0.6, 0.5],  # 物體相似度
                'distractor_count': [2, 4, 6, 8, 10],  # 干擾物數量
                'arrangement_complexity': [1, 2, 3, 4, 5]  # 排列複雜度
            },
            'spatial_concept': {
                'environment_complexity': [1, 2, 3, 4, 5],  # 環境複雜度
                'landmark_visibility': [1.0, 0.8, 0.6, 0.4, 0.2],  # 地標可見度
                'path_complexity': [1, 2, 3, 4, 5]  # 路徑複雜度
            }
        }
        
        # 返回當前遊戲類型的參數，如果不存在則使用默認參數
        return game_params.get(self.game_type, {
            'time_limit': [60, 50, 40, 30, 20],
            'complexity': [1, 2, 3, 4, 5]
        })
        
    def get_difficulty_params(self):
        """獲取當前難度對應的具體參數設置"""
        
        # 獲取難度級別對應的參數值
        level = min(self.current_difficulty, self.difficulty_levels - 1)
        
        params = {}
        for param_name, param_values in self.difficulty_params.items():
            if level < len(param_values):
                params[param_name] = param_values[level]
            else:
                # 如果難度超出預設範圍，使用最難的設置
                params[param_name] = param_values[-1]
                
        return params
        
    def update_difficulty(self, performance_data, deterministic=False):
        """
        基於用戶表現更新難度
        
        參數:
        - performance_data: 用戶表現數據
        - deterministic: 是否使用確定性策略（True用於評估）
        """
        
        # 前處理性能數據，提取相關特徵
        state = self._preprocess_performance(performance_data)
        
        # 存儲性能數據
        self.performance_history.append({
            'state': state,
            'difficulty': self.current_difficulty,
            'timestamp': performance_data.get('timestamp', None)
        })
        
        # 使用模型選擇動作
        action = self.model.get_action(state, deterministic=deterministic)
        
        # 根據動作調整難度
        prev_difficulty = self.current_difficulty
        self._apply_difficulty_action(action)
        
        # 如果是訓練模式，計算獎勵並存儲轉換
        if not deterministic:
            reward = self._calculate_reward(state, action, prev_difficulty)
            self.model.store_transition(state, action, reward)
            
            # 定期訓練模型
            if len(self.model.state_history) >= 20:
                train_info = self.model.train()
                self.logger.info(f"模型訓練完成，損失: {train_info['loss']:.4f}")
                
        return self.current_difficulty, self.get_difficulty_params()
    
    def _preprocess_performance(self, performance_data):
        """從性能數據中提取狀態特徵"""
        
        # 提取基本性能指標
        accuracy = performance_data.get('accuracy', 0.5)
        completion_time = performance_data.get('completion_time', 0.0)
        attempts = performance_data.get('attempts', 1)
        
        # 計算時間效率（歸一化）
        time_limit = self.difficulty_params.get('time_limit', [30]*5)[self.current_difficulty]
        time_efficiency = 1.0 - min(completion_time / time_limit, 1.0)
        
        # 計算挫折或參與度指標
        frustration = performance_data.get('frustration_indicators', 0.0)
        engagement = performance_data.get('engagement_level', 0.5)
        
        # 提取遊戲特定指標
        if self.game_type == 'attention_calculation':
            specific_feature = performance_data.get('calculation_accuracy', accuracy)
        elif self.game_type == 'short_term_memory':
            specific_feature = performance_data.get('memory_retention', 0.5)
        else:
            specific_feature = 0.5  # 默認值
            
        # 歸一化並組合特徵
        state = [
            accuracy,
            time_efficiency,
            min(attempts / 5.0, 1.0),  # 歸一化嘗試次數
            1.0 - frustration,  # 將挫折轉換為正面指標
            engagement,
            specific_feature,
            self.current_difficulty / float(self.difficulty_levels - 1)  # 當前難度（歸一化）
        ]
        
        return np.array(state, dtype=np.float32)
    
    def _apply_difficulty_action(self, action):
        """應用難度調整動作"""
        
        # 動作類型：0=減少難度，1=保持不變，2=增加難度
        if action == 0 and self.current_difficulty > 0:
            self.current_difficulty -= 1
            self.logger.info(f"降低難度至 {self.current_difficulty}")
        elif action == 2 and self.current_difficulty < self.difficulty_levels - 1:
            self.current_difficulty += 1
            self.logger.info(f"提高難度至 {self.current_difficulty}")
        else:
            self.logger.info(f"保持當前難度 {self.current_difficulty}")
            
    def _calculate_reward(self, state, action, prev_difficulty):
        """計算動作獎勵"""
        
        # 從狀態中提取性能指標
        accuracy = state[0]
        time_efficiency = state[1]
        frustration = 1.0 - state[3]  # 轉換回挫折指標
        engagement = state[4]
        
        # 估計用戶滿意度
        satisfaction_input = np.concatenate([state, [float(self.current_difficulty)]])
        predicted_satisfaction = self.satisfaction_estimator(
            tf.convert_to_tensor([satisfaction_input], dtype=tf.float32)).numpy()[0][0]
        
        # 獎勵計算
        # 1. 準確率獎勵：準確率在0.7-0.9範圍內的獎勵最高
        accuracy_reward = -2.0 * (accuracy - 0.8) ** 2 + 1.0
        
        # 2. 挫折度懲罰
        frustration_penalty = -2.0 * frustration
        
        # 3. 參與度獎勵
        engagement_reward = engagement
        
        # 4. 難度變化獎勵/懲罰
        difficulty_change = self.current_difficulty - prev_difficulty
        
        # 如果表現太差且提高了難度，給予懲罰
        if accuracy < 0.5 and difficulty_change > 0:
            diff_reward = -1.0
        # 如果表現太好且降低了難度，給予懲罰
        elif accuracy > 0.9 and difficulty_change < 0:
            diff_reward = -1.0
        # 如果表現適中且保持難度，給予獎勵
        elif 0.6 <= accuracy <= 0.9 and difficulty_change == 0:
            diff_reward = 0.5
        # 其他情況，根據預測滿意度給獎勵
        else:
            diff_reward = predicted_satisfaction - 0.5
            
        # 總獎勵
        reward = (
            accuracy_reward * 0.4 +
            frustration_penalty * 0.2 +
            engagement_reward * 0.2 +
            diff_reward * 0.2
        )
        
        return reward
    
    def train_satisfaction_estimator(self, satisfaction_data):
        """
        使用收集的滿意度數據訓練滿意度估計器
        
        參數:
        - satisfaction_data: 形如 [(state, difficulty, satisfaction_score), ...] 的數據
        """
        if len(satisfaction_data) < 10:
            self.logger.warning("滿意度數據不足，無法進行有效訓練")
            return
            
        # 準備訓練數據
        X = []
        y = []
        
        for state, difficulty, satisfaction in satisfaction_data:
            X.append(np.concatenate([state, [float(difficulty)]]))
            y.append(satisfaction)
            
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        # 訓練滿意度估計器
        self.satisfaction_estimator.compile(
            optimizer='adam', 
            loss='mse',
            metrics=['mae']
        )
        
        history = self.satisfaction_estimator.fit(
            X, y, 
            epochs=20, 
            batch_size=8,
            validation_split=0.2,
            verbose=0
        )
        
        self.logger.info(f"滿意度估計器訓練完成，最終損失: {history.history['loss'][-1]:.4f}")
        
    def get_difficulty_recommendation(self, user_profile, initial_session=False):
        """
        基於用戶配置文件獲取難度推薦
        
        參數:
        - user_profile: 用戶配置文件，包含歷史表現和偏好
        - initial_session: 是否為初始會話
        """
        
        if initial_session or not user_profile.get('performance_history'):
            # 對於新用戶或初始會話，從基本難度開始
            age = user_profile.get('age', 65)
            cognitive_status = user_profile.get('cognitive_status', 'normal')
            
            # 基於年齡和認知狀態調整初始難度
            if cognitive_status == 'MCI' or age > 75:
                recommended_difficulty = max(0, self.difficulty_levels // 2 - 1)
            elif cognitive_status == 'mild_dementia':
                recommended_difficulty = 0  # 最低難度
            else:
                recommended_difficulty = self.difficulty_levels // 2  # 中等難度
                
            self.logger.info(f"為新用戶推薦初始難度: {recommended_difficulty}")
            return recommended_difficulty
            
        else:
            # 使用用戶的歷史表現來預測最佳難度
            recent_performances = user_profile.get('performance_history', [])[-5:]
            
            if not recent_performances:
                return self.difficulty_levels // 2
                
            # 提取最近的性能數據並計算平均表現
            avg_accuracy = np.mean([p.get('accuracy', 0.5) for p in recent_performances])
            avg_engagement = np.mean([p.get('engagement_level', 0.5) for p in recent_performances])
            
            # 基於表現預測最佳難度
            if avg_accuracy > 0.9 and avg_engagement > 0.7:
                # 表現優秀且參與度高，可以提高難度
                return min(user_profile.get('current_difficulty', 2) + 1, self.difficulty_levels - 1)
            elif avg_accuracy < 0.6 or avg_engagement < 0.4:
                # 表現不佳或參與度低，降低難度
                return max(user_profile.get('current_difficulty', 2) - 1, 0)
            else:
                # 保持當前難度
                return user_profile.get('current_difficulty', 2)


def create_difficulty_manager(game_type, num_difficulty_levels=5, state_dim=7, model_path=None):
    """創建難度管理器"""
    return AdaptiveDifficultyManager(
        game_type=game_type,
        difficulty_levels=num_difficulty_levels,
        state_features=state_dim,
        model_path=model_path
    ) 