import json
import numpy as np
import random
import logging
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from models import create_model_manager


# 設置日誌記錄
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# 移除中文字體設置邏輯，改為直接使用英文
class GameSimulator:
    """模擬遊戲環境，用於展示如何調整遊戲難度"""
    
    def __init__(self, user_id, game_type='attention_calculation'):
        self.user_id = user_id
        self.game_type = game_type
        self.current_difficulty = 2  # 中等難度 (範圍 0-4)
        self.session_history = []
        
        # 加載模型管理器
        self.model_manager = create_model_manager()
        
        # 記錄用戶滿意度和表現
        self.satisfaction_history = []
        self.performance_history = []
        self.difficulty_history = []
        
        # 加載用戶資料（如果有）
        self.user_profile = self._load_user_profile()
        
        # 用戶熟練度（會隨著遊戲進行而提高）
        self.user_skill = max(0.3, min(0.7, self.user_profile.get('cognitive_level', 7) / 10))
        
        # 記錄當前難度參數
        self.difficulty_params = self._get_initial_difficulty_params()
        
        logging.info(f"遊戲模擬器初始化：用戶 {user_id}，遊戲類型 {game_type}")
        logging.info(f"初始技能水平: {self.user_skill:.2f}, 初始難度: {self.current_difficulty}")
    
    def _load_user_profile(self):
        """加載用戶資料"""
        try:
            with open('simulated_data/user_profiles.json', 'r', encoding='utf-8') as f:
                user_profiles = json.load(f)
                
            for profile in user_profiles:
                if profile['user_id'] == self.user_id:
                    return profile
            
            # 如果找不到用戶，返回默認資料
            return {
                'user_id': self.user_id,
                'cognitive_level': 7.0
            }
        except Exception as e:
            logging.error(f"加載用戶資料時出錯: {e}")
            return {
                'user_id': self.user_id,
                'cognitive_level': 7.0
            }
    
    def _get_initial_difficulty_params(self):
        """獲取初始難度參數"""
        if self.game_type == 'attention_calculation':
            return {
                'time_limit': 40,  # 秒
                'distraction_level': 2,  # 干擾程度
                'calculation_complexity': 3  # 計算複雜度
            }
        elif self.game_type == 'short_term_memory':
            return {
                'sequence_length': 5,  # 記憶序列長度
                'display_time': 3,  # 顯示時間（秒）
                'interference_level': 2  # 干擾程度
            }
        else:
            return {
                'time_limit': 30,
                'complexity': 2
            }
    
    def simulate_game_round(self):
        """模擬一輪遊戲，根據當前難度和用戶技能獲取模擬表現"""
        
        # 模擬用戶表現（受難度和技能水平影響）
        # 難度越高，表現越差；技能越高，表現越好
        difficulty_factor = self.current_difficulty / 4.0  # 歸一化到0-1
        
        # 基礎準確率（受難度和技能共同影響）
        base_accuracy = max(0.3, min(0.95, 
            self.user_skill * (1.0 - 0.5 * difficulty_factor) + 
            random.normalvariate(0, 0.1)
        ))
        
        # 計算挫折和投入程度（難度太高或太低都會降低投入）
        challenge_match = 1.0 - abs(difficulty_factor - self.user_skill)
        frustration = max(0.0, min(1.0, 
            (difficulty_factor - self.user_skill) * 2 + 
            random.normalvariate(0, 0.1)
        ))
        engagement = max(0.1, min(0.9, 
            challenge_match * 0.8 + 
            random.normalvariate(0, 0.1)
        ))
        
        # 計算滿意度
        satisfaction = max(0.0, min(1.0,
            engagement * 0.7 - frustration * 0.5 +
            random.normalvariate(0, 0.05)
        ))
        
        # 生成表現數據
        performance_data = {
            'accuracy': base_accuracy,
            'completion_time': max(5, 30 + (difficulty_factor * 30) - (self.user_skill * 20) + random.normalvariate(0, 5)),
            'attempts': max(1, min(5, int((1 - base_accuracy) * 5 + random.normalvariate(0, 1)))),
            'frustration_indicators': frustration,
            'engagement_level': engagement,
            'timestamp': datetime.now().isoformat()
        }
        
        # 記錄歷史
        self.performance_history.append(performance_data['accuracy'])
        self.satisfaction_history.append(satisfaction)
        self.difficulty_history.append(self.current_difficulty)
        
        logging.info(f"遊戲回合結果 - 難度: {self.current_difficulty}")
        logging.info(f"  準確率: {performance_data['accuracy']:.2f}")
        logging.info(f"  完成時間: {performance_data['completion_time']:.1f}秒")
        logging.info(f"  挫折程度: {frustration:.2f}")
        logging.info(f"  投入程度: {engagement:.2f}")
        logging.info(f"  滿意度: {satisfaction:.2f}")
        
        # 用戶技能略微提升（學習效果）
        self.user_skill = min(0.95, self.user_skill + 0.01)
        
        return performance_data, satisfaction
    
    def adjust_difficulty(self, performance_data):
        """使用模型調整難度"""
        
        # 使用模型管理器調整難度
        adjustment = self.model_manager.adjust_game_difficulty(
            self.game_type, performance_data, deterministic=True)
        
        # 獲取新難度和參數
        new_difficulty = adjustment['difficulty_level']
        new_params = adjustment['difficulty_params']
        
        # 更新當前難度
        old_difficulty = self.current_difficulty
        self.current_difficulty = new_difficulty
        self.difficulty_params = new_params
        
        logging.info(f"難度調整: {old_difficulty} -> {new_difficulty}")
        logging.info(f"新參數: {new_params}")
        
        return new_difficulty, new_params
    
    def analyze_performance(self):
        """分析用戶表現"""
        
        # 模擬一輪遊戲
        game_data = {
            'user_id': self.user_id,
            'game_type': self.game_type,
            'timestamp': datetime.now().isoformat(),
            'performance_metrics': {
                'accuracy': self.performance_history[-1],
                'response_time': 5.0 - self.performance_history[-1] * 3,
                'completion_rate': self.performance_history[-1] * 0.8,
                'error_rate': 1.0 - self.performance_history[-1],
                'attempts': 1
            }
        }
        
        # 使用行為分析模型分析表現
        analysis_results = self.model_manager.analyze_game_performance(game_data)
        
        logging.info("行為分析結果:")
        if 'cognitive_indicators' in analysis_results:
            # 修正：處理numpy數組，將其轉換為普通列表
            indicators = analysis_results['cognitive_indicators']
            if hasattr(indicators, 'numpy'):
                indicators = indicators.numpy()
            
            if isinstance(indicators, np.ndarray):
                indicators = indicators.flatten().tolist()
            
            for i, score in enumerate(indicators):
                logging.info(f"  認知指標 {i+1}: {float(score):.2f}")
        
        if 'anomaly_score' in analysis_results:
            # 修正：處理anomaly_score，確保它是標量
            anomaly = analysis_results['anomaly_score']
            if hasattr(anomaly, 'numpy'):
                anomaly = anomaly.numpy()
            
            if isinstance(anomaly, np.ndarray):
                anomaly = float(anomaly.item() if anomaly.size == 1 else anomaly[0])
            
            logging.info(f"  異常分數: {float(anomaly):.2f}")
        
        if 'interpretations' in analysis_results:
            logging.info(f"  解釋: {analysis_results['interpretations']}")
        
        return analysis_results
    
    def visualize_session(self):
        """視覺化遊戲會話數據"""
        # 使用英文標題和標籤
        plt.figure(figsize=(12, 8))
        
        # 繪製難度變化
        plt.subplot(3, 1, 1)
        plt.plot(self.difficulty_history, 'r-', linewidth=2)
        plt.title('Game Difficulty Changes')
        plt.ylabel('Difficulty Level')
        plt.ylim(0, 4)
        plt.grid(True)
        
        # 繪製表現變化
        plt.subplot(3, 1, 2)
        plt.plot(self.performance_history, 'g-', linewidth=2)
        plt.title('User Performance Changes')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.grid(True)
        
        # 繪製滿意度變化
        plt.subplot(3, 1, 3)
        plt.plot(self.satisfaction_history, 'b-', linewidth=2)
        plt.title('User Satisfaction Changes')
        plt.ylabel('Satisfaction')
        plt.xlabel('Game Round')
        plt.ylim(0, 1)
        plt.grid(True)
        
        plt.tight_layout()
        
        # 保存圖表
        os.makedirs('results', exist_ok=True)
        plt.savefig(f'results/{self.user_id}_{self.game_type}_session.png')
        plt.close()
        
        logging.info(f"會話視覺化已保存到 results/{self.user_id}_{self.game_type}_session.png")

    def create_combined_visualization(user_ids, game_type='attention_calculation'):
        """創建比較多個用戶的綜合視覺化圖表"""
        fig, axs = plt.subplots(3, 1, figsize=(14, 10))
        
        # 顏色映射
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        
        # 難度變化
        ax = axs[0]
        for i, user_id in enumerate(user_ids):
            try:
                # 嘗試加載用戶會話數據
                with open(f'results/{user_id}_session_data.json', 'r') as f:
                    data = json.load(f)
                    ax.plot(data['difficulty_history'], f'{colors[i % len(colors)]}-', 
                            linewidth=2, label=f'User {user_id}')
            except Exception as e:
                logging.error(f"加載用戶 {user_id} 數據失敗: {e}")
        
        ax.set_title('Game Difficulty Changes Comparison')
        ax.set_ylabel('Difficulty Level')
        ax.set_ylim(0, 4)
        ax.grid(True)
        ax.legend()
        
        # 表現變化
        ax = axs[1]
        for i, user_id in enumerate(user_ids):
            try:
                with open(f'results/{user_id}_session_data.json', 'r') as f:
                    data = json.load(f)
                    ax.plot(data['performance_history'], f'{colors[i % len(colors)]}-', 
                            linewidth=2, label=f'User {user_id}')
            except Exception as e:
                logging.error(f"加載用戶 {user_id} 數據失敗: {e}")
        
        ax.set_title('User Performance Comparison')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend()
        
        # 滿意度變化
        ax = axs[2]
        for i, user_id in enumerate(user_ids):
            try:
                with open(f'results/{user_id}_session_data.json', 'r') as f:
                    data = json.load(f)
                    ax.plot(data['satisfaction_history'], f'{colors[i % len(colors)]}-', 
                            linewidth=2, label=f'User {user_id}')
            except Exception as e:
                logging.error(f"加載用戶 {user_id} 數據失敗: {e}")
        
        ax.set_title('User Satisfaction Comparison')
        ax.set_ylabel('Satisfaction')
        ax.set_xlabel('Game Round')
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend()
        
        plt.tight_layout()
        
        # 保存圖表
        os.makedirs('results', exist_ok=True)
        plt.savefig(f'results/combined_{game_type}_comparison.png')
        plt.close()
        
        logging.info(f"綜合比較視覺化已保存到 results/combined_{game_type}_comparison.png")

    def save_session_data(self):
        """保存會話數據到JSON文件，用於後續分析"""
        session_data = {
            'user_id': self.user_id,
            'game_type': self.game_type,
            'user_skill': self.user_skill,
            'difficulty_history': self.difficulty_history,
            'performance_history': self.performance_history,
            'satisfaction_history': self.satisfaction_history,
            'timestamp': datetime.now().isoformat()
        }
        
        os.makedirs('results', exist_ok=True)
        with open(f'results/{self.user_id}_session_data.json', 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2)
        
        logging.info(f"會話數據已保存到 results/{self.user_id}_session_data.json")


def simulate_user(user_id, game_type='attention_calculation', rounds=30):
    """為單個用戶運行模擬"""
    simulator = GameSimulator(user_id, game_type)
    
    for round_num in range(rounds):
        logging.info(f"\n===== 用戶 {user_id} - 回合 {round_num + 1} =====")
        
        # 模擬遊戲
        performance_data, satisfaction = simulator.simulate_game_round()
        
        # 每3輪進行一次行為分析
        if (round_num + 1) % 3 == 0:
            simulator.analyze_performance()
        
        # 調整難度
        simulator.adjust_difficulty(performance_data)
        
        # 暫停一小段時間
        time.sleep(0.2)
    
    # 視覺化結果
    simulator.visualize_session()
    
    # 保存會話數據
    simulator.save_session_data()
    
    return simulator


def main():
    """主函數：展示如何使用模型調整遊戲難度"""
    game_type = "attention_calculation"
    
    # 定義4個不同認知水平的用戶
    users = ["user_000", "user_001", "user_002", "user_003"]
    
    # 為每個用戶運行模擬
    for user_id in users:
        simulate_user(user_id, game_type, rounds=30)
    
    # 創建綜合比較視覺化
    GameSimulator.create_combined_visualization(users, game_type)
    
    logging.info("所有用戶模擬完成！")


if __name__ == "__main__":
    main() 