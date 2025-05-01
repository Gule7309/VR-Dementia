"""
數據收集示例：展示如何在遊戲中收集數據並轉換為模型輸入格式
"""

import os
import time
import random
from datetime import datetime
import json

from schemas import (
    GameDataCollector, GameType, ModelInputConverter,
    load_session_data, batch_process_game_data
)


def simulate_attention_calculation_game(user_id, difficulty_level):
    """模擬注意力計算遊戲的過程和數據生成"""
    
    # 用戶技能水平（0-1）
    user_skill = random.uniform(0.4, 0.9)
    
    # 根據難度和技能計算表現指標
    base_accuracy = max(0.3, min(0.95, user_skill - 0.1 * difficulty_level + random.uniform(-0.1, 0.1)))
    base_response_time = max(1.0, 2.0 + difficulty_level * 1.5 - user_skill * 2 + random.uniform(-0.5, 1.5))
    
    # 根據難度生成錯誤數量
    errors_count = int(max(0, (1 - base_accuracy) * 10 + random.randint(0, 3)))
    
    # 準備表現數據
    performance_data = {
        'accuracy': base_accuracy,
        'completion_time': base_response_time * 10,  # 總完成時間
        'correct_responses': int(10 * base_accuracy),
        'incorrect_responses': errors_count,
        'missed_responses': max(0, 10 - int(10 * base_accuracy) - errors_count),
        'average_response_time': base_response_time,
        'frustration_indicators': max(0, min(1, (difficulty_level / 4) - user_skill + 0.3 + random.uniform(-0.2, 0.2))),
        'engagement_level': max(0, min(1, 0.7 - abs(difficulty_level / 4 - user_skill) + random.uniform(-0.1, 0.1))),
        'satisfaction_level': max(0, min(1, 0.8 - abs(difficulty_level / 4 - user_skill) - 0.1 * difficulty_level + random.uniform(-0.1, 0.1)))
    }
    
    return performance_data, user_skill


def simulate_memory_game(user_id, difficulty_level):
    """模擬短期記憶遊戲的過程和數據生成"""
    
    # 用戶技能水平（0-1）
    user_skill = random.uniform(0.4, 0.9)
    
    # 根據難度和技能計算表現指標
    base_accuracy = max(0.3, min(0.95, user_skill - 0.12 * difficulty_level + random.uniform(-0.1, 0.1)))
    base_response_time = max(1.0, 1.5 + difficulty_level * 1.2 - user_skill * 1.5 + random.uniform(-0.5, 1.0))
    
    # 準備表現數據
    performance_data = {
        'accuracy': base_accuracy,
        'completion_time': base_response_time * 8,  # 總完成時間
        'correct_responses': int(8 * base_accuracy),
        'incorrect_responses': int(8 * (1 - base_accuracy)),
        'missed_responses': 0,
        'average_response_time': base_response_time,
        'frustration_indicators': max(0, min(1, (difficulty_level / 4) - user_skill + 0.2 + random.uniform(-0.2, 0.2))),
        'engagement_level': max(0, min(1, 0.8 - abs(difficulty_level / 4 - user_skill) + random.uniform(-0.1, 0.1))),
        'satisfaction_level': max(0, min(1, 0.7 - abs(difficulty_level / 4 - user_skill) - 0.05 * difficulty_level + random.uniform(-0.1, 0.1)))
    }
    
    return performance_data, user_skill


def main():
    """主函數：展示數據收集流程"""
    
    # 創建輸出目錄
    os.makedirs('data', exist_ok=True)
    
    # 創建資料收集器
    collector = GameDataCollector()
    
    # 模擬注意力計算遊戲會話
    user_id = f"user_{random.randint(1000, 9999)}"
    game_type = GameType.ATTENTION_CALCULATION.value
    
    print(f"開始為用戶 {user_id} 收集 {game_type} 數據...")
    
    # 開始會話
    session_id = collector.start_session(user_id, game_type)
    
    # 模擬5個回合的遊戲
    difficulty_level = 2  # 起始難度
    user_skill = None
    
    for round_num in range(1, 6):
        print(f"  回合 {round_num}，難度 {difficulty_level}")
        
        # 模擬遊戲表現
        performance_data, user_skill = simulate_attention_calculation_game(user_id, difficulty_level)
        
        # 記錄遊戲表現
        round_id = f"{session_id}_round_{round_num}"
        game_data = collector.record_game_performance(session_id, round_id, performance_data)
        
        # 根據表現調整難度
        if performance_data['accuracy'] > 0.8:
            difficulty_level = min(4, difficulty_level + 1)
            print(f"  表現優秀，難度提高到 {difficulty_level}")
        elif performance_data['accuracy'] < 0.4:
            difficulty_level = max(0, difficulty_level - 1)
            print(f"  表現不佳，難度降低到 {difficulty_level}")
        
        # 記錄難度調整
        collector.record_difficulty_adjustment(session_id, difficulty_level, {
            'time_limit': 60 - difficulty_level * 10,
            'distraction_level': difficulty_level,
            'calculation_complexity': difficulty_level + 1
        })
        
        # 轉換為模型輸入
        behavior_input = ModelInputConverter.prepare_behavior_analysis_input(game_data)
        difficulty_input = ModelInputConverter.prepare_difficulty_adjustment_input(
            game_data.performance_metrics, difficulty_level)
        
        print(f"  表現指標: 準確率={performance_data['accuracy']:.2f}, 完成時間={performance_data['completion_time']:.1f}秒")
        print(f"  情緒指標: 挫折感={performance_data['frustration_indicators']:.2f}, " 
              f"參與度={performance_data['engagement_level']:.2f}, "
              f"滿意度={performance_data['satisfaction_level']:.2f}")
        
        # 模擬遊戲延遲
        time.sleep(0.5)
    
    # 結束會話
    session = collector.end_session(session_id)
    
    # 將用戶技能記錄到會話
    session.user_skill = user_skill
    
    # 保存會話數據
    output_file = f"data/{session_id}.json"
    collector.save_session_data(session_id, output_file)
    print(f"會話數據已保存到 {output_file}")
    
    # 從文件載入會話數據
    loaded_session = load_session_data(output_file)
    print(f"從文件載入的用戶技能: {loaded_session.user_skill:.2f}")
    print(f"難度歷史: {loaded_session.difficulty_history}")
    print(f"表現歷史: {[f'{p:.2f}' for p in loaded_session.performance_history]}")
    
    # 模擬短期記憶遊戲會話
    memory_user_id = f"user_{random.randint(1000, 9999)}"
    memory_game_type = GameType.SHORT_TERM_MEMORY.value
    
    print(f"\n開始為用戶 {memory_user_id} 收集 {memory_game_type} 數據...")
    
    # 開始會話
    memory_session_id = collector.start_session(memory_user_id, memory_game_type)
    
    # 模擬3個回合的遊戲
    difficulty_level = 2  # 起始難度
    
    for round_num in range(1, 4):
        print(f"  回合 {round_num}，難度 {difficulty_level}")
        
        # 模擬遊戲表現
        performance_data, user_skill = simulate_memory_game(memory_user_id, difficulty_level)
        
        # 記錄遊戲表現
        round_id = f"{memory_session_id}_round_{round_num}"
        game_data = collector.record_game_performance(memory_session_id, round_id, performance_data)
        
        # 根據表現調整難度
        if performance_data['accuracy'] > 0.75:
            difficulty_level = min(4, difficulty_level + 1)
            print(f"  表現優秀，難度提高到 {difficulty_level}")
        elif performance_data['accuracy'] < 0.4:
            difficulty_level = max(0, difficulty_level - 1)
            print(f"  表現不佳，難度降低到 {difficulty_level}")
        
        # 記錄難度調整
        collector.record_difficulty_adjustment(memory_session_id, difficulty_level, {
            'sequence_length': 3 + difficulty_level,
            'display_time': 5 - difficulty_level,
            'interference_level': difficulty_level
        })
        
        print(f"  表現指標: 準確率={performance_data['accuracy']:.2f}, 完成時間={performance_data['completion_time']:.1f}秒")
        
        # 模擬遊戲延遲
        time.sleep(0.5)
    
    # 結束會話
    memory_session = collector.end_session(memory_session_id)
    memory_session.user_skill = user_skill
    
    # 保存會話數據
    memory_output_file = f"data/{memory_session_id}.json"
    collector.save_session_data(memory_session_id, memory_output_file)
    print(f"會話數據已保存到 {memory_output_file}")
    
    # 批處理數據
    batch_process_game_data([output_file, memory_output_file], "data/combined_sessions.json")
    print(f"已處理會話數據並保存到 data/combined_sessions.json")
    
    # 顯示組合數據
    with open("data/combined_sessions.json", 'r', encoding='utf-8') as f:
        combined_data = json.load(f)
        print("\n組合數據摘要:")
        for session in combined_data:
            print(f"  用戶: {session['user_id']}, 遊戲: {session['game_type']}")
            print(f"  會話時長: {session['session_duration']:.1f}分鐘")
            print(f"  平均表現: {session['avg_performance']:.2f}")
            print(f"  平均滿意度: {session['avg_satisfaction']:.2f}")
            print(f"  最終難度: {session['final_difficulty']}")
            print()


if __name__ == "__main__":
    main() 