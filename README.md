# VR樂園「憶」壽延年 深度學習模型套件

這個套件包含為Apple Vision Pro平台開發的「VR樂園『憶』壽延年」項目設計的一系列深度學習模型。這些模型專為失智症預防、延緩和復健設計，支持各種遊戲功能的智能化和個人化。

## 模型概覽

套件包含五個核心模型，每個模型負責特定的功能：

1. **用戶行為分析模型** (`behavior_analysis_model.py`)
   - 分析用戶在各種遊戲中的表現模式
   - 識別潛在的認知功能退化信號
   - 為醫生評估系統提供數據支援

2. **動態難度調整模型** (`difficulty_adjustment_model.py`)
   - 基於用戶的實時表現調整遊戲難度
   - 使用強化學習確保適當的挑戰性
   - 個性化訓練體驗

3. **多模態感知模型** (`multimodal_perception_model.py`)
   - 處理視覺輸入（圖形識別、物體配對）
   - 處理聽覺輸入（聲音分類、語音識別）
   - 時空理解與推理

4. **認知功能評估模型** (`cognitive_assessment_model.py`)
   - 評估短期記憶能力
   - 評估注意力和計算能力
   - 評估空間認知能力

5. **數據整合與報告生成模型** (`report_generation_model.py`)
   - 整合各個遊戲的表現數據
   - 生成趨勢分析
   - 提供醫生評估建議

## 目錄結構

```
models/
  ├── __init__.py               # 模型管理器和初始化代碼
  ├── behavior_analysis_model.py    # 用戶行為分析模型
  ├── difficulty_adjustment_model.py # 動態難度調整模型
  ├── multimodal_perception_model.py # 多模態感知模型
  ├── cognitive_assessment_model.py  # 認知功能評估模型
  ├── report_generation_model.py     # 報告生成模型
  ├── test_models.py                # 模型測試腳本
  └── README.md                     # 模型文檔
```

## 系統運作邏輯

整個系統採用模組化設計，由5個核心模型通過`ModelManager`統一管理：

```
   使用者VR互動
       ↓   ↑
+-------------------------------+
|        ModelManager           |
+-------------------------------+
    ↓     ↓     ↓     ↓     ↓
+---+ +-----+ +----+ +----+ +----+
|行為| |難度 | |多模| |認知| |報告|
|分析| |調整 | |感知| |評估| |生成|
+---+ +-----+ +----+ +----+ +----+
```

### 資料流向

1. **輸入層**：從VR遊戲收集用戶互動資料（反應時間、準確率等）
2. **處理層**：通過模型分析和處理資料
3. **輸出層**：向VR系統返回分析結果、調整建議、評估報告等

### 模型管理器 (ModelManager)

`ModelManager`是整個系統的核心，負責：
- 初始化並載入所有模型
- 提供統一的API介面給VR應用調用
- 管理模型之間的資料交換
- 緩存使用者資料和模型狀態

### 核心模型運作邏輯

1. **用戶行為分析模型**
   ```
   遊戲數據 → 特徵提取 → 特徵分析 → 認知指標預測 → 解釋生成
   ```
   - 從不同遊戲中提取時間特徵、錯誤模式和互動序列
   - 使用深度神經網絡分析用戶表現模式
   - 輸出注意力、記憶力等認知指標和異常分數
   - 根據遊戲類型和分析結果生成人類可理解的解釋

2. **動態難度調整模型**
   ```
   表現數據 → 評估當前表現 → 決策網絡 → 新難度參數
   ```
   - 根據準確率、完成時間和挫折指標評估用戶表現
   - 使用強化學習找到最佳難度級別
   - 為不同遊戲類型生成具體難度參數（時間限制、干擾度等）
   - 考慮歷史表現趨勢，避免頻繁變化

3. **多模態感知模型**
   ```
   視覺/聽覺輸入 → 模態特定處理 → 多模態融合 → 感知結果
   ```
   - 處理圖像識別、物體配對等視覺任務
   - 處理語音識別和聲音分類等聽覺任務
   - 整合不同模態的資訊
   - 針對不同任務提供專門處理

4. **認知功能評估模型**
   ```
   遊戲數據 → 領域特徵提取 → 認知評估 → 評分與解釋 → 趨勢分析
   ```
   - 從遊戲資料中提取認知相關特徵
   - 評估注意力、記憶力、語言能力、視空間能力和執行功能
   - 將數值評分轉換為臨床解釋（正常、輕度、中度、嚴重）
   - 記錄用戶歷史表現，計算認知趨勢

5. **報告生成模型**
   ```
   歷史資料 → 特徵融合 → 報告生成 → 圖表視覺化
   ```
   - 整合多個遊戲的表現資料和評估結果
   - 生成摘要、建議和趨勢分析
   - 生成表現趨勢圖和認知領域雷達圖
   - 提供針對性的訓練建議

### 系統運行流程

1. **遊戲開始**：初始化模型管理器和所需模型
2. **遊戲進行中**：
   - 收集用戶表現數據
   - 使用行為分析模型分析認知狀況
   - 使用難度調整模型更新遊戲難度
3. **遊戲結束後**：
   - 使用認知評估模型評估本次表現
   - 更新用戶歷史資料
4. **定期評估**：
   - 整合多次遊戲數據
   - 生成綜合認知評估
   - 使用報告生成模型產生評估報告

## 使用方法

### 初始化模型

```python
from models import create_model_manager

# 創建模型管理器
model_manager = create_model_manager()
```

### 分析遊戲行為

```python
# 遊戲數據
game_data = {
    'game_type': 'attention_calculation',
    'performance_metrics': {
        'accuracy': 0.75,
        'response_time': 3.2,
        'completion_rate': 0.8
    },
    # 其他遊戲特定數據...
}

# 分析行為
analysis_results = model_manager.analyze_game_performance(game_data)
print(analysis_results)
```

### 調整遊戲難度

```python
# 用戶表現數據
performance_data = {
    'accuracy': 0.8,
    'completion_time': 25,
    'attempts': 2,
    'frustration_indicators': 0.2,
    'engagement_level': 0.7
}

# 調整難度
adjustment = model_manager.adjust_game_difficulty('short_term_memory', performance_data)
print(f"新難度: {adjustment['difficulty_level']}")
print(f"難度參數: {adjustment['difficulty_params']}")
```

### 處理多模態輸入

```python
# 物體配對任務
matching_data = {
    'reference_image': reference_img,  # 參考物體圖像
    'candidate_images': [candidate1, candidate2, candidate3]  # 候選物體圖像
}
match_result = model_manager.process_multimodal_input('object_matching', matching_data)

# 語音識別
speech_result = model_manager.process_multimodal_input('speech_recognition', audio_data)
```

### 評估認知功能

```python
# 評估單個遊戲表現
assessment = model_manager.assess_cognitive_function('user123', game_data)
print(f"域評分: {assessment['domain_scores']}")
print(f"整體評分: {assessment['overall_score']}")

# 獲取綜合評估
comprehensive = model_manager.get_comprehensive_assessment('user123')
print(f"認知趨勢: {comprehensive['trends']}")
```

### 生成評估報告

```python
# 生成報告
report = model_manager.generate_assessment_report('user123', '張三', game_data_list)
print(f"摘要: {report['summary']}")
print(f"建議: {report['recommendations']}")

# 報告中包含圖表
if 'charts' in report:
    for chart_name, chart_data in report['charts'].items():
        # 顯示或保存圖表...
        pass
```

## 測試模型

```bash
# 運行測試腳本
python -m models.test_models
```

## 技術實現

這些模型使用TensorFlow實現，並針對VR環境優化。模型設計考慮了以下因素：
- 即時性：確保VR環境中的低延遲反饋
- 可解釋性：提供對模型決策的解釋，特別是醫生評估部分
- 隱私保護：確保用戶數據安全處理
- 個性化：適應不同用戶的需求和認知特點

## 環境要求

- Python 3.8+
- TensorFlow 2.5+
- NumPy
- Pandas
- Matplotlib (用於報告生成) 