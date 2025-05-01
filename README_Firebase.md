# Firebase數據庫整合說明

本文檔說明如何將VR認知訓練系統與Firebase數據庫整合，以實現長期用戶追蹤、多設備同步、臨床研究整合等進階功能。

## 整合概述

Firebase整合方案提供以下核心功能：

1. **長期用戶數據存儲**：追蹤數月甚至數年的認知訓練歷史，支持失智症預防效果研究
2. **多用戶管理和比較**：管理數十至數百位用戶，進行用戶分組、篩選和比較分析
3. **臨床研究數據整合**：將VR認知訓練數據與醫療機構的臨床測量結果關聯
4. **多設備數據同步**：確保用戶在不同設備（家中、診所等）都能訪問完整訓練歷史
5. **複雜報告生成**：生成多維度分析報告，整合多種遊戲類型的表現

## 文件結構

Firebase整合包含以下核心文件：

- `firebase_data_manager.py`：Firebase數據管理類，提供基本CRUD操作
- `firebase_utils.py`：實用工具函數，將GameDataCollector與Firebase整合
- 依賴庫：firebase-admin（Python Firebase SDK）

## 安裝與設置

### 1. 安裝依賴

```bash
pip install firebase-admin
```

### 2. 獲取Firebase認證

1. 前往 [Firebase控制台](https://console.firebase.google.com/)
2. 創建新項目或選擇現有項目
3. 在「項目設置」>「服務帳號」中生成新的私鑰
4. 下載私鑰JSON文件並安全保存

### 3. 初始化Firebase

```python
from firebase_data_manager import FirebaseDataManager

# 方法1：通過憑證文件初始化
firebase = FirebaseDataManager('path/to/your/credentials.json')

# 方法2：通過環境變量初始化（推薦用於生產環境）
# 先設置環境變量：GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/credentials.json
firebase = FirebaseDataManager()
```

## 數據模型

Firebase數據庫使用以下集合（Collections）組織數據：

1. **users**: 存儲用戶基本信息
   - 子集合 `session_history`: 用戶會話歷史
   - 子集合 `clinical_data`: 臨床數據索引

2. **sessions**: 完整會話數據
   - 子集合 `performances`: 單次遊戲表現

3. **game_performances**: 所有遊戲表現數據

4. **cognitive_assessments**: 認知評估結果

5. **clinical_data**: 臨床研究數據

6. **long_term_progress**: 長期進度追蹤

7. **assessment_reports**: 評估報告

## 使用指南

### 基本用戶管理

```python
from schemas import UserProfile, CognitiveStatus
from firebase_data_manager import FirebaseDataManager

# 初始化
firebase = FirebaseDataManager('path/to/credentials.json')

# 創建用戶
user_profile = UserProfile(
    user_id="user123",
    age=65,
    gender="female",
    education_level="bachelor",
    cognitive_status=CognitiveStatus.NORMAL,
    medical_conditions=["hypertension"]
)
firebase.create_user(user_profile)

# 更新用戶信息
firebase.update_user("user123", {
    'age': 66,
    'cognitive_status': CognitiveStatus.MCI.value
})

# 獲取用戶信息
user_data = firebase.get_user("user123")
```

### 遊戲數據收集

將原有的GameDataCollector替換為Firebase版本：

```python
from firebase_utils import FirebaseGameDataCollector

# 初始化
collector = FirebaseGameDataCollector('path/to/credentials.json')

# 使用方式與原GameDataCollector相同，但數據會同步到Firebase
session_id = collector.start_session(user_id, game_type)

# 記錄表現
round_id = f"{session_id}_round_{round_num}"
game_data = collector.record_game_performance(session_id, round_id, performance_data)

# 結束並保存會話
session = collector.end_session(session_id)
```

### 數據查詢與分析

```python
from datetime import datetime, timedelta
from firebase_data_manager import FirebaseDataManager

firebase = FirebaseDataManager()

# 獲取用戶的遊戲會話歷史
six_months_ago = datetime.now() - timedelta(days=180)
sessions = firebase.get_user_sessions(
    user_id="user123",
    start_date=six_months_ago,
    limit=100
)

# 獲取認知評估歷史
assessments = firebase.get_user_assessments("user123")

# 查詢長期進展數據
progress = firebase.get_user_longitudinal_data("user123", months=12)
```

### 臨床研究支持

```python
# 保存外部臨床數據
clinical_data = {
    'test_type': 'MMSE',
    'score': 28,
    'clinician': 'Dr. Lee',
    'location': 'Memory Clinic',
    'notes': '患者表現穩定'
}
firebase.save_clinical_data("user123", "cognitive_test", clinical_data)

# 獲取用戶的臨床數據
clinical_records = firebase.get_user_clinical_data("user123", "cognitive_test")
```

### 用戶組比較分析

```python
# 比較不同年齡組的用戶
comparison_data = firebase.compare_user_groups({
    '老年組': {'age': ['>=', 65]},
    '中年組': {'age': ['>=', 45], 'age': ['<', 65]}
})

# 分析結果包含每組的平均表現、認知領域評分等
for group_name, data in comparison_data['groups'].items():
    print(f"{group_name}: {data['user_count']}人, 平均年齡: {data['avg_age']}")
    for game_type, perf in data['avg_performance'].items():
        print(f"  {game_type}: {perf:.2f}")
```

### 數據遷移

如果您有現有的本地數據需要遷移到Firebase：

```python
from firebase_utils import migrate_to_firebase

# 執行一次性遷移
migrate_to_firebase()
```

## 安全與性能優化

### 安全實踐

1. **避免硬編碼的認證信息**：使用環境變量或安全存儲
2. **設置Firebase安全規則**：限制數據訪問權限
3. **敏感數據加密**：針對醫療和個人識別信息

### 性能優化

1. **批量操作**：使用批量寫入減少網絡請求
2. **索引設計**：為常用查詢添加索引
3. **分頁加載**：大數據集使用分頁載入
4. **離線支持**：啟用Firebase離線功能

## 常見問題

1. **如果遇到認證失敗**：
   - 檢查認證文件路徑
   - 確認專案ID是否正確
   - 驗證服務帳號是否有適當權限

2. **數據同步延遲**：
   - Firebase有設計的延遲以批量處理更新
   - 關鍵操作後可手動等待

3. **錯誤處理**：
   - 所有API調用都應包含在try-except區塊中
   - 記錄詳細錯誤信息以便診斷

## 進階功能

1. **大型資料匯出**：使用批量匯出功能進行研究分析
2. **自動化報告**：設置定期生成用戶進展報告
3. **機器學習整合**：導出數據到TensorFlow或PyTorch進行預測分析 