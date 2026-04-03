"""
手語即時辨識調整檔
==================

這個檔案放在專案根目錄，`run_sentence_interface.py` 會在啟動時自動讀取。

使用方式：
1. 直接修改下面的數值
2. 存檔
3. 重新啟動介面

注意：
- 只需要改你想調的項目，不想改的可以保留原樣
- 路徑請使用完整 Windows 路徑字串，前面加 `r"..."` 最安全
- 若設定檔語法錯誤，程式會退回內建預設值
"""


# 模式預設覆寫
# -------------
# 這裡用來改每個模式的：
# - checkpoint 模型檔
# - sequence_length 序列長度
# - confidence 模型原始信心門檻
# - mirror_input 是否鏡像
# - gesture_profile 手勢範圍 JSON
#
# 常見調整方向：
# - webcam 明明左右相反：把 mirror_input 改 True / False
# - 模型太容易亂猜：把 confidence 稍微調高，例如 0.35 -> 0.45
# - 模型太保守不出字：把 confidence 稍微調低，例如 0.35 -> 0.28
# - 用別的 checkpoint 測試：改 checkpoint
MODE_PRESET_OVERRIDES = {
    "webcam 9 類即時模式": {
        # r"...\multibranch_baseline.pt"
        "checkpoint": r"c:\Users\qqq02\Desktop\99_docs_analysis\integration_workspace\artifacts_webcam9_nosign_seq30s5_iso\models\multibranch_baseline.pt",
        "gesture_profile": r"c:\Users\qqq02\Desktop\99_docs_analysis\metadata\webcam9_gesture_profiles.json",
        "sequence_length": 30,
        "confidence": 0.35,
        "mirror_input": True,
        "notes": "可在根目錄 realtime_tuning_settings.py 中直接調整。",
    },
    # 如果你也想改 30 類句子模式，可取消註解後調整
    # "句子 30 類模式": {
    #     "confidence": 0.55,
    #     "mirror_input": False,
    # },
}


# 即時解碼器參數覆寫
# ------------------
# 這裡調的是「從逐幀分數變成最終輸出單字」的規則，不是模型本身。
#
# default:
#   所有模式都會先套用這組
#
# webcam 9 類即時模式:
#   只在 webcam 9 類模式額外覆蓋
#
# 參數說明：
# - alpha
#   EMA 平滑強度。越大越相信最新幀，越小越平滑。
#   建議範圍：0.30 ~ 0.60
#
# - arm_frames
#   連續幾幀像手勢，才開始進入片段。
#   太小：容易亂觸發
#   太大：出字太慢
#
# - release_frames
#   連續幾幀不像手勢，就結束片段。
#   太小：一個手勢容易被切成兩段
#   太大：不同手勢容易黏在一起
#
# - min_confidence
#   一段片段最後要高於多少分才允許輸出。
#   太低：容易把 no_sign 當手語
#   太高：容易完全不出字
#
# - min_margin
#   top1 和 top2 至少要差多少，才算夠明確。
#   越高越保守。
#
# - min_segment_frames
#   一段至少幾幀才算有效手勢。
#
# - cooldown_frames
#   出完一個詞之後，幾幀內不允許再重複出同詞。
#
# - min_valid_ratio
#   MediaPipe 有效骨架比例門檻。
#   若你鏡頭品質差、常抓不到手，可略微降低。
#
# - min_signal_score
#   綜合有效性分數門檻。
#   越高越不容易亂出字。
#
# - min_motion_energy
#   動作量門檻。
#   若靜態手勢不容易被抓出來，可略微降低。
DECODER_PRESET_OVERRIDES = {
    "default": {
        "alpha": 0.45,
        "arm_frames": 4,
        "release_frames": 5,
        "min_confidence": 0.45,
        "min_margin": 0.12,
        "min_segment_frames": 6,
        "cooldown_frames": 10,
        "min_valid_ratio": 0.45,
        "min_signal_score": 0.42,
        "min_motion_energy": 0.06,
    },
    "webcam 9 類即時模式": {
        # 如果待機中常亂跳成手語：提高這三個
        # "min_confidence": 0.50,
        # "min_margin": 0.15,
        # "min_signal_score": 0.48,

        # 如果靜態手勢不容易出字：降低這三個
        # "arm_frames": 3,
        # "min_segment_frames": 4,
        # "min_motion_energy": 0.008,

        "alpha": 0.60,
        "arm_frames": 2,
        "release_frames": 3,
        "min_confidence": 0.30,
        "min_margin": 0.12,
        "min_segment_frames": 3,
        "cooldown_frames": 4,
        "min_valid_ratio": 0.35,
        "min_signal_score": 0.25,
        "min_motion_energy": 0.06,
    },
}


# Rule-based 消歧義總開關
# -----------------------
# False:
#   完全信任模型原始機率，不套用 father/mother 等規則覆寫
#
# True:
#   啟用 class_specific_disambiguation.py 中的 rule-based 規則
ENABLE_RULE_BASED_DISAMBIGUATION = False


# Dynamic no_sign 抑制
# --------------------
# 在連續推論的 raw probabilities 階段，先把 no_sign 壓低，再交給後續 decoder。
ENABLE_DYNAMIC_NOSIGN_SUPPRESSION = False
NOSIGN_PENALTY_FACTOR = 0.4


# Trigger-based Spotter 參數
# -------------------------
# 只影響 --engine-mode trigger_based。
#
# - TRIGGER_PATIENCE
#   起步防抖：motion_energy 需連續超過觸發閾值幾幀才開始收錄。
#
# - IDLE_PATIENCE
#   動作中斷寬容：motion_energy 需連續低於靜止閾值幾幀才結束。
#
# - MIN_ACTION_FRAMES
#   最短壽命過濾：結束後若片段幀數低於此值，直接丟棄不推論。
# - MIN_MOTION_ENERGY
#   進入 trigger 收錄前所需的最小 motion energy。
#   越高越保守，越能抑制動作突波帶來的誤觸發。
# - PRE_CONTEXT_FRAMES
#   action start 前要補回去的歷史幀數，避免起手式被裁掉。
# - TRIGGER_MAX_BUFFER_FRAMES
#   trigger segment 的硬上限。太小會在手勢尚未結束前被 force close。
TRIGGER_PATIENCE = 2
IDLE_PATIENCE = 10
MIN_ACTION_FRAMES = 10
MIN_MOTION_ENERGY = 0.08
EMIT_CONFIDENCE_THRESHOLD = 0.60
MIN_TOP_MARGIN = 0.50
PRE_CONTEXT_FRAMES = 5
TRIGGER_MAX_BUFFER_FRAMES = 96


# 手型/位置輔助權重
# -----------------
# 這一組影響的是：
# 1. 靜態手勢輔助分數
# 2. gesture profile 比對分數
#
# 如果你發現：
# - 手型明明對，但只因為位置稍微偏掉就辨識錯
#   => 提高 shape，降低 position
#
# 目前已預設成「手型優先」。
GESTURE_ASSIST_SETTINGS = {
    # 靜態輔助層：用來微調模型原始分數
    "assist_shape_weight": 0.85,
    "assist_position_weight": 0.15,

    # profile 比對層：用來估計各詞和目前手勢的相似度
    "profile_shape_weight": 0.85,
    "profile_position_weight": 0.15,

    # 輔助倍率範圍
    "assist_multiplier_base": 0.92,
    "assist_multiplier_scale": 0.18,
    "assist_multiplier_min": 0.92,
    "assist_multiplier_max": 1.10,
}


# 訓練/重訓參數覆寫
# -----------------
# integration_workspace/config.py 會在載入時讀取這一組，用來覆蓋訓練相關設定。
#
# - use_spatial_translation_aug
#   是否啟用 sequence-level 空間平移增強。
#
# - spatial_translation_min_offset / spatial_translation_max_offset
#   每個 sequence 固定套用一次 XY 偏移，範圍以 normalized 座標比例表示。
#
# - location_dropout_prob
#   只在 training 時生效。以這個機率把整個 batch 的 location branch 輸出歸零。
#
# - use_weighted_loss
#   啟用 class-balanced CrossEntropyLoss，讓樣本少的類別擁有更高 loss 權重。
#
# - weight_decay
#   Optimizer 權重衰減，用來抑制過大的參數並降低過擬合。
TRAINING_CONFIG_OVERRIDES = {
    "use_spatial_translation_aug": True,
    "spatial_translation_min_offset": 0.05,
    "spatial_translation_max_offset": 0.15,
    "use_weighted_loss": True,
    "location_dropout_prob": 0.60,
    "weight_decay": 1e-4,
    "num_epochs": 36,
    "lr_scheduler_type": "cosine",
    "lr_scheduler_min_lr": 1e-5,
}


# 建議你優先測這幾組
# -------------------
# 1. 待機中亂跳成手語：
#    min_confidence = 0.50
#    min_margin = 0.15
#    min_signal_score = 0.48
#
# 2. 你有做手勢但系統不出字：
#    arm_frames = 3
#    min_segment_frames = 4
#    min_motion_energy = 0.008
#
# 3. 一個手勢被切成兩個詞：
#    release_frames = 7
#    cooldown_frames = 14
#
# 4. 不同手勢黏在一起：
#    release_frames = 4
#    min_margin = 0.15
#
# 5. 手型是對的，但位置一偏就認錯：
#    assist_shape_weight = 0.90
#    assist_position_weight = 0.10
#    profile_shape_weight = 0.90
#    profile_position_weight = 0.10
