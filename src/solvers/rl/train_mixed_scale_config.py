"""Configuration for mixed-scale training validation and checkpointing.

Edit this file to change validation segmentation, weighting schemes, and
checkpoint/log behavior. Weight vectors do not need to sum to one; the training
script normalizes them before computing weighted validation distance.
"""

VAL_CONFIG = {
    # Validation interval is inherited from --num_loc_min / --num_loc_max.
    "num_segments": 9,
    "scales_per_segment": 5,
    "instances_per_scale": 4,
    "scale_seed": 42,
}

# One best checkpoint is maintained for every enabled scheme.
# Each weight corresponds to one validation segment, from smallest to largest.
VAL_WEIGHT_SCHEMES = {
    "uniform": {
        "enabled": True,
        "weights": [1, 1, 1, 1, 1, 1, 1, 1, 1],
    },
    "fix30": {
        "enabled": True,
        # "weights": [
        #     152,  # <30，其中包含141个小于20的订单池
        #     5,    # 30-39
        #     4,    # 40-49
        #     13,   # 50-59
        #     3,    # 60-69
        #     5,    # 70-79
        #     3,    # 80-89
        #     3,    # 90-99
        #     3,    # 100-109
        #     2,    # 110-119
        #     3,    # 120-129
        #     1,    # 130-139
        #     1,    # 140-149
        #     2,    # 150-159
        #     0,    # 160-169
        #     1,    # 170-179
        #     1,    # 180-189
        #     0,    # 190-200
        # ],
        "weights": [
            157,  # <40，其中包含141个小于20的订单池
            17,   # 40-59
            8,    # 60-79
            6,    # 80-99
            5,    # 100-119
            4,    # 120-139
            3,    # 140-159
            1,    # 160-179
            1,    # 180-200
        ],
    },
    "fix60": {
        "enabled": False,
        "weights": [
            73,  # <40，包含58个小于20的订单池
            9,   # 40-59
            8,   # 60-79
            3,   # 80-99
            2,   # 100-119
            4,   # 120-139
            2,   # 140-159
            1,   # 160-179
            6,   # >=180，包含6个大于200的订单池
        ],
    },
    "demand5": {
        "enabled": True,
        "weights": [
            29,  # <40
            16,  # 40-59
            11,  # 60-79
            8,   # 80-99
            4,   # 100-119
            4,   # 120-139
            3,   # 140-159
            2,   # 160-179
            1,   # 180-200
        ],
    },
    "demand2": {
        "enabled": False,
        "weights": [
            66,  # <40, 其中包含59个小于20的订单池
            17,  # 40-59
            9,   # 60-79
            6,   # 80-99
            5,   # 100-119
            4,   # 120-139
            3,   # 140-159
            1,   # 160-179
            1,   # 180-200
        ],
    },
}

CHECKPOINT_CONFIG = {
    "base_dir": "checkpoints",
    "save_true_last": True,
    "save_on_keyboard_interrupt": True,
    # All files from one run share the same run-start timestamp.
    "timestamp_format": "%Y%m%d_%H%M%S",
}

LOG_CONFIG = {
    "base_dir": "logs",
}
