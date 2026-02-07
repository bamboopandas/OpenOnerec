"""
Recommendation Task Configurations

This module contains configurations for all recommendation tasks including:
- label_cond: Predict next video given specified consumption behavior
- video: Next video prediction
- product: Predict next clicked product
- ad: Predict next clicked advertisement
"""

# Common prompt config for recommendation tasks
RECOMMENDATION_PROMPT_CONFIG = {
    "enable_thinking": False,
    "custom_chat_template": "qwen3_soft_switch.jinja2",
}

# Common generation config for recommendation tasks
RECOMMENDATION_GENERATION_CONFIG = {
    "num_return_sequences": 128,
    "max_new_tokens": 3,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 50,  
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "prompt_token": "<|sid_begin|>",  # Token to append for two-stage generation
    "max_new_thinking_tokens": 1000,
    "num_return_thinking_sequences": 8,  # Number of thinking candidates to generate in stage 1
    "num_beams": 16,
}

# Common evaluation config for recommendation tasks
RECOMMENDATION_EVALUATION_CONFIG = {
    "metrics": ["pass@k", "position1_pass@k", "recall@k"],
    "k_values": [1, 5, 10, 16, 32, 64, 128],
    "select_k": "first_k",  # Strategy for selecting k predictions: 'first_k' or 'random_k'

    # PID-based evaluation settings
    "evaluation_mode": "both",  # Evaluation mode: 'sid', 'pid', or 'both'
    "sid_to_pid_strategy": "most_popular_after_downsampling",  # Strategy for SID->PID conversion: 'most_popular_originally', 'most_popular_after_downsampling', or 'random'
}

# Label Cond Task Configuration
LABEL_COND_CONFIG = {
    "name": "label_cond",
    "source": "Kuaishou Internal",
    "splits": ["test"],
    "size": 34891,
    "sample_size": 34891,
    "description": "Predict next video given specified consumption behavior",
    "data_fields": {
        "messages_field": "messages",
        "metadata_field": "metadata",
    },
    "prompt_config": RECOMMENDATION_PROMPT_CONFIG.copy(),
    "generation_config": RECOMMENDATION_GENERATION_CONFIG.copy(),
    "evaluation_config": RECOMMENDATION_EVALUATION_CONFIG.copy(),
}

# SID USER Doc Task Configuration
VIDEO_CONFIG = {
    "name": "video",
    "source": "Kuaishou Internal",
    "splits": ["test"],
    "size": 38781,
    "sample_size": 38781,
    "description": "Next video prediction",
    "data_fields": {
        "messages_field": "messages",
        "metadata_field": "metadata",
    },
    "prompt_config": RECOMMENDATION_PROMPT_CONFIG.copy(),
    "generation_config": RECOMMENDATION_GENERATION_CONFIG.copy(),
    "evaluation_config": RECOMMENDATION_EVALUATION_CONFIG.copy(),
}

# Product Task Configuration
PRODUCT_CONFIG = {
    "name": "product",
    "source": "Kuaishou Internal",
    "splits": ["test"],
    "size": 27910,
    "sample_size": 27910,
    "description": "Predict next clicked product",
    "data_fields": {
        "messages_field": "messages",
        "metadata_field": "metadata",
    },
    "prompt_config": RECOMMENDATION_PROMPT_CONFIG.copy(),
    "generation_config": RECOMMENDATION_GENERATION_CONFIG.copy(),
    "evaluation_config": RECOMMENDATION_EVALUATION_CONFIG.copy(),
}

# Ad Task Configuration
AD_CONFIG = {
    "name": "ad",
    "source": "Kuaishou Internal",
    "splits": ["test"],
    "size": 27677,
    "sample_size": 27677,
    "description": "Predict next clicked advertisement",
    "data_fields": {
        "messages_field": "messages",
        "metadata_field": "metadata",
    },
    "prompt_config": RECOMMENDATION_PROMPT_CONFIG.copy(),
    "generation_config": RECOMMENDATION_GENERATION_CONFIG.copy(),
    "evaluation_config": RECOMMENDATION_EVALUATION_CONFIG.copy(),
}

# Interactive Task Configuration
INTERACTIVE_CONFIG = {
    "name": "interactive",
    "source": "Kuaishou Internal",
    "splits": ["test"],
    "size": 1000,
    "sample_size": 1000,
    "description": "Predict next interacted video",
    "data_fields": {
        "messages_field": "messages",
        "metadata_field": "metadata",
    },
    "prompt_config": RECOMMENDATION_PROMPT_CONFIG.copy(),
    "generation_config": RECOMMENDATION_GENERATION_CONFIG.copy(),
    "evaluation_config": RECOMMENDATION_EVALUATION_CONFIG.copy(),
}

# Task configuration mapping
RECOMMENDATION_TASK_CONFIGS = {
    "label_cond": LABEL_COND_CONFIG,
    "video": VIDEO_CONFIG,
    "product": PRODUCT_CONFIG,
    "ad": AD_CONFIG,
    "interactive": INTERACTIVE_CONFIG,
}

