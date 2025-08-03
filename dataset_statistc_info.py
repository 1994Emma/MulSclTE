# Average number of **distinct action classes** per video in each dataset
mean_gt_action_classes_datasets = {
    "Breakfast": 5,
    "YTI": 8,
    "YTI-clean": 7,       # With background frames removed
    "50Salads": 16,
    "50Salads-eval": 10,  # Evaluation protocol version
    "epich": 52
}

# Average number of **total actions** (including repetitions) per video in each dataset
mean_gt_action_num_datasets = {
    "Breakfast": 7,
    "YTI": 13,
    "YTI-clean": 12,
    "50Salads": 20,
    "50Salads-eval": 19,
    "epich": 199
}

# Maximum number of **unique action classes** per video in each activity
max_gt_activity_action_classes_num_datasets = {
    "Breakfast": {
        'cereals': 5, 'coffee': 6, 'friedegg': 8, 'juice': 8, 'milk': 5,
        'pancake': 12, 'salat': 7, 'sandwich': 8, 'scrambledegg': 10, 'tea': 6
    },
    "YTI": {
        'changing_tire': 12, 'coffee': 11, 'cpr': 7, 'jump_car': 13, 'repot': 9
    },
    "YTI-clean": {
        'changing_tire': 11, 'coffee': 10, 'cpr': 6, 'jump_car': 12, 'repot': 8
    },
    "50Salads": {"rgb": 19},
    "50Salads-eval": {"rgb": 12},
    "epich": {"rgb": 284}
}

# Mean number of **unique action classes** per video for each activity (from TW-FINCH reference)
mean_gt_activity_action_classes_num_datasets = {
    "Breakfast": {
        'cereals': 4, 'coffee': 4, 'friedegg': 6, 'juice': 5, 'milk': 4,
        'pancake': 9, 'salat': 5, 'sandwich': 5, 'scrambledegg': 7, 'tea': 4
    },
    "YTI": {
        'changing_tire': 10, 'coffee': 8, 'cpr': 6, 'jump_car': 10, 'repot': 7
    },
    "YTI-clean": {
        'changing_tire': 9, 'coffee': 7, 'cpr': 5, 'jump_car': 9, 'repot': 6
    },
    "50Salads": {"rgb": 18},
    "50Salads-eval": {"rgb": 12},
    "epich": {"rgb": 52}
}

# Mean number of **actions (total instances)** per video for each activity
mean_gt_activity_action_num_datasets = {
    "Breakfast": {
        'cereals': 5, 'coffee': 4, 'friedegg': 7, 'juice': 7, 'milk': 5,
        'pancake': 11, 'salat': 10, 'sandwich': 6, 'scrambledegg': 9, 'tea': 5
    },
    "YTI": {
        'changing_tire': 18, 'coffee': 17, 'cpr': 14, 'jump_car': 19, 'repot': 19
    },
    "YTI-clean": {
        'changing_tire': 9, 'coffee': 8, 'cpr': 7, 'jump_car': 10, 'repot': 10
    },
    "50Salads": {"rgb": 20},
    "50Salads-eval": {"rgb": 19},
    "epich": {"rgb": 52}
}
