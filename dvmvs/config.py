import time


class Config:
    # training settings
    train_image_width = 256
    train_image_height = 256
    # question from yang: in what unit is the depth? ans: depth value in millimeters
    train_min_depth = 0.25
    train_max_depth = 20.0
    train_n_depth_levels = 64
    train_minimum_pose_distance = 0.125
    train_maximum_pose_distance = 0.325
    train_crawl_step = 3
    train_subsequence_length = None
    train_predict_two_way = None
    train_freeze_batch_normalization = False
    train_data_pipeline_workers = 8
    train_epochs = 100000
    train_print_frequency = 5000
    train_validate = True
    train_seed = int(round(time.time()))

    # test settings
    test_image_width = 320
    test_image_height = 256
    test_distortion_crop = 0
    test_perform_crop = False
    test_visualize = True
    test_n_measurement_frames = 2
    test_keyframe_buffer_size = 30
    test_keyframe_pose_distance = 0.1
    test_optimal_t_measure = 0.15
    test_optimal_R_measure = 0.0

    # SET THESE: TRAINING FOLDER LOCATIONS
    # TODO add new dataset here to test train
    dataset = "/home/yang/Workspace/dataset/"
    train_run_directory = "/home/yang/Workspace/deep-video-mvs/training-runs"

    # SET THESE: TESTING FOLDER LOCATIONS
    # for run-testing-online.py (evaluate a single scene, WITHOUT keyframe indices, online selection)
    
    # here are scenes from blendermvs and desk dataset, for testing purpose
    # test_online_scene_path = "/home/yang/Workspace/git/deep-video-mvs/sample-data/own-dataset/5a3ca9cb270f0e3f14d0eddb"
    # test_online_scene_path = "/home/yang/Workspace/git/deep-video-mvs/sample-data/own-dataset/5a3cb4e4270f0e3f14d12f43"
    # test_online_scene_path = "/home/yang/Workspace/git/deep-video-mvs/sample-data/own-dataset/5a3f4aba5889373fbbc5d3b5"
    # test_online_scene_path = "/home/yang/Workspace/git/deep-video-mvs/sample-data/own-dataset/5a4a38dad38c8a075495b5d2"

    # test data from ARKitScene
    test_online_scene_path = "/home/yang/Workspace/git/deep-video-mvs/sample-data/own-dataset/40753679_frames"

    # for run-testing.py (evaluate all available scenes, WITH pre-calculated keyframe indices)
    test_offline_data_path = "/home/yang/Coding/deep-video-mvs/sample-data"

    # below give a dataset name like tumrgbd, i.e. folder or None
    # if None, all datasets will be evaluated given that
    # their keyframe index files are in Config.test_offline_data_path/indices folder
    test_dawtaset_name = "hololens-dataset"  # or None

    test_result_folder = "/home/yang/Workspace/result/"
