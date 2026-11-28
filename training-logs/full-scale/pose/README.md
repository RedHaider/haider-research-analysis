# Coarse-to-fine CAD keypoint codes

[CheckerPose](https://openaccess.thecvf.com/content/ICCV2023/html/Lian_CheckerPose_Progressive_Dense_Keypoint_Localization_for_Object_Pose_Estimation_with_ICCV_2023_paper.html), co-authored by Ruyi Lian, motivates localizing projected CAD points through binary image coordinates and a graph over 3D points. This independent study compares direct coordinate regression with weighted coarse-to-fine bit prediction under the same CNN and graph readout. It samples 32 vertices per object, uses six graph neighbors per point, and fits the camera pose by PnP with RANSAC. The code predicts the bit positions together and emphasizes coarse bits in the loss.

The aligned experiment generates 6,000 CAD-based renderings with perturbed viewpoints, backgrounds and occlusion. The real-data companion uses annotated RGB frames from the [BOP TUD-L dataset](https://huggingface.co/datasets/bop-benchmark/tudl). It uses annotated object crops and object identity. Groups are contiguous frame blocks within a scene. The evaluation measures crop-coordinate error and pose recovery under a custom split of these frames. The benchmark differs from the official BOP object detection and pose tasks.

    python training-logs/full-scale/pose/train.py
    python training-logs/full-scale/pose/train.py --real

Install PyTorch, NumPy, OpenCV, SciPy, Pillow and scikit-learn. The trainer expects datasets/tudl_models.zip and the real run also reads datasets/tudl_test_bop19.zip. It saves model weights, split and CAD keypoint coordinates, validation histories and exact test predictions.
