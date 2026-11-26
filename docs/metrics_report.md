Report on Metrics and Evaluation

This report provides an evaluation of system performance based on the accuracy of helmet detection model, confirmation logic efficiency to reduce false detections, and computational efficiency on conventional hardware.
Evaluation of the helmet detection model has been done by testing 151 images containing 409 labeled samples. All tests were performed on Intel Core i7-1355U (mobile CPU) with no GPU acceleration.

Model Accuracy

The accuracy metrics of the helmet detection model are as follows:

- Precision: 0.9336
- Recall: 0.8179
- mAP50: 0.9034
- mAP50-95: 0.6184

Performance per class:

- With Helmet:
  - Precision: 0.9252
  - Recall: 0.8934
  - mAP50: 0.9592

- Without Helmet:
  - Precision: 0.9419
  - Recall: 0.7424
  - mAP50: 0.8

-High precision (93–94%) means that if the model decides that there is a helmet in a frame/instance, it will be correct almost all times. However, recall is rather low (82% in general), and the classes are quite imbalanced "without helmet" :74% versus and "with helmet" : 89%.

-That is quite expected because of the reasons outlined above. The shape of a helmet is the same each time which helps the model learn its features better. On the contrary, the category "without helmet" includes a lot of variability and differences in hairstyle, position of the head, lighting, occlusion by another rider. There were not too many samples in the training set, which means that the model did not have enough data on this variability. Therefore, that is the main limitation of the current model since additional diversity of training set would help in achieving higher recall while with the current training set we achieve only 74% of recall.

-Therefore, one can expect that about 1 quarter of actual riders without helmets will be missed. The confusion matrix with more detailed information about the classification results is provided in runs/detect/val-4/confusion_matrix.png.

Confirmation logic

The system uses the "70% agreement" rule , a violation is only logged when it occurs in at least 70% of frames for the tracked object. This method should decrease the number of false positives caused by errors of detection, but the real efficiency of this rule is not estimated now. The hypothesis is that the difference in false positives with using confirmation and without confirmation on the same video would be big, although such test has not been done.

#Performance

Performance evaluation was done on a system consisting of a Core i7-1355U using a 150 frame sample. The reported FPS for the relevant components are:

- Object detection only (person/motorcycle): 3.74
- Object detection and tracking: 5.37
- Helmet model on full frame: 10.67
- Helmet model on cropped region (two-stage process): 8.66

There are some unexpected results: the tracking process shows better throughput compared to detection only, and the crop process takes longer than processing the full frame. This may be caused by the warm-up phenomenon. Each benchmark is run sequentially and the first run contains the one-time overheads (such as memory allocation and compilation of the graph). A more proper benchmarking approach involves multiple runs of each measurement in a randomized order and taking the average value. Thus, the reported numbers should be viewed as approximations and not precise numbers.

The whole pipeline (detection + tracking + helmet crops + rule-based reasoning) runs at about 5 FPS on this CPU. It is sufficient for offline analysis of the videos but not enough for 30 FPS live monitoring.

Design Decisions

As seen above, the helmet model is run on the cropped regions, not the full frame. The helmet model was trained on high-resolution faces. Running the model on full CCTV frame where the riders take up 50 pixels in height did not yield any detections. Properly scaling the region helped to restore the detectable signal. While it increases the number of computations, the increased accuracy justifies the added computation.

70% confirmation threshold is a purposeful decision. Unanimous requirement on all frames would allow a single motion-blurred frame to hide a violation, while a single-frame violation would cause too many false positives because of detection artifacts. 70% threshold is a compromise that allows minimizing missing violations but allows them once in a while.

The system uses a 960-pixel inference resolution instead of YOLO’s standard 640 pixels. It allows detecting small objects better (e.g., riders distant from the camera) at the cost of lower throughput. It was calibrated specifically for the use case where most objects are small (CCTV cameras).

Limitations
Wrong-way detection assumes straight, unidirectional roads and may produce false positives at intersections.
Triple-riding detection is geometry-based and may incorrectly identify nearby pedestrians as riders in crowded scenes.
Helmet violation detection achieves 74% recall, with missed detections under challenging poses, lighting, and occlusions.
The CPU-based implementation runs at 5–10 FPS, limiting the system to offline analysis. Real-time deployment would require GPU acceleration.

Summary
The proposed system achieves high precision and moderate recall, making it suitable for offline CCTV analysis. Its main limitations are the limited training data and CPU-only processing.
