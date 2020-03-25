# Human Activity recognition

In `data/` it is stored the data set of skeletal poses for the Human 
Activity Recognition RNN-

The weights of the model is in `model/` as a tensorflow session.

### Repository Exploration

| Repository | Technologies | Notes |
| --- | --- | --- |
| [Spatial Temporal GCN](https://github.com/yysijie/st-gcn/) | ~~CUDA~~ | - use OpenPose as preliminary stage|
| [Deep_SORT](https://github.com/TianzhongSong/Real-Time-Action-Recognition) | | TO CHECK |
| [RNN for HAR](https://github.com/stuarteiffert/RNN-for-Human-Activity-Recognition-using-2D-Pose-Input/) | Tensorflow | IMPLEMENTED: use OpenPose skeletons to feed RNN|
| [Deep HAR](https://github.com/dluvizon/deephar) |  | to check |


### Available Data sets

| Name | Data representation | Actions of interest | Notes |
| --- | --- | --- | --- |
| [NTU RGB+D](http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp) | - RGB videos<br> - 3D skeletal data<br> - Depth Map sequences<br> - IR videos | - A8: sit down<br> - A9: stand up<br> - A28: phone call<br> - A29: play with phone/tablet<br> - A31: point to something<br> - A38: salute<br> - A67: hush<br> - A96: cross arms<br> | There are mutual actions too, but not interesting for my case |


### Model obtained using RNN for HAR on ITW

Comparison between the model obtainable using all the individual action tags.

On the left, no further pre-processing is done. On the right: skeletons have been proportionally normalised into 100x100 px bounding boxes.
As we can notice the model benefits from normalization in terms of accuracy and "discrimating power" between classes, i.e. less False Positives and False Negatives (especially for 'walking_slow' class).

<div>
  <table>
    <tr>
      <td><img src="../training_sessions/redux_vs_norm/no_redux_no_norm/no_redux_no_norm_acc.png" width=400></td>
      <td><img src="../training_sessions/redux_vs_norm/no_redux_norm/no_redux_norm_acc.png" width=400></td>
    </tr>
    <tr>
    <td><img src="../training_sessions/redux_vs_norm/no_redux_no_norm/no_redux_no_norm_cm_col.png" width=400 align="left"></td>
    <td><img src="../training_sessions/redux_vs_norm/no_redux_norm/no_redux_norm_cm_col.png" width=400 align="right"></td>
    </tr>
  </table>
</div>

Reducing the number of classes by clustering those with few samples and qualitatively similar (e.g. walking_slow and walking_fast differs by velocity, which can be easily computed independently from the HAR task), the benefit of normalisation becomes more evident:


<div>
  <table>
    <tr>
      <td><img src="../training_sessions/redux_vs_norm/redux_no_norm/redux_no_norm_acc.png" width=400></td>
      <td><img src="../training_sessions/redux_vs_norm/redux_norm/redux_norm_acc_acc.png" width=400></td>
    </tr>
    <tr>
      <td><img src="../training_sessions/redux_vs_norm/redux_no_norm/redux_no_norm_cm_col.png" width=400></td>
      <td><img src="../training_sessions/redux_vs_norm/redux_norm/redux_norm_cm_col.png" width=400></td>
    </tr>
  </table>
</div>
