# Human activities tags in video

## Carosello Data set description

Of the recorded videos at Carosello shopping mall, five have been tagged using the tool CVAT. The equipment consists of a single RGB Camera with resolution 640 × 426.
Three of them have been recorded with a fixed point of view; for other two videos, the camera has been mounted on the front of a caddy, so they are characterized by a dynamic point of view.

The CVAT tool allows to draw rectangular bounding boxes at specific frames (keyframes), then it automatically tags
the frames in between using linear interpolation.
Each time-sequence of frames containing a single person tagged is called *trace*. The following table shows 
some statistics of the tagging work. 

Brief summary of the tagged videos:
- Number of frames: 135518
- Frame rate: 30 frame/s, constant
- Total lenght: 75.29 minutes

Since in each frame there are potentially multiple people and in turn multiple tags, it follows a summary
of the dataset after the tagging task:
- Number of individual frames: 220260 (= Total Tags)
- Total lenght of tagged videos: 146.84 minutes


Legend:
- Avg. FPT: Average number of Frames Per Trace.

| Class                 | Traces   | Manual Tags | Interp. Tags | Total Tags | Percentage  | Avg. FPT   |
|-----------------------|----------|-------------|--------------|------------|-------------|------------|
| sitting               |  109     | 995         | 54537        | 55532      | 25.21 %     | 509.47     |
| walking_slow          |  406     | 3163        | 53865        | 57028      | 25.89 %     | 140.46     |
| walking_fast          |  207     | 1616        | 19819        | 21435      | 9.73  %     | 103.55     |
| wandering             |  50      | 317         | 7684         | 8001       | 3.63  %     | 160.02     |
| walking_caddy         |  56      | 484         | 8015         | 8499       | 3.86  %     | 151.77     |
| standing_free         |  247     | 1494        | 54749        | 56243      | 25.53 %     | 227.70     |
| standing_busy         |  20      | 72          | 2537         | 2609       | 1.18  %     | 130.45     |
| window_shopping       |  4       | 17          | 1126         | 1143       | 0.52  %     | 285.75     |
| walking_phone         |  49      | 364         | 4437         | 4801       | 2.18  %     | 97.98      |
| walking_phone_talking |  28      | 269         | 3980         | 4249       | 1.93  %     | 151.75     |
| sitting_phone_talking |  3       | 47          | 673          | 720        | 0.33  %     | 240.00     |
| **Total**             | **1179** | **8838**    | **211422**   | **220260** | **100   %** | **186.82** |

After the extraction of fixed length sliding windows, as explained in the 'Data set pre-processing' section, the data set used
for training and testing has the following number of samples per class:

| Class                 | Sequences |
|-----------------------|-----------|
| sitting               |   8664    |
| walking_slow          |   7547    |
| walking_fast          |   2588    |
| wandering             |   1096    |
| walking_caddy         |   1133    |
| standing_free         |   8184    |
| standing_busy         |   339     |
| window_shopping       |   172     |
| walking_phone         |   573     |
| walking_phone_talking |   560     |
| sitting_phone_talking |   107     |
| **Total**             | **30963** | 
