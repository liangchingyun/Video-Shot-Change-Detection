# Video Shot Change Detection

This project develops an algorithm to detect shot changes in three videos (**news**, **climate**, **ngc**) and assesses its performance. Notably, the **climate** and **ngc** videos involve gradual transitions.

## Visual Features Used

1. **Grayscale Image**
2. **Histogram (64-bin gray-level)**

## Shot Change Detection Algorithm

The following algorithm is designed to detect shot changes in the videos:

1. Convert each frame to a grayscale image and compute its histogram.
2. Calculate the **Bhattacharyya distance** between histograms of consecutive frames.
3. Detect a shot change if the distance exceeds a threshold t<sub>b</sub>.
4. Detect a shot change if cumulative distances between t<sub>s</sub> and t<sub>b</sub> exceed t<sub>b</sub>

## Usage

1. Clone or download this project.
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the main script:
   ```
   python main.py
   ```

## Results and Analysis

1. News\
   **Precision**: 0.8571428571428571\
   **Recall**: 0.8571428571428571
2. Climate\
   ![image](https://github.com/liangchingyun/img-folder/blob/main/Video-Shot-Change-Detection_PRcurve_Climate.png)
3. NGC\
   ![image](https://github.com/liangchingyun/img-folder/blob/main/Video-Shot-Change-Detection_PRcurve_NGC.png)
