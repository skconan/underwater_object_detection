# Underwater Object Detection

## Usage

### Overview and Setup

* This repo have code for 2 machines. Please clone this repo to`JetsonTX2` and `Nuc Intel`
* In JetsonTX2
    - Run `semantic_segmentation
* In NucIntel 
    - Run **[Color Range Selection](#color-range-selection)**
    - Run **[Object Detection](#object-detection)**

### Background Subtraction with Kmean

* File: [bg_sub_by_kmean.py](https://github.com/skconan/underwater_object_detection/blob/master/src/bg_sub_by_kmean.py)

* File description :
    - Find `max_iter` variable. The variable is maximum number of iterations of algorithm.
      - If increase `max_iter` the accuracy is increase but spend more time.
      - If decrease `max_iter` is opposite above case.
    
    - Find parameter `mode` of bg_subtraction()
      - Use `neg` when background has intensity higher than foreground (object).
      - Use `pos` in otherwise.
    
    - `bg_k` and `fg_k` is number of color of result image.
    
 
 ### Color Range Selection
 
 * Files:
    - [object_color_range.py](https://github.com/skconan/underwater_object_detection/blob/master/src/object_color_range.py)
    - [object_color_range.launch](https://github.com/skconan/underwater_object_detection/blob/master/launch/object_color_range.launch)
    - [constants.py](https://github.com/skconan/underwater_object_detection/blob/master/src/constants.py) - Insert mission in mission list. (The first letter must be unique letter. cannot use z,x,s,c,q)
    
 * Execution
    - roslaunch object_detection object_color_range.launch
    - Press `y` or `Y`
 
 * In program
    - In the `image` window display HSV image.
    - And `window status` display by trackbar name `m<->c`. (m = mask = 0, c = color = 2)
    - First step, If you want to select the color range of `gate` press `g` (press first letter of mission name). Now `window status` is `m`
    - Second, focus on `image` and `mask` window. Then click on the `gate` for select the color range of `gate`. The mask result display on `mask` window.
    - Third, you can slide the tracebar for select the color range of `gate`
    - Finally, save color range. press `g` again. check `window status` on the right side (c=color). Then press `s`. If saved, terminal will show `<------------ save ------------>`
    
  * Command in program  
    - `press z` undo
    - `press x` redo
    - `press s` save
    - `press c` clear color value (lower: 179, 255, 255 and upper: 0, 0, 0) 
    - `press q`	exit program. if not save cannot exit but you can `Ctrl+C` in termnal for exit.

### Object Detection

* Before use this program, you need to done in [Color Range Selection](#color-range-selection) part.
* In [object_detection_front.py](https://github.com/skconan/underwater_object_detection/blob/master/src/object_detection_front.py)
    - This file is a `server` node. It has service name is `object_detection_front`
    - Example of client call this server [see this](https://github.com/skconan/underwater_object_detection/blob/master/src/call_obj_detection.py) 
    - The result of service 
        - return `appear` > True or False that mean appear or disappear
        - return `mask`  > Binary image if appear is True. 
        - 
