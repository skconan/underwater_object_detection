# underwater_object_detection

## Usage

### Background Subtraction with Kmean

* [bg_sub_by_kmean.py](https://github.com/skconan/underwater_object_detection/blob/master/src/bg_sub_by_kmean.py)

* File description :
    - Find `max_iter` variable. The variable is maximum number of iterations of algorithm.
      - If increase `max_iter` the accuracy is increase but spend more time.
      - If decrease `max_iter` is opposite above case.
    
    - Find parameter `mode` of bg_subtraction()
      - Use `neg` when background has intensity higher than foreground (object).
      - Use `pos` in otherwise.
    
    - `bg_k` and `fg_k` is number of color of result image.
