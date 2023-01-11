# Case2

1. We should data cleaning. Since most raw data is messy, it is necessary to rearrange the data and remove duplicate data and so on.

2. Before performing the analysis, the following functions are required.  
    
    * __template_fun__ : Change the image data as follows. The background is 0 and the content is 1. In this case, a group of positions with a value of 1 is referred to as a mask region.

    * __full_to_mask__ : It is a function of extracting mask region from our original image data.

    * __mask_to_full__ : It is a function that converts the image data of the mask region into an original image data format 

![original_img](https://user-images.githubusercontent.com/71793706/211830305-8afcad4a-153a-4534-b1f5-0bfdc5abd0f5.png) *Original image*
![mask_region_img](https://user-images.githubusercontent.com/71793706/211830264-6c85ca5a-e282-4699-b13a-11db173a0255.png)  *Mask region image*


3. 

