# Case2

1. We should data cleaning. Since most raw data is messy, it is necessary to rearrange the data and remove duplicate data and so on.

2. Before performing the analysis, the following functions are required.  
    
    * __template_fun__ : Change the image data as follows. The background is 0 and the content is 1. In this case, a group of positions with a value of 1 is referred to as a mask region.

    * __full_to_mask__ : It is a function of extracting mask region from our original image data.

    * __mask_to_full__ : It is a function that converts the image data of the mask region into an original image data format 

3. 

