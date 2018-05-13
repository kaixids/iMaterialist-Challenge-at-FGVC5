# iMaterialist-Challenge-at-FGVC5
Kaggle competition - image classification 

Progress so far:
- Reduced 1 million+ training images to fewer than 20k 
- For the final filtered training set, loaded images and labels into arrays
- Loaded the 20k images into a series of numpy arrays but ran into memory issues while trying to stack them input the desired input shape 

Currenty working on:
May 10
- ~~build multi-label generator to work with the large dataset    ~~
  https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html    
  http://www.kubacieslik.com/extending-keras-imagedatagenerator-handle-multilable-classification-tasks/    
May 13
- Build transfer learning model architecture using trained model  
