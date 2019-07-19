# Jigsaw Puzzle Piece Image Segmentation & Placement Prediction

## Motivation

Despite having only rudimentary exposure to image classification and no exposure to semantic/instance segmentation, I found myself gravitating towards instance segmentation.  Inspired by [this writeup](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46) about using Matterport's Mask R-CNN for pixel-level balloon identification, I started mulling ideas with my good friend and founder of the [codebug bootcamp](https://www.codebug.us/).  A couple rabbit holes later, we stumbled upon a couple of puzzles under the coffee table and came up with an initial business question:

***Can you take a photo of a puzzle piece and the photo of its box and predict where in the puzzle it belongs?***

As an avid puzzler growing up, I thought this would be a fun challenge that had several checkpoints (and stretch goals) that allowed me to gauge the feasibility of the task along the way and adjust as needed.

## Project Organization

### Dataset Creation

The dataset was created by taking pieces from 5 puzzles and photographing them in *expected* situations (in the puzzle's box, in one's hand, on a table, etc.).  Given the business application of the desired solution, it did not make sense to photograph these pieces in random situations.  For the training and validation sets, the neural network requires the object outlines to be annotated and classified, so I used the [VGG Image Annotator (VIA)](http://www.robots.ox.ac.uk/~vgg/software/via/) to create these in JSON.

###### Annotation Example
![annotation-gif](https://i.imgur.com/k5ku6OZ.mp4)


- 5 puzzles (2x 100-piece, 2x 200-piece, 1x 1000-piece)
- 93 annotated training images
- 22 annotated validation images
- Ever increasing amount of test images

**The goal of predicting the location of a puzzle piece was broken down into four parts:**

#### Part I: Instance Segmentation

In computer vision, image identification can be broken down into at least four tiers:

1. ***Classification***: Identifying if there is, or is not, a puzzle piece in the image.
2. ***Semantic Segmentation***: Identifying all the pixels of puzzle piece(s) in the image.
3. ***Object Detection***: Quantifying and Locating the number of pieces in an image (accounting for overlaps)
4. ***Instance Segmentation***: Quantifying and locating all instances of a puzzle piece, at the pixel level, in the image.

The first part of the project focused on image segmentation and being able to accurately classify and locate a puzzle piece in an image.  To do this, I used a Mask R-CNN pretrained on the COCO dataset, and provided the model with the annotated dataset.  A handful of models were trained (freezing the base layers) using different configuration parameters for varying epochs, and the models were evaluated based on their `val_mask_rcnn_loss` and Intersect over Union (IoU) scores.  Intersect over Union measures the percent overlap between the ground truth mask/bounding box (annotated) and predicted mask/bounding box.  Despite adjusting weights in an effort to increase the IoU scores, the average mask IoU plateaued at around 14%.

**Average Mask IoU: 84%**

**Average Box IoU: 87%**

######  IoU Comparison
![take-that-for-data](https://i.imgur.com/Of4j8bV.png)

#### Part II: Segmentation Extraction

After tuning the model and getting accurate (enough) predicted masks, the next step was to create a new image with only the puzzle piece.  Using the bounding box coordinates, only the region of interest (ROI) was extracted from the original input image.  With the piece isolated and now consuming most of the picture, the next step was to remove the predicted background by changing their pixels black, adding an alpha channel, and setting the background transparent.  To do this, I used the predicted mask, and applied these changes to all the pixels in the ROI image not encompassed by the object's mask.

###### Segmentation Extraction Example
![image-seg-i](https://i.imgur.com/AP0qL5Q.png)
![image-seg](https://i.imgur.com/4aBecLk.gif)

#### Part III: Feature Matching

With the extracted puzzle piece and an image of the puzzle, the next step involved passing both images to a feature matching algorithm.  Given the expectation that the pieces in the images were randomly photographed, the piece's rotation and tilt with respect to the camera had to be considered unknowns.  As a result, sliding window algorithms were incompatible, and ultimately, I settled on using the SIFT (Scale Invariant Feature Transform) algorithm to detect features.  SIFT can be rather slow when dealing with large images (which was the case), and even slower when dealing with more complex puzzle scenes (ex. ocean floor with hundreds of animals = more features to sift through).

###### Feature Matching Example
![fm-bb](https://i.imgur.com/HogNJrV.gif)


#### Part IV: Location Prediction

If enough features were matched between the piece and the box, and the piece's location could be determined, the next step was to draw the outline of the piece where it belongs in the puzzle.  To do this, OpenCV's  `findHomography` and `perspectiveTransform` were used to find the orientation (scale, rotation, skew, etc.) of the piece in the box, apply this perspective transformation to the vertices of the piece's outline, and draw them on the box.

###### Location Prediction Example
![fm](https://i.imgur.com/Ge0k4gX.gif)

## Future Work & Takeaways

### Areas of Improvement

* Sometimes, when applying the perspective transformation on the puzzle piece's contours, **the piece's outline can be distorted** into several lines across the image.  This is something that is most definitely fixable given a little more time.

* When feature matching fails to find enough keypoints, consider applying **a second feature matching algorithm** or applying SIFT again with a new set of parameters.

* While SIFT is scale invariant, one way to improve the feature matching would be to, if possible, **adjust its invariance to scale**.  That is, a puzzle piece can only be 1/100, 1/200, 1/500, 1/1000 the size of the box image (depending on the number of pieces in the puzzle).  If I could provide SIFT with the knowledge that it does not have to consider scales in which the puzzle piece could not exist, the algorithm should, at the very least, have a faster performance.  Additionally, being cognizant of maximum scale for the piece, it might also extract more features.

* Again, when feature matching fails to find enough keypoints, consider passing in **multiple images of the same piece** into SIFT to see if different orientations and scales provide more successful results.  Though I had not anticipated the need to take photos of the same piece in different orientations/scales, this is probably the easiest to implement as it only requires more test images and adjusting the feature matching to loop through a batch of images until a successful match (or the end of the batch).

### Alternative Image Segmentation Model & Process

While the final model had a respectable Mask IoU of ~84% and Box Iou of ~87%, I believe that this could break the 90% threshold by:


1. Requiring a photo of the backside of the puzzle piece (in addition to the front side and box)
2. Instead of training the model on the front sides, train the model on the backs of puzzle pieces
3. Pass in backside of puzzle piece into model when doing image segmentation (inference mode)
4. Horizontal flip the extracted ROI of the backside, so that it's flipped shape matches the outlines of the front side
5. Apply SIFT feature detection between the flipped, backside and the frontside piece, isolating the frontside.
6. Extract match from SIFT detection and create instance segmentation mask on the frontside image.
7. Take this final mask and apply SIFT feature detection between this and the box image.

Given that puzzle pieces have a significant amount of internal edges and features that may distract or mislead an image segmentation model, training on the backside, and then horizontally flipping the extracted mask should result in a higher intersect over union for both the mask and box predictions. Sometimes, for example, when an object was only partially contained in the piece, the model did not include this part in its prediction of the piece, and thus had a lower predicted mask IoU.  That being said, I do not believe that the model's accuracy of image segmentation held back the final performance of this endeavor as much as the feature matching aspect did.

### Future Work

* If I can find a way to speed up the feature detection, host the final result online, with the test images and boxes so that it is interactive.  The next step would be to allow piece/box image uploads so that new puzzles can be tested.

* Fine tuning SIFT's scale invariance or adding a fallback feature matching algorithm to improve the rate of successful matches.

##### Minor Headaches

*   First time working in google colab, so there was a learning curve in just figuring out file navigation, shortcuts, and how to properly install and reference parts of the project. Plus, a couple crashes during model training (my fault).

*  Not enough exposure to or intimate knowledge of `measure.find_contours`, and `cv2.perspectiveTransform` to handle cases when the transformed piece outline broke when drawing on the final image.  Time was spent here, but unsuccessfully.

*   Knowledge creep between classes and functions over the life of the project, and wanting to be able to access things like `box_image` or `box_name` in places where it should not necessarily live.  This occurred mostly because, after getting the overall pipeline to work, I wanted to iterate across all the test and validation images by box and save them in their appropriate places.

* Desire to do much more than time allowed, and having to accept certain parts of the process as "completed" even though I would have enjoyed improving/cleaning/changing them.

## Resources

#### Instance Segmentation & Mask RCNN

- [matterport - Splash of Color: Instance Segmentation with Mask R-CNN](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46)
- [github - Mask R_CNN](https://github.com/matterport/Mask_RCNN)
- [hackernoon - Object Detection in Google Colab with Custom Dataset](https://hackernoon.com/object-detection-in-google-colab-with-custom-dataset-5a7bb2b0e97e)
- [hackernoon - Instance Segmentation in Google Colab with Custom Dataset](https://hackernoon.com/instance-segmentation-in-google-colab-with-custom-dataset-b3099ac23f35)
- [towardsdatascience - Mask R-CNN for Ship Detection & Segmentation](https://towardsdatascience.com/mask-r-cnn-for-ship-detection-segmentation-a1108b5a083)
- [pyimagesearch - Mask R-CNN with OpenCV](https://www.pyimagesearch.com/2018/11/19/mask-r-cnn-with-opencv/)
- [pyimagesearch - Mask R-CNN](https://www.pyimagesearch.com/2019/06/10/keras-mask-r-cnn/)
- [vidhya - Step-by-Step Introduction to Image Segmentation Techniques](https://www.analyticsvidhya.com/blog/2019/04/introduction-image-segmentation-techniques-python/)
- [medium - Review of Deep Learning Algorithms](https://medium.com/@arthur_ouaknine/review-of-deep-learning-algorithms-for-image-semantic-segmentation-509a600f7b57)
- [jeremyjordan - An overview of semantic image segmentation](https://www.jeremyjordan.me/semantic-segmentation/)
- [github - Image Segmentation with tf.keras](https://github.com/tensorflow/models/blob/master/samples/outreach/blogs/segmentation_blogpost/image_segmentation.ipynb)
- [colab - maskrcnn_custom_tf_colab.ipynb](https://colab.research.google.com/github/RomRoc/maskrcnn_train_tensorflow_colab/blob/master/maskrcnn_custom_tf_colab.ipynb#scrollTo=X7iSzccTL9hM)
- [towardsdatascience - CNN Application - Detecting Car Exterior Damage](https://towardsdatascience.com/cnn-application-detecting-car-exterior-damage-full-implementable-code-1b205e3cb48c)

#### Feature Matching

- [opencv-python - Feature Matching + Homography to Find Objects](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html)
- [washington.edu - SIFT (pdf)](https://courses.cs.washington.edu/courses/cse576/11sp/notes/SIFT_white2011.pdf)
- [stackoverflow - How to get pixel coordinates from Feature Matching in OpenCV](https://stackoverflow.com/questions/30716610/how-to-get-pixel-coordinates-from-feature-matching-in-opencv-python)
- [kaggle - From Masks to Rotating Bounding Boxes using OpenCV](https://www.kaggle.com/frappuccino/from-masks-to-rotating-bounding-boxes-using-opencv)
- [stackoverflow - how to straighten a rotated rectangle area of an image using opencv](https://stackoverflow.com/questions/11627362/how-to-straighten-a-rotated-rectangle-area-of-an-image-using-opencv-in-python)
- [github - Including Non-Free Algorithms in OpenCV](https://github.com/skvark/opencv-python/issues/126)
- [opencv-python - Introduction to SIFT](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_sift_intro/py_sift_intro.html)
- [opencv-python - Feature Matching + Homography to Find Objects](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html)
- [docs.opencv - Brute Force & Flann Based Matcher](https://docs.opencv.org/trunk/dc/dc3/tutorial_py_matcher.html)
- [github.io - cv.findHomography docs](https://kyamagu.github.io/mexopencv/matlab/findHomography.html)
- [learnopencv - Image Alignment using OpenCV C++](https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/)
- [opencv-python - Template Matching](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html)
- [washington.edu - Features & Image Matching - PDF](https://courses.cs.washington.edu/courses/cse455/09wi/Lects/lect6.pdf)

#### Plotting & Utility

- [Long explanation of using plt subplots to create small multiples](http://jonathansoma.com/lede/data-studio/classes/small-multiples/long-explanation-of-using-plt-subplots-to-create-small-multiples/)
- [pythoncentral - How to Recursively Copy A Folder In Python](https://www.pythoncentral.io/how-to-recursively-copy-a-directory-folder-in-python/)
- [towardsdatascience - Getting Started with TensorFlow in Google Colaboratory](https://towardsdatascience.com/getting-started-with-tensorflow-in-google-colaboratory-9a97458e1014)
- [docs.opencv - Drawing Functions](https://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html)
- [scikit-image.org - Contour Finding](https://scikit-image.org/docs/dev/auto_examples/edges/plot_contours.html)
- [docs.opencv - Contour Features](https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html)
- [stackoverflow - Python hide ticks but show tick labels](https://stackoverflow.com/questions/29988241/python-hide-ticks-but-show-tick-labels/29988431)