import numpy as np
import pickle
import cv2
import glob

import matplotlib.image as mpimg
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from skimage.feature import hog

from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

heat = None

def extract_features(imgs):
	# Create a list to append feature vectors to
	features = []
	# Iterate through the list of images
	for file in imgs:
		file_features = []

		# Read in each one by one
		image = mpimg.imread(file)
		feature_image = np.copy(image)      

		spatial_features = bin_spatial(feature_image, size=(32, 32))
		file_features.append(spatial_features)
		
		# Apply color_hist()
		hist_features = color_hist(feature_image, nbins=32)
		file_features.append(hist_features)

		# Call get_hog_features() with vis=False, feature_vec=True
		#if hog_channel == 'ALL':
		hog_features = []
		for channel in range(feature_image.shape[2]):
			hog_features.append(get_hog_features(feature_image[:,:,channel], 
								9, 8, 2, 
								vis=False, feature_vec=True))
		hog_features = np.ravel(hog_features)        
		#else:
			#hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
			#			pix_per_cell, cell_per_block, vis=False, feature_vec=True)
				
		# Append the new feature vector to the features list
		file_features.append(hog_features)
		#print(file_features)
		features.append(np.concatenate(file_features))

	# Return list of feature vectors
	return features

def process_image(img):
	orient = 9  # HOG orientations
	pix_per_cell = 8 # HOG pixels per cell
	cell_per_block = 2 # HOG cells per block
	hog_channel = 0 # Can be 0, 1, 2, or "ALL"
	spatial_size = (32, 32) # Spatial binning dimensions
	hist_bins = 32    # Number of histogram bins

	return find_cars(img, 480, img.shape[0], 1, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

	#heat = np.zeros_like(img[:,:,0]).astype(np.float)

	# Add heat to each box in box list
	#heat = add_heat(heat, box_list)
	
	# Apply threshold to help remove false positives
	#heat = apply_threshold(heat,1)

def add_heat(heatmap, bbox_list):
	# Iterate through list of bboxes
	for box in bbox_list:
		# Add += 1 for all pixels inside each bbox
		# Assuming each "box" takes the form ((x1, y1), (x2, y2))
		heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

	# Return updated heatmap
	return heatmap# Iterate through list of bboxes
	
def apply_threshold(heatmap, threshold):
	# Zero out pixels below the threshold
	heatmap[heatmap <= threshold] = 0
	# Return thresholded map
	return heatmap

def draw_labeled_bboxes(img, labels):
	# Iterate through all detected cars
	for car_number in range(1, labels[1]+1):
		# Find pixels with each car_number label value
		nonzero = (labels[0] == car_number).nonzero()
		# Identify x and y values of those pixels
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		# Define a bounding box based on min/max x and y
		bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
		# Draw the box on the image
		cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
	# Return the image
	return img

def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
						vis=False, feature_vec=True):
	# Call with two outputs if vis==True
	if vis == True:
		features, hog_image = hog(img, orientations=orient, 
								  pixels_per_cell=(pix_per_cell, pix_per_cell),
								  cells_per_block=(cell_per_block, cell_per_block), 
								  transform_sqrt=False, 
								  visualise=vis, feature_vector=feature_vec)
		return features, hog_image

	# Otherwise call with one output
	else:      
		features = hog(img, orientations=orient, 
					   pixels_per_cell=(pix_per_cell, pix_per_cell),
					   cells_per_block=(cell_per_block, cell_per_block), 
					   transform_sqrt=False, 
					   visualise=vis, feature_vector=feature_vec)
		return features

def bin_spatial(img, size=(32, 32)):
	color1 = cv2.resize(img[:,:,0], size).ravel()
	color2 = cv2.resize(img[:,:,1], size).ravel()
	color3 = cv2.resize(img[:,:,2], size).ravel()
	return np.hstack((color1, color2, color3))
						
def color_hist(img, nbins=32):    #bins_range=(0, 256)
	# Compute the histogram of the color channels separately
	channel1_hist = np.histogram(img[:,:,0], bins=nbins)
	channel2_hist = np.histogram(img[:,:,1], bins=nbins)
	channel3_hist = np.histogram(img[:,:,2], bins=nbins)

	# Concatenate the histograms into a single feature vector
	hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

	# Return the individual histograms, bin_centers and feature vector
	return hist_features

def find_cars(img, ystart, ystop, scale, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins): 
	draw_img = np.copy(img)
	img = img.astype(np.float32)/255
	
	ctrans_tosearch = img[ystart:ystop,:,:]
	if scale != 1:
		imshape = ctrans_tosearch.shape
		ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
		
	ch1 = ctrans_tosearch[:,:,0]
	ch2 = ctrans_tosearch[:,:,1]
	ch3 = ctrans_tosearch[:,:,2]

	# Define blocks and steps as above
	nxblocks = (ch1.shape[1] // pix_per_cell) - 1
	nyblocks = (ch1.shape[0] // pix_per_cell) - 1 
	nfeat_per_block = orient * cell_per_block ** 2

	# 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
	window = 64
	nblocks_per_window = (window // pix_per_cell) - 1 
	cells_per_step = 2  # Instead of overlap, define how many cells to step
	nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
	nysteps = (nyblocks - nblocks_per_window) // cells_per_step
	
	# Compute individual channel HOG features for the entire image
	hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
	hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
	hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

	for xb in range(nxsteps):
		for yb in range(nysteps):
			ypos = yb * cells_per_step
			xpos = xb * cells_per_step

			# Extract HOG for this patch
			hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel() 
			hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel() 
			hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel() 
			hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

			xleft = xpos * pix_per_cell
			ytop = ypos * pix_per_cell

			# Extract the image patch
			subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64,64))
			
			# Get color features
			spatial_features = bin_spatial(subimg, size=spatial_size)
			hist_features = color_hist(subimg, nbins=hist_bins)

			#svr = SVC()
			parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}

			#clf = GridSearchCV(svr, parameters)

			# Scale features and make a prediction
			features = []
			features.append(spatial_features)
			features.append(hist_features)
			features.append(hog_features)

			#X = np.hstack((spatial_features, hist_features, hog_features))
			test_features = X_scaler.transform(np.array(np.concatenate(features)).reshape(1, -1))
			test_prediction = clf.predict(test_features)
			
			#print(test_prediction)

			if test_prediction == 1:
				xbox_left = np.int(xleft * scale)
				ytop_draw = np.int(ytop * scale)
				win_draw = np.int(window * scale)
				cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0,0,255), 6) 
				
	mpimg.imsave("out.png", draw_img)
	return draw_img

images = glob.glob('training_imgs/*.png')
cars = []
notcars = []
for image in images:
	if 'image' in image or 'extra' in image:
		notcars.append(image)
	else:
		cars.append(image)

cars = cars[0:2000]
notcars = notcars[0:2000]
car_features = extract_features(cars)
notcar_features = extract_features(notcars)

X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

clf = SVC()
clf.fit(scaled_X, y)
# Visualize the heatmap when displaying    
#heatmap = np.clip(heat, 0, 255)

# Find final boxes from heatmap using label function
#labels = label(heatmap)
#draw_img = draw_labeled_bboxes(np.copy(image), labels)

white_output = './test_images/project_video.mp4'

clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)