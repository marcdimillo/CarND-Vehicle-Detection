import numpy as np
import pickle
import cv2
import glob

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from skimage.feature import hog

from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

def extract_features(imgs):
	# Create a list to append feature vectors to
	features = []
	# Iterate through the list of images
	for file in imgs:
		file_features = []

		# Read in each one by one
		image = cv2.imread(file)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		feature_image = np.copy(image)      

		spatial_features = bin_spatial(feature_image, size=(32, 32))
		file_features.append(spatial_features)
		
		# Apply color_hist()
		hist_features = color_hist(feature_image, nbins=32)
		file_features.append(hist_features)

		# Call get_hog_features() with vis=False, feature_vec=True
		hog_features = []
		for channel in range(feature_image.shape[2]):
			hog_features.append(get_hog_features(feature_image[:, :, channel], 
								9, 8, 2, 
								vis=False, feature_vec=True))
		hog_features = np.ravel(hog_features)        
		
		# Append the new feature vector to the features list
		file_features.append(hog_features)

		features.append(np.concatenate(file_features))

	# Return list of feature vectors
	return features

def process_image(input_img):
	orient = 9  # HOG orientations
	pix_per_cell = 8 # HOG pixels per cell
	cell_per_block = 2 # HOG cells per block
	spatial_size = (32, 32) # Spatial binning dimensions
	hist_bins = 32    # Number of histogram bins

	hot_windows = []

	img = np.copy(input_img)
	img = img.astype(np.float32) / 255.0
	windows1 = slide_window(img, x_start_stop=[700, 1280], y_start_stop=[400, 600], xy_window=(96, 96), xy_overlap=(0.75, 0.75))
	hot_windows += search_windows(img, windows1, X_scaler)

	#windows1 = slide_window(img, y_start_stop=[380, 500], xy_window=(240, 240), xy_overlap=(0.75, 0.75))
	#hot_windows += search_windows(img, windows1, X_scaler)

	#windows2 = slide_window(img, y_start_stop=[380, 470], xy_window=(180, 180), xy_overlap=(0.75, 0.75))
	#hot_windows += search_windows(img, windows2, X_scaler)

	#windows3 = slide_window(img, y_start_stop=[395, 455], xy_window=(120, 120), xy_overlap=(0.75, 0.75))
	#hot_windows += search_windows(img, windows3, X_scaler)

	#windows4 = slide_window(img, y_start_stop=[405, 440], xy_window=(70, 70), xy_overlap=(0.75, 0.75))
	#hot_windows += search_windows(img, windows4, X_scaler)
	
	heatmap = np.zeros_like(img[:, :, 0]).astype(np.float)
	heatmap = add_heat(heatmap, hot_windows)

	heatmaps.insert(0, heatmap)
	
	if(len(heatmaps) > 25):
		heatmaps.pop()
	
	all_frames = np.array(heatmaps)
	heatmap_sum = np.sum(all_frames, axis=0)

	heatmap = apply_threshold(heatmap, 15)

	labels = label(heatmap_sum)

	cv2.imwrite("heatmap.jpg", heatmap)
	
	window_image = draw_labeled_bboxes(input_img, labels)

	output_image = cv2.cvtColor(window_image, cv2.COLOR_BGR2RGB)
	cv2.imwrite("out.jpg", output_image)
	return window_image

def add_heat(heatmap, bbox_list):
	if bbox_list:		# Iterate through list of bboxes
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

def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
						hist_bins=32, orient=9, 
						pix_per_cell=8, cell_per_block=2, hog_channel=0,
						spatial_feat=True, hist_feat=True, hog_feat=True):    
	#1) Define an empty list to receive features
	img_features = []

	#2) Apply color conversion if other than 'RGB'
	feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

	#3) Compute spatial features if flag is set
	spatial_features = bin_spatial(feature_image, size=spatial_size)

	#4) Append features to list
	img_features.append(spatial_features)

	#5) Compute histogram features if flag is set
	hist_features = color_hist(feature_image, nbins=hist_bins)

	#6) Append features to list
	img_features.append(hist_features)

	#7) Compute HOG features if flag is set
	#if hog_channel == 'ALL':
	hog_features = []
	for channel in range(feature_image.shape[2]):
		hog_features.extend(get_hog_features(feature_image[:,:,channel], 
							orient, pix_per_cell, cell_per_block, 
								vis=False, feature_vec=True))      
	#else:
		#hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
		#			pix_per_cell, cell_per_block, vis=False, feature_vec=True)

	#8) Append features to list
	img_features.append(hog_features)

	#9) Return concatenated array of features
	return np.concatenate(img_features)

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, scaler, color_space='RGB', 
					spatial_size=(32, 32), hist_bins=32, 
					hist_range=(0, 256), orient=9, 
					pix_per_cell=8, cell_per_block=2, 
					hog_channel='ALL'):

	#1) Create an empty list to receive positive detection windows
	on_windows = []

	#2) Iterate over all windows in the list
	for window in windows:
		#3) Extract the test window from original image
		#print(window[0][1])
		#print(window[0][0])
		#print(window[1][0])
		#print(window[1][1])
		test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
		
		#4) Extract features for that window using single_img_features()
		features = single_img_features(test_img, color_space=color_space, 
							spatial_size=spatial_size, hist_bins=hist_bins, 
							orient=orient, pix_per_cell=pix_per_cell, 
							cell_per_block=cell_per_block, 
							hog_channel=hog_channel)

		#5) Scale extracted features to be fed to classifier
		test_features = scaler.transform(np.array(features).astype(np.float64).reshape(1, -1))
		
		#6) Predict using your classifier
		prediction = clf.predict(test_features)
		
		#7) If positive (prediction == 1) then save the window
		if prediction == 1:
			on_windows.append(window)

	#8) Return windows for positive detections
	return on_windows

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
					xy_window=(64, 64), xy_overlap=(0.5, 0.5)):

	# If x and/or y start/stop positions not defined, set to image size
	if x_start_stop[0] == None:
		x_start_stop[0] = 0
	if x_start_stop[1] == None:
		x_start_stop[1] = img.shape[1]
	if y_start_stop[0] == None:
		y_start_stop[0] = 0
	if y_start_stop[1] == None:
		y_start_stop[1] = img.shape[0]

	# Compute the span of the region to be searched    
	xspan = np.int(x_start_stop[1] - x_start_stop[0])
	yspan = np.int(y_start_stop[1] - y_start_stop[0])

	# Compute the number of pixels per step in x/y
	nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
	ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))

	# Compute the number of windows in x/y
	nx_windows = np.int(xspan/nx_pix_per_step) - 1
	ny_windows = np.int(yspan/ny_pix_per_step) - 1

	# Initialize a list to append window positions to
	window_list = []

	# Loop through finding x and y window positions
	# Note: you could vectorize this step, but in practice
	# you'll be considering windows one by one with your
	# classifier, so looping makes sense
	for ys in range(ny_windows):
		for xs in range(nx_windows):
			# Calculate window position
			startx = np.int(xs*nx_pix_per_step + x_start_stop[0])
			endx = startx + xy_window[0]
			starty = ys*ny_pix_per_step + y_start_stop[0]
			endy = starty + xy_window[1]
			
			# Append window position to list
			window_list.append(((startx, starty), (endx, endy)))

	# Return the list of windows
	return window_list

images = glob.glob('training_imgs/*.png')
cars = []
notcars = []
for image in images:
	if 'image' in image or 'extra' in image:
		notcars.append(image)
	else:
		cars.append(image)

heatmaps = []

car_features = extract_features(cars)
notcar_features = extract_features(notcars)

X = np.vstack((car_features, notcar_features)).astype(np.float64)

# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)

# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

clf = LinearSVC()
clf.fit(scaled_X, y)
# Visualize the heatmap when displaying    
#heatmap = np.clip(heat, 0, 255)

# Find final boxes from heatmap using label function
#labels = label(heatmap)
#draw_img = draw_labeled_bboxes(np.copy(image), labels)

white_output = './test_images/project_video.mp4'

clip1 = VideoFileClip("project_video.mp4").subclip(15, 19)
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)