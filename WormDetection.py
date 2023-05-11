import torch
import cv2

# How to capture data from webcam on Windows
#cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
#num_frames_to_record = 100
#fps_for_vid_output = 2

# How to capture data from video file
cap = cv2.VideoCapture("CFL3.mp4") # Capture from file
num_frames_to_record = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # record all frames from video
fps_for_vid_output = int(cap.get(cv2.CAP_PROP_FPS)) # match output fps to input video fps

# Make sure the camera is available
assert cap.isOpened()

# cuda means we will use GPU, otherwise we will be using CPU 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.set_printoptions(sci_mode=False)

# Load our model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='worm_best_050123.pt', force_reload=True)
model.to(device)

window_width = 800
window_height = 800

output_video = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps_for_vid_output, (window_width, window_height))

while num_frames_to_record > 0:
    # Get the current frame
    ret, frame = cap.read()
    assert ret
    
    frame = cv2.resize(frame, (window_width, window_height))

    # YOLOv5 uses RGB colour format for their images whereas OpenCV uses BGR, so we need to convert
    # between them to ensure we get the right results.
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Red colour in BGR format
    red_colour = (0, 0, 255)
    thickness = 2

    # Draw a rectangle around every detection in the results
    for result in results.xyxy[0]:
      x1 = result[0]
      y1 = result[1]
      x2 = result[2]
      y2 = result[3]
      confidence = result[4]

      start_point = (int(x1), int(y1))
      end_point = (int(x2), int(y2))
      
      cv2.rectangle(frame, start_point, end_point, red_colour, thickness)

    #cv2.imshow('Worm Detection', frame)

    # waitKey() is needed otherwise imshow() does not work correctly
    cv2.waitKey(5)    

    # Write the current frame
    output_video.write(frame)

    num_frames_to_record = num_frames_to_record - 1

cap.release()
output_video.release() # save video

cv2.destroyAllWindows()