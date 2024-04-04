import cv2

def extract_frames(video_directory, output_dir):
    # Open the video file
    total_frame_count = 0
    frame_count = [0] * 12
    for i in range(1, 12) :
        video_capture = cv2.VideoCapture(video_directory + f"\\v{i:03d}.mp4")
        print(video_directory + f"\\v00{i}.mp4")
        # Check if the video file opened successfully
        if not video_capture.isOpened():
            print("Error: Unable to open video file.")
            return

        # Read the video frame by frame
        success, frame = video_capture.read()
        target_width = 1920
        target_height = 1080
    
        while success:
            # Save the frame as a JPEG image
            if(frame_count[i] % 15 == 0) :
                resized_frame = cv2.resize(frame, (target_width, target_height))
                frame_filename = f"{output_dir}/{int(total_frame_count / 15):06d}.jpg"
            cv2.imwrite(frame_filename, resized_frame)  # Save frame as JPEG
            frame_count[i] += 1
            total_frame_count +=1
            # Read the next frame
            success, frame = video_capture.read()

        # Release the video capture object
        video_capture.release()

        print(f"Extracted {frame_count[i]} frames successfully.")
    print(f"total number of extracted frames : {total_frame_count}")
# Example usage
if __name__ == "__main__":
    video_directory = "D:\cap_video_source"  # Path to your input video file
    output_directory = "D:\captured_pics"  # Directory where frames will be saved
    extract_frames(video_directory, output_directory)