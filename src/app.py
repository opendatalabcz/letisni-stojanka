import cv2
import model
import constants


IMG_W = 1278
IMG_H = 720


CONFIG_PATH = "../model/yolo-obj.cfg"
WEIGHTS_PATH = "../model/yolo-obj_last.weights"
LABELS_PATH = "../model/obj.names"
TEST_FRAMES = "/home/oliver/School/THESIS/data/hong_kong/test_frames"
TRAINING_DATA_PATH = "../model/train.txt"
TEST_DATA_PATH = "../model/test.txt"

#TEST_VIDEO_PATH = "/home/oliver/School/THESIS/data/test_videos/hong_kong_train.mp4"
#TEST_VIDEO_PATH = "/home/oliver/School/THESIS/data/japan/Data/japan_letiste/raw_mp4/japan_test.mp4"
TEST_VIDEO_PATH = "/home/oliver/School/THESIS/data/japan_2_batch/chosen_test.mp4"
#TEST_VIDEO_PATH = "/home/oliver/School/THESIS/data/japan_2_batch/chosen_test2.mp4"
#TEST_VIDEO_PATH = "/home/oliver/School/THESIS/data/test_videos/hong_kong_train.mp4"
#TEST_VIDEO_PATH = "/home/oliver/School/THESIS/data/test_videos/japan_test_3.mp4"
OUTPUT_TEST_VIDEO = "/home/oliver/School/THESIS/data/test_videos/japan_test3_output.mp4"

class App:
    def __init__(self, model, video_path):
        self.model = model
        self.video_path = video_path
        self.stream = cv2.VideoCapture(video_path)
        self.buffer = []
        for i in range(1000):
            self.buffer.append(None)
        self.shown_frames = 0
    
    
    def run_batch(self):
        buffer = []
        c = 0
        while c <= 100:
            (grabbed, frame) = self.stream.read()

            if not grabbed:
                break

            if c % 50 == 0:
                In = self.model.inference_img(frame, "frame")
            
            In.img = frame
            In.show(False)

            color = (34,139,34)
            cv2.putText(In.img, "frame {}".format(self.shown_frames), (50, 50),                  
            cv2.FONT_HERSHEY_SIMPLEX,1.5, color, 2)

            buffer.append(In.img)
            c+=1
            self.shown_frames += 1    

        return buffer
    
    def run(self):
        stop = False
        while not stop:
            #run for 100 frames
            buffer = self.run_batch()
            for i in range(len(buffer)):
                cv2.imshow('Frame',buffer[i])
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    stop = True
        self.stream.release()

if __name__ == "__main__":
    m = model.Model(CONFIG_PATH, WEIGHTS_PATH, LABELS_PATH)
    app = App(m, TEST_VIDEO_PATH)
    app.run()