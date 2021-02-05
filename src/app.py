import cv2
import model
import constants
import argparse
import imutils
import object_tracking
import time

class App:
    """Class representing the key object airport apron detector application"""
    def __init__(self, model):
        self.args = self.parse_args()
        self.model = model
        self.video_path = self.args.input
        self.stream = cv2.VideoCapture(self.video_path)
        self.shown_frames = 0
        self.tracker = object_tracking.Tracker()
        self.fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.writer = cv2.VideoWriter("./output.avi", self.fourcc, 30,(constants.IMG_W, constants.IMG_H), True)

    def parse_args(self):
        """Parsing the input arguments."""
        parser = argparse.ArgumentParser(description = "Key object airport apron detector application app")
        parser.add_argument("-i", "--input", help="path to the input video file", required=True)
        parser.add_argument("-c", "--confidence", help="confidence threshold for predicting", type=float, default=0.5)
        return parser.parse_args()


    def draw_frame_num(self, frame):
        """Draws current frame number to the given frame"""
        color = (34,139,34)
        cv2.putText(frame, "frame {}".format(self.shown_frames), (50, 50),            
                    cv2.FONT_HERSHEY_SIMPLEX,1.5, color, 2)
        return frame

    def total_frames(self):
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
        total = int(self.stream.get(prop))
        return total
    
    def print_time(self, start, end, total_frames):
        elap = (end - start)
        print("[INFO] single frame took {:.4f} seconds".format(elap))
        print("[INFO] estimated total time to finish: {:.4f}".format(elap * total_frames)) 



    def run(self):
        total_frames = self.total_frames()
        info_printed = False
        while True:
            start = time.time()
            bboxes = []
            (grabbed, frame) = self.stream.read()

            if not grabbed:
                break

            In = self.model.inference_img(frame, "frame")
            for d in In.detections:
                start_x, start_y, w, h = d.bbox.unwrap()
                bboxes.append([start_x, start_y, start_x + w, start_y + h])
            In.show(False)
            In.img = self.draw_frame_num(In.img)
            

            objects = self.tracker.update(bboxes)
            for(object_id, centroid) in objects.items():
                text = "ID {}".format(object_id)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            #cv2.imshow("frame",In.img)
            #key = cv2.waitKey(1) & 0xFF
	        
            #if key == ord("q"):
            #    break
            
            self.writer.write(In.img)
            self.shown_frames += 1 
            end = time.time()
            
            if not info_printed:
                self.print_time(start, end, total_frames)
                print('Width = ', self.stream.get(3),' Height = ', self.stream.get(4),' fps = ', self.stream.get(5))
                info_printed = True

        cv2.destroyAllWindows()
        self.stream.release()
        self.writer.release()

if __name__ == "__main__":
    m = model.Model(constants.CONFIG_PATH, constants.WEIGHTS_PATH, constants.LABELS_PATH)
    app = App(m)
    app.run()