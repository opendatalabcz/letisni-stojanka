import cv2
import model
import constants
import argparse
import imutils
import object_tracking
import time
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import os

class App:
    """Class representing the key object airport apron detector application"""
    def __init__(self):
        self.args = self.parse_args()
        self.model = self.load_model()
        self.video_path = self.args.input
        self.stream = cv2.VideoCapture(self.video_path)
        self.shown_frames = 0
        self.tracker = object_tracking.Tracker()
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(constants.OUT_PATH + "res2.mp4", self.fourcc, 30,(constants.IMG_W, constants.IMG_H), True)
        
    def parse_args(self):
        """Parsing the input arguments."""
        parser = argparse.ArgumentParser(description = "Key object airport apron detector")
        parser.add_argument("-i", "--input", help="path to the input video file", required=True)
        parser.add_argument("-m", "--model", help="trained model to be used, options: tiny/yolov4", default="yolov4")
        parser.add_argument("-f", "--freq", help="inference will be performed only on every n-th frame for speed improvement", type=int, default=1)
        

        
        return parser.parse_args()

    def load_model(self):
        """Loads the chosen trained model by user"""
        if self.args.model == "yolov4":
            m = model.Model(constants.YOLOV4_CONFIG_PATH, constants.YOLOV4_WEIGHTS_PATH, constants.YOLO_V4_LABELS_PATH)
        elif self.args.model == "tiny":
            m = model.Model(constants.TINY_CONFIG_PATH, constants.TINY_WEIGHTS_PATH, constants.TINY_LABELS_PATH)
        else:
            raise ValueError("Input model not recognized.")
        return m

    def draw_frame_num(self, frame):
        """Draws current frame number to the given frame."""
        #green
        color = (34,139,34)
        cv2.putText(frame, "frame {}".format(self.shown_frames), (50, 50),            
                    cv2.FONT_HERSHEY_SIMPLEX,1.5, color, 2)
        return frame

    def total_frames(self):
        """Returns the number of total frames in the video."""
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
        total = int(self.stream.get(prop))
        return total
    
    def print_time(self, start, end, total_frames):
        """Outputs info about video and estimated inference time."""
        elap = (end - start)
        if self.args.model == "tiny":
            mod = "YoloV4 Tiny"
        else:
            mod = "YoloV4"

        if self.args.freq == 1:
            skip = "[INFO] Running inference on every frame"
        else:
            skip = "[INFO] Running inference on every {}-th frame".format(self.args.freq)

        print(skip)
        print("[INFO] loaded model {}".format(mod))
        print("[INFO] frame width - {} | frame height - {} | FPS: {}".format(self.stream.get(3),self.stream.get(4),self.stream.get(5)))
        print("[INFO] processing one frame in {:.4f} seconds".format(elap))
        print("[INFO] estimated total time to finish: {:.4f} seconds".format((elap * total_frames)/self.args.freq)) 

    def print_final_info(self, elap):
        """Final statistics."""
        print("[INFO] video processed in: {:.4f} minutes".format(elap/60.0))
        print("[INFO] saving output...")
        print("[INFO] visualize results by running: cd ../front_end/my-app ; npm start")
        
    def change_codec(self):
        """Changes codec of the final video so it can be rendered by web"""
        in_file = constants.OUT_PATH + "res2.mp4"
        out_file = constants.OUT_PATH + "res.mp4"

        os.system("yes 2>/dev/null | ffmpeg -i {} -vcodec libx264 {} > /dev/null 2>&1".format(in_file, out_file))

    def run(self):
        """Main loop of the application"""
        run_start = time.time()
        skip = self.args.freq
        c = 0 
        
        total_frames = self.total_frames()
        info_printed = False
        
        while True:
            start = time.time()
            bboxes = []
            labels = []
            (grabbed, frame) = self.stream.read()

            if not grabbed:
                break
            
            #running inference only on every n-th frame for speed improvement, given by user
            if c % skip == 0:
                In = self.model.inference_img(frame, "frame")
            else:
                #current frame with previous detections
                In.img = frame
            
            for d in In.detections:
                start_x, start_y, w, h = d.bbox.unwrap()
                bboxes.append([start_x, start_y, start_x + w, start_y + h])
                labels.append(d.label)

            
            #save detections
            In.show(False)

            In.img = self.draw_frame_num(In.img)
            
            #object tracking visualization
            objects = self.tracker.update(bboxes, labels)
            for(object_id, centroid) in objects.items():
                text = "ID {}".format(object_id)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            #save to output video file
            self.writer.write(In.img)
            self.shown_frames += 1 
            end = time.time()
            c += 1
            
            #print info anout inference time only at the begining
            if not info_printed:
                self.print_time(start, end, total_frames)
                info_printed = True

        self.tracker.end_vp_tracking()
        cv2.destroyAllWindows()
        self.stream.release()
        self.writer.release()
        run_end = time.time()
        self.change_codec()
        self.print_final_info(run_end - run_start)

    def visualize_timelime(self):
        print("[INFO] generating dashboard.")
        lfrom = []
        lto = []
        lid = []

        #get the data from the tracker
        dps = self.tracker.vis_data_points
        for id, dp in dps.items():
            for app in dp.appereances:
                lfrom.append(app[0]/30)
                lto.append(app[1]/30)
                lid.append(str(dp.label))
        
        #timeline chart
        df = pd.DataFrame(list(zip(lfrom, lto, lid)), columns=["from", "to", "label"])
        chart = alt.Chart(df).mark_bar().encode(alt.X("from", title="Timeline(seconds)"), alt.X2("to", title = ""), y="label", color=alt.Color("label", \
                                                scale=alt.Scale(scheme='dark2')))
        chart.save(constants.OUT_PATH + "chart_timeline.png")


if __name__ == "__main__":
    try:
        app = App()
        app.run()
        app.visualize_timelime()
    except ValueError as err:
        print(constants.HELP_MSG)
    