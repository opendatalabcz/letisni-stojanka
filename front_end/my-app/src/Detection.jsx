import React from "react";
import VideoPlayer from "react-video-js-player"
import Video from "./video/res.mp4"
import "./Detection.css";

export default function Detection() {
    const srcVideo = Video;
    return (
        <div className="div-video">
            <VideoPlayer
                src ={srcVideo}
                type="video/mp4"
                width ="1280"
                height ="720"
                playbackRates={[1, 3.85, 8, 16]}
            />
        </div>
    )
}