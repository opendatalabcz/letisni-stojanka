import React from "react";
import Chart from "./video/chart_timeline.png"

const div_styles = {
    height:"100%",
    width:"100%"
}

const img_styles = {
    position:"absolute",
    top:0,
    left:0,
    right:0,
    bottom:0,
    margin:"auto",
    width:"50%",
    height:"50%"
}



const Dashboard = () => {
    return (
        <div style={div_styles}>
            <img src={Chart} alt="Timeline graph" style={img_styles}/>
        </div>
    );
}

export default Dashboard;