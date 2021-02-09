import React from "react";
import {Tabs, Tab, AppBar} from "@material-ui/core";
import Dashboard from "./Dashboard"
import Detection from "./Detection"; 


const Home = props => {
    const { match, history } = props;
    const { params } = match;
    const { page } = params;
    
    const tabNameToIndex = {
        0: "Detection",
        1: "Dashboard"
    }

    const indexToTabName = {
        "Detection": 0,
        "Dashboard": 1
    }

    const [selectedTab, setSelectedTab] = React.useState(indexToTabName[page])

    const handleChange = (event, newValue) => {
        history.push(`/home/${tabNameToIndex[newValue]}`);
        setSelectedTab(newValue);
    }

    return (
      <>
        <AppBar>
          <Tabs value={selectedTab} onChange={handleChange}>
            <Tab label="Detection" />
            <Tab label="Dashboard" />
          </Tabs>
        </AppBar>
      {selectedTab === 0 && <Detection/>}
      {selectedTab === 1 && <Dashboard/>}
      </>
    );
}

export default Home;
    