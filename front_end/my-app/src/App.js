import './App.css';
import React from "react";
import {Route, Switch} from "react-router-dom";
import { Redirect } from 'react-router';
import Home from "./Home";


function App() {
  return (
    <Switch>
       <Redirect exact from="/" to="home/Detection"/>
       <Route exact path="/home/:page?" render={props => <Home {...props} />}/> 
    </Switch>
  );
}

export default App;
