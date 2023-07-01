import React from "react";
import PropTypes from "prop-types";

export const YoutubeEmbed = ({ embedId }) => (
  <div className="video-responsive">
    <iframe
      width="853"
      height="480"
      src={`https://www.youtube.com/watch?v=rg5Hwety4RU&pp=ygUIcG90aG9sZXM%3D`}
      frameBorder="0"
      allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
      allowFullScreen
      title="Embedded youtube"
    />
   
  </div>
);



