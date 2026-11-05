import { useEffect, useRef, useState } from "react";
import createPlotlyComponent from "react-plotly.js/factory";
import type { PlotParams } from "react-plotly.js";
import Plotly from "plotly.js-dist-min";

const Plot = createPlotlyComponent(Plotly);

/** Plotly's window resize handler misses changes caused by the adjacent panes. */
export function ResponsivePlot(props: PlotParams) {
  const container = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState<number>();

  useEffect(() => {
    const element = container.current;
    if (!element) return;
    let frame = 0;
    const observer = new ResizeObserver(() => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        if (element.clientWidth > 0) setWidth(element.clientWidth);
      });
    });
    observer.observe(element);
    return () => {
      observer.disconnect();
      cancelAnimationFrame(frame);
    };
  }, []);

  return <div ref={container} style={{ width: "100%", minWidth: 0 }}>
    <Plot {...props} layout={{ ...props.layout, autosize: true, ...(width === undefined ? {} : { width }) }} />
  </div>;
}
